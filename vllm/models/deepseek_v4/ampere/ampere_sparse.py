# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA attention for SM8x (Ampere: A100/A800).

Reuses the ROCm Triton sparse-MLA implementation wholesale: its kernels,
ragged metadata builders, and bf16 o_proj reference path are plain
Triton/torch (the aiter-only preshuffle GEMMs self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.

DCP (``VLLM_SM86_DCP`` + ``decode_context_parallel_size > 1``): the decode
path of compressed layers (C4A/C128A) runs cross-shard attention.  Every DCP
rank all-gathers the group's query heads, attends its LOCAL compressed-KV
shard (plus, on exactly one owner rank per query, the replicated SWA
window), and emits raw PRE-sink partials ``(o, m, l)``.  The partials are
merged in fp32 with a fixed rank order via the a2a LSE reduce, the attention
sink is applied exactly once at the global max, and inverse RoPE follows
later in ``_o_proj`` — after the merge (ARCHITECTURE.md section 10 rules
1/2/9/10, section 6).

Compressed-layer PREFILL (P2d) takes the opposite shape: the compressed
prefix is tiny, so each rank all-gathers the raw entry bytes of every shard
into a dense GLOBAL-entry-order buffer and runs the unchanged single-softmax
prefill pipeline against it (no output merge; replicated compute across the
DCP group).  With the gate unset or dcp == 1 every path below falls through
to the unchanged parent implementation.
"""

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator, get_dcp_group
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterMLASparseMetadata,
    DeepseekV4ROCMAiterSparseSWAMetadata,
    combine_topk_swa_indices,
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.models.deepseek_v4.ampere.dcp_delta_tracker import (
    Sm86DcpDeltaTracker,
)
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
    sm86_dcp_allgather_k_entries,
    sm86_dcp_delta_gather_k_entries,
    sm86_dcp_identity_block_table,
    sm86_pack_swa_window_entries,
)
from vllm.models.deepseek_v4.common.ops.dcp import (
    dcp_merge_flashmla_output,
    softmax_stats_to_lse,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


def _maybe_gather_dcp_q(
    layer: "DeepseekV4AmpereMLAAttention",
    q: torch.Tensor,
) -> tuple[torch.Tensor, int, "GroupCoordinator", bool]:
    """All-gather the real query heads across the DCP group.

    Mirrors the Lasimeri reference (``_maybe_gather_dcp_q``): TP shards the
    query heads while DCP shards the compressed KV *within* the TP group, so
    each DCP rank must attend its KV shard with ALL query heads of the
    group; the LSE merge afterwards redistributes each head's output back to
    its owning rank.  The Q all-gather IS required in this base — same TP
    head layout as the reference (``n_local_heads = n_heads // tp_size``,
    and the Ampere path pads no heads, so the slice is exact).
    """
    dcp_group = get_dcp_group()
    if dcp_group.world_size == 1:
        return q, layer.n_local_heads, dcp_group, False
    q = dcp_group.all_gather(q[:, : layer.n_local_heads, :].contiguous(), dim=1)
    return q, q.shape[1], dcp_group, True


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """SM8x DeepSeek V4 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV4AmpereMLASparseBackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parallel_config = get_current_vllm_config().parallel_config
        self._dcp_size = parallel_config.decode_context_parallel_size
        self._cp_interleave = parallel_config.cp_kv_cache_interleave_size
        if (
            envs.VLLM_SM86_DCP
            and self._dcp_size > 1
            and parallel_config.dcp_comm_backend != "a2a"
        ):
            # Reference behavior (Lasimeri).  The a2a merge reduces in fp32
            # with a fixed rank order inside a single Triton kernel
            # (ARCHITECTURE.md section 10 rule 2); the ag_rs path
            # reduce-scatters partial outputs in bf16 through NCCL, which
            # breaks the fp32 deterministic-merge rule.
            raise ValueError("DeepseekV4 Ampere DCP requires dcp_comm_backend='a2a'.")
        # Token positions of the in-flight forward, stashed by forward_mqa
        # for the DCP decode branch (SWA/owner-rank selection needs the
        # absolute query position; `_forward_decode` does not receive it).
        self._dcp_positions: torch.Tensor | None = None
        # P7 (VLLM_DSV4_DELTA_GATHER): per-LAYER persistent staging tracker
        # for the delta compressed-entry gather (this module IS one layer,
        # so layer identity is implicit; under PP each stage only tracks its
        # own layers). Created lazily on the first delta-eligible prefill.
        self._delta_gather_tracker: "Sm86DcpDeltaTracker | None" = None

    def _dcp_group_or_none(self) -> "GroupCoordinator | None":
        """The DCP group when the SM86 DCP path is active, else None."""
        if not (envs.VLLM_SM86_DCP and self._dcp_size > 1):
            return None
        dcp_group = get_dcp_group()
        return dcp_group if dcp_group.world_size > 1 else None

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if not (envs.VLLM_SM86_DCP and self._dcp_size > 1):
            super().forward_mqa(q, kv, positions, output)
            return
        self._dcp_positions = positions
        try:
            super().forward_mqa(q, kv, positions, output)
        finally:
            self._dcp_positions = None

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        dcp_group = None if attn_metadata is None else self._dcp_group_or_none()
        if dcp_group is None:
            # Default path — gate unset, dcp == 1, or a SWA-only layer
            # (attn_metadata is None): the SWA ring is replicated
            # (dcp_exempt), so SWA-only layers prefill normally.
            super()._forward_prefill(
                q=q,
                positions=positions,
                compressed_k_cache=compressed_k_cache,
                swa_k_cache=swa_k_cache,
                output=output,
                attn_metadata=attn_metadata,
                swa_metadata=swa_metadata,
            )
            return
        self._forward_prefill_dcp(
            q=q,
            compressed_k_cache=compressed_k_cache,
            swa_k_cache=swa_k_cache,
            output=output,
            attn_metadata=attn_metadata,
            swa_metadata=swa_metadata,
            dcp_group=dcp_group,
        )

    def _forward_prefill_dcp(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        dcp_group: "GroupCoordinator",
    ) -> None:
        """DCP compressed-layer prefill (P2d W1): gather, then run unchanged.

        The compressed prefix is tiny (584 B/entry C4A — a 128K-token prefix
        is ~19 MB/layer; C128A is 32x smaller), so instead of local-shard
        attention + LSE merge (the Lasimeri approach, which additionally
        requires the sharded-SWA index filtering that P1's replicated ring
        makes impossible) each rank all-gathers the raw fp8 entries + UE8M0
        scales of every shard over the DCP group, reordered into GLOBAL
        entry order via the shared inverse layout formula (see
        ``dequantize_and_gather_k_cache``'s DCP branch), and then runs the
        EXISTING non-DCP prefill pipeline against the dense buffer:

        - ``combine_topk_swa_indices`` consumes the GLOBAL entry indices the
          P2b indexer prefill merge already emits (C4A) or the identity
          ``[0..n-1]`` rows of the C128A prefill metadata — both index the
          dense buffer directly, so it needs no CP branch;
        - the replicated SWA gather and window arithmetic are unchanged
          (rule 7);
        - ``rocm_sparse_attn_prefill`` computes ONE full softmax per query
          row over full information on every rank, so the attention sink is
          applied exactly once per output (rule 1) and no cross-rank output
          merge exists; each rank keeps its own TP heads (replicated
          compute across the DCP group, identical by construction).

        The body below is the parent ``_forward_prefill`` with only the
        compressed-cache gather call changed (DCP kwargs); global absolute
        positions/entry indices flow through untouched (rule 6), and the
        bytes are dequantized exactly once in the existing kernel (rule 8).
        """
        if torch.cuda.is_current_stream_capturing():
            # The ONE narrow eager-only guard left after P2f.  This is not
            # conservatism: the per-chunk entry bound below is a genuine host
            # sync (`seq_lens_cpu[...].max().item()`) and it SIZES the DCP
            # all-gather, so both the launch shape and the collective payload
            # are data-dependent — neither can be baked into a graph.  Decode
            # (the path that matters for throughput) is capture-safe; run
            # cudagraph_mode=FULL_DECODE_ONLY, which never captures prefill.
            raise RuntimeError(
                "VLLM_SM86_DCP compressed-layer prefill is eager-only (the "
                "per-chunk staging bound is a host sync that sizes the DCP "
                "all-gather). Use cudagraph_mode=FULL_DECODE_ONLY; DCP decode "
                "is capture-safe as of P2f."
            )
        assert compressed_k_cache is not None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None
        # CPU twin for the chunk-max entry count (rank-invariant; avoids a
        # GPU sync per chunk).
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        assert seq_lens_cpu is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            topk_indices = topk_indices[:num_prefill_tokens]
        else:
            topk_indices = attn_metadata.c128a_prefill_topk_indices
        assert topk_indices is not None
        top_k = topk_indices.shape[-1]
        N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        # P7: the per-layer delta tracker, or None -> full re-gather per
        # chunk (flag off, or no request identities: dummy/warmup runs).
        # Rank-invariant either way, so the collective plans stay symmetric.
        delta_tracker = self._maybe_delta_tracker(
            swa_metadata, compressed_k_cache.device
        )

        if envs.VLLM_DSV4_FLASH_PREFILL:
            # P6: same chunking, same metadata, but the dequant workspace,
            # combine_topk_swa_indices and the Triton prefill kernel are
            # replaced by one fused flash-mla op per chunk. The compressed
            # pack + all-gather is IDENTICAL to the path below (shared
            # helper); only the dequant is skipped -- the op dequantizes the
            # staging bytes in-kernel, exactly once (rule 8).
            self._forward_prefill_dcp_flash(
                q=q,
                compressed_k_cache=compressed_k_cache,
                swa_k_cache=swa_k_cache,
                output=output,
                attn_metadata=attn_metadata,
                swa_metadata=swa_metadata,
                dcp_group=dcp_group,
                topk_indices=topk_indices,
                num_chunks=num_chunks,
                delta_tracker=delta_tracker,
            )
            return

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start

            block_table = attn_metadata.block_table[num_decodes:]
            if delta_tracker is None:
                # DCP delta vs the parent: the compressed cache holds only
                # this rank's shard; all-gather + reorder to GLOBAL entry
                # order inside the gather (chunk max entries from CPU seq
                # lens keeps the collective shape identical on every rank).
                max_entries = int(
                    seq_lens_cpu[chunk_start:chunk_end].max().item()
                ) // self.compress_ratio
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end]
                    // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                    dcp_group=dcp_group,
                    dcp_interleave=self._cp_interleave,
                    dcp_max_entries=max_entries,
                )
            else:
                # P7: gather only the NEW entries of tracked requests into
                # their persistent GLOBAL-order stagings, then dequantize
                # each request's prefix [0, n) straight out of its staging
                # (identity block table, block_size=1 -- same kernel, same
                # per-row 576+scales@576 layout). Untracked requests keep
                # the full re-gather (one subset collective per chunk).
                self._delta_fill_compressed_chunk(
                    kv=kv,
                    compressed_k_cache=compressed_k_cache,
                    block_table_chunk=block_table[chunk_start:chunk_end],
                    attn_metadata=attn_metadata,
                    swa_metadata=swa_metadata,
                    dcp_group=dcp_group,
                    delta_tracker=delta_tracker,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )

            # Replicated (dcp_exempt) SWA cache: unchanged parent code.
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )

    def _maybe_delta_tracker(
        self,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        device: torch.device,
    ) -> "Sm86DcpDeltaTracker | None":
        """The per-layer P7 delta tracker when active, else None; runs GC.

        Active iff VLLM_DSV4_DELTA_GATHER is set AND the builder supplied
        prefill request identities (it does not on dummy/warmup/capture
        runs -- those take the full re-gather path, which is always
        correct). GC frees every tracked id absent from this step's prefill
        rows: finish/abort/preemption manifest as absence, and a request
        that reached decode never prefills again, so freeing it early is a
        strict refinement of the absence rule. GC only runs here (eager
        prefill steps -- captured decode replays execute no Python), which
        is safe: staleness is caught structurally by the tracker's
        prefix-continuity check, and the budget is only ever consulted on
        prefill steps, after this GC.
        """
        if not envs.VLLM_DSV4_DELTA_GATHER:
            return None
        prefill_req_ids = swa_metadata.prefill_req_ids
        if prefill_req_ids is None:
            return None
        if self._delta_gather_tracker is None:
            self._delta_gather_tracker = Sm86DcpDeltaTracker(device)
        self._delta_gather_tracker.gc(prefill_req_ids)
        return self._delta_gather_tracker

    def _plan_delta_chunk(
        self,
        delta_tracker: "Sm86DcpDeltaTracker",
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        chunk_start: int,
        chunk_end: int,
    ) -> tuple[
        "list[tuple[int, int, int, torch.Tensor]]",
        "dict[int, tuple[int, torch.Tensor]]",
        "list[int]",
    ]:
        """Plan one chunk's delta gather. Pure host math (CPU lens are
        precise for prefill rows), identical on every DCP rank.

        Returns ``(jobs, tracked, fallback_rows)``:
          - ``jobs``: ``(chunk_row, prev, new, staging)`` for
            ``sm86_dcp_delta_gather_k_entries`` (only rows with a non-empty
            delta);
          - ``tracked``: ``{chunk_row: (new_entries, staging)}`` -- every
            row consumed from its persistent staging this chunk;
          - ``fallback_rows``: chunk-relative rows on the full re-gather
            path (untracked / budget-blocked / no identity).
        """
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        query_lens_cpu = swa_metadata.prefill_query_lens_cpu
        prefill_req_ids = swa_metadata.prefill_req_ids
        assert seq_lens_cpu is not None
        assert query_lens_cpu is not None
        assert prefill_req_ids is not None

        jobs: list[tuple[int, int, int, torch.Tensor]] = []
        tracked: dict[int, tuple[int, torch.Tensor]] = {}
        fallback_rows: list[int] = []
        for i in range(chunk_end - chunk_start):
            row = chunk_start + i
            seq_len = int(seq_lens_cpu[row])
            new_entries = seq_len // self.compress_ratio
            state = None
            if row < len(prefill_req_ids):
                prefix_tokens = seq_len - int(query_lens_cpu[row])
                state = delta_tracker.plan_request(
                    prefill_req_ids[row], prefix_tokens, new_entries
                )
            if state is None:
                fallback_rows.append(i)
                continue
            assert state.staging is not None
            if new_entries > state.upto:
                jobs.append((i, state.upto, new_entries, state.staging))
            tracked[i] = (new_entries, state.staging)
            # Advance even on an empty delta so the prefix-continuity chain
            # stays unbroken across chunks that complete no entry.
            delta_tracker.advance(state, new_entries, seq_len)
        return jobs, tracked, fallback_rows

    def _delta_prepare_chunk(
        self,
        compressed_k_cache: torch.Tensor,
        block_table_chunk: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        dcp_group: "GroupCoordinator",
        delta_tracker: "Sm86DcpDeltaTracker",
        chunk_start: int,
        chunk_end: int,
    ) -> tuple[
        "dict[int, tuple[int, torch.Tensor]]",
        "tuple[list[int], torch.Tensor | None, torch.Tensor | None, "
        "torch.Tensor, int, int] | None",
    ]:
        """Move one chunk's compressed bytes: delta for tracked requests,
        one subset full re-gather for the rest.

        Collective order is fixed (delta first, then the fallback subset)
        and both plans derive from rank-invariant host state, so every DCP
        rank issues identical, symmetric collectives.

        Returns ``(tracked, fallback_ctx)`` where ``fallback_ctx`` is None
        when no row fell back, else ``(fallback_rows, gathered_rows,
        virtual_block_table, sub_entry_lens, max_local, max_entries_sub)``
        with ``gathered_rows`` None when the fallback rows have no
        completed entries (skipped symmetrically, like the non-delta path).
        """
        seq_lens = swa_metadata.prefill_seq_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        assert seq_lens is not None and seq_lens_cpu is not None
        entry_block_size = attn_metadata.block_size // self.compress_ratio

        jobs, tracked, fallback_rows = self._plan_delta_chunk(
            delta_tracker, swa_metadata, chunk_start, chunk_end
        )
        if jobs:
            sm86_dcp_delta_gather_k_entries(
                compressed_k_cache,
                block_table_chunk,
                entry_block_size,
                dcp_group,
                self._cp_interleave,
                jobs,
            )

        fallback_ctx = None
        if fallback_rows:
            entry_lens_chunk = (
                seq_lens[chunk_start:chunk_end] // self.compress_ratio
            )
            max_entries_sub = max(
                int(seq_lens_cpu[chunk_start + i]) // self.compress_ratio
                for i in fallback_rows
            )
            if len(fallback_rows) == chunk_end - chunk_start:
                # Whole chunk fell back: contiguous slices, no index copy.
                sub_entry_lens = entry_lens_chunk
                sub_block_table = block_table_chunk
            else:
                # Mixed chunk (rare: budget-blocked rows next to tracked
                # ones). The tiny H2D for the row index list is accepted on
                # this fallback-only path.
                sub_index = torch.tensor(
                    fallback_rows,
                    dtype=torch.int64,
                    device=compressed_k_cache.device,
                )
                sub_entry_lens = entry_lens_chunk.index_select(0, sub_index)
                sub_block_table = block_table_chunk.index_select(0, sub_index)
            if max_entries_sub > 0:
                gathered_rows, virtual_block_table, max_local = (
                    sm86_dcp_allgather_k_entries(
                        compressed_k_cache,
                        sub_entry_lens,
                        sub_block_table,
                        entry_block_size,
                        dcp_group,
                        self._cp_interleave,
                        max_entries_sub,
                    )
                )
            else:
                gathered_rows, virtual_block_table, max_local = None, None, 0
            fallback_ctx = (
                fallback_rows,
                gathered_rows,
                virtual_block_table,
                sub_entry_lens,
                max_local,
                max_entries_sub,
            )
        return tracked, fallback_ctx

    def _delta_fill_compressed_chunk(
        self,
        kv: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        block_table_chunk: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        dcp_group: "GroupCoordinator",
        delta_tracker: "Sm86DcpDeltaTracker",
        chunk_start: int,
        chunk_end: int,
    ) -> None:
        """P7 Triton-path consumer: fill ``kv[i, 0:n_i)`` per chunk request.

        Tracked requests dequantize straight from their persistent GLOBAL-
        order staging through the identity block table (block_size=1 -- the
        per-row 576-data + scales@576 layout the existing kernel already
        derives, P2d); fallback requests read the subset re-gather through
        its virtual block table. Same dequant kernel per row either way, so
        the bf16 workspace contents are byte-identical to the non-delta
        path (the batch dimension is embarrassingly parallel in the
        kernel).
        """
        device = compressed_k_cache.device
        seq_lens = swa_metadata.prefill_seq_lens
        assert seq_lens is not None
        entry_lens_chunk = seq_lens[chunk_start:chunk_end] // self.compress_ratio

        tracked, fallback_ctx = self._delta_prepare_chunk(
            compressed_k_cache,
            block_table_chunk,
            attn_metadata,
            swa_metadata,
            dcp_group,
            delta_tracker,
            chunk_start,
            chunk_end,
        )

        for i in sorted(tracked):
            new_entries, staging = tracked[i]
            if new_entries == 0:
                continue
            dequantize_and_gather_k_cache_triton(
                kv[i : i + 1],
                staging,
                seq_lens=entry_lens_chunk[i : i + 1],
                gather_lens=None,
                block_table=sm86_dcp_identity_block_table(new_entries, device),
                block_size=1,
                offset=0,
                use_fnuz=False,
            )

        if fallback_ctx is not None:
            (
                fallback_rows,
                gathered_rows,
                virtual_block_table,
                sub_entry_lens,
                _max_local,
                _max_entries_sub,
            ) = fallback_ctx
            if gathered_rows is not None:
                assert virtual_block_table is not None
                for s, i in enumerate(fallback_rows):
                    dequantize_and_gather_k_cache_triton(
                        kv[i : i + 1],
                        gathered_rows,
                        seq_lens=sub_entry_lens[s : s + 1],
                        gather_lens=None,
                        block_table=virtual_block_table[s : s + 1],
                        block_size=1,
                        offset=0,
                        use_fnuz=False,
                    )

    def _forward_prefill_dcp_flash(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        dcp_group: "GroupCoordinator",
        topk_indices: torch.Tensor,
        num_chunks: int,
        delta_tracker: "Sm86DcpDeltaTracker | None" = None,
    ) -> None:
        """P6 (VLLM_DSV4_FLASH_PREFILL): fused flash-mla DCP prefill.

        Per chunk: pack + all-gather the compressed shard bytes (the SAME
        helper the Triton path uses -- dequant skipped), byte-pack the
        replicated SWA window, translate the producer's GLOBAL entry ids to
        flat staging rows through the memoized P4 map, and run ONE
        ``fwd_sparse_prefill_mla`` call: single softmax over SWA window +
        top-k entries, sink folded once in-kernel (rules 1/6/8; section 6).
        No bf16 dequant workspace, no combine_topk_swa_indices.

        Eager-only like the parent (same host-sync-sized collective); the
        capture guard already fired in ``_forward_prefill_dcp``.
        """
        from vllm.models.deepseek_v4.ampere.flash_mla_prefill import (
            sparse_prefill_via_flash_mla,
        )

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert seq_lens_cpu is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        block_table = attn_metadata.block_table[num_decodes:]
        swa_block_table = swa_metadata.block_table[num_decodes:]

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)

            # Host bounds (CPU tensors; rank-invariant; seq_lens_cpu is an
            # upper bound, which only ever ENLARGES staging).
            qsl_cpu = query_start_loc_cpu[
                num_decodes + chunk_start : num_decodes + chunk_end + 1
            ]
            max_qlen = int((qsl_cpu[1:] - qsl_cpu[:-1]).max().item())
            max_seq = int(seq_lens_cpu[chunk_start:chunk_end].max().item())
            max_entries = max_seq // self.compress_ratio
            # gather_len = query_len + min(prefix, window-1) per the
            # sparse_swa builder, so this bounds every chunk row.
            max_gather = min(max_seq, max_qlen + self.window_size - 1)

            if delta_tracker is not None:
                # P7: delta gather into persistent GLOBAL-order stagings +
                # per-request flash op calls (extra_indices become the
                # identity for tracked requests). Splitting the chunk call
                # per request changes launch geometry only -- each query
                # row's softmax is independent of its batch neighbors.
                self._flash_delta_chunk(
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=swa_k_cache,
                    output=output,
                    attn_metadata=attn_metadata,
                    swa_metadata=swa_metadata,
                    dcp_group=dcp_group,
                    delta_tracker=delta_tracker,
                    topk_indices=topk_indices,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    max_gather=max_gather,
                    block_table_chunk=block_table[chunk_start:chunk_end],
                    swa_block_table_chunk=swa_block_table[
                        chunk_start:chunk_end
                    ],
                    prefill_token_base=int(prefill_token_base),
                )
                continue

            # Compressed stream: pack + all-gather (skip symmetric at 0).
            gathered_rows = None
            virtual_block_table = None
            max_local = 0
            if max_entries > 0:
                gathered_rows, virtual_block_table, max_local = (
                    sm86_dcp_allgather_k_entries(
                        compressed_k_cache,
                        seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                        block_table[chunk_start:chunk_end],
                        attn_metadata.block_size // self.compress_ratio,
                        dcp_group,
                        self._cp_interleave,
                        max_entries,
                    )
                )

            # Replicated (dcp_exempt) SWA window: byte-pack, no collective.
            swa_staging_rows = sm86_pack_swa_window_entries(
                swa_k_cache,
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                swa_block_table[chunk_start:chunk_end],
                swa_metadata.block_size,
                max_gather,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            sparse_prefill_via_flash_mla(
                q[query_start:query_end],
                swa_staging_rows=swa_staging_rows,
                max_gather=max_gather,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                window_size=self.window_size,
                compressed_staging_rows=gathered_rows,
                virtual_block_table=virtual_block_table,
                max_local=max_local,
                max_entries=max_entries,
                topk_indices=(
                    topk_indices[query_start:query_end]
                    if gathered_rows is not None
                    else None
                ),
                compress_ratio=self.compress_ratio,
                query_start_loc=query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
                forbidden_pools=(compressed_k_cache, swa_k_cache),
            )

    def _flash_delta_chunk(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        dcp_group: "GroupCoordinator",
        delta_tracker: "Sm86DcpDeltaTracker",
        topk_indices: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        max_gather: int,
        block_table_chunk: torch.Tensor,
        swa_block_table_chunk: torch.Tensor,
        prefill_token_base: int,
    ) -> None:
        """P7 flash-path consumer: one flash-mla op call PER REQUEST.

        Tracked requests hand the op their persistent GLOBAL-order staging
        sliced to ``[n, 584]`` with ``extra_indices`` = the producer's
        GLOBAL entry ids as-is (identity translation -- the P6 formula
        change is exactly and only this, gated on the new layout).
        Fallback requests read the chunk's subset re-gather through its
        virtual block table row, i.e. the P6 formula unchanged. The SWA
        stream is the P6 chunk pack, sliced per request (rows
        ``[i*max_gather, (i+1)*max_gather)`` are exactly request ``i``'s).
        Per-request op calls only change launch geometry: each query row's
        single softmax is independent of its batch neighbors, and the
        in-op dequant pre-pass now touches exactly one request's staging
        (less work than the chunk-wide buffer, except for fallback rows,
        which share -- and each re-dequantize -- the subset buffer).
        """
        from vllm.models.deepseek_v4.ampere.flash_mla_prefill import (
            sparse_prefill_via_flash_mla,
        )

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        num_decodes = swa_metadata.num_decodes
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc is not None and query_start_loc_cpu is not None

        tracked, fallback_ctx = self._delta_prepare_chunk(
            compressed_k_cache,
            block_table_chunk,
            attn_metadata,
            swa_metadata,
            dcp_group,
            delta_tracker,
            chunk_start,
            chunk_end,
        )

        # Replicated (dcp_exempt) SWA window: byte-pack, no collective
        # (unchanged P6 kernel, chunk-level).
        swa_staging_rows = sm86_pack_swa_window_entries(
            swa_k_cache,
            seq_lens[chunk_start:chunk_end],
            gather_lens[chunk_start:chunk_end],
            swa_block_table_chunk,
            swa_metadata.block_size,
            max_gather,
        )

        fallback_pos: dict[int, int] = {}
        fb_gathered = None
        fb_vbt = None
        fb_max_local = 0
        fb_max_entries = 0
        if fallback_ctx is not None:
            (
                fallback_rows,
                fb_gathered,
                fb_vbt,
                _fb_lens,
                fb_max_local,
                fb_max_entries,
            ) = fallback_ctx
            fallback_pos = {row: s for s, row in enumerate(fallback_rows)}

        for i in range(chunk_end - chunk_start):
            row = chunk_start + i
            query_start = (
                int(query_start_loc_cpu[num_decodes + row]) - prefill_token_base
            )
            query_end = (
                int(query_start_loc_cpu[num_decodes + row + 1])
                - prefill_token_base
            )
            if query_end <= query_start:
                continue
            swa_rows_i = swa_staging_rows[i * max_gather : (i + 1) * max_gather]
            if i in tracked:
                new_entries, staging = tracked[i]
                has_compressed = new_entries > 0
                sparse_prefill_via_flash_mla(
                    q[query_start:query_end],
                    swa_staging_rows=swa_rows_i,
                    max_gather=max_gather,
                    seq_lens=seq_lens[row : row + 1],
                    gather_lens=gather_lens[row : row + 1],
                    window_size=self.window_size,
                    compressed_staging_rows=(
                        staging[:new_entries] if has_compressed else None
                    ),
                    virtual_block_table=None,
                    max_local=0,
                    max_entries=new_entries,
                    topk_indices=(
                        topk_indices[query_start:query_end]
                        if has_compressed
                        else None
                    ),
                    compress_ratio=self.compress_ratio,
                    staging_is_global_order=True,
                    query_start_loc=query_start_loc[
                        num_decodes + row : num_decodes + row + 2
                    ],
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                    forbidden_pools=(compressed_k_cache, swa_k_cache),
                )
            else:
                s = fallback_pos[i]
                has_compressed = fb_gathered is not None
                sparse_prefill_via_flash_mla(
                    q[query_start:query_end],
                    swa_staging_rows=swa_rows_i,
                    max_gather=max_gather,
                    seq_lens=seq_lens[row : row + 1],
                    gather_lens=gather_lens[row : row + 1],
                    window_size=self.window_size,
                    compressed_staging_rows=(
                        fb_gathered if has_compressed else None
                    ),
                    virtual_block_table=(
                        fb_vbt[s : s + 1] if has_compressed else None
                    ),
                    max_local=fb_max_local,
                    max_entries=fb_max_entries,
                    topk_indices=(
                        topk_indices[query_start:query_end]
                        if has_compressed
                        else None
                    ),
                    compress_ratio=self.compress_ratio,
                    query_start_loc=query_start_loc[
                        num_decodes + row : num_decodes + row + 2
                    ],
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                    forbidden_pools=(compressed_k_cache, swa_k_cache),
                )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        dcp_group = None if swa_only else self._dcp_group_or_none()
        if dcp_group is None:
            # Default path — gate unset, dcp == 1, or a SWA-only layer
            # (L0/L1/DSpark): the SWA ring is replicated across DCP ranks
            # (dcp_exempt), so every rank already sees the full window for
            # its own heads and the in-kernel sink is applied exactly once.
            # No cross-rank merge is needed or wanted there.
            super()._forward_decode(
                q=q,
                kv_cache=kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=attn_metadata,
                swa_only=swa_only,
                output=output,
            )
            return
        assert attn_metadata is not None
        self._forward_decode_dcp(
            q=q,
            kv_cache=kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            dcp_group=dcp_group,
            output=output,
        )

    def _forward_decode_dcp(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata,
        dcp_group: "GroupCoordinator",
        output: torch.Tensor,
    ) -> None:
        """DCP decode: local pre-sink partial + fp32 cross-rank merge.

        Per the P2 interface contract / ARCHITECTURE.md section 10:
        each rank attends its local shard of selected compressed entries;
        the replicated SWA window joins the ONE owner rank's partial (a
        single online softmax with its top-k shard, section 6 — never
        normalized standalone); partials with ``l == 0`` carry a finite
        sentinel LSE and drop out of the merge (rule 9); the sink is added
        exactly once at the global max (rule 1); the merge is fp32 with a
        fixed rank order (rule 2, a2a); inverse RoPE runs in ``_o_proj``
        after the merge (rule 10).

        CUDA-graph capture (P2f — lifts P2a's eager-only guard).  Every step
        below is fixed-shape and sync-free, and everything a captured kernel
        dereferences lives at an address that survives replay:

        - metadata inputs are persistent builder buffers (``is_valid_token``,
          ``token_to_req_indices``, ``decode_swa_indices``/``lens``,
          ``block_table``, ``topk_indices_buffer``,
          ``c128a_global_decode_topk_indices``) — the discipline the SWA
          builder already states ("Ensure all metadata tensors maintain fixed
          memory addresses for CUDA graph compatibility");
        - both ragged builds write into the persistent ``dcp_decode_*_buffer``
          scratch the two builders own: the ``_copy_ragged_to_graph_buffers``
          pattern of the non-DCP path, reached through ``out=`` buffers so the
          copy disappears;
        - row WIDTHS are init-time constants (``index_topk`` for C4A, the
          builder's pinned per-rank bound for C128A — see P2f in sparse_mla.py:
          the default ``active_topk_width`` tracks ``cm.max_seq_len``, which is
          ``max_model_len`` at capture but the batch maximum at replay, so a
          captured consumer would read the dense rows with the wrong stride);
        - the two collectives (Q all-gather over the DCP group, a2a LSE
          reduce) are the same calls mainline captures on the standard DCP MLA
          path (``mla_attention.py``: ``get_dcp_group().all_gather(mqa_q,
          dim=1)`` then ``dcp_a2a_lse_reduce``), and ``dcp_alltoall``'s
          send/recv buffers are deliberately ``torch.empty`` so they live in
          the graph's private pool.

        No host sync (``.item()`` / ``.cpu()``) and no data-dependent control
        flow exists here; every branch is on a Python constant (compress
        ratio, world size, and the token counts that are fixed per captured
        shape).  Compressed-layer PREFILL stays eager-only — see
        ``_forward_prefill_dcp``.
        """
        assert kv_cache is not None
        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # ---- Local top-k selection (this rank's KV shard only) ----
        # Interface with the P2 producers: under DCP both compressed-layer
        # kinds deliver rank-LOCAL SHARD ENTRY COORDINATES of the selected
        # entries this rank owns, owned entries first (prefix-compact),
        # -1-padded to a fixed width:
        #   - C4A: P2b's indexer decode merge (`topk_indices_buffer`, the
        #     globally-top-512 entries; reference `_topk_per_row_decode_dcp`
        #     ends in `layout.global_to_local`);
        #   - C128A (P2d W2): the sparse_mla.py builder's gated branch
        #     (`c128a_global_decode_topk_indices` holds `[0..count-1, -1]`
        #     rank-local rows -- C128A attends ALL completed entries, so
        #     the owned set is a contiguous local prefix).  The amd/rocm.py
        #     builder's dense->ragged copy is skipped under the gate (it
        #     would pack these coordinates as if they were global slots).
        # The entry coordinate -> physical slot translation therefore goes
        # through the P1-sharded block table with the SAME kernel as the
        # non-DCP C4A path; -1 rows are counted out by the ragged pack and
        # `is_valid` (from the REPLICATED SWA group's unsharded slot
        # mapping) zeroes padding rows on every rank.
        is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            local_entry_indices = self.topk_indices_buffer[:num_decode_tokens]
        else:
            dense_local = attn_metadata.c128a_global_decode_topk_indices
            assert dense_local is not None, (
                "VLLM_SM86_DCP C128A decode requires the sparse_mla.py "
                "builder's rank-local metadata (P2d W2)."
            )
            local_entry_indices = dense_local.reshape(num_decode_tokens, -1)
        # P2f: pack straight into the builder's persistent scratch so a FULL
        # decode graph bakes in addresses that are still ours on replay.  The
        # widths the buffers were sized for are init-time constants, so this
        # slice bound only ever shrinks (assert = loud, never silent OOB).
        topk_ragged_buffer = attn_metadata.dcp_decode_topk_ragged_indices_buffer
        topk_indptr_buffer = attn_metadata.dcp_decode_topk_ragged_indptr_buffer
        topk_lens_buffer = attn_metadata.dcp_decode_topk_lens_buffer
        assert topk_ragged_buffer is not None
        assert topk_indptr_buffer is not None
        assert topk_lens_buffer is not None
        assert (
            num_decode_tokens * local_entry_indices.shape[-1]
            <= topk_ragged_buffer.numel()
        ), (
            "VLLM_SM86_DCP decode top-k rows exceed the builder's graph "
            f"buffer: {num_decode_tokens} x {local_entry_indices.shape[-1]} > "
            f"{topk_ragged_buffer.numel()}"
        )
        (
            topk_ragged_indices,
            topk_ragged_indptr,
            _topk_lens,
        ) = compute_global_topk_ragged_indices_and_indptr(
            local_entry_indices,
            swa_metadata.token_to_req_indices,
            attn_metadata.block_table[:num_decodes],
            attn_metadata.block_size // self.compress_ratio,
            is_valid,
            out_ragged=topk_ragged_buffer,
            out_indptr=topk_indptr_buffer,
            out_lens=topk_lens_buffer,
        )

        # ---- SWA owner selection ----
        # The SWA ring is replicated (dcp_exempt, P1), unlike the reference
        # which shards it — so Lasimeri's per-position SWA filtering is
        # impossible here.  Instead exactly ONE rank per query contributes
        # the whole window: the rank that owns the query position's newest
        # compressed-KV block under the P1 slot-mapping formula
        # (block_table.py::_compute_slot_mapping_kernel):
        #   owner(pos) = ((pos % (block_size * ws)) // interleave) % ws
        # This keeps SWA+sink counted exactly once globally and rotates the
        # extra window work across ranks as positions advance.
        assert self._dcp_positions is not None, (
            "forward_mqa must stash positions for the DCP decode branch"
        )
        positions = self._dcp_positions[:num_decode_tokens]
        ws = dcp_group.world_size
        virtual_block = attn_metadata.block_size * ws
        owner = ((positions % virtual_block) // self._cp_interleave) % ws
        # P2f: masked lens and their ragged expansion land in the SWA
        # builder's DCP scratch (separate from decode_swa_ragged_*, which the
        # SWA-only layers still read UNMASKED in the same step).
        swa_lens_buffer = swa_metadata.dcp_decode_swa_lens_buffer
        swa_ragged_buffer = swa_metadata.dcp_decode_swa_ragged_indices_buffer
        swa_indptr_buffer = swa_metadata.dcp_decode_swa_ragged_indptr_buffer
        assert swa_lens_buffer is not None
        assert swa_ragged_buffer is not None
        assert swa_indptr_buffer is not None
        swa_lens = swa_lens_buffer[:num_decode_tokens]
        torch.where(
            owner == dcp_group.rank_in_group,
            swa_metadata.decode_swa_lens,
            torch.zeros_like(swa_metadata.decode_swa_lens),
            out=swa_lens,
        )
        swa_k_cache = self.swa_cache_layer.kv_cache
        swa_dense_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, -1
        )
        assert (
            num_decode_tokens * swa_dense_indices.shape[-1]
            <= swa_ragged_buffer.numel()
        ), (
            "VLLM_SM86_DCP decode SWA rows exceed the builder's graph buffer: "
            f"{num_decode_tokens} x {swa_dense_indices.shape[-1]} > "
            f"{swa_ragged_buffer.numel()}"
        )
        swa_ragged_indices, swa_ragged_indptr = build_ragged_indices_from_dense(
            swa_dense_indices,
            swa_lens,
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
            out_indices=swa_ragged_buffer,
            out_indptr=swa_indptr_buffer,
        )

        # ---- Attend the local shard with the group's gathered heads ----
        q, num_real_heads, dcp_group, use_dcp = _maybe_gather_dcp_q(self, q)
        assert use_dcp
        partials = rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_only=False,
            topk_indices=None,
            topk_lens=None,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_lens,
            swa_ragged_indices=swa_ragged_indices,
            swa_ragged_indptr=swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            # PRE-sink partial: the sink joins exactly once, post-merge.
            attn_sink=None,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=None,
            return_softmax_stats=True,
        )
        assert partials is not None
        out_attn, rowmax, sumexp = partials

        # ---- fp32 cross-rank merge, sink once at the global max ----
        # NOTE: deviation from the reference diff, per the P2 contract: the
        # reference gates the trained sink behind VLLM_SM86_SINK (dropping
        # it by default under DCP).  The sink is part of the trained model
        # (Eq.27) and the contract mandates it be added EXACTLY ONCE — not
        # optionally — so the true sink is always applied here.
        lse = softmax_stats_to_lse(rowmax, sumexp)
        dcp_merge_flashmla_output(
            out_attn[:, :num_real_heads, :],
            lse[:, :num_real_heads],
            self.attn_sink,
            output,
            dcp_group,
        )
