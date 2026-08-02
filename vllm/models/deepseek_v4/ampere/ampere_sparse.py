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
1/2/9/10, section 6).  With the gate unset or dcp == 1 every path below
falls through to the unchanged parent implementation.
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
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.models.deepseek_v4.common.ops.dcp import (
    dcp_merge_flashmla_output,
    softmax_stats_to_lse,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_sparse_attn_decode,
)


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
        if attn_metadata is not None and self._dcp_group_or_none() is not None:
            # P2a implements the DECODE merge only.  Compressed-layer
            # prefill reads the sharded compressed-KV cache through
            # dequantize_and_gather_k_cache / combine_topk_swa_indices
            # (common/ops/cache_utils.py), which have no CP-layout support
            # in this base yet — running them against the P1-sharded block
            # tables would silently gather wrong entries.  SWA-only layers
            # (attn_metadata is None) prefill normally: their ring is
            # replicated (dcp_exempt).  Tracked as a P2a blocker.
            raise NotImplementedError(
                "VLLM_SM86_DCP: compressed-layer prefill is not "
                "context-parallel on the SM8x path yet."
            )
        super()._forward_prefill(
            q=q,
            positions=positions,
            compressed_k_cache=compressed_k_cache,
            swa_k_cache=swa_k_cache,
            output=output,
            attn_metadata=attn_metadata,
            swa_metadata=swa_metadata,
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
        """
        if torch.cuda.is_current_stream_capturing():
            # Documented eager-only guard: the per-step ragged index build
            # below allocates fresh tensors (no persistent-address graph
            # buffers) — eager correctness first per the P2 plan.  Run DCP
            # with CUDA graphs disabled for attention.
            raise RuntimeError(
                "VLLM_SM86_DCP decode is eager-only in P2a; disable CUDA "
                "graph capture for attention (enforce_eager / cudagraph "
                "mode NONE)."
            )
        if self.compress_ratio != 4:
            # Fail closed (review B/D, round 2): the C128A decode top-k
            # metadata producer (sparse_mla.py builder + amd/rocm.py ragged
            # copy) enumerates GLOBAL compressed-entry slots and has no DCP
            # awareness yet — consuming them here as local-shard entries
            # would compute silently wrong attention.  Mirrors the prefill
            # guard above; lift once the builder emits rank-local entries
            # via the shared ownership formulas (owner(e) = (e//I) % W).
            raise NotImplementedError(
                "VLLM_SM86_DCP: C128A (compress_ratio=128) decode is not "
                "context-parallel on the SM8x path yet; sparse_mla.py's "
                "top-k metadata builder must emit rank-local entries first."
            )
        assert kv_cache is not None
        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # ---- Local top-k selection (this rank's KV shard only) ----
        # Interface with the P2 indexer (reference `_topk_per_row_decode_dcp`
        # ends in `layout.global_to_local`): under DCP `topk_indices_buffer`
        # holds LOCAL-SHARD ENTRY COORDINATES of the globally-top-512 entries
        # this rank owns, -1-padded to the fixed 512 width.  The entry
        # coordinate -> physical slot translation below therefore goes
        # through the P1-sharded block table with the SAME kernel as the
        # non-DCP path; -1 rows are counted out by the ragged pack.
        is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            (
                topk_ragged_indices,
                topk_ragged_indptr,
                _topk_lens,
            ) = compute_global_topk_ragged_indices_and_indptr(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                attn_metadata.block_size // self.compress_ratio,
                is_valid,
            )
        else:
            # C128A: indices are materialized by the FlashMLA metadata
            # builder; under DCP its producer must already emit
            # local-shard slots (P2a blocker note — not owned by this
            # workstream's files).
            topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
            topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr
        assert topk_ragged_indices is not None
        assert topk_ragged_indptr is not None

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
        swa_lens = torch.where(
            owner == dcp_group.rank_in_group,
            swa_metadata.decode_swa_lens,
            torch.zeros_like(swa_metadata.decode_swa_lens),
        )
        swa_k_cache = self.swa_cache_layer.kv_cache
        swa_ragged_indices, swa_ragged_indptr = build_ragged_indices_from_dense(
            swa_metadata.decode_swa_indices.reshape(num_decode_tokens, -1),
            swa_lens,
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
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
