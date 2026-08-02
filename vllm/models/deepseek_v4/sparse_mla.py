# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 FlashMLA sparse backend, metadata, and metadata builder."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# Pad C128A topk width to this alignment. 128 covers both h_q=64 (B_TOPK=64) and
# h_q=128 (B_TOPK=128). FlashMLA decode asserts extra_topk % B_TOPK == 0;
# unaligned widths (e.g. 17 = ceil(2136/128)) crash the sm100 head64 kernel.
# Padded slots stay -1 and decode_lens caps them via topk_length, so the pad is a
# no-op at kernel level. Mirrors _SPARSE_PREFILL_TOPK_ALIGNMENT in cache_utils.py.
_C128A_TOPK_ALIGNMENT = 128


class DeepseekV4FlashMLABackend(AttentionBackend):
    """DeepSeek-V4 sparse-MLA backend.

    Subclasses ``AttentionBackend`` directly (not the V3.2
    ``FlashMLASparseBackend``): DeepSeek-V4 runs its own attention layer
    (``DeepseekV4Attention``), so it does not reuse the V3.2 builder or impl, and
    only needs to declare its own metadata builder, KV-cache layout, and the
    sparse-MLA capability flags.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4FlashMLAMetadataBuilder"]:
        return DeepseekV4FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        # DeepSeek-V4 runs its attention through ``DeepseekV4Attention.forward``,
        # not the generic ``Attention``/``MLAAttention`` layer, so the backend's
        # impl class is never instantiated.
        raise NotImplementedError(
            "DeepseekV4FlashMLABackend has no separate impl class; DeepSeek-V4 "
            "attention runs through DeepseekV4Attention."
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512.
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class DeepseekV4FlashMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    topk_tokens: int

    # Pre-computed C128A metadata (compress_ratio == 128 only).
    # Decode: global slot ids + valid-entry counts (fused from positions).
    # SM86 DCP EXCEPTION (VLLM_SM86_DCP + dcp>1, P2d W2): the decode buffer
    # instead holds rank-LOCAL COMPRESSED-ENTRY COORDINATES [0..count-1, -1
    # pad] of this rank's owned entries (owner(e) = (e // I) % W, the shared
    # P2 layout), NOT global slot ids, and the lens hold the raw owned
    # counts (validity is applied by the consumer). The only consumer under
    # the gate -- ampere_sparse._forward_decode_dcp -- translates them to
    # physical slots through the P1-SHARDED block table, exactly like the
    # C4A indexer output.
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill: local topk indices (used by combine_topk_swa_indices).
    # Under SM86 DCP these stay as-is: [0..n-1] in GLOBAL entry order, which
    # is exactly how the P2d dense all-gathered prefill buffer is indexed.
    c128a_prefill_topk_indices: torch.Tensor | None = None


class DeepseekV4FlashMLAMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4FlashMLAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        # Classify single-token queries (plus num_speculative_tokens via
        # supports_spec_as_decode=True) as decodes; longer queries go to prefill.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.topk_tokens = self.model_config.hf_config.index_topk

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,), dtype=torch.int32, device=device
        )

        assert hasattr(self.kv_cache_spec, "compress_ratio")
        self.compress_ratio = self.kv_cache_spec.compress_ratio

        # SM86 DCP (P2d W2): C128A decode metadata must be rank-localized --
        # the default kernel walks the GLOBAL entry range through the
        # P1-SHARDED block table (rows sized 1/dcp of the global range),
        # which would read out of the owned range. Init-time snapshot, like
        # the indexer builder's use_sm86_dcp, so a mid-run env flip cannot
        # desynchronize producer and consumer.
        parallel_config = vllm_config.parallel_config
        self._sm86_dcp_c128a = (
            envs.VLLM_SM86_DCP
            and parallel_config.decode_context_parallel_size > 1
            and self.compress_ratio == 128
        )
        if self._sm86_dcp_c128a:
            from vllm.distributed import get_dcp_group

            self._dcp_world_size = parallel_config.decode_context_parallel_size
            self._dcp_rank = get_dcp_group().rank_in_group
            self._cp_interleave = parallel_config.cp_kv_cache_interleave_size

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
        # stability when compress_ratio > 1.
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )

        # Pre-allocate C128A topk buffers for CUDA graph address stability.
        if self.compress_ratio == 128:
            c128a_max_compressed = cdiv(
                self.model_config.max_model_len, self.compress_ratio
            )
            c128a_max_compressed = (
                cdiv(c128a_max_compressed, _C128A_TOPK_ALIGNMENT)
                * _C128A_TOPK_ALIGNMENT
            )
            # Stored so _build_c128a_metadata passes it as the kernel's
            # max_compressed_tokens, matching the buffer stride. Otherwise the
            # kernel's default 8192 iterates past row width and spills writes
            # into adjacent rows (present in both decode and prefill branches of
            # _build_c128a_topk_metadata_kernel).
            self.c128a_max_compressed = c128a_max_compressed
            self.c128a_global_decode_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=device,
            )
            self.c128a_decode_lens_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int32, device=device
            )
            self.c128a_prefill_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4FlashMLAMetadata:
        cm = common_attn_metadata
        req_id_per_token = cm.token_to_req_indices(self.req_id_per_token_buffer)

        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.query_start_loc,
                cm.seq_lens,
                cm.block_table_tensor.clamp_(min=0),
                int(self.kv_cache_spec.storage_block_size),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        c128a_fields: dict[str, torch.Tensor | None] = {}
        if self.compress_ratio == 128:
            c128a_fields = self._build_c128a_metadata(cm, req_id_per_token)

        return DeepseekV4FlashMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            c128a_global_decode_topk_indices=c128a_fields.get(
                "c128a_global_decode_topk_indices"
            ),
            c128a_decode_topk_lens=c128a_fields.get("c128a_decode_topk_lens"),
            c128a_prefill_topk_indices=c128a_fields.get("c128a_prefill_topk_indices"),
        )

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Pre-compute C128A topk indices for DeepseekV4 (compress_ratio >= 128)."""
        # Must match SWA's decode split (no `require_uniform=True`) so
        # `c128a_global_decode_topk_indices.shape[0]` lines up with q in
        # `_forward_decode`. The per-token C128A kernel handles non-uniform
        # query lengths.
        (num_decodes, _, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                cm,
                decode_threshold=self.reorder_batch_threshold or 1,
            )
        )

        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return {}

        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        active_topk_width = min(
            max(
                triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)),
                _C128A_TOPK_ALIGNMENT,
            ),
            self.c128a_max_compressed,
        )
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        if self._sm86_dcp_c128a:
            global_decode, decode_lens, prefill_local = (
                self._build_c128a_metadata_sm86_dcp(
                    cm,
                    req_id_per_token,
                    num_decodes,
                    num_decode_tokens,
                    num_total,
                    active_topk_width,
                    block_size,
                )
            )
        else:
            global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
                cm.positions[:num_total],
                self.compress_ratio,
                num_decode_tokens,
                req_id_per_token,
                cm.block_table_tensor[:num_decodes],
                block_size,
                cm.slot_mapping,
                self.c128a_global_decode_buffer,
                self.c128a_decode_lens_buffer,
                self.c128a_prefill_buffer,
                max_compressed_tokens=active_topk_width,
            )

        result: dict[str, torch.Tensor | None] = {}
        if num_decode_tokens > 0:
            result["c128a_global_decode_topk_indices"] = global_decode.view(
                num_decode_tokens, 1, -1
            )
            result["c128a_decode_topk_lens"] = decode_lens
        if num_prefill_tokens > 0:
            result["c128a_prefill_topk_indices"] = prefill_local
        return result

    def _build_c128a_metadata_sm86_dcp(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        num_total: int,
        active_topk_width: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """C128A metadata under VLLM_SM86_DCP + dcp>1 (P2d W2).

        Decode rows are rank-LOCALIZED: for a token at global position
        ``pos`` with ``n = (pos + 1) // 128`` completed entries (causal rule
        ``(t+1)//m``, rule 3), this rank owns the entries
        ``{e < n : (e // I) % W == rank}`` (the shared P2 entry-space
        layout). Their rank-local coordinates under
        ``local(e) = (e // (I*W))*I + e % I`` are exactly ``0..count-1``:
        the local enumeration is an order-preserving bijection onto a
        contiguous prefix, so the row is ``[0..count-1, -1 pad]`` -- already
        owned-first prefix-compacted, matching P2b's decode convention. The
        owned count comes from ``get_dcp_local_seq_lens`` (the base layout
        helper P2b validated against the shared formulas; no third
        reimplementation of the algebra).

        Deltas vs the default builder path:
        - the fused kernel runs in PREFILL-ONLY mode: its decode branch
          resolves GLOBAL entry indices through the block table, but the
          P1-sharded C128A table rows are 1/dcp of the global entry range,
          so global-width walks would read past the owned rows. Prefill
          rows are unchanged (``[0..n-1]`` identity, no table access) and
          index the P2d dense global-order gathered buffer directly.
        - physical-slot translation moves to the consumer
          (``ampere_sparse._forward_decode_dcp``), which pushes these local
          coordinates through the SAME kernel as the C4A indexer output
          (``compute_global_topk_ragged_indices_and_indptr`` with the
          sharded block table).
        - ``decode_lens`` hold the RAW owned counts. The default path
          zeroes lens for padding via ``cm.slot_mapping < 0``, but under
          DCP that mapping is the P1-SHARDED token-space mapping whose -1s
          mark NON-OWNED tokens (not padding) -- using it would drop real
          tokens. The only gated consumer recomputes lens from the -1
          layout with the replicated SWA group's ``is_valid_token``, which
          also covers padding rows (their garbage positions are bounded by
          the clamp below and never dereferenced: length 0 packs nothing).

        All writes land in the same persistent buffers as the default path
        (CUDA-graph address stability; the fill is fixed-shape given the
        same token counts and width).
        """
        num_prefill_tokens = num_total - num_decode_tokens
        _, _, prefill_local = build_c128a_topk_metadata(
            cm.positions[num_decode_tokens:num_total],
            self.compress_ratio,
            0,  # prefill-only: never run the kernel's decode branch
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=active_topk_width,
        )
        assert prefill_local.shape[0] == num_prefill_tokens

        width = active_topk_width
        global_decode = self.c128a_global_decode_buffer.view(-1)[
            : num_decode_tokens * width
        ].view(num_decode_tokens, width)
        decode_lens = self.c128a_decode_lens_buffer[:num_decode_tokens]
        if num_decode_tokens > 0:
            positions = cm.positions[:num_decode_tokens]
            # Global entry count, clamped exactly like the kernel's
            # max_compressed_tokens bound (also bounds padding-row garbage).
            num_entries = torch.clamp(
                (positions.to(torch.int64) + 1) // self.compress_ratio,
                max=width,
            )
            local_counts = get_dcp_local_seq_lens(
                num_entries,
                self._dcp_world_size,
                self._dcp_rank,
                self._cp_interleave,
            )
            entry_range = torch.arange(
                width, dtype=torch.int32, device=positions.device
            )
            rows = torch.where(
                entry_range.unsqueeze(0) < local_counts.unsqueeze(1),
                entry_range.unsqueeze(0).expand(num_decode_tokens, width),
                torch.full(
                    (), -1, dtype=torch.int32, device=positions.device
                ),
            )
            global_decode.copy_(rows)
            decode_lens.copy_(local_counts)
        return global_decode, decode_lens, prefill_local


def build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single kernel for all C128A tokens (decode + prefill).

    Decode tokens: position → block_table lookup → global slot ids + topk_lens.
    Prefill tokens: position → local indices [0, ..., n-1, -1, ...].

    Writes into packed views of pre-allocated buffers for CUDA graph stability.
    """
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens

    # view(-1) as 1-d array and then expanded to
    # [num_decode_tokens, max_compressed_tokens]
    global_decode = global_decode_buffer.view(-1)[
        : num_decode_tokens * max_compressed_tokens
    ].view(num_decode_tokens, max_compressed_tokens)
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer.view(-1)[
        : num_prefill_tokens * max_compressed_tokens
    ].view(num_prefill_tokens, max_compressed_tokens)

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    _build_c128a_topk_metadata_kernel[(num_tokens,)](
        global_decode_buffer,
        max_compressed_tokens,
        decode_lens_buffer,
        prefill_buffer,
        max_compressed_tokens,
        positions,
        compress_ratio,
        max_compressed_tokens,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        BLOCK_SIZE=1024,
    )
    return global_decode, decode_lens, prefill_local


@triton.jit
def _build_c128a_topk_metadata_kernel(
    # Decode outputs
    global_decode_ptr,
    global_decode_stride,
    decode_lens_ptr,
    # Prefill output
    prefill_local_ptr,
    prefill_local_stride,
    # Inputs
    positions_ptr,
    compress_ratio,
    max_compressed_tokens,
    num_decode_tokens,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx)
    num_compressed = (position + 1) // compress_ratio
    num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
    is_decode = token_idx < num_decode_tokens

    if is_decode:
        # --- Decode: block-table lookup → global slot ids + count ---
        is_valid_token = tl.load(slot_mapping_ptr + token_idx) >= 0
        req_idx = tl.load(token_to_req_indices_ptr + token_idx)
        count = tl.zeros((), dtype=tl.int32)
        for i in range(0, max_compressed_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < max_compressed_tokens
            is_valid = offset < num_compressed

            block_indices = offset // block_size
            block_numbers = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_indices,
                mask=mask & is_valid,
            )
            block_offsets = offset % block_size
            slot_ids = block_numbers * block_size + block_offsets
            slot_ids = tl.where(is_valid, slot_ids, -1)
            tl.store(
                global_decode_ptr + token_idx * global_decode_stride + offset,
                slot_ids,
                mask=mask,
            )
            count += tl.sum(is_valid.to(tl.int32), axis=0)

        tl.store(
            decode_lens_ptr + token_idx,
            tl.where(is_valid_token, count, 0),
        )
    else:
        # --- Prefill: write local indices ---
        pfx_idx = token_idx - num_decode_tokens
        for i in range(0, max_compressed_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < max_compressed_tokens
            tl.store(
                prefill_local_ptr + pfx_idx * prefill_local_stride + offset,
                tl.where(offset < num_compressed, offset, -1),
                mask=mask,
            )
