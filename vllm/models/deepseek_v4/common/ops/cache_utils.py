# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernels for DeepseekV4 paged K-cache management and sparse-attention index
preparation.

- quantize_and_insert_k_cache: quantize bf16 K to UE8M0 FP8 and insert into
  the paged cache.
- dequantize_and_gather_k_cache: gather and dequantize FP8 K from the paged
  cache for sparse/SWA prefill.
- compute_global_topk_indices_and_lens: map local topk indices to global KV
  cache slots and count valid entries.
- combine_topk_swa_indices: concatenate topk compressed indices with SWA
  window indices for sparse prefill.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonPointerInputVariant,
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import is_cutedsl_supported
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.mla.sm86_dcp_layout import (
    sm86_dcp_global_to_local,
    sm86_dcp_local_count,
    sm86_dcp_owner,
)
from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32, _encode_fp8_u8

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@triton.jit
def quantize_and_insert_k_kernel(
    # Input tensors
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    # Output tensor
    k_cache_ptr,  # [num_blocks, block_bytes] as uint8 (flattened view)
    # Dimensions
    num_tokens,
    input_dim: tl.constexpr,  # 512
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded)
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 8 (7 real + 1 padding)
    use_fnuz: tl.constexpr = False,
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - [0, 64*576): Token data, each token has 448 fp8 + 128 bf16
    - [64*576, 64*576 + 64*8): Scales, each token has 8 uint8 scales
    - [64*576 + 64*8, block_stride): Padding

    One program per token.

    ``use_fnuz=True`` selects FNUZ (``tl.float8e4b8``); default OCP
    (``tl.float8e4nv``) matches every production caller.
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Get slot mapping
    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Input pointer for this token
    input_row_ptr = k_ptr + pid * input_dim

    # int64: block_idx * block_stride can exceed 2^31 with many KV-cache blocks
    # (e.g. >= 57K at block_stride ~37K). Matches gather path below.
    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride

    # Token data pointer: token data is stored contiguously at start of block
    # Each token's data is at offset pos_in_block * token_data_size
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

    # Scale pointer: scales are stored after ALL token data in the block
    # Scale for this token is at offset (64 * 576) + pos_in_block * 8
    token_scale_ptr = (
        cache_block_ptr + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    # Token data layout: [0:448] fp8, [448:576] bf16
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    # ========== Quantize and store FP8 portion (first 448 elements) ==========
    # Using UE8M0 quantization strategy (scale is power of 2, stored as uint8 exponent)
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block

        if qblock_start < fp8_dim:
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim

            # Load bf16 input
            x = tl.load(input_row_ptr + offsets, mask=mask, other=0.0)

            # Compute absmax scale (same as CUDA kernel)
            abs_x = tl.abs(x)
            block_max = tl.max(abs_x, axis=0)
            block_max = tl.maximum(block_max, 1e-4)  # Match CUDA: fmaxf(amax, 1e-4)

            # UE8M0: Round scale UP to next power of 2
            # scale = 2^ceil(log2(block_max / fp8_max))
            raw_scale = block_max / fp8_max
            log_scale = tl.log2(raw_scale)
            exponent = tl.ceil(log_scale)  # Round UP to next integer exponent
            scale = tl.exp2(exponent)  # scale = 2^exponent (power of 2)

            # Quantize to fp8: fp8_value = bf16_value / scale
            x_scaled = x / scale
            x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)

            # Convert to fp8 (FNUZ on gfx942, OCP elsewhere) as raw bytes.
            x_uint8 = _encode_fp8_u8(x_clamped, use_fnuz)

            # Store as uint8 (1 byte each)
            tl.store(token_fp8_ptr + offsets, x_uint8, mask=mask)

            # UE8M0 scale encoding: stored_value = exponent + 127 (bias)
            # During dequant: scale = 2^(stored_value - 127)
            encoded_scale = exponent + 127.0
            encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
            tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # Padding scale at index 7
    tl.store(token_scale_ptr + 7, tl.zeros((), dtype=tl.uint8))

    # ========== Store BF16 portion (last 64 elements, no quantization) ==========
    bf16_input_offset = fp8_dim

    # Process bf16 in chunks of 16
    bf16_out_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        chunk_offsets = i * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(input_row_ptr + bf16_input_offset + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


def quantize_and_insert_k_cache(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    block_size: int = 64,
    is_ue8m0: bool = True,
    use_fnuz: bool = False,
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576

    ``use_fnuz=True`` selects FNUZ E4M3 cache encoding and is only valid on
    platforms whose FP8 format is FNUZ. ``use_fnuz=False`` selects OCP E4M3,
    which is used by OCP-encoded caches even on gfx942.

    DCP note (VLLM_SM86_DCP, P2c): this per-token K insert targets the SWA
    ring / raw-K caches, which are dcp_exempt (replicated) groups under P1;
    their slot_mapping comes from an unsharded (shard_dcp=False) BlockTable,
    so under DCP every rank performs the identical write for every token and
    no sharding translation is needed here. The ``slot_idx == -1`` skip
    covers CUDA-graph pads. The same replication argument covers the fused
    csrc SWA insert (fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu, built
    for SM80+): it consumes the same replicated SWA slot_mapping, so its
    behavior under the gate is byte-identical to the non-DCP path and it
    intentionally is NOT gated off.
    """
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert is_ue8m0, "Only support ue8m0 quantization."

    # NOTE: When using DP, slot_mapping.shape[0] can be less than k.shape[0] due to
    # padding. Always use slot_mapping.shape[0] as the token count.
    num_tokens = slot_mapping.shape[0]
    block_stride = k_cache.stride(0)  # bytes per block

    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    if use_fnuz:
        if not current_platform.is_fp8_fnuz():
            raise ValueError("use_fnuz=True requires a platform using FNUZ FP8")
        _, FP8_MAX = get_fp8_min_max()
    else:
        FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    grid = (num_tokens,)

    quantize_and_insert_k_kernel[grid](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        input_dim=512,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=block_stride,
        fp8_max=FP8_MAX,
        n_quant_blocks=8,
        use_fnuz=use_fnuz,
    )


# P9: `max_blocks_per_seq` is the block-table ROW STRIDE and nothing else
# (`block_table_ptr + batch_idx * max_blocks_per_seq`, below) -- pure integer
# address arithmetic, so moving it out of the constexpr set cannot change a
# single computed value.  As a constexpr it was the SM8x prefill JIT tax: the
# DCP all-gather path passes `_sm86_dcp_virtual_block_table`'s width, which is
# `max_entries` and therefore GROWS with the chunk index, and the P7 delta path
# passes `sm86_dcp_identity_block_table(new_entries)`, which varies per request
# per chunk -- so every new context-length bucket paid a fresh ~7-8 s Triton
# compile of this kernel (the leaf under all six py-spy caller frames:
# ampere_sparse.py {148,177,344,623}, cache_utils.py 377, amd/rocm.py 748).
# `do_not_specialize` also suppresses Triton's implicit divisible-by-16 /
# equals-1 specialization on the value, collapsing the family to ONE compile
# per (cache_block_size, block_stride, use_fnuz) -- all of which take two
# values at most.  The sibling kernels in this file already pass their block
# table stride as a plain runtime arg (see `_combine_topk_swa_indices_kernel`).
@triton.jit(do_not_specialize=["max_blocks_per_seq"])
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq,  # block-table row stride (runtime; see note above)
    # Constants
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 or 128 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded) int32
    output_dim: tl.constexpr,  # 512
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 7 real blocks
    use_fnuz: tl.constexpr = False,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        # Gather all tokens
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        # Calculate the actual token index in the sequence
        pos = start_pos + i

        # Calculate which block and position within block
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks (e.g. >= 57K at block_stride ~37K).
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        # Token data pointer
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        # Scale pointer: after all token data
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        # Token data layout: [0:448] fp8, [448:576] bf16
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        # Output pointer for this token (flattened)
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        # ========== Dequantize FP8 portion using UE8M0 ==========
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block

            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim

                # Load quantized fp8 values (stored as uint8)
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)

                # Decode fp8 bytes (FNUZ on gfx942, OCP elsewhere) to f32.
                x_float = _decode_fp8_f32(x_uint8, use_fnuz)

                # Load and decode UE8M0 scale
                # UE8M0: scale = 2^(stored_value - 127)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)

                # Dequantize: bf16_value = fp8_value * scale
                x_dequant = x_float * scale

                # Store as bf16
                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

        # ========== Copy BF16 portion directly ==========
        bf16_output_offset = fp8_dim  # After 448 elements in output

        # Read bf16 from cache
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))

        # Process in chunks of 16
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_and_gather_k_cache_triton(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=FP8_MAX,
        n_quant_blocks=7,
        use_fnuz=use_fnuz,
    )


# --------------------------------------------------------------------------
# P2d (SM8x DSV4 DCP, VLLM_SM86_DCP): compressed-shard all-gather for
# prefill. Under DCP the compressed-KV cache holds only this rank's
# round-robin shard of entries (owner(e) = (e // I) % W -- the shared P2
# layout written by P2c and scored by P2b). The compressed prefix is tiny
# (584 B/entry at C4A: a 128K-token prefix is ~19 MB/layer; C128A is 32x
# smaller), so prefill all-gathers every rank's shard into a dense buffer in
# GLOBAL entry order and runs the UNCHANGED prefill kernels against it with
# the GLOBAL entry indices the P2b indexer merge already emits.
# Rule 8 (ARCHITECTURE.md section 10): the fp8 payload and UE8M0 scales move
# VERBATIM as bytes; dequantization happens exactly once, in the same
# existing kernel the non-DCP path uses.
# --------------------------------------------------------------------------

_SM86_DCP_ENTRY_DATA_BYTES = 576  # 448 fp8 + 64 bf16 (128 bytes)
_SM86_DCP_ENTRY_SCALE_BYTES = 8  # 7 UE8M0 scales + 1 pad
_SM86_DCP_ENTRY_BYTES = _SM86_DCP_ENTRY_DATA_BYTES + _SM86_DCP_ENTRY_SCALE_BYTES
_SM86_DCP_PACK_NUM_WORKERS = 128

# P4 / OPT-2: the virtual block table below is a pure function of five HOST
# ints (max_entries, max_local, num_reqs, world, interleave) plus the device.
# It carries no per-layer and no per-request data, yet the DCP prefill gather
# runs once per COMPRESSED LAYER per chunk (41 layers for Flash) and rebuilt it
# every time.  Memoize it so a forward pass builds it at most once per distinct
# shape.  Bounded LRU, because chunked prefill walks max_entries upward and the
# keys DO churn across a long prefill -- an unbounded dict would leak device
# memory.  Two independent bounds: an entry count (comfortably covers the few
# distinct compress-ratio x chunk shapes in one forward, so all 41 compressed
# layers hit) and a total-element budget, so a long-context batch cannot pin an
# unbounded amount.  The just-built table is always kept, even if it alone
# exceeds the budget.
# 64 MiB ceiling: two decimal orders below the all-gather staging buffers this
# same function already allocates (num_reqs x max_local x 584 B, times world),
# and large enough that even a 1M-token C4A prefix (262144 entries) keeps
# several tables resident instead of thrashing.
_SM86_DCP_VBT_CACHE_MAXSIZE = 8
_SM86_DCP_VBT_CACHE_MAX_ELEMS = 16 * 1024 * 1024  # int32 -> 64 MiB
_SM86_DCP_VBT_CACHE: "OrderedDict[tuple[Any, ...], torch.Tensor]" = OrderedDict()


def _sm86_dcp_virtual_block_table(
    max_entries: int,
    max_local: int,
    num_reqs: int,
    world: int,
    dcp_interleave: int,
    device: torch.device,
) -> torch.Tensor:
    """Virtual ``cache_block_size=1`` block table over the all-gathered buffer.

    Row ``(request c, global entry e)`` must address the flat gathered row of
    the rank that OWNS ``e``:

        row(c, e) = owner(e) * num_reqs * max_local
                    + c * max_local
                    + local_entry(e)

    P4 / OPT-1: this used to be built by looping over ranks and scattering
    with ``row_base[global_e[in_range]] = ...``.  Boolean-mask advanced
    indexing lowers to ``nonzero()``, which is a hard ``cudaStreamSynchronize``
    -- twice per rank iteration (16 syncs at W=8) plus ~120 tiny launches --
    all to compute a table with no data dependence whatsoever.  It is now the
    CLOSED-FORM FORWARD map evaluated once over ``e = arange(max_entries)``:
    fixed shapes, no ``nonzero``, no host sync.  ``sm86_dcp_owner`` and
    ``sm86_dcp_global_to_local`` are the shared single source of truth
    (``sm86_dcp_layout.py``) -- the algebra is not restated here.

    RETURNS A SHARED, CACHED TENSOR.  Callers must treat it as READ-ONLY; it
    is only ever passed as the ``block_table`` input of
    ``dequantize_and_gather_k_cache_triton``, which reads it and never writes
    it (the kernel's only stores go to ``out``).
    """
    key = (max_entries, max_local, num_reqs, world, dcp_interleave, device)
    cached = _SM86_DCP_VBT_CACHE.get(key)
    if cached is not None:
        _SM86_DCP_VBT_CACHE.move_to_end(key)
        return cached

    entries = torch.arange(max_entries, dtype=torch.int64, device=device)
    owner = sm86_dcp_owner(entries, world, dcp_interleave)
    # Per-element owning rank -> every entry's OWN local index in one pass.
    local_entry = sm86_dcp_global_to_local(entries, owner, world, dcp_interleave)
    row_base = owner * (num_reqs * max_local) + local_entry
    request_offsets = (
        torch.arange(num_reqs, dtype=torch.int64, device=device) * max_local
    )
    virtual_block_table = (
        row_base.unsqueeze(0) + request_offsets.unsqueeze(1)
    ).to(torch.int32)

    _SM86_DCP_VBT_CACHE[key] = virtual_block_table
    total_elems = sum(t.numel() for t in _SM86_DCP_VBT_CACHE.values())
    while len(_SM86_DCP_VBT_CACHE) > 1 and (
        len(_SM86_DCP_VBT_CACHE) > _SM86_DCP_VBT_CACHE_MAXSIZE
        or total_elems > _SM86_DCP_VBT_CACHE_MAX_ELEMS
    ):
        _, evicted = _SM86_DCP_VBT_CACHE.popitem(last=False)
        total_elems -= evicted.numel()
    return virtual_block_table


@triton.jit(do_not_specialize=["local_start", "const_local_len"])
def _sm86_dcp_pack_k_entries_kernel(
    staging_ptr,  # [num_reqs, max_local_entries, 584] uint8 contiguous
    k_cache_ptr,  # paged cache (uint8 bytes)
    local_lens_ptr,  # [num_reqs] int32 owned-entry counts, or None (P7 delta)
    block_table_ptr,  # [num_reqs, max_blocks_per_seq] int32 (P1-sharded)
    max_blocks_per_seq,
    staging_stride0,
    staging_stride1,
    cache_block_size: tl.constexpr,  # entries per cache page (64 C4A / 2 C128A)
    token_data_size: tl.constexpr,  # 576
    scale_dim: tl.constexpr,  # 8
    block_stride: tl.constexpr,  # cache bytes per page
    local_start=0,  # P7: first local entry to pack (0 = full prefix)
    const_local_len=-1,  # P7: entry COUNT as a host scalar when lens ptr is None
):
    """Copy this rank's compressed entries out of the paged cache, VERBATIM.

    Local entry ``j`` lives at page ``block_table[req][j // page_entries]``
    offset ``j % page_entries`` -- exactly the layout P2c's DCP-aware insert
    writes. Each staging row is ``[576 data bytes | 8 scale bytes]``, i.e. a
    valid ``cache_block_size=1`` paged-cache block, so the existing
    ``_dequantize_and_gather_k_kernel`` can read the gathered buffer without
    any new dequant code (rule 8: bytes + scales move untouched).

    P7 delta form: staging row ``i`` holds local entry ``local_start + i``
    for ``i in [0, count)`` -- the caller passes ``local_lens_ptr=None`` and
    the count as ``const_local_len`` (a host int, so the delta launch needs
    no device lens tensor and no H2D copy). With the defaults
    (``local_start=0``, lens from the pointer) the addressing is identical
    to the original kernel: ``j == i``.
    """
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    if local_lens_ptr is not None:
        local_len = tl.load(local_lens_ptr + batch_idx)
    else:
        local_len = const_local_len
    for i in range(worker_id, local_len, num_workers):
        j = local_start + i
        block_in_seq = j // cache_block_size
        pos_in_block = j % cache_block_size
        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        # int64: physical_block_idx * block_stride can exceed 2^31 (matches
        # the insert/gather kernels above).
        cache_block_ptr = (
            k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        )
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        out_ptr = staging_ptr + batch_idx * staging_stride0 + i * staging_stride1

        data_offsets = tl.arange(0, 64)
        for chunk_idx in tl.static_range(token_data_size // 64):
            vals = tl.load(token_data_ptr + chunk_idx * 64 + data_offsets)
            tl.store(out_ptr + chunk_idx * 64 + data_offsets, vals)

        scale_offsets = tl.arange(0, scale_dim)
        scales = tl.load(token_scale_ptr + scale_offsets)
        tl.store(out_ptr + token_data_size + scale_offsets, scales)


@triton.jit
def _sm86_pack_swa_window_kernel(
    staging_ptr,  # [num_reqs, max_gather, 584] uint8 contiguous
    k_cache_ptr,  # paged fp8_ds_mla cache (uint8 bytes)
    seq_lens_ptr,  # [num_reqs] int32: GLOBAL sequence lengths
    gather_lens_ptr,  # [num_reqs] int32: SWA window token counts
    block_table_ptr,  # [num_reqs, max_blocks_per_seq] int32 (replicated SWA)
    max_blocks_per_seq,
    staging_stride0,
    staging_stride1,
    cache_block_size: tl.constexpr,  # tokens per SWA cache page
    token_data_size: tl.constexpr,  # 576
    scale_dim: tl.constexpr,  # 8
    block_stride: tl.constexpr,  # cache bytes per page
):
    """Copy the SWA window tokens out of the paged cache, VERBATIM (P6).

    The byte twin of ``_dequantize_and_gather_k_kernel``'s SWA gather:
    staging row ``i`` of request ``c`` holds the fp8_ds_mla bytes of token
    position ``(seq_len - gather_len) + i`` -- same ``start_pos`` arithmetic,
    same page addressing, but bytes are moved untouched (rule 8) so the
    flash-mla prefill op can dequantize them in-kernel exactly once. Each
    staging row is ``[576 data bytes | 8 scale bytes]``, i.e. a valid
    ``block_size=1`` fp8_ds_mla page.
    """
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - gather_len
    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        # int64: physical_block_idx * block_stride can exceed 2^31 (matches
        # the insert/gather kernels above).
        cache_block_ptr = (
            k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        )
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        out_ptr = staging_ptr + batch_idx * staging_stride0 + i * staging_stride1

        data_offsets = tl.arange(0, 64)
        for chunk_idx in tl.static_range(token_data_size // 64):
            vals = tl.load(token_data_ptr + chunk_idx * 64 + data_offsets)
            tl.store(out_ptr + chunk_idx * 64 + data_offsets, vals)

        scale_offsets = tl.arange(0, scale_dim)
        scales = tl.load(token_scale_ptr + scale_offsets)
        tl.store(out_ptr + token_data_size + scale_offsets, scales)


def sm86_pack_swa_window_entries(
    swa_k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_gather: int,
) -> torch.Tensor:
    """Pack the SWA window tokens into a compact ``[R * max_gather, 584]``
    uint8 staging buffer (P6 flash-mla prefill).

    Staging row of (request ``c``, window slot ``i``) is
    ``c * max_gather + i`` where ``i`` counts from ``seq_len - gather_len``
    -- the identical index space ``combine_topk_swa_indices`` uses for the
    SWA half (``pos - gather_start``), just flat over rows instead of
    offset into a bf16 workspace. ``max_gather`` is a host upper bound on
    every ``gather_lens`` value (rows past ``gather_len`` are uninitialized
    and must never be addressed inside a row's ``lens``).
    """
    num_reqs = seq_lens.shape[0]
    assert max_gather > 0
    staging = torch.empty(
        (num_reqs, max_gather, _SM86_DCP_ENTRY_BYTES),
        dtype=torch.uint8,
        device=swa_k_cache.device,
    )
    _sm86_pack_swa_window_kernel[(num_reqs, _SM86_DCP_PACK_NUM_WORKERS)](
        staging,
        swa_k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_table.shape[-1],
        staging.stride(0),
        staging.stride(1),
        cache_block_size=block_size,
        token_data_size=_SM86_DCP_ENTRY_DATA_BYTES,
        scale_dim=_SM86_DCP_ENTRY_SCALE_BYTES,
        block_stride=swa_k_cache.stride(0),
    )
    return staging.reshape(num_reqs * max_gather, _SM86_DCP_ENTRY_BYTES)


def sm86_dcp_allgather_k_entries(
    k_cache: torch.Tensor,
    entry_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    dcp_group: "GroupCoordinator",
    dcp_interleave: int,
    max_entries: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack + all-gather the DCP-sharded compressed entries, VERBATIM bytes.

    Steps (per prefill chunk; eager -- prefill is never captured):
      1. pack this rank's local entries (verbatim fp8 bytes + UE8M0 scales)
         into a contiguous ``[num_reqs, max_local, 584]`` staging buffer via
         the P1-sharded block table;
      2. all-gather the staging buffers over the DCP group (uint8, fixed
         NCCL rank order); every shape below derives from GLOBAL seq lens,
         so all ranks issue an identical, symmetric collective;
      3. build a virtual ``cache_block_size=1`` block table over the
         gathered buffer that maps GLOBAL entry ``e`` to the owning rank's
         staging row via the shared forward formulas ``sm86_dcp_owner`` /
         ``sm86_dcp_global_to_local`` (single source of truth with
         P2b/P2c -- never reimplemented here).  P4 memoizes this table: it
         depends only on host ints, not on the layer or the KV data.

    Returns ``(gathered_rows, virtual_block_table, max_local)``:
      - ``gathered_rows``: ``[world * num_reqs * max_local, 584]`` uint8.
        Flat row of (rank r, request c, local j) is
        ``(r * num_reqs + c) * max_local + j``; each 584-byte row is a valid
        ``cache_block_size=1`` page (576 data bytes, 8 scale bytes at +576).
      - ``virtual_block_table``: ``[num_reqs, max_entries]`` int32,
        ``vbt[c, e]`` = flat gathered row of GLOBAL entry ``e`` for request
        ``c`` (memoized + SHARED: callers must treat it as READ-ONLY).
      - ``max_local``: the rank-invariant staging width.

    ``entry_lens`` are GLOBAL compressed-entry counts (``seq_lens //
    compress_ratio``); ``max_entries`` is their chunk-wide max, computed
    from CPU seq lens by the caller (rank-invariant).
    """
    # Lazy import: backends.utils is a large module; keep the default import
    # path of this file lean (same pattern as the cutedsl import below).
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    world = dcp_group.world_size
    rank = dcp_group.rank_in_group
    num_reqs = entry_lens.shape[0]
    device = k_cache.device

    # This rank's owned-entry count per request (base layout helper -- the
    # same algebra as owner/local above, in count form).
    local_lens = get_dcp_local_seq_lens(entry_lens, world, rank, dcp_interleave)

    # Rank-invariant staging width: rank 0 always owns the most entries
    # (base + min(rem, I)), so this bounds every rank's local count.
    interleave_cycle = dcp_interleave * world
    max_local = (
        max_entries // interleave_cycle * dcp_interleave
        + min(max_entries % interleave_cycle, dcp_interleave)
    )
    assert max_local > 0  # caller skips the max_entries == 0 case

    staging = torch.empty(
        (num_reqs, max_local, _SM86_DCP_ENTRY_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    _sm86_dcp_pack_k_entries_kernel[(num_reqs, _SM86_DCP_PACK_NUM_WORKERS)](
        staging,
        k_cache,
        local_lens,
        block_table,
        block_table.shape[-1],
        staging.stride(0),
        staging.stride(1),
        cache_block_size=block_size,
        token_data_size=_SM86_DCP_ENTRY_DATA_BYTES,
        scale_dim=_SM86_DCP_ENTRY_SCALE_BYTES,
        block_stride=k_cache.stride(0),
    )

    # [world * num_reqs, max_local, 584], rank-major (fixed NCCL rank order).
    gathered = dcp_group.all_gather(staging, dim=0)
    gathered_rows = gathered.reshape(
        world * num_reqs * max_local, _SM86_DCP_ENTRY_BYTES
    )

    # Virtual block table in GLOBAL entry order. (rank, local) -> global is a
    # bijection, and for every global e < max_entries the owning rank's local
    # index is < max_local, so every one of the max_entries slots is defined.
    # P4: closed-form + memoized; see _sm86_dcp_virtual_block_table. The
    # returned tensor is SHARED -- read-only for every caller.
    virtual_block_table = _sm86_dcp_virtual_block_table(
        max_entries, max_local, num_reqs, world, dcp_interleave, device
    )
    return gathered_rows, virtual_block_table, max_local


# --------------------------------------------------------------------------
# P7 (VLLM_DSV4_DELTA_GATHER): delta compressed-entry gather. Compressed
# entries are written ONCE at their block boundary and never mutated, so the
# per-chunk full re-gather above moves O(P^2/(m*F)) redundant bytes over a
# chunked prefill. Instead each tracked request keeps a persistent per-layer
# staging buffer in GLOBAL entry order (row e = global entry e's 584 bytes --
# row ids stable across chunks, so index translation is the identity for
# both consumers), and each chunk only gathers the NEW entries
# [prev_count, new_count) and scatters them into place. Rule 8 throughout:
# the same pack kernel moves the same verbatim bytes; the only dequant is
# still the existing kernel (Triton path) or the flash op's in-kernel
# pre-pass (P6 path).
# --------------------------------------------------------------------------

# Memoized per-device identity "virtual block table" (an int32 arange) for
# reading a GLOBAL-entry-order staging with the existing dequant kernel at
# cache_block_size=1: entry e lives at staging row e, so the table is the
# identity. Grown geometrically, sliced per call; READ-ONLY for callers
# (same contract as _sm86_dcp_virtual_block_table -- the dequant kernel
# never writes its block table).
_SM86_DCP_IDENTITY_BT: dict[torch.device, torch.Tensor] = {}


def sm86_dcp_identity_block_table(
    num_entries: int, device: torch.device
) -> torch.Tensor:
    """``[1, num_entries]`` int32 view of a cached identity arange."""
    assert num_entries > 0
    cached = _SM86_DCP_IDENTITY_BT.get(device)
    if cached is None or cached.numel() < num_entries:
        capacity = max(next_power_of_2(num_entries), 4096)
        cached = torch.arange(capacity, dtype=torch.int32, device=device)
        _SM86_DCP_IDENTITY_BT[device] = cached
    return cached[:num_entries].view(1, num_entries)


def sm86_dcp_delta_gather_k_entries(
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    dcp_group: "GroupCoordinator",
    dcp_interleave: int,
    jobs: "list[tuple[int, int, int, torch.Tensor]]",
) -> None:
    """Gather ONLY the delta entries of tracked requests into their staging.

    ``jobs`` is a list of ``(row, prev, new, staging)``:
      - ``row``: this request's row in ``block_table`` (chunk-relative);
      - ``prev``/``new``: HOST-int global compressed-entry counts before /
        after this chunk (from CPU prefill seq lens, which are precise for
        prefill rows -- see CommonAttentionMetadata.seq_lens_cpu_upper_bound);
      - ``staging``: the persistent ``[capacity >= new, 584]`` uint8 buffer
        in GLOBAL entry order; rows ``[prev, new)`` are written, rows
        ``[0, prev)`` are never touched (append-only, so row ids -- and the
        bytes behind them -- are stable across chunks).

    Steps (eager only; the callers sit behind the prefill capture guard):
      1. per job, pack this rank's owned entries of the delta range. The
         owned subset of global ``[prev, new)`` is the CONTIGUOUS local range
         ``[local_count(prev), local_count(new))`` (the local enumeration is
         order-preserving), so the existing pack kernel handles it with just
         a start offset + host-scalar count (P7 kernel extension);
      2. one all-gather over the DCP group with equal padded per-rank widths
         (``max_pad`` = max delta local count over ALL jobs AND ranks --
         pure host math from rank-invariant CPU lens, so every rank issues
         an identical, symmetric collective);
      3. per job, scatter the gathered rows to their GLOBAL positions with a
         precomputed index_copy_ (fixed shapes -- ``new - prev`` is a host
         int -- elementwise layout algebra only, no ``nonzero()``, no host
         sync, no H2D copy).

    The layout algebra is the shared single source of truth
    (``sm86_dcp_owner`` / ``sm86_dcp_global_to_local`` /
    ``sm86_dcp_local_count`` in ``sm86_dcp_layout.py``) -- never re-derived
    here. Bytes and scales move verbatim (rule 8): the staging contents for
    rows ``[0, new)`` are byte-identical to what the full re-gather above
    produces for the same global entries (proven in
    ``scratchpad/sim_p7_delta.py``).
    """
    world = dcp_group.world_size
    rank = dcp_group.rank_in_group
    device = k_cache.device
    num_jobs = len(jobs)
    assert num_jobs > 0

    # Host planning: per-(job, rank) delta local start/count. Rank-invariant
    # inputs => every rank computes identical shapes.
    starts = [
        [
            sm86_dcp_local_count(prev, r, world, dcp_interleave)
            for r in range(world)
        ]
        for (_, prev, _, _) in jobs
    ]
    counts = [
        [
            sm86_dcp_local_count(new, r, world, dcp_interleave) - starts[t][r]
            for r in range(world)
        ]
        for t, (_, _, new, _) in enumerate(jobs)
    ]
    max_pad = max(cnt for per_rank in counts for cnt in per_rank)
    if max_pad == 0:
        # No rank owns any delta entry anywhere (all deltas empty); skipped
        # BEFORE the collective, symmetric on every rank.
        return

    send = torch.empty(
        (num_jobs, max_pad, _SM86_DCP_ENTRY_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    for t, (row, _prev, _new, _staging) in enumerate(jobs):
        cnt = counts[t][rank]
        if cnt == 0:
            # Nothing owned in this delta on this rank; the padding rows are
            # never scattered by any receiver (the scatter indices below only
            # address rows < counts[t][owner]).
            continue
        _sm86_dcp_pack_k_entries_kernel[(1, _SM86_DCP_PACK_NUM_WORKERS)](
            send[t],
            k_cache,
            None,  # local_lens_ptr: count comes from the host scalar below
            block_table[row : row + 1],
            block_table.shape[-1],
            0,  # staging_stride0: single-request launch (batch_idx == 0)
            send.stride(1),
            cache_block_size=block_size,
            token_data_size=_SM86_DCP_ENTRY_DATA_BYTES,
            scale_dim=_SM86_DCP_ENTRY_SCALE_BYTES,
            block_stride=k_cache.stride(0),
            local_start=starts[t][rank],
            const_local_len=cnt,
        )

    # [world * num_jobs, max_pad, 584], rank-major (fixed NCCL rank order).
    gathered = dcp_group.all_gather(send, dim=0)
    gathered_rows = gathered.reshape(
        world * num_jobs * max_pad, _SM86_DCP_ENTRY_BYTES
    )

    cycle = world * dcp_interleave
    for t, (_row, prev, new, staging) in enumerate(jobs):
        delta = new - prev
        if delta <= 0:
            continue
        assert staging.shape[0] >= new
        e = torch.arange(prev, new, dtype=torch.int64, device=device)
        owner = sm86_dcp_owner(e, world, dcp_interleave)
        local = sm86_dcp_global_to_local(e, owner, world, dcp_interleave)
        # Per-rank delta local start as a device tensor, computed by the same
        # closed form as sm86_dcp_local_count from host-int `prev` (no H2D).
        r = torch.arange(world, dtype=torch.int64, device=device)
        start_per_rank = (prev // cycle) * dcp_interleave + torch.clamp(
            (prev % cycle) - r * dcp_interleave, 0, dcp_interleave
        )
        src = (owner * num_jobs + t) * max_pad + (local - start_per_rank[owner])
        staging.index_copy_(0, e, gathered_rows.index_select(0, src))


def _sm86_dcp_allgather_dequantize_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    entry_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    dcp_group: "GroupCoordinator",
    dcp_interleave: int,
    max_entries: int,
    use_fnuz: bool,
) -> None:
    """All-gather the DCP-sharded compressed entries; dequantize in GLOBAL order.

    ``sm86_dcp_allgather_k_entries`` does the byte movement (see its
    docstring); this wrapper then runs the EXISTING dequant kernel against
    the gathered buffer: the dense bf16 output lands in GLOBAL entry order,
    so the unchanged prefill pipeline (combine_topk_swa_indices with P2b's
    GLOBAL topk indices + sparse prefill attention) consumes it as if dcp
    were 1.
    """
    gathered_rows, virtual_block_table, _ = sm86_dcp_allgather_k_entries(
        k_cache,
        entry_lens,
        block_table,
        block_size,
        dcp_group,
        dcp_interleave,
        max_entries,
    )

    # Existing dequant kernel over the gathered buffer viewed as a
    # cache_block_size=1 paged cache (each 584-byte row: 576 data + 8 scales
    # at row offset 576 -- exactly what the kernel derives for block size 1).
    dequantize_and_gather_k_cache_triton(
        out,
        gathered_rows,
        seq_lens=entry_lens,
        gather_lens=None,
        block_table=virtual_block_table,
        block_size=1,
        offset=offset,
        use_fnuz=use_fnuz,
    )


def dequantize_and_gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
    dcp_group: "GroupCoordinator | None" = None,
    dcp_interleave: int = 1,
    dcp_max_entries: int | None = None,
) -> None:
    """Dequantize and gather a paged DSv4 K cache.

    ``use_fnuz`` MUST match the encoder of the specific cache being read:
    ``False`` for ``compressed_k_cache`` (Triton encoder is OCP everywhere),
    ``current_platform.is_fp8_fnuz()`` for ``swa_k_cache`` (C++ encoder
    writes FNUZ on gfx942 and OCP on gfx950).

    DCP (VLLM_SM86_DCP, P2d): pass ``dcp_group`` (world > 1) when ``k_cache``
    is a DCP-SHARDED compressed-KV cache read through a P1-sharded block
    table. ``seq_lens`` must then be GLOBAL compressed-entry counts,
    ``gather_lens`` must be None (the compressed gather always reads the
    full prefix) and ``dcp_max_entries`` their chunk max (an int derived
    from CPU seq lens, identical on all ranks). The entries are
    all-gathered over the DCP group and dequantized into GLOBAL entry
    order -- ``out`` is then bit-identical in meaning to the dcp=1 gather.
    The replicated (dcp_exempt) SWA cache must NOT pass ``dcp_group``.
    ``combine_topk_swa_indices`` needs no CP branch on this design: topk
    indices stay GLOBAL and the gathered buffer is dense global order.
    """
    if dcp_group is not None and dcp_group.world_size > 1:
        assert gather_lens is None, (
            "SM86 DCP compressed gather reads the full prefix; the "
            "replicated SWA gather must not pass dcp_group."
        )
        assert dcp_max_entries is not None
        if dcp_max_entries <= 0:
            # No completed compressed entries anywhere in the chunk; the
            # non-DCP path would gather nothing either. Skipped on every
            # rank symmetrically (derived from global CPU seq lens).
            return
        # The cutedsl gather has no CP-layout support (and SM8x never takes
        # it); the DCP branch always runs the Triton path.
        _sm86_dcp_allgather_dequantize_k_cache(
            out,
            k_cache,
            seq_lens,
            block_table,
            block_size,
            offset,
            dcp_group,
            dcp_interleave,
            dcp_max_entries,
            use_fnuz,
        )
        return

    if is_cutedsl_supported():
        # lazily import, otherwise some tests fail due to CUDA driver init failure.
        from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
            dequantize_and_gather_k_cache_cutedsl,
        )

        dequantize_and_gather_k_cache_cutedsl(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
        )
        return

    dequantize_and_gather_k_cache_triton(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=use_fnuz,
    )


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV cache slots and count valid entries.

    Fuses three operations into a single kernel:
    1. Block-table lookup (local index → global slot id)
    2. Valid-entry counting (topk_lens per token)
    3. Masking padding tokens to length 0
    """
    num_tokens = topk_indices.shape[0]
    if output_buffers is None:
        global_topk_indices = torch.empty_like(topk_indices)
        topk_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        global_topk_indices, topk_lens = output_buffers
        assert global_topk_indices.shape == topk_indices.shape
        assert topk_lens.shape == (num_tokens,)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        is_valid = local_idx >= 0

        block_indices = local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & is_valid,
        )
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=0)

    # Zero out length for padding tokens.
    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


# FlashMLA sparse prefill asserts `params.topk % B_TOPK == 0` (see
# flashmla/csrc/sm100/prefill/sparse/fwd/head{64,128}/phase1.cuh). B_TOPK is
# 64 for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
# The extra slots stay as -1 sentinels and `combined_lens` caps the valid
# range via `topk_length`, so padding is a no-op at kernel level.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate topk compressed indices with SWA window indices.

    DCP note (VLLM_SM86_DCP, P2d): deliberately NO CP branch. Under DCP the
    prefill design all-gathers the compressed shards into a dense buffer in
    GLOBAL entry order (see ``dequantize_and_gather_k_cache``) and the P2b
    indexer merge emits GLOBAL entry indices, so the topk arithmetic here
    (``e + M * batch``) is already correct; the SWA cache is replicated
    (dcp_exempt), so the window arithmetic is unchanged too. The Lasimeri
    reference localized both here only because it sharded the SWA cache and
    ran local-KV prefill attention with an LSE merge -- a different design.
    """
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    if out is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            fill_value=-1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        combined_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        combined_indices, combined_lens = out

    _COMBINE_TOPK_SWA_INDICES_KERNEL(
        combined_indices,
        combined_lens,
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
    )
    return combined_indices, combined_lens


_COMBINE_TOPK_SWA_NUM_WORKERS = 128


# Representative pointer alignment variants for Triton pointer specialization.
_COMBINE_TOPK_SWA_POINTER_INPUTS = zip_inputs(
    dict(
        topk_indices=True,
        query_start_loc=True,
        seq_lens=True,
        gather_lens=True,
    ),
    dict(
        topk_indices=True,
        query_start_loc=False,
        seq_lens=False,
        gather_lens=True,
    ),
    dict(
        topk_indices=False,
        query_start_loc=False,
        seq_lens=False,
        gather_lens=False,
    ),
)


_DSV4_COMBINE_TOPK_SWA_WARMUP_INPUTS = zip_inputs(
    # DSv4-Flash / SWA-only and C4A.
    dict(compress_ratio=1, topk=0, topk_width=512),
    dict(compress_ratio=4, topk=512, topk_width=512),
    # DSv4-Pro C4A.
    dict(compress_ratio=4, topk=1024, topk_width=1024),
    # DSv4-Pro C128A.
    dict(compress_ratio=128, topk=8192, topk_width=8192),
)


def _hf_config_int(vllm_config: Any, name: str, default: int) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return int(getattr(hf_config, name, default) or default)


def _scheduler_config_int(vllm_config: Any, name: str, default: int) -> int:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return int(getattr(scheduler_config, name, default) or default)


class CombineTopkSwaIndicesKernel(
    VllmJitKernel["CombineTopkSwaIndicesKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        TOP_K: int
        COMPRESS_RATIO: int
        WINDOW_SIZE: int
        PADDED_TOP_K: int
        input_variant: TritonPointerInputVariant

    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "combined_indices_stride",
            "topk_indices_stride",
            "M",
            "N",
        ]
    )
    def kernel(
        combined_indices_ptr,
        combined_indices_stride,
        combined_lens_ptr,
        topk_indices_ptr,
        topk_indices_stride,
        query_start_loc_ptr,
        seq_lens_ptr,
        gather_lens_ptr,
        M,
        N,
        TOP_K: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        PADDED_TOP_K: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        worker_id = tl.program_id(1)
        num_workers = tl.num_programs(1)

        # query_start_loc is a global tensor; rebase to chunk-local offsets
        # by subtracting the chunk's starting value.
        base = tl.load(query_start_loc_ptr)
        query_start = tl.load(query_start_loc_ptr + batch_idx) - base
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
        query_len = query_end - query_start
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        gather_len = tl.load(gather_lens_ptr + batch_idx)
        start_pos = seq_len - query_len
        # The SWA portion of the gathered buffer starts from position
        # (seq_len - gather_len), not position 0. We need this offset
        # to correctly index into the gathered buffer.
        gather_start = seq_len - gather_len

        for token_idx in range(query_start + worker_id, query_end, num_workers):
            # topk_len is fully determined by the query token's absolute position:
            # both the C4A indexer and the C128A metadata builder emit
            # min((pos + 1) // compress_ratio, topk_tokens) valid entries.
            # Caller passes TOP_K=0 for SWA-only layers to zero this out.
            token_idx_in_query = token_idx - query_start
            pos = start_pos + token_idx_in_query
            topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
            swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

            offset = tl.arange(0, PADDED_TOP_K)
            mask = offset < topk_len
            topk_indices = tl.load(
                topk_indices_ptr + token_idx * topk_indices_stride + offset,
                mask=mask,
            )
            tl.store(
                combined_indices_ptr + token_idx * combined_indices_stride + offset,
                topk_indices + M * batch_idx,
                mask=mask,
            )
            offset = tl.arange(0, WINDOW_SIZE)
            # Index into gathered buffer: N + (position - gather_start)
            # For positions [pos - swa_len + 1, pos], the buffer indices are:
            # [N + pos - swa_len + 1 - gather_start, N + pos - gather_start]
            tl.store(
                combined_indices_ptr
                + token_idx * combined_indices_stride
                + topk_len
                + offset,
                M * batch_idx + N + offset + pos - swa_len + 1 - gather_start,
                mask=offset < swa_len,
            )

            combined_len = topk_len + swa_len
            tl.store(combined_lens_ptr + token_idx, combined_len)

    def dispatch(  # type: ignore[override]
        self,
        *,
        topk_width: int,
        topk_indices: bool,
        query_start_loc: bool,
        seq_lens: bool,
        gather_lens: bool,
        topk: int,
        compress_ratio: int,
        WINDOW_SIZE: int,
    ) -> CompileKey:
        padded_topk = next_power_of_2(topk_width)
        input_variant = TritonPointerInputVariant.from_alignment(
            topk_indices=topk_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
        )
        return self.CompileKey(
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=WINDOW_SIZE,
            PADDED_TOP_K=padded_topk,
            input_variant=input_variant,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        if _scheduler_config_int(vllm_config, "max_num_batched_tokens", 0) <= 0:
            return []

        window_size = _hf_config_int(vllm_config, "sliding_window", 128)
        return self._trace_dispatch(self.dispatch)(
            _DSV4_COMBINE_TOPK_SWA_WARMUP_INPUTS,
            _COMBINE_TOPK_SWA_POINTER_INPUTS,
            WINDOW_SIZE=window_size,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        input_variant = compile_key.input_variant
        warmup(
            int32_ptr,
            1,  # do not specialize combined_indices_stride
            int32_ptr,
            input_variant.pointer("topk_indices", torch.int32),
            1,  # do not specialize topk_indices_stride
            input_variant.pointer("query_start_loc", torch.int32),
            input_variant.pointer("seq_lens", torch.int32),
            input_variant.pointer("gather_lens", torch.int32),
            1,  # do not specialize M
            1,  # do not specialize N
            TOP_K=compile_key.TOP_K,
            COMPRESS_RATIO=compile_key.COMPRESS_RATIO,
            WINDOW_SIZE=compile_key.WINDOW_SIZE,
            PADDED_TOP_K=compile_key.PADDED_TOP_K,
            grid=(1, _COMBINE_TOPK_SWA_NUM_WORKERS),
        )

    def __call__(
        self,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        topk_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor,
        M: int,
        N: int,
        *,
        TOP_K: int,
        COMPRESS_RATIO: int,
        WINDOW_SIZE: int,
    ) -> None:
        num_reqs = seq_lens.shape[0]
        self.kernel[(num_reqs, _COMBINE_TOPK_SWA_NUM_WORKERS)](
            combined_indices,
            combined_indices.stride(0),
            combined_lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            TOP_K=TOP_K,
            COMPRESS_RATIO=COMPRESS_RATIO,
            WINDOW_SIZE=WINDOW_SIZE,
            PADDED_TOP_K=next_power_of_2(topk_indices.shape[-1]),
        )


_COMBINE_TOPK_SWA_INDICES_KERNEL = CombineTopkSwaIndicesKernel()


def build_flashinfer_mixed_sparse_indices(
    decode_swa_indices: torch.Tensor,
    decode_compressed_indices: torch.Tensor | None,
    decode_compressed_topk_lens: torch.Tensor | None,
    prefill_topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_block_size: int,
    compressed_block_table: torch.Tensor | None,
    compressed_block_size: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    decode_compressed_indices_are_local: bool = False,
    decode_is_valid_token: torch.Tensor | None = None,
    swa_block_span: int | None = None,
    compressed_block_span: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the FlashInfer DSV4 sparse-index matrix for decode-first batches.

    Produces ``sparse_indices`` of shape ``[num_tokens, window_size +
    padded_topk]`` (the first ``window_size`` columns are SWA slot ids, the rest
    are compressed/top-k slot ids) and ``sparse_topk_lens`` (active length per
    token). Decode tokens read precomputed SWA/compressed indices; prefill tokens
    derive their SWA window from the position and translate local compressed
    indices to global slots via the block tables.
    """
    assert decode_swa_indices.dtype == torch.int32
    assert decode_swa_indices.dim() == 2
    assert decode_swa_indices.shape[-1] == window_size
    if decode_compressed_topk_lens is not None:
        assert decode_compressed_topk_lens.dtype == torch.int32
    assert prefill_topk_indices.dtype == torch.int32
    assert prefill_topk_indices.dim() == 2
    assert query_start_loc.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert token_to_req_indices.dtype == torch.int32
    assert swa_block_table.dtype == torch.int32

    num_decode_tokens = decode_swa_indices.shape[0]
    num_prefill_tokens = prefill_topk_indices.shape[0]
    num_tokens = num_decode_tokens + num_prefill_tokens
    assert token_to_req_indices.shape[0] >= num_tokens
    if decode_compressed_topk_lens is not None:
        assert decode_compressed_topk_lens.shape[0] >= num_decode_tokens

    decode_compressed_topk = 0
    if decode_compressed_indices is None:
        decode_compressed_indices = prefill_topk_indices
    else:
        assert decode_compressed_indices.dtype == torch.int32
        assert decode_compressed_indices.dim() == 2
        assert decode_compressed_indices.shape[0] == num_decode_tokens
        decode_compressed_topk = decode_compressed_indices.shape[-1]
    if decode_compressed_topk > 0 and decode_compressed_indices_are_local:
        assert decode_is_valid_token is not None
        assert decode_is_valid_token.dtype == torch.bool
        assert decode_is_valid_token.shape[0] >= num_decode_tokens
    else:
        decode_is_valid_token = token_to_req_indices

    if compressed_block_table is None:
        compressed_block_table = swa_block_table
    assert compressed_block_table.dtype == torch.int32
    has_decode_compressed_lens = decode_compressed_topk_lens is not None
    if decode_compressed_topk_lens is None:
        decode_compressed_topk_lens = token_to_req_indices

    # The FlashInfer TRTLLM-gen sparse-MLA kernels require every per-token topk
    # index row to start on a 16-byte boundary: the kernel loads the compressed
    # indices with 128-bit (16-byte) vectorized loads, so a misaligned row would
    # fault or read across rows. 16 bytes = 4 int32 indices, so round the topk
    # width (and hence the row stride, since the SWA columns are fixed-width) up
    # to a multiple of 4. The extra columns are filled with -1 (invalid) and bounded
    # by ``sparse_topk_lens``, so padding never changes the attention result.
    padded_topk = max(topk, decode_compressed_topk)
    padded_topk = (padded_topk + 3) // 4 * 4
    sparse_indices = torch.empty(
        (num_tokens, window_size + padded_topk),
        dtype=torch.int32,
        device=decode_swa_indices.device,
    )
    sparse_topk_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=decode_swa_indices.device
    )
    if num_tokens == 0:
        return sparse_indices, sparse_topk_lens

    window_block_size = triton.next_power_of_2(max(window_size, 1))
    topk_block_size = triton.next_power_of_2(max(padded_topk, 1))
    max_block_size = max(window_block_size, topk_block_size)
    num_warps = 4 if max_block_size >= 256 else 1

    # block_span = page_stride / token_stride; == block_size (no-op) for unpacked KV.
    swa_span = swa_block_size if swa_block_span is None else swa_block_span
    compressed_span = (
        compressed_block_size
        if compressed_block_span is None
        else compressed_block_span
    )
    _build_flashinfer_mixed_sparse_indices_kernel[(num_tokens,)](
        sparse_indices,
        sparse_indices.stride(0),
        sparse_topk_lens,
        decode_swa_indices,
        decode_swa_indices.stride(0),
        decode_compressed_indices,
        decode_compressed_indices.stride(0),
        decode_compressed_topk_lens,
        decode_is_valid_token,
        prefill_topk_indices,
        prefill_topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        swa_block_table,
        swa_block_table.stride(0),
        swa_block_size,
        swa_span,
        compressed_block_table,
        compressed_block_table.stride(0),
        compressed_block_size,
        compressed_span,
        NUM_DECODE_TOKENS=num_decode_tokens,
        WINDOW_SIZE=window_size,
        COMPRESS_RATIO=compress_ratio,
        TOP_K=topk,
        PADDED_TOP_K=padded_topk,
        PREFILL_TOPK_STRIDE=prefill_topk_indices.shape[-1],
        DECODE_COMPRESSED_TOPK=decode_compressed_topk,
        DECODE_COMPRESSED_INDICES_ARE_LOCAL=decode_compressed_indices_are_local,
        HAS_DECODE_COMPRESSED_LENS=has_decode_compressed_lens,
        WINDOW_BLOCK_SIZE=window_block_size,
        TOPK_BLOCK_SIZE=topk_block_size,
        num_warps=num_warps,
    )
    return sparse_indices, sparse_topk_lens


@triton.jit
def _remap_flashinfer_index(values, block_size, block_span):
    # FlashInfer's DSv4 kernel indexes sparse KV by physical token stride, so
    # packed pages (#44577) need block*block_size+off -> block*block_span+off.
    # TODO: remove once flashinfer-ai/flashinfer#3856 is fixed.
    is_valid = values >= 0
    safe_values = tl.where(is_valid, values, 0)
    values = (safe_values // block_size) * block_span
    values += safe_values % block_size
    return tl.where(is_valid, values, -1)


@triton.jit(
    do_not_specialize=[
        "sparse_indices_stride",
        "decode_swa_stride",
        "decode_compressed_stride",
        "prefill_topk_stride",
        "swa_block_table_stride",
        "swa_block_size",
        "swa_block_span",
        "compressed_block_table_stride",
        "compressed_block_size",
        "compressed_block_span",
        "NUM_DECODE_TOKENS",
        "PREFILL_TOPK_STRIDE",
    ]
)
def _build_flashinfer_mixed_sparse_indices_kernel(
    sparse_indices_ptr,
    sparse_indices_stride,
    sparse_topk_lens_ptr,
    decode_swa_indices_ptr,
    decode_swa_stride,
    decode_compressed_indices_ptr,
    decode_compressed_stride,
    decode_compressed_topk_lens_ptr,
    decode_is_valid_token_ptr,
    prefill_topk_indices_ptr,
    prefill_topk_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    swa_block_table_ptr,
    swa_block_table_stride,
    swa_block_size,
    swa_block_span,
    compressed_block_table_ptr,
    compressed_block_table_stride,
    compressed_block_size,
    compressed_block_span,
    NUM_DECODE_TOKENS,
    WINDOW_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOP_K: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PREFILL_TOPK_STRIDE,
    DECODE_COMPRESSED_TOPK: tl.constexpr,
    DECODE_COMPRESSED_INDICES_ARE_LOCAL: tl.constexpr,
    HAS_DECODE_COMPRESSED_LENS: tl.constexpr,
    WINDOW_BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    if token_idx < NUM_DECODE_TOKENS:
        for i in range(0, WINDOW_SIZE, WINDOW_BLOCK_SIZE):
            offset = i + tl.arange(0, WINDOW_BLOCK_SIZE)
            mask = offset < WINDOW_SIZE
            values = tl.load(
                decode_swa_indices_ptr + token_idx * decode_swa_stride + offset,
                mask=mask,
                other=-1,
            )
            values = _remap_flashinfer_index(values, swa_block_size, swa_block_span)
            tl.store(
                sparse_indices_ptr + token_idx * sparse_indices_stride + offset,
                values,
                mask=mask,
            )

        compressed_len = tl.zeros((), dtype=tl.int32)
        for i in range(0, PADDED_TOP_K, TOPK_BLOCK_SIZE):
            offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
            mask = offset < PADDED_TOP_K
            values = tl.load(
                decode_compressed_indices_ptr
                + token_idx * decode_compressed_stride
                + offset,
                mask=offset < DECODE_COMPRESSED_TOPK,
                other=-1,
            )
            if DECODE_COMPRESSED_INDICES_ARE_LOCAL:
                token_valid = tl.load(decode_is_valid_token_ptr + token_idx)
                is_valid = values >= 0
                req_idx = tl.load(token_to_req_indices_ptr + token_idx)
                block_indices = values // compressed_block_size
                block_numbers = tl.load(
                    compressed_block_table_ptr
                    + req_idx * compressed_block_table_stride
                    + block_indices,
                    mask=mask & is_valid,
                    other=-1,
                )
                block_offsets = values % compressed_block_size
                values = block_numbers * compressed_block_size + block_offsets
                values = tl.where(is_valid, values, -1)
                compressed_len += tl.sum((is_valid & token_valid).to(tl.int32), axis=0)
            values = _remap_flashinfer_index(
                values, compressed_block_size, compressed_block_span
            )
            tl.store(
                sparse_indices_ptr
                + token_idx * sparse_indices_stride
                + WINDOW_SIZE
                + offset,
                values,
                mask=mask,
            )

        if DECODE_COMPRESSED_TOPK == 0:
            compressed_len = tl.zeros((), dtype=tl.int32)
        elif not DECODE_COMPRESSED_INDICES_ARE_LOCAL:
            if HAS_DECODE_COMPRESSED_LENS:
                compressed_len = tl.load(decode_compressed_topk_lens_ptr + token_idx)
            else:
                compressed_len = tl.full((), DECODE_COMPRESSED_TOPK, dtype=tl.int32)

        tl.store(sparse_topk_lens_ptr + token_idx, WINDOW_SIZE + compressed_len)
        return

    prefill_idx = token_idx - NUM_DECODE_TOKENS
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    start_pos = seq_len - query_len
    token_idx_in_query = token_idx - query_start
    pos = start_pos + token_idx_in_query
    swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
    swa_start_pos = pos - swa_len + 1
    topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)

    for i in range(0, WINDOW_SIZE, WINDOW_BLOCK_SIZE):
        offset = i + tl.arange(0, WINDOW_BLOCK_SIZE)
        mask = offset < WINDOW_SIZE
        pos_offset = swa_start_pos + offset
        block_indices = pos_offset // swa_block_size
        block_numbers = tl.load(
            swa_block_table_ptr + req_idx * swa_block_table_stride + block_indices,
            mask=mask & (offset < swa_len),
            other=-1,
        )
        block_offsets = pos_offset % swa_block_size
        slot_ids = block_numbers * swa_block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        slot_ids = _remap_flashinfer_index(slot_ids, swa_block_size, swa_block_span)
        tl.store(
            sparse_indices_ptr + token_idx * sparse_indices_stride + offset,
            slot_ids,
            mask=mask,
        )

    for i in range(0, PADDED_TOP_K, TOPK_BLOCK_SIZE):
        offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
        mask = offset < PADDED_TOP_K
        local_idx = tl.load(
            prefill_topk_indices_ptr + prefill_idx * prefill_topk_stride + offset,
            mask=(offset < PREFILL_TOPK_STRIDE) & (offset < topk_len),
            other=-1,
        )
        is_valid = local_idx >= 0
        block_indices = local_idx // compressed_block_size
        block_numbers = tl.load(
            compressed_block_table_ptr
            + req_idx * compressed_block_table_stride
            + block_indices,
            mask=mask & is_valid,
            other=-1,
        )
        block_offsets = local_idx % compressed_block_size
        slot_ids = block_numbers * compressed_block_size + block_offsets
        slot_ids = tl.where((offset < topk_len) & is_valid, slot_ids, -1)
        slot_ids = _remap_flashinfer_index(
            slot_ids, compressed_block_size, compressed_block_span
        )
        tl.store(
            sparse_indices_ptr
            + token_idx * sparse_indices_stride
            + WINDOW_SIZE
            + offset,
            slot_ids,
            mask=mask,
        )

    tl.store(sparse_topk_lens_ptr + token_idx, WINDOW_SIZE + topk_len)
