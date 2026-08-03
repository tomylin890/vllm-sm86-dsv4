# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    is_deep_gemm_supported,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.mla.sm86_dcp_layout import (
    sm86_dcp_global_to_local,
    sm86_dcp_local_to_global,
    sm86_dcp_owns,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.sm86_det_topk import (
    det_top_k_per_row_decode,
    det_top_k_per_row_flat_lengths,
    det_top_k_per_row_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


# --------------------------------------------------------------------------
# P2b (SM8x DSV4 DCP, gated by VLLM_SM86_DCP via
# DeepseekV32IndexerMetadata.use_sm86_dcp_topk): pure-torch deterministic
# global top-k for compressed-entry indexer shards. The CuteDSL merge above
# is Hopper+-only; these helpers are its SM8x replacement and additionally
# implement the explicit fp32 tie-break (lower global entry index wins)
# required by ARCHITECTURE.md section 10 rule 4.
#
# Index spaces: "local" indices are this rank's compressed-entry positions
# under the interleave-aware round-robin DCP layout
# (owner(e) = (e // interleave) % world); "global" indices are absolute
# compressed-entry positions. -1 is the invalid sentinel throughout.
#
# P2d refactor (no logic change): the pure ownership algebra
# (sm86_dcp_local_to_global / sm86_dcp_owns / sm86_dcp_global_to_local)
# moved verbatim to vllm/v1/attention/backends/mla/sm86_dcp_layout.py so
# the prefill all-gather (cache_utils.py) shares the exact same formulas.
# --------------------------------------------------------------------------

_SM86_DCP_INVALID_SCORE = float("-inf")


def _sm86_dcp_identity_selection(
    row_lens: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    """``[0, 1, ..., n-1, -1, -1, ...]`` per row, int32, width ``topk_tokens``.

    P4 / OPT-3.  When a row's candidate count ``n`` is <= ``topk_tokens``,
    "top-k of n" is the IDENTITY selection: every candidate is chosen, so the
    merge that computes *which* ones is pure overhead.  This is the same
    substitution the rest of the stack already makes -- the compiled selectors'
    own ``rowLen <= topK`` shortcut emits the valid entries in ascending column
    order (see ``v1/attention/ops/sm86_det_topk.py``), and DSV4's short-context
    fast path (``attention.py::_fill_short_context_topk_indices``) and the
    C128A DCP decode metadata (``[0..count-1, -1]`` rank-local rows) both
    produce exactly this shape.

    Order note: the selection is the same SET as the merge's, but in ascending
    index order instead of (score desc, global index asc).  Every consumer
    treats the row as a set plus a length -- ``combine_topk_swa_indices`` reads
    ``min((pos+1)//m, topk)`` LEADING slots and
    ``compute_global_topk_ragged_indices_and_indptr`` counts the ``>= 0``
    slots -- so prefix-compactness and the valid COUNT are what matter, and
    both hold here (ARCHITECTURE.md section 10 rule 4: set equivalence).

    Fixed shapes, no collectives, no host sync, capture-safe.
    """
    columns = torch.arange(
        topk_tokens, dtype=torch.int32, device=row_lens.device
    ).unsqueeze(0)
    lens = row_lens.to(torch.int32).reshape(-1, 1)
    return torch.where(columns < lens, columns, torch.full_like(columns, -1))


def _sm86_dcp_global_topk(
    local_values: torch.Tensor,
    local_global_indices: torch.Tensor,
    topk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather fixed-width per-rank candidates and take the deterministic
    global top-k.

    Exactness: any entry in the true global top-k has fewer than topk_tokens
    entries ranked above it globally, hence fewer than topk_tokens above it on
    its own rank, so it appears in its owning rank's local top-k candidates --
    merging only ``topk_tokens * dcp_world_size`` candidates is equivalent to
    a topk over the full concatenated score row.

    Determinism / tie-break (ARCHITECTURE.md section 10 rule 4): scores are
    compared in fp32 and ties are broken by LOWER global entry index. Plain
    torch.topk does not guarantee any tie order across different candidate
    orderings, so this is implemented with two stable argsorts: first
    ascending global index, then a stable descending sort on scores -- equal
    scores retain ascending-index order. This is a documented deviation from
    the Lasimeri reference (_dcp_global_topk used plain torch.topk); the
    selected SET only differs from the reference when fp32 scores tie at the
    top-k boundary. All ops are fixed-shape and sync-free (capture-safe); the
    all-gather concatenates in fixed rank order.

    Invalid candidates carry score -inf and index -1 on input; they are
    remapped to index int32-max for the tie-break sort so that a genuine
    candidate always wins, and any -inf survivor is masked back to -1.
    """
    dcp_group = get_dcp_group()
    cand_values = dcp_group.all_gather(local_values.contiguous(), dim=1)
    cand_indices = dcp_group.all_gather(local_global_indices.contiguous(), dim=1)
    invalid = cand_indices < 0
    cand_values = torch.where(
        invalid,
        torch.full_like(cand_values, _SM86_DCP_INVALID_SCORE),
        cand_values,
    )
    sort_indices = torch.where(
        invalid,
        torch.full_like(cand_indices, torch.iinfo(torch.int32).max),
        cand_indices,
    )
    idx_order = torch.argsort(sort_indices, dim=-1, stable=True)
    values_by_idx = torch.gather(cand_values, -1, idx_order)
    indices_by_idx = torch.gather(cand_indices, -1, idx_order)
    score_order = torch.argsort(
        values_by_idx, dim=-1, descending=True, stable=True
    )[..., :topk_tokens]
    top_values = torch.gather(values_by_idx, -1, score_order)
    top_indices = torch.gather(indices_by_idx, -1, score_order)
    top_indices = torch.where(
        top_values == _SM86_DCP_INVALID_SCORE,
        torch.full_like(top_indices, -1),
        top_indices,
    )
    return top_values, top_indices


def _sm86_dcp_topk_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    local_topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    has_local_kv: bool,
    identity_row_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge per-rank prefill local top-k into the global top-k.

    ``local_topk_indices`` come from ops.top_k_per_row_prefill and are
    relative to each row's band start ``cu_seqlen_ks`` -- under DCP the band
    holds this rank's LOCAL compressed entries, so the relative index IS the
    local entry index. Returns int32 GLOBAL entry indices padded with -1
    (matching both the Lasimeri reference prefill convention and the base
    _merge_dcp_topk_global output contract: the attention side localizes).
    Every rank must call this for every chunk (chunk splits are derived from
    global CPU seq lens, hence rank-invariant) so the all-gather stays
    symmetric even when has_local_kv is False.

    P4 / OPT-3: ``identity_row_lens`` (per row, the GLOBAL compressed-entry
    count) is passed ONLY when the caller has established host-side that no
    row can exceed ``topk_tokens`` candidates.  The union of every rank's
    shard is then the whole causal prefix, i.e. global entries
    ``0 .. n_global-1``, so both all-gathers and both stable argsorts are
    skipped for an identity fill in the SAME index space (GLOBAL, request
    relative -- exactly what ``combine_topk_swa_indices`` consumes).  The gate
    is a global host int, so every rank skips together and the collectives
    stay symmetric.
    """
    num_rows = local_topk_indices.shape[0]
    if identity_row_lens is not None:
        return _sm86_dcp_identity_selection(
            identity_row_lens[:num_rows], topk_tokens
        )
    if has_local_kv:
        gather_indices = torch.clamp(local_topk_indices, min=0).to(torch.int64)
        gather_indices = gather_indices + cu_seqlen_ks.to(torch.int64).unsqueeze(1)
        gather_indices = torch.clamp(gather_indices, max=logits.shape[1] - 1)
        gathered = torch.gather(logits, 1, gather_indices)
        local_values = torch.where(
            local_topk_indices >= 0,
            gathered,
            torch.full_like(gathered, _SM86_DCP_INVALID_SCORE),
        )
        local_indices = local_topk_indices
    else:
        # This rank holds no KV for this chunk: contribute all-invalid
        # candidates so the collective stays aligned across ranks.
        local_values = torch.full(
            (num_rows, topk_tokens),
            _SM86_DCP_INVALID_SCORE,
            dtype=torch.float32,
            device=local_topk_indices.device,
        )
        local_indices = torch.full(
            (num_rows, topk_tokens),
            -1,
            dtype=torch.int32,
            device=local_topk_indices.device,
        )
    global_candidates = sm86_dcp_local_to_global(
        local_indices, dcp_rank, dcp_world_size, cp_interleave
    )
    _, top_indices = _sm86_dcp_global_topk(
        local_values, global_candidates, topk_tokens
    )
    return top_indices


def _sm86_dcp_topk_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    local_topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    identity_selection: bool = False,
) -> torch.Tensor:
    """Merge per-rank decode local top-k into the global top-k, then keep only
    this rank's owned entries as LOCAL indices (the P2 interface contract).

    ``seq_lens`` are this rank's LOCAL compressed-entry counts (the metadata
    builder localizes divide-then-shard); ``local_topk_indices`` come from the
    local decode top-k kernels and index this rank's local entries. Output is
    int32 (rows, topk_tokens): owned entries first, ordered by (score desc,
    global index asc) via a stable compaction sort, padded with -1. Only the
    ORDER of the owned prefix differs from the Lasimeri reference (which
    compacted with plain torch.topk); the selected set is identical.

    P4 / OPT-3: with ``identity_selection`` the caller has established
    host-side that no row's GLOBAL entry count exceeds ``topk_tokens``.  Then
    the global top-k selects every entry, and after the owned-filter this
    rank keeps exactly its own shard -- whose LOCAL prefix-compact
    coordinates are ``0 .. n_local-1`` and whose count is precisely
    ``seq_lens`` (already the localized count).  So the whole merge collapses
    to the identity fill, in the rank-LOCAL index space the DCP decode
    consumer (``ampere_sparse._forward_decode_dcp``) requires -- the same
    ``[0..count-1, -1]`` rows the C128A branch already hands it.
    """
    num_rows = local_topk_indices.shape[0]
    local_lens = seq_lens.reshape(-1)[:num_rows]
    if identity_selection:
        return _sm86_dcp_identity_selection(local_lens, topk_tokens)
    valid = (local_topk_indices >= 0) & (
        local_topk_indices < local_lens.unsqueeze(1)
    )
    safe_indices = torch.clamp(
        local_topk_indices, min=0, max=max(logits.shape[1] - 1, 0)
    ).to(torch.int64)
    gathered = torch.gather(logits, 1, safe_indices)
    local_values = torch.where(
        valid,
        gathered,
        torch.full_like(gathered, _SM86_DCP_INVALID_SCORE),
    )
    masked_local = torch.where(
        valid,
        local_topk_indices,
        torch.full_like(local_topk_indices, -1),
    )
    global_candidates = sm86_dcp_local_to_global(
        masked_local, dcp_rank, dcp_world_size, cp_interleave
    )
    global_values, global_indices = _sm86_dcp_global_topk(
        local_values, global_candidates, topk_tokens
    )
    owned = sm86_dcp_owns(global_indices, dcp_rank, dcp_world_size, cp_interleave)
    owned_values = torch.where(
        owned,
        global_values,
        torch.full_like(global_values, _SM86_DCP_INVALID_SCORE),
    )
    # Stable compaction: owned entries first, preserving the deterministic
    # (score desc, global index asc) order from the global top-k.
    owned_order = torch.argsort(owned_values, dim=-1, descending=True, stable=True)
    owned_sorted = torch.gather(owned, -1, owned_order)
    global_sorted = torch.gather(global_indices, -1, owned_order)
    local_out = sm86_dcp_global_to_local(
        global_sorted, dcp_rank, dcp_world_size, cp_interleave
    )
    return torch.where(
        owned_sorted,
        local_out,
        torch.full_like(local_out, -1),
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    # DeepGEMM availability is constant per process; check once for both branches.
    use_deep_gemm = is_deep_gemm_supported()
    # P4 / OPT-3 (SM8x DSV4 DCP). `max_global_compressed_entries` is
    # `max_seq_len // compress_ratio` computed by the indexer metadata builder
    # from the GLOBAL (never DCP-localized) seq lens, so it is a HOST int that
    # is identical on every DCP rank. When it does not exceed topk_tokens, no
    # row in this batch can have more global compressed entries than the
    # selection width, "global top-k of n <= k" is the identity selection, and
    # the cross-rank merge (2 all-gathers + 2 stable argsorts per chunk per
    # compressed layer) is provably redundant. Every rank evaluates the same
    # condition, so the skipped collectives stay symmetric. `-1` means the
    # builder did not supply the bound: stay on the merge.
    sm86_dcp_identity_topk = (
        attn_metadata_narrowed.use_sm86_dcp_topk
        and 0 <= attn_metadata_narrowed.max_global_compressed_entries <= topk_tokens
    )
    if not use_deep_gemm:
        assert not use_fp4_cache, (
            "Triton sparse-MLA fallback does not support FP4 KV cache"
        )
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                elif use_deep_gemm:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                else:
                    # SM80/SM121 Triton fallback (DeepGEMM unavailable).
                    logits = fp8_mqa_logits_triton(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                if envs.VLLM_SM86_DET_TOPK:
                    # P2e debug gate: deterministic pure-torch selection
                    # (score desc, ties by lower index). Same band
                    # (cu_seqlen_ks/ke), same band-relative int32 output with
                    # -1 padding, so both the plain dcp=1 path and the P2b
                    # per-rank local selection below consume it unchanged.
                    det_top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        topk_tokens,
                    )
                else:
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

            if attn_metadata_narrowed.use_sm86_dcp_topk:
                # P2b SM8x DSV4 DCP path (VLLM_SM86_DCP): pure-torch
                # deterministic global top-k; the CuteDSL merge below is
                # Hopper+-only. Called for EVERY chunk on every rank (chunk
                # splits are rank-invariant) so collectives stay symmetric.
                if sm86_dcp_identity_topk:
                    # P4 / OPT-3: identity selection (see the gate above).
                    # The builder attaches the per-row GLOBAL entry counts
                    # under exactly the same condition that sets
                    # use_sm86_dcp_topk, so this is never None here.
                    assert chunk.global_row_entry_lens is not None, (
                        "VLLM_SM86_DCP prefill identity short-circuit needs "
                        "the builder's global_row_entry_lens (P4)."
                    )
                    identity_row_lens = chunk.global_row_entry_lens
                else:
                    identity_row_lens = None
                topk_indices.copy_(
                    _sm86_dcp_topk_prefill(
                        logits,
                        cu_seqlen_ks,
                        topk_indices,
                        topk_tokens,
                        dcp_rank,
                        dcp_world_size,
                        cp_kv_cache_interleave_size,
                        has_local_kv=chunk.local_total_seq_lens > 0,
                        identity_row_lens=identity_row_lens,
                    )
                )
            else:
                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=chunk.cu_seqlen_ks,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        elif use_deep_gemm:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            # SM80/SM121 Triton fallback. Downstream topk reads only up to
            # `seq_lens`, so size the buffer to the active batch max rather
            # than the configured model max.
            active_max_model_len = attn_metadata_narrowed.max_seq_len
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len=active_max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if envs.VLLM_SM86_DET_TOPK:
            # P2e debug gate: deterministic selection replaces whichever of
            # the three decode kernels below would have run. They agree on
            # output (absolute int32 column indices, -1 padded) and differ
            # only in how the per-row column bound is derived, so mirror the
            # dispatch: cooperative_topk/persistent_topk always flat-index
            # `lengths` per row (no next_n argument), while
            # top_k_per_row_decode applies the 1-D speculative-offset rule
            # unless seq_lens is 2-D. At this call site seq_lens is always
            # 2-D (B, next_n), so all three coincide. The P2b merge below
            # consumes the result unchanged.
            if use_cooperative_topk or use_persistent_topk:
                det_top_k_per_row_flat_lengths(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_tokens,
                )
            else:
                det_top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    topk_tokens,
                )
        elif use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if attn_metadata_narrowed.use_sm86_dcp_topk:
            # P2b SM8x DSV4 DCP path (VLLM_SM86_DCP): seq_lens here are this
            # rank's LOCAL compressed-entry counts (builder localizes
            # divide-then-shard); the merged output is LOCAL entry indices,
            # owned entries first, -1 padded (P2 interface contract).
            topk_indices.copy_(
                _sm86_dcp_topk_decode(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    # P4 / OPT-3: same host gate as the prefill branch. Here
                    # the per-row length is already `seq_lens` (this rank's
                    # LOCAL compressed-entry counts), so nothing extra is
                    # needed from the builder.
                    identity_selection=sm86_dcp_identity_topk,
                )
            )
        elif decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        num_heads: int,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        # On SM80/SM121 (A100, GB10) DeepGEMM is unavailable — fall back to
        # the Triton sparse-MLA path. is_deep_gemm_supported() encodes the
        # SM-arch + has_deep_gemm() gate; if not supported, downgrade the
        # hard error from upstream to a one-time warning so the indexer
        # routes through the Triton kernels in `mqa_logits_triton.py`.
        if current_platform.is_cuda() and not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM not supported on this platform; "
                "using Triton fallback for sparse attention indexer."
            )
            # Prime the autotune caches (and, as a side effect of the first
            # launch, the e4m3 decode LUT) here rather than in a warmup hook:
            # memory profiling captures cudagraphs before any hook runs, and
            # the autotuner's synchronizing benchmark is illegal under
            # capture.
            from vllm.v1.attention.ops.mqa_logits_triton import (
                warmup_fp8_mqa_logits_triton,
                warmup_fp8_paged_mqa_logits_triton,
            )

            if not use_fp4_cache:
                device = topk_indices_buffer.device
                warmup_fp8_mqa_logits_triton(num_heads, head_dim, device)
                # 64/256 are the V3.2 and V4 indexer kernel block sizes; the
                # configured cache block size covers user-chosen values, which
                # the backends accept as any MultipleOf(64).
                block_sizes = {
                    64,
                    256,
                    get_current_vllm_config().cache_config.block_size,
                }
                for kernel_block_size in sorted(block_sizes):
                    warmup_fp8_paged_mqa_logits_triton(
                        num_heads, head_dim, kernel_block_size, device
                    )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
