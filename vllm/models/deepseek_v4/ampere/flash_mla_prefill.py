# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flash-mla fused sparse-prefill adapter for the SM8x DSV4 DCP path (P6).

Routes the compressed-layer (C4A/C128A) DCP prefill attention through the
``flash_mla`` CUDA op ``fwd_sparse_prefill_mla`` instead of the Triton
dequant-workspace pipeline. One op call computes the SAME single softmax the
Triton path computes (ARCHITECTURE.md section 6: the SWA window and the
top-k compressed entries share one softmax; section 10 rule 1: the per-head
attention sink is folded exactly once, at softmax init, inside the kernel).

Index spaces (the crux -- see sim_p6_lens.py for the proof):

- The op takes FLAT SLOT IDS over each cache view: slot ``s`` addresses
  block ``s // cache.size(1)``, position ``s % cache.size(1)``. Both caches
  handed to it here are compact ``[N, 1, 584]`` uint8 stagings, so slot ids
  ARE row ids.
- Compressed stream (``extra_*``): the P2b/P2d producer emits GLOBAL entry
  ids (request-relative, -1 padded). The P4 closed-form virtual map
  ``row(c, e) = owner(e)*num_reqs*max_local + c*max_local + local(e)`` is
  the memoized virtual block table's row ``c`` -- applied to the id tensor
  directly (``vbt[0][e] + c*max_local``), never rebuilt here.
- P7 delta layout (``staging_is_global_order=True``): the cache is one
  request's persistent staging in GLOBAL entry order (row e = entry e), so
  the extra_indices formula degenerates to the identity -- the producer's
  GLOBAL ids ARE the flat slot ids. Single-request calls only; nothing
  else about the op contract changes.
- SWA stream (``swa_*``): staging row of (request ``c``, position ``p``) is
  ``c*max_gather + (p - gather_start(c))`` -- the same ``pos -
  gather_start`` arithmetic as ``combine_topk_swa_indices``.

Lens are TIGHT by construction (never derived by counting -1s):
``extra_lens[t] = min((pos_t+1)//compress_ratio, top_k)`` -- the producer's
own validity contract -- and ``swa_lens[t] = min(pos_t+1, window)``. The
kernel adds a zero row to the softmax denominator for any invalid-but-
inside-lens slot, so a loose len is a silent numerics bug; slots at or past
``lens`` are never dereferenced, so the -1 padding is replaced with row 0
only defensively (AFTER lens are fixed).

HAZARD (hard guard below): ``fwd_sparse_prefill_mla`` allocates a bf16
dequant buffer sized by the WHOLE of every cache tensor passed to it
(flash_api.cpp: ``kv = new_empty(total_slots, 512)``). The compact stagings
are ~1 KiB/slot of bf16 (32 MiB at a 128K-token C4A prefix); the local
paged pools are gigabytes. Passing a paged pool would OOM-or-thrash AND,
under DCP, index the wrong bytes (global entry ids do not address the
P1-sharded local pool). The adapter therefore refuses any cache whose
data_ptr matches a paged pool and structurally requires the exact staging
geometry.
"""

from typing import Any

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _SM86_DCP_ENTRY_BYTES,
)

_FLASH_MLA_BUILD_HINT = (
    "flash_mla with the SM86 sparse prefill op is required for "
    "VLLM_DSV4_FLASH_PREFILL. Build it from source into this venv with the "
    "arch pinned to the 3090s, e.g.:\n"
    "  cd ~/dsv4-dcp/flash-mla-int && \\\n"
    "  FLASH_MLA_CUDA_ARCHS=86 python -m pip install -v "
    "--no-build-isolation .\n"
    "(torch-stable ABI: the wheel works on torch >= 2.9; verify the venv's "
    "torch at build time, not at boot.)"
)

_flash_mla_sparse_prefill: Any = None


def _get_flash_mla_sparse_prefill():
    """Lazy `import flash_mla`; a missing/old build fails with the fix."""
    global _flash_mla_sparse_prefill
    if _flash_mla_sparse_prefill is None:
        try:
            from flash_mla import sparse_mla_prefill
        except ImportError as e:
            raise ImportError(f"{_FLASH_MLA_BUILD_HINT}\n(import error: {e}") from e
        if not hasattr(torch.ops.flash_mla, "fwd_sparse_prefill_mla"):
            raise ImportError(
                "the installed flash_mla lacks fwd_sparse_prefill_mla "
                f"(built without the SM86 sparse ops?).\n{_FLASH_MLA_BUILD_HINT}"
            )
        _flash_mla_sparse_prefill = sparse_mla_prefill
    return _flash_mla_sparse_prefill


def _assert_staging_only(
    name: str,
    staging_rows: torch.Tensor,
    expected_rows: int,
    forbidden_pools: tuple[torch.Tensor, ...],
) -> None:
    """The staging-buffer-only hard guard (see module docstring)."""
    assert staging_rows.dtype == torch.uint8 and staging_rows.dim() == 2, (
        f"{name}: expected a flat uint8 staging view, got "
        f"{staging_rows.dtype} dim={staging_rows.dim()}"
    )
    assert (
        staging_rows.shape[0] == expected_rows
        and staging_rows.shape[1] == _SM86_DCP_ENTRY_BYTES
        and staging_rows.stride(1) == 1
        and staging_rows.stride(0) == _SM86_DCP_ENTRY_BYTES
    ), (
        f"{name}: staging must be contiguous [{expected_rows}, "
        f"{_SM86_DCP_ENTRY_BYTES}], got shape={tuple(staging_rows.shape)} "
        f"strides={tuple(staging_rows.stride())}. NEVER hand this op a "
        "paged KV pool: its in-op dequant pre-pass allocates bf16 sized by "
        "the WHOLE cache."
    )
    for pool in forbidden_pools:
        assert staging_rows.data_ptr() != pool.data_ptr(), (
            f"{name}: cache tensor aliases a paged KV pool. The flash-mla "
            "prefill op dequantizes the WHOLE passed cache into a bf16 "
            "buffer; only the compact per-chunk staging buffers may be "
            "passed."
        )


def _row_geometry(
    query_start_loc: torch.Tensor,  # [R+1] chunk-sliced, GLOBAL values
    seq_lens: torch.Tensor,  # [R] (device)
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (request row, absolute position) for one prefill chunk.

    Identical arithmetic to the ``combine_topk_swa_indices`` kernel:
    ``pos = (seq_len - query_len) + token_idx_in_query`` with the chunk's
    ``query_start_loc`` rebased to its first value. Pure elementwise torch,
    fixed shapes, no host sync (eager prefill only).
    """
    qsl = (query_start_loc - query_start_loc[0]).to(torch.int64)
    tok = torch.arange(num_tokens, dtype=torch.int64, device=seq_lens.device)
    req = torch.searchsorted(qsl[1:], tok, right=True)
    query_len = qsl[req + 1] - qsl[req]
    idx_in_query = tok - qsl[req]
    pos = seq_lens.to(torch.int64)[req] - query_len + idx_in_query
    return req, pos


def sparse_prefill_via_flash_mla(
    q: torch.Tensor,  # [T, H, 512] bf16
    *,
    # SWA window stream (always present; replicated dcp_exempt cache).
    swa_staging_rows: torch.Tensor,  # [R*max_gather, 584] uint8
    max_gather: int,
    seq_lens: torch.Tensor,  # [R] chunk GLOBAL seq lens (device)
    gather_lens: torch.Tensor,  # [R] chunk SWA gather lens (device)
    window_size: int,
    # Compressed top-k stream (None when the chunk has zero entries).
    compressed_staging_rows: torch.Tensor | None,  # [W*R*max_local, 584]
    virtual_block_table: torch.Tensor | None,  # [R, max_entries] int32
    max_local: int,
    max_entries: int,
    topk_indices: torch.Tensor | None,  # [T, top_k] int32 GLOBAL ids, -1 pad
    compress_ratio: int,
    # P7 delta layout: staging is ONE request's [max_entries, 584] buffer in
    # GLOBAL entry order (extra_indices = ids, identity). vbt must be None.
    staging_is_global_order: bool = False,
    # Shared.
    query_start_loc: torch.Tensor,  # [R+1] chunk-sliced (device)
    scale: float,
    attn_sink: torch.Tensor | None,  # [>=H] float32 (padded heads = -inf)
    output: torch.Tensor,  # [T, H, 512]
    forbidden_pools: tuple[torch.Tensor, ...],
) -> None:
    """One fused sparse-prefill call for one chunk (see module docstring)."""
    sparse_mla_prefill = _get_flash_mla_sparse_prefill()

    num_tokens, num_heads, head_dim = q.shape
    num_reqs = seq_lens.shape[0]
    assert head_dim == 512 and q.dtype == torch.bfloat16, (
        f"flash-mla sparse prefill wants q [T, H, 512] bf16, got "
        f"{tuple(q.shape)} {q.dtype}"
    )

    _assert_staging_only(
        "swa_staging_rows",
        swa_staging_rows,
        num_reqs * max_gather,
        forbidden_pools,
    )

    req, pos = _row_geometry(query_start_loc, seq_lens, num_tokens)

    # ---- SWA stream: flat staging rows + TIGHT lens --------------------
    # Window of token pos: positions [pos - swa_len + 1, pos]; staging row
    # of position p (request c) is c*max_gather + (p - gather_start(c)).
    swa_len = torch.clamp(pos + 1, max=window_size)  # int64 [T]
    gather_start = (seq_lens - gather_lens).to(torch.int64)[req]  # [T]
    j = torch.arange(window_size, dtype=torch.int64, device=q.device)
    win_pos = (pos - swa_len + 1).unsqueeze(1) + j.unsqueeze(0)  # [T, W]
    swa_rows = (req * max_gather).unsqueeze(1) + (
        win_pos - gather_start.unsqueeze(1)
    )
    # Slots at or past swa_len are never dereferenced (masked by lens);
    # keep them in-range anyway (defensive, matches the -1 -> 0 policy).
    swa_rows = torch.where(
        j.unsqueeze(0) < swa_len.unsqueeze(1), swa_rows, torch.zeros_like(swa_rows)
    )
    swa_indices = swa_rows.to(torch.int32)
    swa_lens_i32 = swa_len.to(torch.int32)

    # ---- compressed stream: P4 map applied to the id tensor ------------
    extra_cache = None
    extra_indices = None
    extra_lens = None
    if compressed_staging_rows is not None:
        assert topk_indices is not None
        assert max_entries > 0
        world_rows = compressed_staging_rows.shape[0]
        _assert_staging_only(
            "compressed_staging_rows",
            compressed_staging_rows,
            world_rows,
            forbidden_pools,
        )
        top_k = topk_indices.shape[-1]
        # TIGHT lens from the producer contract (never by counting -1s):
        # min((pos+1) // m, top_k) valid GLOBAL ids occupy the row prefix.
        extra_len = torch.clamp(
            (pos + 1) // compress_ratio, max=top_k
        )  # int64 [T]
        safe_ids = topk_indices.to(torch.int64).clamp_(0, max_entries - 1)
        if staging_is_global_order:
            # P7 delta layout: staging row e IS global entry e, so the flat
            # slot id is the producer's GLOBAL id itself (identity). Only
            # meaningful for a single request's staging.
            assert virtual_block_table is None
            assert num_reqs == 1, (
                "global-order staging is per-request; call the op per request"
            )
            assert world_rows == max_entries, (
                f"global-order staging must be sliced to [max_entries, 584]: "
                f"{world_rows} != {max_entries}"
            )
            flat = safe_ids
        else:
            assert virtual_block_table is not None
            assert max_local > 0
            assert world_rows % (num_reqs * max_local) == 0, (
                "gathered staging rows must be world * num_reqs * max_local"
            )
            # row(c, e) = vbt[0][e] + c * max_local (P4 closed form, memoized
            # -- vbt row c is vbt[0] + c*max_local by construction, so row 0
            # IS the request-invariant base map; never rebuilt here).
            row_base = virtual_block_table[0].to(torch.int64)  # [max_entries]
            flat = row_base[safe_ids] + (req * max_local).unsqueeze(1)
        extra_indices = flat.to(torch.int32)
        extra_lens = extra_len.to(torch.int32)
        extra_cache = compressed_staging_rows.view(
            world_rows, 1, _SM86_DCP_ENTRY_BYTES
        )

    sink = attn_sink
    if sink is not None:
        sink = sink[:num_heads]
        assert sink.dtype == torch.float32, (
            "flash-mla folds attn_sink as raw float32; got " + str(sink.dtype)
        )

    out = sparse_mla_prefill(
        q=q,
        swa_cache=swa_staging_rows.view(
            num_reqs * max_gather, 1, _SM86_DCP_ENTRY_BYTES
        ),
        swa_indices=swa_indices,
        swa_lens=swa_lens_i32,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
    )
    output.copy_(out)
