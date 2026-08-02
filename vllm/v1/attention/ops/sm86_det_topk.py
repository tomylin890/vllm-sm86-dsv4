# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic pure-torch replacement for the indexer top-k kernels
(debug gate ``VLLM_SM86_DET_TOPK``, P2e).

WHY
---
The compiled selectors (``top_k_per_row_prefill`` / ``top_k_per_row_decode``
in ``csrc/libtorch_stable/sampler.cu``, ``persistent_topk`` in
``csrc/libtorch_stable/persistent_topk.cuh``, ``cooperative_topk`` in
``csrc/libtorch_stable/cooperative_topk.cuh``) are all radix/histogram
selectors that resolve the *threshold bin* with ``atomicAdd`` slot handout.
Elements that compare exactly equal therefore enter the output in
launch-nondeterministic order, and when the threshold bin is larger than the
number of remaining slots the SELECTED SET itself is nondeterministic --
which entry of a tie group survives depends on atomic arrival order.

DeepSeek-V4 indexer scores are ``sum_h w_h * ReLU(q_h . k)``: every entry
that matches no head scores EXACTLY 0.0.  With k=512 and fewer than 512
positive-scoring entries the tail of the selection is drawn from a huge
exact-0.0 tie pool, so the same prompt at temperature 0 selects different
entries run to run -- observed on the rack as ~0.09 nats of first-token
logprob flutter AT dcp=1, i.e. the baseline cannot reproduce itself and a
token-for-token dcp=1 vs dcp>1 A/B is meaningless.

WHAT
----
Under the gate every indexer top-k selection is replaced by a stable-sort
selector with a total order:

    score DESC, ties broken by LOWER column index

``torch.argsort(..., descending=True, stable=True)`` is documented to
preserve the relative order of equivalent elements and the ``stable`` flag is
supported on CUDA, so equal scores come out in ascending column order --
lower index wins, deterministically, on every rank and every run.  With
dcp=1 and dcp>1 both selecting under the same total order, and the P2b
cross-rank merge already keying on (score desc, lower GLOBAL entry index),
the dcp=1 baseline and each rank's local selection agree by construction and
an exact selected-set A/B becomes meaningful (ARCHITECTURE.md section 10
rule 4).

This path is DEBUG-ONLY: it materializes a full (rows x cols) bool mask and
sorts every column of every row.  Performance is irrelevant; masking and
padding fidelity is everything.

KERNEL SEMANTICS REPLICATED (see P2E-NOTES.md for the full table)
----------------------------------------------------------------
* ``logits`` are fp32, 2-D ``(num_rows, num_cols)``; the kernels honour
  ``stride0``/``stride1`` explicitly, ordinary torch indexing honours the
  same strides here.
* Visibility is a per-row half-open COLUMN BAND, never a score threshold:
  prefill uses ``[cu_seqlen_ks[r], cu_seqlen_ke[r])``; decode uses
  ``[0, len(r))``.  Out-of-band columns are not read at all by the kernels
  (the logits buffers are produced with ``clean_logits=False``, so they hold
  garbage) -- here they are masked out of the selection by the band mask,
  never merely down-weighted.
* ``k`` is not clamped to the band length: when the band holds fewer than
  ``topk`` entries the kernels emit the valid entries followed by ``-1``
  padding out to the full ``topk`` output width.
* Output is int32 ``(num_rows, topk)``.  Prefill emits BAND-RELATIVE indices
  (``column - cu_seqlen_ks[r]``); decode emits absolute column indices
  (its band starts at 0, so the two conventions coincide).

DIFFERENCES FROM THE KERNELS (intended, gate-only)
--------------------------------------------------
1. Tie order / tie membership is fixed (that is the entire point).
2. The kernels' ``rowLen <= topK`` shortcut emits the valid entries in
   ASCENDING COLUMN order; this selector always emits them in
   (score desc, column asc) order.  Both fill the same prefix with the same
   SET and pad the same tail with ``-1``; every consumer treats the prefix as
   a set plus a length (``combine_topk_swa_indices`` reads
   ``min((pos+1)//m, topk)`` leading slots), so only the *within-row order*
   differs -- deliberately, so that the short-row case obeys the same total
   order as the long-row case on both sides of the A/B.
"""

import torch

# Sentinel written into unused output slots by every indexer top-k kernel.
_PAD_INDEX = -1
_NEG_INF = float("-inf")


def _select_deterministic(
    scores: torch.Tensor,
    band_mask: torch.Tensor,
    topk_tokens: int,
    out_indices: torch.Tensor,
    num_rows: int,
    index_offset: torch.Tensor | None,
) -> None:
    """Write the deterministic per-row top-k into ``out_indices`` in place.

    ``scores``     -- (num_rows, num_cols) fp32 logits.
    ``band_mask``  -- (num_rows, num_cols) bool, True where the kernel would
                      have read the column (its visibility band).
    ``index_offset`` -- optional (num_rows,) subtracted from the selected
                      column index (prefill's band-relative convention).

    Order: score DESC, ties by LOWER column index.  Valid entries occupy the
    row prefix; the remainder is ``-1``, matching the kernels' padding.
    """
    out = out_indices[:num_rows, :topk_tokens]
    out.fill_(_PAD_INDEX)
    num_cols = scores.shape[1]
    if num_rows == 0 or num_cols == 0 or topk_tokens == 0:
        return

    neg_inf = torch.tensor(_NEG_INF, dtype=torch.float32, device=scores.device)
    keys = torch.where(band_mask, scores.to(torch.float32), neg_inf)
    # Pass 1: score descending; `stable=True` keeps equal scores in ascending
    # column order, which IS the "lower index wins" tie-break.
    order = torch.argsort(keys, dim=-1, descending=True, stable=True)
    # Pass 2: push every out-of-band column behind every in-band one while
    # preserving the pass-1 order inside each group (stable again).  Only
    # observable if an in-band score is itself -inf; the kernels never read
    # out-of-band columns, so such a column must never displace a real
    # candidate.
    visible_sorted = torch.gather(band_mask, 1, order).to(torch.int32)
    order = torch.gather(
        order,
        1,
        torch.argsort(visible_sorted, dim=-1, descending=True, stable=True),
    )

    width = min(topk_tokens, num_cols)
    selected = order[:, :width]
    valid = torch.gather(band_mask, 1, selected)
    if index_offset is not None:
        selected = selected - index_offset.reshape(-1, 1).to(selected.dtype)
    selected = selected.to(out.dtype)
    out[:, :width] = torch.where(
        valid, selected, torch.full_like(selected, _PAD_INDEX)
    )


def det_top_k_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out_indices: torch.Tensor,
    num_rows: int,
    topk_tokens: int,
) -> None:
    """Deterministic stand-in for ``ops.top_k_per_row_prefill``.

    Band is ``[cu_seqlen_ks[r], cu_seqlen_ke[r])``; output is BAND-RELATIVE
    int32 indices padded with ``-1`` (the kernel subtracts ``rowStart`` on
    store, and under DCP the band holds this rank's LOCAL compressed entries,
    so the relative index is the local entry index -- P2b relies on that).

    The ``stride0``/``stride1`` arguments of the kernel are omitted: they only
    describe ``logits``' own layout, which torch indexing already honours.
    """
    num_cols = logits.shape[1]
    cols = torch.arange(num_cols, device=logits.device, dtype=torch.int64)
    ks = cu_seqlen_ks.reshape(-1)[:num_rows].to(torch.int64)
    ke = cu_seqlen_ke.reshape(-1)[:num_rows].to(torch.int64)
    band_mask = (cols.unsqueeze(0) >= ks.unsqueeze(1)) & (
        cols.unsqueeze(0) < ke.unsqueeze(1)
    )
    _select_deterministic(
        logits[:num_rows],
        band_mask,
        topk_tokens,
        out_indices,
        num_rows,
        index_offset=ks,
    )


def _decode_row_ends(
    seq_lens: torch.Tensor,
    next_n: int,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-row exclusive column bound, replicating ``topKPerRowDecode``.

    2-D ``seq_lens`` (``seqLensIs2D=1``): row ``r`` -> ``seqLens[r]`` under
    C-contiguous flattening, i.e. ``(batch, next_n)`` indexed by the flat row.
    1-D ``seq_lens``: row ``r`` of batch ``b = r // next_n``, position
    ``j = r % next_n`` -> ``max(0, seqLens[b] - next_n + j + 1)`` (the causal
    per-position bound inside a speculative group).
    """
    flat = seq_lens.reshape(-1).to(torch.int64)
    if seq_lens.dim() == 2:
        return torch.clamp(flat[:num_rows], min=0)
    rows = torch.arange(num_rows, device=device, dtype=torch.int64)
    batch_idx = rows // max(next_n, 1)
    next_n_idx = rows % max(next_n, 1)
    return torch.clamp(flat[batch_idx] - next_n + next_n_idx + 1, min=0)


def det_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    out_indices: torch.Tensor,
    num_rows: int,
    topk_tokens: int,
) -> None:
    """Deterministic stand-in for ``ops.top_k_per_row_decode``.

    Band is ``[0, row_end(r))`` with ``row_end`` from ``_decode_row_ends``;
    output is absolute int32 column indices padded with ``-1``.  Under DCP
    the columns are this rank's LOCAL compressed entries (the metadata
    builder localizes divide-then-shard), matching P2b's decode contract.
    """
    num_cols = logits.shape[1]
    row_end = _decode_row_ends(seq_lens, next_n, num_rows, logits.device)
    cols = torch.arange(num_cols, device=logits.device, dtype=torch.int64)
    band_mask = cols.unsqueeze(0) < row_end.unsqueeze(1)
    _select_deterministic(
        logits[:num_rows],
        band_mask,
        topk_tokens,
        out_indices,
        num_rows,
        index_offset=None,
    )


def det_top_k_per_row_flat_lengths(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Deterministic stand-in for ``persistent_topk`` / ``cooperative_topk``.

    Both kernels take ``num_rows = logits.size(0)`` and read
    ``lengths[row_idx]`` FLAT regardless of the tensor's declared rank (there
    is no ``next_n`` argument), then select over ``[0, lengths[row])`` with
    the same ``-1`` padding and the same trivial ``seq_len <= k`` shortcut as
    ``top_k_per_row_decode``.  At the indexer call site ``seq_lens`` is always
    2-D ``(B, next_n)``, so this coincides with ``det_top_k_per_row_decode``;
    it is kept separate to mirror each kernel's own contract exactly.
    """
    num_rows = logits.shape[0]
    num_cols = logits.shape[1]
    row_end = torch.clamp(
        lengths.reshape(-1)[:num_rows].to(torch.int64), min=0
    )
    cols = torch.arange(num_cols, device=logits.device, dtype=torch.int64)
    band_mask = cols.unsqueeze(0) < row_end.unsqueeze(1)
    _select_deterministic(
        logits[:num_rows],
        band_mask,
        topk_tokens,
        out_indices,
        num_rows,
        index_offset=None,
    )
