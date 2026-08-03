# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared DCP compressed-entry ownership algebra for SM8x DeepSeek-V4
(``VLLM_SM86_DCP``).

Single source of truth for the round-robin entry layout used by every P2
workstream (indexer top-k merge, compressed-entry cache writes, prefill
all-gather, C128A decode metadata).  Ownership and translation live in
**compressed-ENTRY index space** with ``I = cp_kv_cache_interleave_size``
applied directly to entry indices (P2c-NOTES "Shared entry-layout
convention"; mirrors the Lasimeri ``ContextParallelLayout`` /
``cp_utils.cp_global_to_local_pos`` formulas):

    owner(e)       = (e // I) % W
    local_entry(e) = (e // (I * W)) * I + e % I          (owned e only)
    global(r, j)   = (j // I) * (I * W) + r * I + j % I  (inverse)

with ``W = dcp_world_size``.  Owned entries fill each rank's pages
contiguously in local-entry order; ``-1`` is the invalid sentinel and passes
through every helper unchanged.

History: these helpers were introduced by P2b inside
``vllm/model_executor/layers/sparse_attn_indexer.py`` (private names) and
moved here VERBATIM (no-logic-change refactor) in P2d so that
``models/deepseek_v4/common/ops/cache_utils.py`` (prefill all-gather) and
future consumers reuse the same algebra instead of reimplementing it.
The per-rank owned-entry COUNT lives in
``vllm.v1.attention.backends.utils.get_dcp_local_seq_lens`` (same layout;
apply it to entry counts).

P4 added ``sm86_dcp_owner`` (the ``owner(e)`` formula above, previously only
reachable one-rank-at-a-time through ``sm86_dcp_owns``) and widened
``sm86_dcp_global_to_local``'s ``dcp_rank`` to accept a per-element tensor,
so a caller can build the whole global->(rank, local) map with pure
elementwise ops.  No formula changed.
"""

import torch


def sm86_dcp_local_to_global(
    local_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Map rank-local compressed-entry indices to global entry indices.

    Mirrors the Lasimeri ContextParallelLayout.local_to_global formula;
    -1 passes through unchanged.
    """
    safe = torch.clamp(local_indices, min=0)
    global_indices = (
        (safe // cp_interleave) * (cp_interleave * dcp_world_size)
        + dcp_rank * cp_interleave
        + safe % cp_interleave
    )
    return torch.where(
        local_indices >= 0,
        global_indices,
        torch.full_like(global_indices, -1),
    )


def sm86_dcp_owns(
    global_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """True where this rank owns the global compressed-entry index (>= 0)."""
    safe = torch.clamp(global_indices, min=0)
    owner = (safe // cp_interleave) % dcp_world_size
    return (global_indices >= 0) & (owner == dcp_rank)


def sm86_dcp_owner(
    global_indices: torch.Tensor,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Owning rank of each global compressed-entry index (``owner(e)`` above).

    Forward companion of :func:`sm86_dcp_owns`, which only answers the
    predicate for one rank at a time: ``sm86_dcp_owns(e, r, W, I)`` is true
    iff ``sm86_dcp_owner(e, W, I) == r`` (for ``e >= 0``).  Added by P4 so
    consumers that need the map for ALL entries at once can evaluate it
    elementwise instead of looping over ranks with boolean-mask indexing
    (which lowers to ``nonzero()``, i.e. a hard host sync).  ``-1`` passes
    through unchanged, like every helper here.
    """
    safe = torch.clamp(global_indices, min=0)
    owner = (safe // cp_interleave) % dcp_world_size
    return torch.where(
        global_indices >= 0,
        owner,
        torch.full_like(owner, -1),
    )


def sm86_dcp_global_to_local(
    global_indices: torch.Tensor,
    dcp_rank: "int | torch.Tensor",
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Map global entry indices to this rank's local indices.

    Exact inverse of sm86_dcp_local_to_global for indices this rank owns;
    callers must mask non-owned entries (the formula returns the local
    PREFIX COUNT for those, matching get_dcp_local_seq_lens semantics).
    -1 passes through unchanged.

    ``dcp_rank`` accepts an int (one rank for the whole tensor, the original
    and still dominant use) or a broadcastable tensor of per-element owning
    ranks -- pass ``sm86_dcp_owner(...)`` to get every entry's OWN local
    index in a single elementwise pass.  The formula is untouched either
    way; only the operand type widens.
    """
    safe = torch.clamp(global_indices, min=0)
    rank_stride = dcp_world_size * cp_interleave
    base = safe // rank_stride * cp_interleave
    remainder = safe - base * dcp_world_size
    extra = torch.clamp(remainder - dcp_rank * cp_interleave, 0, cp_interleave)
    return torch.where(
        global_indices >= 0,
        base + extra,
        torch.full_like(global_indices, -1),
    )
