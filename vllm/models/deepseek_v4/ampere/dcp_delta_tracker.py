# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer persistent staging tracker for the SM8x DSV4 DCP delta gather
(P7, ``VLLM_DSV4_DELTA_GATHER``).

Each compressed attention layer (C4A/C128A) owns one :class:`Sm86DcpDeltaTracker`
mapping ``req_id -> _ReqDeltaState``: a persistent uint8 staging buffer in
GLOBAL compressed-entry order (row ``e`` = global entry ``e``'s 584 raw
bytes) plus ``gathered_upto`` -- how many leading global entries the buffer
already holds. Layer identity is implicit in the per-layer ownership (one
tracker per attention module), which also makes the design PP-clean: each
pipeline stage's worker only tracks the layers it hosts.

Lifecycle WITHOUT scheduler hooks:

- state advances only on prefill steps (the only place the delta gather
  runs);
- GC rule: any tracked ``req_id`` absent from the current step's PREFILL
  rows is freed. Finish, abort and preemption all manifest as absence; a
  request that moved on to decode is also freed (it can never prefill
  again, so its staging is dead weight against the budget). A preemption-
  resumed request is re-admitted fresh and full-gathers once through the
  delta machinery (``prev=0`` -- correct, just slower);
- a PREFIX-CONTINUITY check makes staleness structurally impossible even
  when GC is delayed (it only runs on eager prefill steps; captured
  decode-only steps replay no Python): every sighting must present a
  prefix-token count equal to the seq len recorded at the previous
  sighting -- chunked prefill guarantees exactly that, and any violation
  (request-id reuse after finish, preempt-resume with different chunking)
  resets the state to a fresh admission. The one case that passes the
  check without having been continuously tracked is a preempt-resume that
  replays the identical chunk sequence -- and recompression of the same
  tokens is deterministic (fp32, ARCHITECTURE.md section 4), so the staged
  bytes are byte-identical to a re-gather anyway.

Budget (``VLLM_DSV4_DELTA_GATHER_BUDGET_MB``): one per-worker byte budget
across all tracked (request, layer) pairs. Admission and growth both charge
it; when a charge fails, that (request, layer) is dropped to a BLOCKED
sentinel and the request falls back to the existing full re-gather path on
that layer (graceful, per request per layer). Every decision is a pure
function of rank-invariant host state (CPU seq lens, batch order, layer
visit order, the env budget), so all ranks of a DCP group make identical
decisions and the delta collectives stay symmetric.
"""

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _SM86_DCP_ENTRY_BYTES,
)
from vllm.utils.math_utils import next_power_of_2

# Minimum staging capacity in entries. Keeps C128A buffers (tiny counts)
# from thrashing through many small reallocations while costing at most
# 256 * 584 B = 146 KiB per (request, layer).
_MIN_CAPACITY_ENTRIES = 256


class Sm86DeltaGatherBudget:
    """Per-worker byte budget across all tracked (request, layer) stagings."""

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        self.used_bytes = 0

    def try_charge(self, nbytes: int) -> bool:
        if self.used_bytes + nbytes > self.limit_bytes:
            return False
        self.used_bytes += nbytes
        return True

    def release(self, nbytes: int) -> None:
        self.used_bytes -= nbytes
        assert self.used_bytes >= 0, "delta-gather budget released below zero"


_BUDGET: Sm86DeltaGatherBudget | None = None


def get_delta_gather_budget() -> Sm86DeltaGatherBudget:
    """The process-wide budget singleton (limit read once, at first use)."""
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = Sm86DeltaGatherBudget(
            envs.VLLM_DSV4_DELTA_GATHER_BUDGET_MB * 1024 * 1024
        )
    return _BUDGET


@dataclass
class _ReqDeltaState:
    """Per-(request, layer) staging state.

    ``staging is None`` marks the BLOCKED sentinel: the budget rejected this
    (request, layer); it stays on the full re-gather path until GC removes
    it (re-admission while continuously present would thrash the budget).
    """

    staging: torch.Tensor | None  # [capacity, 584] uint8, GLOBAL entry order
    capacity: int  # entries
    upto: int  # global entries gathered; rows [0, upto) are valid
    expected_prefix: int  # next sighting's prefix-token count must equal this


def _initial_capacity(num_entries: int) -> int:
    return next_power_of_2(max(num_entries, _MIN_CAPACITY_ENTRIES))


class Sm86DcpDeltaTracker:
    """One per compressed attention layer (see module docstring)."""

    # Every live tracker, so the metadata builder can free stale staging
    # across ALL layers before any layer's forward (and therefore before
    # any layer's indexer transients) executes. Attention modules live for
    # the process lifetime, so plain references are fine.
    _instances: "list[Sm86DcpDeltaTracker]" = []

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._states: dict[str, _ReqDeltaState] = {}
        self._budget = get_delta_gather_budget()
        Sm86DcpDeltaTracker._instances.append(self)

    @classmethod
    def gc_all(cls, live_prefill_req_ids: "list[str]") -> None:
        """Step-level GC over every layer's tracker (rank-symmetric)."""
        freed_any = False
        for tracker in cls._instances:
            before = len(tracker._states)
            tracker.gc(live_prefill_req_ids)
            freed_any = freed_any or len(tracker._states) < before
        if freed_any:
            # Request boundary: hundreds of MiB of staging plus a long
            # prefill's transients were just freed, but the caching
            # allocator pins freed blocks to the stream that used them
            # (DSV4 runs dual streams via maybe_execute_in_parallel) and
            # graph pools are separate, so the NEXT long request's first
            # allocations can OOM with gigabytes 'reserved but
            # unallocated' (observed: 16 MiB requests failing on the 2nd
            # consecutive 200k prompt). empty_cache() syncs the free
            # events and returns the blocks to CUDA. Runs once per
            # request transition on the eager prefill path -- never under
            # capture -- and is a local op on every rank (no collective).
            torch.cuda.empty_cache()

    def gc(self, live_prefill_req_ids: "list[str]") -> None:
        """Free every tracked id absent from the current step's prefill rows."""
        live = set(live_prefill_req_ids)
        for req_id in list(self._states):
            if req_id not in live:
                self._free(req_id)

    def _free(self, req_id: str) -> None:
        state = self._states.pop(req_id)
        if state.staging is not None:
            self._budget.release(state.capacity * _SM86_DCP_ENTRY_BYTES)

    def _block(self, req_id: str) -> None:
        self._states[req_id] = _ReqDeltaState(
            staging=None, capacity=0, upto=0, expected_prefix=-1
        )

    def plan_request(
        self, req_id: str, prefix_tokens: int, new_entries: int
    ) -> "_ReqDeltaState | None":
        """Admit/continue tracking; None means take the full re-gather path.

        On success the caller must (a) delta-gather ``[state.upto,
        new_entries)`` into ``state.staging`` and (b) call :meth:`advance`
        -- even when the delta is empty, so the continuity chain stays
        unbroken.
        """
        state = self._states.get(req_id)
        if state is not None and state.staging is None:
            return None  # blocked by the budget; stays blocked until GC
        if state is not None and (
            prefix_tokens != state.expected_prefix or new_entries < state.upto
        ):
            # Discontinuity: preempt-resume with different chunking, or a
            # reused request id. Reset to a fresh admission (full-gathers
            # once through the delta machinery, prev=0).
            self._free(req_id)
            state = None

        if state is None:
            if new_entries <= 0:
                return None  # nothing to stage yet; admit at a later chunk
            capacity = _initial_capacity(new_entries)
            if not self._budget.try_charge(capacity * _SM86_DCP_ENTRY_BYTES):
                self._block(req_id)
                return None
            state = _ReqDeltaState(
                staging=torch.empty(
                    (capacity, _SM86_DCP_ENTRY_BYTES),
                    dtype=torch.uint8,
                    device=self._device,
                ),
                capacity=capacity,
                upto=0,
                expected_prefix=prefix_tokens,
            )
            self._states[req_id] = state
            return state

        if new_entries > state.capacity:
            new_capacity = next_power_of_2(new_entries)
            grow_bytes = (new_capacity - state.capacity) * _SM86_DCP_ENTRY_BYTES
            if not self._budget.try_charge(grow_bytes):
                self._free(req_id)
                self._block(req_id)
                return None
            assert state.staging is not None
            new_staging = torch.empty(
                (new_capacity, _SM86_DCP_ENTRY_BYTES),
                dtype=torch.uint8,
                device=self._device,
            )
            if state.upto > 0:
                # Row ids are stable (global entry order): a realloc is a
                # verbatim prefix copy, amortized O(1) copies per entry by
                # the power-of-two growth.
                new_staging[: state.upto].copy_(state.staging[: state.upto])
            state.staging = new_staging
            state.capacity = new_capacity
        return state

    @staticmethod
    def advance(
        state: "_ReqDeltaState", new_entries: int, seq_len_tokens: int
    ) -> None:
        """Record a processed chunk: rows [0, new_entries) are now staged and
        the next sighting of this request must start at ``seq_len_tokens``."""
        state.upto = new_entries
        state.expected_prefix = seq_len_tokens
