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
that layer (graceful, per request per layer). The sentinel outlives further
chunks of its own request -- re-admitting one that never left would thrash
the budget -- but never the request itself: the first sighting that is not a
continuation drops it, so a later request reusing the id is planned on its
own merits (GC alone cannot do this, since a request occupies a prefill row
on every one of its own steps). Every decision is a pure
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
    A sentinel has no staging to chain, but it keeps the SAME continuity
    record a tracked row keeps: ``upto`` is the last sighting's
    ``new_entries`` and ``expected_prefix`` is the last sighting's seq len,
    written by :meth:`_block` where :meth:`advance` would have written it.
    That is what lets :meth:`plan_request` tell a further chunk of the
    blocked request from a different request that reused its id -- including
    one whose first sighting carries a large prefix-cache hit.
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

    # Consecutive-absence counts, shared across layers (all trackers see the
    # same live set each step, so one counter map suffices). A request is
    # freed only after _GRACE consecutive steps absent from the FULL
    # scheduled set (prefill AND decode rows): the scheduler legitimately
    # skips a live request for a step (PP decode cadence, DP deferral,
    # zero-token steps), so first-absence freeing is unsound the moment
    # prefills co-schedule. Finished/aborted requests never reappear, so
    # they are freed exactly _GRACE steps late -- bounded, tiny.
    _absence: "dict[str, int]" = {}
    _GRACE = 3

    # Requests that have already taken their one allocator flush, plus the
    # consecutive-absence counter that retires those latches.
    #
    # Keyed per REQUEST, deliberately NOT as a set difference against the
    # previous step's prefill rows: a co-scheduled prefill can be given zero
    # tokens for a step and then does not appear in ``req_ids`` at all
    # ("possible at 3+ concurrent prefills under
    # long_prefill_token_threshold" -- `ampere_sparse._maybe_delta_tracker`),
    # so an edge trigger re-fires for it on the way back in. That is an
    # unbounded number of device-synchronising flushes per request, landing
    # mid-prefill -- the exact placement this whole change exists to remove.
    # A latch fires once per request id and is immune to the gap.
    #
    # Retirement reuses _GRACE for the same reason the staging GC does: one
    # absent step proves nothing. It must also survive an EMPTY id set --
    # gc_all IS reached on dummy/warmup/capture builds (gpu_model_runner
    # populates `cm_req_ids` whenever VLLM_SM86_DCP and
    # VLLM_DSV4_DELTA_GATHER are on, with no `for_cudagraph_capture` gate),
    # and clearing the latches there would re-fire the flush for every
    # request still mid-prefill.
    _flushed: "set[str]" = set()
    _flush_absence: "dict[str, int]" = {}

    @classmethod
    def gc_all(
        cls,
        live_req_ids: "list[str]",
        prefill_req_ids: "list[str]",
    ) -> None:
        """Step-level GC over every layer's tracker (rank-symmetric).

        ``live_req_ids`` is the FULL scheduled set (decode rows included);
        ``prefill_req_ids`` the prefill subset. Three verdicts per tracked
        request:
        - in the prefill rows: alive, staging in use -- keep;
        - present ONLY as a decode row: prefill provably finished (decode
          never reads staging, and a request cannot return to prefill) --
          free IMMEDIATELY;
        - absent entirely: could be finished OR merely skipped this step
          (PP decode cadence, DP deferral, zero-token steps) -- free only
          after _GRACE consecutive absences. Runs on decode-only steps
          too, fixing the budget leak where dead stagings survived until
          the next step that happened to carry a prefill row.

        Also flushes the caching allocator once per request, on the step
        that request enters prefill (see the empty_cache() call below).
        """
        live = set(live_req_ids)
        prefilling = set(prefill_req_ids)
        entering = prefilling - cls._flushed
        cls._flushed |= prefilling
        for req_id in list(cls._flush_absence):
            if req_id in live or req_id not in cls._flushed:
                cls._flush_absence.pop(req_id, None)
        for req_id in list(cls._flushed):
            if req_id in live:
                continue
            n = cls._flush_absence.get(req_id, 0) + 1
            cls._flush_absence[req_id] = n
            if n >= cls._GRACE:
                cls._flushed.discard(req_id)
                cls._flush_absence.pop(req_id, None)
        tracked: set[str] = set()
        for tracker in cls._instances:
            tracked.update(tracker._states)
        for req_id in list(cls._absence):
            if req_id in live or req_id not in tracked:
                cls._absence.pop(req_id, None)
        doomed: list[str] = []
        for req_id in tracked:
            if req_id in prefilling:
                continue
            if req_id in live:
                doomed.append(req_id)  # decoding: staging is dead weight
                continue
            n = cls._absence.get(req_id, 0) + 1
            cls._absence[req_id] = n
            if n >= cls._GRACE:
                doomed.append(req_id)
        for tracker in cls._instances:
            for req_id in doomed:
                if req_id in tracker._states:
                    tracker._free(req_id)
        for req_id in doomed:
            cls._absence.pop(req_id, None)
        if entering:
            # Request boundary. By now hundreds of MiB of a previous
            # request's staging plus a long prefill's transients have been
            # freed, but the caching allocator pins freed blocks to the
            # stream that used them (DSV4 runs dual streams via
            # maybe_execute_in_parallel) and graph pools are separate, so
            # the entering request's first allocations can OOM with
            # gigabytes 'reserved but unallocated' (observed: 16 MiB
            # requests failing on the 2nd consecutive 200k prompt).
            # empty_cache() syncs the free events and returns the blocks to
            # CUDA, here right before that request's first layer allocates
            # anything (gc_all runs from the metadata builder, ahead of
            # every layer's forward).
            #
            # Keyed on the ENTERING request and not on a free having
            # happened: a free is a function of the PREVIOUS request's
            # lifecycle -- with _GRACE it lands around the third step of
            # the next request's prefill -- and flushing returns every
            # cached block to CUDA, which moves the address and the
            # residual bytes of every torch.empty later in that step. Keyed
            # that way the second of two identical consecutive requests
            # takes the flush mid-prefill and the first, having no
            # predecessor, never takes it at all. Every request now takes
            # exactly one flush -- `_flushed` is a per-request latch, not a
            # prefill-set edge, so a prefill that is skipped for a step and
            # comes back does not pay a second one.
            #
            # Never runs inside a captured region: capture wraps only the
            # model forward (`with torch.cuda.graph(...)` in
            # compilation/cuda_graph.py) and gc_all runs from the metadata
            # builder, which completes before that wrapper is entered. It IS
            # reached on dummy/warmup/capture BUILDS, so it must stay
            # tolerant of an empty or stale id set -- hence the graced
            # retirement above and not a plain intersection. Note
            # cuda_graph.py neutralises only `torch.accelerator.empty_cache`
            # during capture, not this call, so there is no safety net if it
            # ever migrates inside.
            #
            # The DECISION is rank-symmetric (`prefill_req_ids` is a slice of
            # the broadcast scheduler output and `_flushed` is advanced by
            # the same call sequence on every rank), so no collective is
            # skewed. The LATENCY is not: each rank's flush costs what its
            # own fragmentation costs, and the group pays max-over-ranks at
            # the next delta all-gather. That is the argument for keeping the
            # count at one per request rather than for gating on per-rank
            # allocator state, which would make the decision itself
            # rank-dependent.
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

    def _block(self, req_id: str, new_entries: int, seq_len_tokens: int) -> None:
        # Same continuity contract `advance()` records for a tracked row: the
        # next sighting of THIS request must present prefix_tokens equal to
        # this sighting's seq len. A blocked row never reaches advance(), so
        # plan_request records it here instead.
        self._states[req_id] = _ReqDeltaState(
            staging=None,
            capacity=0,
            upto=new_entries,
            expected_prefix=seq_len_tokens,
        )

    def plan_request(
        self,
        req_id: str,
        prefix_tokens: int,
        new_entries: int,
        seq_len_tokens: int,
    ) -> "_ReqDeltaState | None":
        """Admit/continue tracking; None means take the full re-gather path.

        On success the caller must (a) delta-gather ``[state.upto,
        new_entries)`` into ``state.staging`` and (b) call :meth:`advance`
        -- even when the delta is empty, so the continuity chain stays
        unbroken.

        ``seq_len_tokens`` is this sighting's seq len, i.e. what the next
        sighting of this request will present as ``prefix_tokens``. It is
        what :meth:`advance` records for an admitted row; it is passed in
        here because a BLOCKED row never reaches advance() and still needs
        the same continuity record to tell its own next chunk from a
        different request that reused its id.
        """
        state = self._states.get(req_id)
        if state is not None and state.staging is None:
            # BLOCKED sentinel. It must outlive re-asking by the SAME
            # request (re-admitting one that is continuously present would
            # thrash the budget) but it must NOT outlive the request: gc_all
            # never sees a blocked id absent while its request keeps
            # occupying a prefill row, so without a freshness check here a
            # later request that reuses the id inherits the block for its
            # whole life. `_block()` records the same next-expected seq len
            # `advance()` records for a tracked row, so the freshness test is
            # the tracked path's EQUALITY test, not a monotonicity test: with
            # prefix caching on (P11) a fresh admission's first sighting
            # presents prefix_tokens = H for a cache hit at H tokens, which
            # is routinely larger than a dead predecessor's last prefix. Any
            # `>=` rule keeps the sentinel exactly in the case this reorder
            # exists to fix.
            if prefix_tokens == state.expected_prefix and new_entries >= state.upto:
                state.expected_prefix = seq_len_tokens
                state.upto = new_entries
                return None  # still the same request: stays blocked
            self._free(req_id)
            state = None
        elif state is not None and (
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
                self._block(req_id, new_entries, seq_len_tokens)
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
                self._block(req_id, new_entries, seq_len_tokens)
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
