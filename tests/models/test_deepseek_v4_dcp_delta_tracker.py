# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P11 step 6: pin the P7 delta tracker's FRESH-ADMISSION path.

With prefix caching on (PROFILE-CACHE), a resumed request reaches the SM8x
DSV4 DCP delta gather for the first time already carrying a large computed
prefix: the scheduler hands the worker a cache hit at ``H`` tokens, so the
first sighting presents ``prefix_tokens = H`` with no tracker history at all.
`Sm86DcpDeltaTracker.plan_request` must treat that exactly like any other
fresh admission -- ``state is None`` skips the prefix-continuity check, the
staging buffer is sized from ``new_entries``, and ``upto`` stays 0 so the
caller performs ONE full gather over ``[0, new_entries)`` before the delta
chain resumes.

No code change was expected here; these tests are the guard. A future
``assert prefix_tokens == 0`` on the fresh-admission arm would look harmless
(it holds for every cold request) while silently breaking every cache hit at
high context.

No GPU is required -- the tracker's planning is host-side bookkeeping and the
staging buffers are allocated on CPU here -- but the module-level
``vllm.models.deepseek_v4`` import pulls the fused-MoE / quantization chain,
so a box without Triton importable cannot collect this file at all.

Budget consequence, recorded so a capacity-math regression is visible: at a
200k-token C4 hit the staging capacity rounds up to 65,536 entries x 584 B =
38.3 MB per (request, layer), so the default 512 MiB per-worker budget covers
only the first handful of layers and the rest fall to the BLOCKED sentinel
and the pre-existing full re-gather path. That is the same steady state the
tracker reaches today after growth -- a cache hit just reaches it in one
step instead of many.
"""

import pytest
import torch

from vllm import envs
from vllm.models.deepseek_v4.ampere import dcp_delta_tracker
from vllm.models.deepseek_v4.ampere.dcp_delta_tracker import (
    Sm86DcpDeltaTracker,
    get_delta_gather_budget,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _SM86_DCP_ENTRY_BYTES,
)
from vllm.utils.math_utils import next_power_of_2

pytestmark = pytest.mark.cpu_test

# A 200k-token prefix-cache hit under PROFILE-CACHE. `H` is whatever
# `find_longest_cache_hit` reconciled, hence a multiple of the dcp=4
# scheduler block size (1024); the compressor entry count is a pure function
# of the seq len, so the tracker only ever sees `seq_len // compress_ratio`.
_COMPRESS_RATIO = 4  # C4A attention + indexer compressors
_HIT_TOKENS = 196 * 1024
_CHUNK_TOKENS = 512  # PROFILE-CACHE max_num_batched_tokens


def _entries(seq_len_tokens: int) -> int:
    return seq_len_tokens // _COMPRESS_RATIO


@pytest.fixture
def isolated_budget(monkeypatch):
    """A private budget + tracker registry, rebuilt from the env defaults.

    `Sm86DcpDeltaTracker` charges a process-wide singleton and appends itself
    to a class-level instance list, both of which would otherwise leak across
    tests.
    """
    monkeypatch.delenv("VLLM_DSV4_DELTA_GATHER_BUDGET_MB", raising=False)
    monkeypatch.setattr(dcp_delta_tracker, "_BUDGET", None)
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_instances", [])
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_absence", {})
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_flushed", set())
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_flush_absence", {})
    return get_delta_gather_budget()


@pytest.fixture
def tiny_budget(monkeypatch):
    """As `isolated_budget`, but 1 MiB: seven min-capacity stagings fit.

    Small enough to reach the BLOCKED sentinel without building the 14
    layers a 200k cache hit needs to exhaust the 512 MiB default.
    """
    monkeypatch.setenv("VLLM_DSV4_DELTA_GATHER_BUDGET_MB", "1")
    monkeypatch.setattr(dcp_delta_tracker, "_BUDGET", None)
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_instances", [])
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_absence", {})
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_flushed", set())
    monkeypatch.setattr(Sm86DcpDeltaTracker, "_flush_absence", {})
    return get_delta_gather_budget()


@pytest.fixture
def empty_cache_calls(monkeypatch):
    """Counts `torch.cuda.empty_cache()` calls (a no-op without CUDA)."""
    calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(None))
    return calls


def test_plan_request_admits_fresh_admission_with_nonzero_prefix(isolated_budget):
    """A cache-hit resume looks like a fresh admission with a huge prefix."""
    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))
    seq_len = _HIT_TOKENS + _CHUNK_TOKENS
    new_entries = _entries(seq_len)

    state = tracker.plan_request("r1", _HIT_TOKENS, new_entries, seq_len)

    # Admitted, not refused: nothing on this path may require prefix_tokens
    # to be 0 just because there is no history to be continuous with.
    assert state is not None
    assert state.staging is not None
    # upto=0 => the caller delta-gathers [0, new_entries), i.e. one full
    # gather, and only then does the chain become incremental.
    assert state.upto == 0
    # The continuity check is skipped, not satisfied: the tracker adopts the
    # presented prefix as the chain's origin.
    assert state.expected_prefix == _HIT_TOKENS
    assert state.capacity == next_power_of_2(new_entries)
    assert state.staging.shape == (state.capacity, _SM86_DCP_ENTRY_BYTES)
    assert isolated_budget.used_bytes == state.capacity * _SM86_DCP_ENTRY_BYTES


def test_plan_request_resumes_delta_chain_after_a_cache_hit_admission(
    isolated_budget,
):
    """The chunk after the resume gathers a DELTA, not a second full gather."""
    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))
    seq_len = _HIT_TOKENS + _CHUNK_TOKENS
    new_entries = _entries(seq_len)

    state = tracker.plan_request("r1", _HIT_TOKENS, new_entries, seq_len)
    assert state is not None
    tracker.advance(state, new_entries, seq_len)
    charged_after_admission = isolated_budget.used_bytes

    next_seq_len = seq_len + _CHUNK_TOKENS
    next_state = tracker.plan_request(
        "r1", seq_len, _entries(next_seq_len), next_seq_len
    )

    assert next_state is state
    assert next_state.upto == new_entries
    assert next_state.expected_prefix == seq_len
    # next_power_of_2(new_entries) already covers the next chunk, so the
    # resume costs no realloc and no extra budget.
    assert next_state.capacity == state.capacity
    assert isolated_budget.used_bytes == charged_after_admission


def test_delta_gather_budget_covers_a_fixed_number_of_layers_at_200k(
    isolated_budget,
):
    """The budget is charged per (request, layer); a 200k hit charges all at once.

    Every compressed attention layer owns its own tracker, so the resumed
    request is admitted layer by layer until the byte budget refuses; the
    remaining layers keep the pre-existing full re-gather path. Asserted as
    ``limit // per-layer capacity bytes`` rather than a literal so a change
    to `_initial_capacity`'s power-of-two rounding is visible here.
    """
    seq_len = _HIT_TOKENS + _CHUNK_TOKENS
    new_entries = _entries(seq_len)
    per_layer_bytes = next_power_of_2(new_entries) * _SM86_DCP_ENTRY_BYTES
    expected_layers = isolated_budget.limit_bytes // per_layer_bytes

    # The env budget is read as MiB, not decimal MB: 512 MiB / 38.3 MB is 14
    # layers, not the 13 a decimal reading gives.
    assert isolated_budget.limit_bytes == (
        envs.VLLM_DSV4_DELTA_GATHER_BUDGET_MB * 1024 * 1024
    )
    assert expected_layers >= 1

    trackers = [
        Sm86DcpDeltaTracker(torch.device("cpu")) for _ in range(expected_layers + 2)
    ]
    admitted = [
        tracker
        for tracker in trackers
        if tracker.plan_request("r1", _HIT_TOKENS, new_entries, seq_len) is not None
    ]

    assert len(admitted) == expected_layers
    assert isolated_budget.used_bytes == expected_layers * per_layer_bytes
    # The refused layers latch the BLOCKED sentinel: re-asking on the NEXT
    # chunk (the only thing the caller ever does) must not thrash the budget.
    next_seq_len = seq_len + _CHUNK_TOKENS
    for tracker in trackers[expected_layers:]:
        assert (
            tracker.plan_request("r1", seq_len, _entries(next_seq_len), next_seq_len)
            is None
        )
    assert isolated_budget.used_bytes == expected_layers * per_layer_bytes


# ---------------------------------------------------------------------------
# Cross-request state leaks (pre-existing, found during P11 rack verification).
# ---------------------------------------------------------------------------

_PREFILL_STEPS = 4  # chunks per request in the step simulations below


def _prefill_step(
    tracker: Sm86DcpDeltaTracker, req_id: str, chunk: int
) -> None:
    """One scheduler step in which ``req_id`` is the only prefill row.

    Mirrors the real call order: the metadata builder GCs across all layer
    trackers BEFORE any layer runs (sparse_swa.py), then the layer plans and
    advances its chunk.
    """
    Sm86DcpDeltaTracker.gc_all([req_id], [req_id])
    prefix_tokens = chunk * _CHUNK_TOKENS
    seq_len = prefix_tokens + _CHUNK_TOKENS
    state = tracker.plan_request(req_id, prefix_tokens, _entries(seq_len), seq_len)
    if state is not None:
        tracker.advance(state, _entries(seq_len), seq_len)


def _run_prefill(
    tracker: Sm86DcpDeltaTracker, req_id: str, empty_cache_calls: "list[None]"
) -> "list[int]":
    """Drive a whole prefill; return the chunk indices that flushed."""
    flushed_at: list[int] = []
    for chunk in range(_PREFILL_STEPS):
        before = len(empty_cache_calls)
        _prefill_step(tracker, req_id, chunk)
        if len(empty_cache_calls) > before:
            flushed_at.append(chunk)
    return flushed_at


def test_empty_cache_is_symmetric_across_consecutive_identical_requests(
    isolated_budget, empty_cache_calls
):
    """Two identical back-to-back requests must see the same allocator events.

    `gc_all` runs from the metadata builder on every step, and the flush it
    performs frees every cached block back to CUDA -- which moves the address
    (and the residual bytes) of every `torch.empty` later in that same step.
    Tying it to a free makes it land in the middle of the NEXT request's
    prefill (`_GRACE` steps after its predecessor vanished), so the first
    request of a fresh process never sees it and its identical successor
    does.
    """
    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))

    first = _run_prefill(tracker, "r1", empty_cache_calls)
    # "r1" finished inside its last prefill chunk (max_tokens=1 / abort), so
    # it never appears as a decode row and only the grace path can free it.
    second = _run_prefill(tracker, "r2", empty_cache_calls)

    assert first == second, (
        "empty_cache() landed on different steps of two identical requests: "
        f"{first} vs {second}"
    )
    # ...and not by never running at all: the flush is load-bearing (it is
    # what keeps the 2nd consecutive 200k prompt from OOMing).
    assert first, "empty_cache() must still run once per request"


def test_empty_cache_runs_at_a_request_boundary_before_any_allocation(
    isolated_budget, empty_cache_calls
):
    """The flush belongs to the request entering prefill, not to a free.

    `gc_all` is called by the builder before any layer's forward, so a flush
    issued there for the entering request precedes every allocation that
    request makes -- exactly where the recorded OOM ("16 MiB requests failing
    on the 2nd consecutive 200k prompt") needs it, instead of some steps into
    the prefill that has already started allocating.
    """
    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))
    _run_prefill(tracker, "r1", empty_cache_calls)

    before = len(empty_cache_calls)
    Sm86DcpDeltaTracker.gc_all(["r2"], ["r2"])  # r2's FIRST prefill step
    assert len(empty_cache_calls) > before
    # Nothing of r2 has been planned yet, so the flush cannot have moved any
    # of r2's own buffers.
    assert "r2" not in tracker._states

    # A continuing chunk is not a boundary: no further flush.
    before = len(empty_cache_calls)
    state = tracker.plan_request("r2", 0, _entries(_CHUNK_TOKENS), _CHUNK_TOKENS)
    assert state is not None
    tracker.advance(state, _entries(_CHUNK_TOKENS), _CHUNK_TOKENS)
    Sm86DcpDeltaTracker.gc_all(["r2"], ["r2"])
    assert len(empty_cache_calls) == before


def test_blocked_sentinel_is_dropped_when_a_new_request_reuses_the_id(
    tiny_budget,
):
    """A budget-blocked (request, layer) must not outlive its request.

    `gc_all` cannot clear the sentinel while the id is in the prefill rows,
    and a request is in the prefill rows on every one of its own steps -- so
    if the early return fires before the prefix-continuity reset, a later
    request that reuses the id inherits the block for its entire life and
    silently runs the full re-gather path its predecessor-less twin would
    not.
    """
    first_entries = _entries(_CHUNK_TOKENS)
    # A first chunk is far under the `_MIN_CAPACITY_ENTRIES` floor, so every
    # admission charges the same 256 x 584 B whatever the chunk size.
    per_hog_bytes = (
        next_power_of_2(max(first_entries, dcp_delta_tracker._MIN_CAPACITY_ENTRIES))
        * _SM86_DCP_ENTRY_BYTES
    )
    hog_layers = tiny_budget.limit_bytes // per_hog_bytes
    assert hog_layers >= 1

    hogs = [Sm86DcpDeltaTracker(torch.device("cpu")) for _ in range(hog_layers)]
    for hog in hogs:
        assert hog.plan_request("hog", 0, first_entries, _CHUNK_TOKENS) is not None

    # This layer is refused and latches the BLOCKED sentinel on its first,
    # cold chunk (prefix_tokens == 0).
    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))
    assert tracker.plan_request("x", 0, first_entries, _CHUNK_TOKENS) is None

    # The hogs finish; their staging is freed after the grace period, so the
    # budget is wide open again. "x" is in the prefill rows throughout, so
    # its sentinel is never a GC candidate.
    for _ in range(Sm86DcpDeltaTracker._GRACE):
        Sm86DcpDeltaTracker.gc_all(["x"], ["x"])
    assert tiny_budget.used_bytes == 0

    # Same request, next chunk: the sentinel must survive. Re-admitting a
    # continuously present request would thrash the budget.
    assert (
        tracker.plan_request(
            "x", _CHUNK_TOKENS, _entries(2 * _CHUNK_TOKENS), 2 * _CHUNK_TOKENS
        )
        is None
    )
    assert tiny_budget.used_bytes == 0

    # A NEW request reuses the id and starts cold. This is a fresh admission
    # -- no history for it to be continuous with -- and it must be planned on
    # the budget's merits, not on its predecessor's.
    state = tracker.plan_request("x", 0, first_entries, _CHUNK_TOKENS)
    assert state is not None
    assert state.staging is not None
    assert state.upto == 0
    assert state.expected_prefix == 0
    assert tiny_budget.used_bytes == per_hog_bytes


def test_empty_cache_fires_once_per_request_across_a_skipped_prefill_step(
    isolated_budget, empty_cache_calls
):
    """A prefill that gets zero tokens for a step must not re-pay the flush.

    `prefill_req_ids` is a slice of the step's `req_ids`, so a co-scheduled
    prefill that is scheduled zero tokens is absent from the prefill rows
    entirely and reappears on the next step -- documented in this fork at
    `ampere_sparse._maybe_delta_tracker` ("possible at 3+ concurrent prefills
    under long_prefill_token_threshold") and made likely by P10's adaptive
    long-prefill threshold. A set-difference trigger charges that request a
    second device-synchronising `empty_cache()` in the MIDDLE of its own
    prefill, with its staging and indexer transients already allocated.
    """
    Sm86DcpDeltaTracker(torch.device("cpu"))

    Sm86DcpDeltaTracker.gc_all(["a", "b"], ["a", "b"])
    assert len(empty_cache_calls) == 1  # both entered on the same step

    # "b" is given zero tokens this step: it keeps its blocks and its place in
    # the batch, but it carries no prefill row.
    Sm86DcpDeltaTracker.gc_all(["a", "b"], ["a"])
    # ...and comes back on the next one.
    Sm86DcpDeltaTracker.gc_all(["a", "b"], ["a", "b"])

    assert len(empty_cache_calls) == 1, (
        "a skipped prefill step re-fired the allocator flush mid-prefill"
    )


def test_empty_cache_latch_survives_a_dummy_build_with_no_requests(
    isolated_budget, empty_cache_calls
):
    """`gc_all` is reached on dummy/warmup/capture builds with an empty set.

    `gpu_model_runner` populates `cm_req_ids` whenever VLLM_SM86_DCP and
    VLLM_DSV4_DELTA_GATHER are on, with no `for_cudagraph_capture` gate, so
    an empty id set reaches gc_all. Clearing the latches there would re-fire
    the flush for every request still mid-prefill.
    """
    Sm86DcpDeltaTracker(torch.device("cpu"))

    Sm86DcpDeltaTracker.gc_all(["a"], ["a"])
    assert len(empty_cache_calls) == 1

    Sm86DcpDeltaTracker.gc_all([], [])  # dummy batch
    Sm86DcpDeltaTracker.gc_all(["a"], ["a"])

    assert len(empty_cache_calls) == 1


def test_empty_cache_latch_is_retired_after_the_request_is_gone(
    isolated_budget, empty_cache_calls
):
    """The latch set must not grow without bound over a server's lifetime."""
    Sm86DcpDeltaTracker(torch.device("cpu"))

    Sm86DcpDeltaTracker.gc_all(["a"], ["a"])
    assert Sm86DcpDeltaTracker._flushed == {"a"}
    for _ in range(Sm86DcpDeltaTracker._GRACE):
        Sm86DcpDeltaTracker.gc_all(["b"], ["b"])
    assert "a" not in Sm86DcpDeltaTracker._flushed


def test_blocked_sentinel_is_dropped_when_the_reusing_request_hits_the_cache(
    tiny_budget,
):
    """The reuse case P11 actually produces: a first sighting with a prefix.

    With prefix caching on, a fresh admission presents ``prefix_tokens = H``
    for a cache hit at H tokens -- routinely far larger than a dead
    predecessor's last recorded prefix. A monotonicity test (``>=``) keeps the
    stale sentinel in exactly this case; only the tracked path's equality
    contract drops it.
    """
    first_entries = _entries(_CHUNK_TOKENS)
    per_hog_bytes = (
        next_power_of_2(max(first_entries, dcp_delta_tracker._MIN_CAPACITY_ENTRIES))
        * _SM86_DCP_ENTRY_BYTES
    )
    hog_layers = tiny_budget.limit_bytes // per_hog_bytes
    assert hog_layers >= 1

    hogs = [Sm86DcpDeltaTracker(torch.device("cpu")) for _ in range(hog_layers)]
    for hog in hogs:
        assert hog.plan_request("hog", 0, first_entries, _CHUNK_TOKENS) is not None

    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))
    assert tracker.plan_request("x", 0, first_entries, _CHUNK_TOKENS) is None

    for _ in range(Sm86DcpDeltaTracker._GRACE):
        Sm86DcpDeltaTracker.gc_all(["x"], ["x"])
    assert tiny_budget.used_bytes == 0

    # A NEW request reuses the id and its FIRST sighting carries a cache hit.
    # The hit is kept small only so the 1 MiB test budget can admit it; what
    # matters is that it is >= the dead sentinel's recorded prefix (512) and
    # its entry count >= the sentinel's, i.e. every `>=` rule keeps the block.
    hit_tokens = 4 * _CHUNK_TOKENS
    hit_seq_len = hit_tokens + _CHUNK_TOKENS
    assert hit_tokens >= _CHUNK_TOKENS  # the sentinel's recorded prefix
    state = tracker.plan_request("x", hit_tokens, _entries(hit_seq_len), hit_seq_len)
    assert state is not None, "stale BLOCKED sentinel survived a cache-hit reuse"
    assert state.staging is not None
    assert state.upto == 0
    assert state.expected_prefix == hit_tokens
