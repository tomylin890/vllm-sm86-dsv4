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
    return get_delta_gather_budget()


def test_plan_request_admits_fresh_admission_with_nonzero_prefix(isolated_budget):
    """A cache-hit resume looks like a fresh admission with a huge prefix."""
    tracker = Sm86DcpDeltaTracker(torch.device("cpu"))
    seq_len = _HIT_TOKENS + _CHUNK_TOKENS
    new_entries = _entries(seq_len)

    state = tracker.plan_request("r1", _HIT_TOKENS, new_entries)

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

    state = tracker.plan_request("r1", _HIT_TOKENS, new_entries)
    assert state is not None
    tracker.advance(state, new_entries, seq_len)
    charged_after_admission = isolated_budget.used_bytes

    next_seq_len = seq_len + _CHUNK_TOKENS
    next_state = tracker.plan_request("r1", seq_len, _entries(next_seq_len))

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
        if tracker.plan_request("r1", _HIT_TOKENS, new_entries) is not None
    ]

    assert len(admitted) == expected_layers
    assert isolated_budget.used_bytes == expected_layers * per_layer_bytes
    # The refused layers latch the BLOCKED sentinel: re-asking on the next
    # chunk must not thrash the budget.
    for tracker in trackers[expected_layers:]:
        assert tracker.plan_request("r1", _HIT_TOKENS, new_entries) is None
    assert isolated_budget.used_bytes == expected_layers * per_layer_bytes
