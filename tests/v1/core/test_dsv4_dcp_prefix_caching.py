# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P11 step 10: offline tests for prefix caching under the DeepseekV4 SM8x
DCP hybrid KV cache layout (PROFILE-CACHE).

The production group set is six groups: three MLAAttentionSpec-family groups
(compressed-KV C4A, compressed-KV C128A, sparse-indexer k_cache) at block
size 256, the SWA KV window at (64, 128), and the two fp32 compressor-state
families at (4, 8) -- covering BOTH the attention C4 compressors and the
indexer's own compressor, which is instantiated only on compress_ratio == 4
layers -- and (8, 128). At dcp=4 the MLA groups are round-robin sharded so
one logical block covers 256 * 4 tokens, while every sliding-window group is
dcp_exempt (REPLICATED per rank) and keeps its unsharded block size.

Two startup blockers had to be cleared for that configuration to boot, and
they are what most of these tests pin:

* the scheduler-side group block sizes must be the dcp_exempt-aware ones,
  otherwise ``hash_block_size = gcd(1024, 256, 16, 32) = 16`` while the
  managers really use 4 and the coordinator's divisibility assert fires;
* the coordinator's ``dcp > 1`` spec-type gate must accept sliding-window
  groups when ``VLLM_SM86_DCP`` is set -- and only then, and only for groups
  the replication predicate also accepts.

No GPU is required: allocation, hashing and hit reconciliation are entirely
scheduler-side (invariant I5). The tests that reach the compressor's own
config hooks do need the ``vllm.models.deepseek_v4`` package to be
importable, which pulls the fused-MoE / quantization chain and hence Triton;
those imports are function-local so a Triton-less box still collects this
module and runs the rest.

Assertions are written in terms of ``cdiv(window - 1, block_size)`` and the
LCM/GCD of the actual specs rather than today's numbers, so a future block
size or compress ratio cannot move silently.
"""

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

_DCP_WORLD_SIZE = 4
_MAX_MODEL_LEN = 262144
# PROFILE-CACHE caps max_num_batched_tokens at 512 (the compressor-state
# reservation is O(F) once the P8 ring is off).
_MAX_IN_FLIGHT_TOKENS = 512
_POOL_BLOCKS = 4096

_MLA_BLOCK_SIZE = 256
_SWA_BLOCK_SIZE = 64
_SWA_WINDOW = 128
# `coff = 1 + (compress_ratio == 4)` gives the C4 family an OVERLAPPED
# window of 2 * 4 = 8 rows; C128 (and the indexer's C128-side state) is not
# overlapped, so its window is its compress ratio.
_C4_STATE_BLOCK_SIZE = 4
_C4_STATE_WINDOW = 8
_C128_STATE_BLOCK_SIZE = 8
_C128_STATE_WINDOW = 128
# VLLM_DSV4_COMPRESSOR_WINDOW default (PROFILE-P8 only).
_RING_WINDOW = 512
# The three sliding-window families the model really has: SWA KV, C4 state,
# C128 state. Stated as a count rather than recomputed from the fixture so a
# loop that silently stops visiting groups fails instead of passing vacuously
# -- this is a property of the model, not a geometry constant, so it is not
# the kind of literal step 10 bans.
_NUM_SLIDING_WINDOW_GROUPS = 3


@pytest.fixture(autouse=True)
def _init_hash_fn():
    init_none_hash(sha256)


def _mla_kv_spec(**kwargs) -> MLAAttentionSpec:
    """A DeepseekV4 compressed-KV / indexer group: MLA, hence sharded."""
    return MLAAttentionSpec(
        block_size=_MLA_BLOCK_SIZE,
        num_kv_heads=1,
        dtype=torch.uint8,
        **kwargs,
    )


def _dsv4_kv_cache_groups(
    extra_specs: tuple[KVCacheSpec, ...] = (),
) -> list[KVCacheGroupSpec]:
    """The scheduler-side KV cache groups DeepseekV4 produces on this branch.

    ``group_and_unify_kv_cache_specs`` splits the sliding-window layers by
    ``(block_size, sliding_window)``, so the two compressor-state families
    are separate groups even though both are SlidingWindowMLASpec.
    """
    specs: list[KVCacheSpec] = [
        # Compressed KV, C4A layers.
        _mla_kv_spec(
            head_size=512,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            compress_ratio=4,
        ),
        # Compressed KV, C128A layers.
        _mla_kv_spec(
            head_size=512,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            compress_ratio=128,
        ),
        # Sparse-indexer k_cache: same block size, much smaller page.
        _mla_kv_spec(head_size=132),
        # Real SWA KV window (uint8), not compressor state.
        SlidingWindowMLASpec(
            block_size=_SWA_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=_SWA_WINDOW,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
        ),
        # fp32 compressor state, C4 family (attention C4A compressors AND
        # the indexer compressors, which are handed compress_ratio 4).
        SlidingWindowMLASpec(
            block_size=_C4_STATE_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=2048,
            dtype=torch.float32,
            sliding_window=_C4_STATE_WINDOW,
        ),
        # fp32 compressor state, C128 family.
        SlidingWindowMLASpec(
            block_size=_C128_STATE_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=2048,
            dtype=torch.float32,
            sliding_window=_C128_STATE_WINDOW,
        ),
    ]
    specs.extend(extra_specs)
    return [KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)]


def _kv_cache_config(
    groups: list[KVCacheGroupSpec], num_blocks: int = _POOL_BLOCKS
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


def _vllm_config(
    dcp_world_size: int,
    enable_prefix_caching: bool,
    prefix_match_unit: int | None = None,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=_MLA_BLOCK_SIZE,
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_world_size),
        # Only read by the compressor's config hook, for the startup log.
        scheduler_config=SimpleNamespace(max_num_batched_tokens=_MAX_IN_FLIGHT_TOKENS),
        kv_transfer_config=None,
    )


def _replicated_block_sizes(
    groups: list[KVCacheGroupSpec], dcp_world_size: int
) -> list[int]:
    """The block size every consumer must agree on (invariants I3/I4).

    Sliding-window groups are dcp_exempt -- a full copy of their tokens lives
    on every DCP rank -- so one logical block still covers ``block_size``
    tokens. Every other attention group is round-robin sharded, so one
    logical block covers ``block_size * dcp``. Written out here rather than
    imported so the test states the invariant instead of restating the
    implementation.
    """
    return [
        g.kv_cache_spec.block_size
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
        else g.kv_cache_spec.block_size * dcp_world_size
        for g in groups
    ]


def _unexempted_block_sizes(
    groups: list[KVCacheGroupSpec], dcp_world_size: int
) -> list[int]:
    """What the scheduler side computed before the dcp_exempt gate: every
    AttentionSpec scaled, replicated or not."""
    return [
        g.kv_cache_spec.block_size * dcp_world_size
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]


def _dsv4_scheduler_block_size() -> int:
    groups = _dsv4_kv_cache_groups()
    return math.lcm(*_replicated_block_sizes(groups, _DCP_WORLD_SIZE))


def _build_coordinator(
    groups: list[KVCacheGroupSpec],
    *,
    dcp_world_size: int = _DCP_WORLD_SIZE,
    enable_caching: bool = True,
    num_blocks: int = _POOL_BLOCKS,
) -> HybridKVCacheCoordinator:
    kv_cache_config = _kv_cache_config(groups, num_blocks)
    vllm_config = _vllm_config(dcp_world_size, enable_caching)
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    return HybridKVCacheCoordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=_MAX_MODEL_LEN,
        max_in_flight_tokens=_MAX_IN_FLIGHT_TOKENS,
        use_eagle=False,
        enable_caching=enable_caching,
        enable_kv_cache_events=False,
        dcp_world_size=dcp_world_size,
        pcp_world_size=1,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )


def _make_request(request_id: str, num_tokens: int, hash_block_size: int):
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


# --------------------------------------------------------------------------
# Scheduler-side block sizes (P11 step 2 / blocker B2)
# --------------------------------------------------------------------------


def test_resolve_block_sizes_exempts_replicated_groups_at_dcp4(monkeypatch):
    """hash_block_size must be the GCD of the REAL (manager) block sizes.

    Scaling the replicated groups by dcp is inert for the LCM but not for the
    GCD: it lifts the hash granularity above the 4-token compressor-state
    block size, and the coordinator then refuses every manager whose real
    block size does not divide it.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    real = _replicated_block_sizes(groups, _DCP_WORLD_SIZE)
    unexempted = _unexempted_block_sizes(groups, _DCP_WORLD_SIZE)

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        _kv_cache_config(groups),
        _vllm_config(_DCP_WORLD_SIZE, enable_prefix_caching=True),
    )

    assert scheduler_block_size == math.lcm(*real)
    assert hash_block_size == math.gcd(*real)
    # I3: every manager's real block size divides the hash granularity.
    assert all(bs % hash_block_size == 0 for bs in real)
    # I2: cache-hit boundaries are multiples of the scheduler block size, so
    # they are multiples of every window and compress ratio in the model.
    assert all(scheduler_block_size % bs == 0 for bs in real)
    # The gate is inert for the LCM ...
    assert math.lcm(*unexempted) == scheduler_block_size
    # ... and load-bearing for the GCD: without it the coordinator's
    # divisibility assert fires at engine init.
    assert math.gcd(*unexempted) != hash_block_size
    assert any(bs % math.gcd(*unexempted) != 0 for bs in real)


def test_resolve_block_sizes_at_dcp1_uses_the_raw_spec_sizes(monkeypatch):
    """At dcp=1 nothing is sharded, so both aggregations are over the specs."""
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    spec_sizes = [g.kv_cache_spec.block_size for g in groups]

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        _kv_cache_config(groups),
        _vllm_config(1, enable_prefix_caching=True),
    )

    assert scheduler_block_size == math.lcm(*spec_sizes)
    assert hash_block_size == math.gcd(*spec_sizes)
    assert scheduler_block_size % 128 == 0


def test_resolve_block_sizes_with_caching_off_keeps_the_scheduler_lcm(
    monkeypatch,
):
    """PROFILE-P8 / PROFILE-CONTROL must see byte-identical alignment.

    Block hashes are inert without prefix caching or a KV connector, so the
    function returns the scheduler granularity for both values; the point of
    the assertion is that the exemption did not move that granularity.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()

    off = resolve_kv_cache_block_sizes(
        _kv_cache_config(groups),
        _vllm_config(_DCP_WORLD_SIZE, enable_prefix_caching=False),
    )
    on = resolve_kv_cache_block_sizes(
        _kv_cache_config(groups),
        _vllm_config(_DCP_WORLD_SIZE, enable_prefix_caching=True),
    )

    scheduler_block_size = math.lcm(*_unexempted_block_sizes(groups, _DCP_WORLD_SIZE))
    assert off == (scheduler_block_size, scheduler_block_size)
    assert on[0] == off[0]


def test_prefix_match_unit_override_is_a_valid_fallback(monkeypatch):
    """The config-only workaround for B2 must keep validating.

    ``--prefix-match-unit <gcd>`` satisfies the divisibility check for the
    scaled and the exempt block sizes alike, so it stays available whether or
    not the scheduler-side gate is in place.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    real = _replicated_block_sizes(groups, _DCP_WORLD_SIZE)
    unit = math.gcd(*real)

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        _kv_cache_config(groups),
        _vllm_config(_DCP_WORLD_SIZE, True, prefix_match_unit=unit),
    )

    assert hash_block_size == unit
    assert scheduler_block_size == math.lcm(*real)


# --------------------------------------------------------------------------
# Coordinator spec-type gate (P11 step 3 / blocker B1)
# --------------------------------------------------------------------------


def test_hybrid_coordinator_constructs_at_dcp4_with_caching(monkeypatch):
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()

    coordinator = _build_coordinator(groups)

    real = _replicated_block_sizes(groups, _DCP_WORLD_SIZE)
    assert [m.block_size for m in coordinator.single_type_managers] == real
    assert coordinator.scheduler_block_size == math.lcm(*real)
    assert coordinator.hash_block_size == math.gcd(*real)
    # Partial (hash-granular) hits need full attention + mamba without
    # context parallelism, so hits stay scheduler-block aligned.
    assert not coordinator.enable_partial_hash_hits
    assert coordinator._cache_hit_alignment_tokens == math.lcm(*real)
    # I4: "skips the dcp scale", "is a sliding-window group" and "is not a
    # FullAttentionSpec" must be the same set of groups.
    unscaled = {
        i
        for i, manager in enumerate(coordinator.single_type_managers)
        if manager.block_size == groups[i].kv_cache_spec.block_size
    }
    sliding_window = {
        i
        for i, g in enumerate(groups)
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
    }
    not_full_attention = {
        i
        for i, g in enumerate(groups)
        if not isinstance(g.kv_cache_spec, FullAttentionSpec)
    }
    assert unscaled == sliding_window == not_full_attention


def test_hybrid_coordinator_rejects_sliding_window_without_the_gate(
    monkeypatch,
):
    """Upstream behavior is preserved for every model that is not DSV4-DCP."""
    monkeypatch.delenv("VLLM_SM86_DCP", raising=False)
    groups = _dsv4_kv_cache_groups()

    with pytest.raises(AssertionError, match="full-attention and Mamba"):
        _build_coordinator(groups)


def test_hybrid_coordinator_rejects_a_non_replicated_spec_under_the_gate(
    monkeypatch,
):
    """The whitelist is not a blanket bypass: only replicated kinds pass.

    A chunked-local group is an AttentionSpec that is neither full attention
    nor sliding window, so it is still DCP-sharded and must still be
    rejected even with VLLM_SM86_DCP set.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    chunked_local = ChunkedLocalAttentionSpec(
        block_size=_MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        attention_chunk_size=1024,
    )
    groups = _dsv4_kv_cache_groups(extra_specs=(chunked_local,))

    with pytest.raises(AssertionError, match="full-attention and Mamba"):
        _build_coordinator(groups)


# --------------------------------------------------------------------------
# Cache-hit geometry (invariants I1 / I2)
# --------------------------------------------------------------------------


def _block_hashes(num_hash_blocks: int) -> list[BlockHash]:
    return [BlockHash(f"dsv4-{i}".encode()) for i in range(num_hash_blocks)]


def _seed_prefix(
    coordinator: HybridKVCacheCoordinator,
    groups: list[KVCacheGroupSpec],
    block_hashes: list[BlockHash],
    hit_tokens: int,
    *,
    evicted_group_ids: tuple[int, ...] = (),
) -> None:
    """Populate the hash map exactly as a finished request would leave it.

    Sharded groups cache every full block; replicated sliding-window groups
    only cache what ``reachable_block_mask`` keeps -- the ``need``-block tail
    of each scheduler-block segment -- so the seeding mirrors the write side
    instead of over-caching.
    """
    block_pool = coordinator.block_pool
    free_block_ids = iter(range(1, block_pool.num_gpu_blocks))
    for group_id, group in enumerate(groups):
        if group_id in evicted_group_ids:
            continue
        spec = group.kv_cache_spec
        block_size = coordinator.single_type_managers[group_id].block_size
        num_blocks = hit_tokens // block_size
        block_mask = (
            SlidingWindowManager.reachable_block_mask(
                start_block=0,
                end_block=num_blocks,
                alignment_tokens=coordinator.scheduler_block_size,
                kv_cache_spec=spec,
                use_eagle=False,
            )
            if isinstance(spec, SlidingWindowSpec)
            else None
        )
        scale = block_size // block_pool.hash_block_size
        for i in range(num_blocks):
            if block_mask is not None and not block_mask[i]:
                continue
            block_pool.cached_block_hash_to_block.insert(
                make_block_hash_with_group_id(
                    block_hashes[(i + 1) * scale - 1], group_id
                ),
                block_pool.blocks[next(free_block_ids)],
            )


def _record_finder_dcp_world_size(monkeypatch) -> dict[int, int]:
    """Spy on the ``dcp_world_size`` each group's finder is handed."""
    seen: dict[int, int] = {}
    for manager_cls in (FullAttentionManager, SlidingWindowManager):
        original = manager_cls.find_longest_cache_hit

        def spy(*, _original=original, **kwargs):
            for group_id in kwargs["kv_cache_group_ids"]:
                seen[group_id] = kwargs["dcp_world_size"]
            return _original(**kwargs)

        monkeypatch.setattr(manager_cls, "find_longest_cache_hit", staticmethod(spy))
    return seen


def test_find_longest_cache_hit_reserves_the_compression_lookback(monkeypatch):
    """Every sliding-window group hands back exactly its lookback tail.

    That tail is what replaces the rejected block trim: at a hit boundary H
    the first recomputed C4 boundary is p = H + m - 1 and its gather starts
    at H - (L - m), so the resume is correct iff the cached tail reaches at
    least ``L - 1`` tokens below H.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    coordinator = _build_coordinator(groups)
    hit_tokens = 4 * coordinator.scheduler_block_size
    block_hashes = _block_hashes(hit_tokens // coordinator.hash_block_size)
    _seed_prefix(coordinator, groups, block_hashes, hit_tokens)
    dcp_seen = _record_finder_dcp_world_size(monkeypatch)

    hit_blocks, hit_length, num_uncached = coordinator.find_longest_cache_hit(
        block_hashes, hit_tokens
    )

    # I2: the reconciled hit is a whole number of scheduler blocks.
    assert hit_length == hit_tokens
    assert hit_length % coordinator.scheduler_block_size == 0
    assert num_uncached == 0

    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        block_size = coordinator.single_type_managers[group_id].block_size
        blocks = hit_blocks[group_id]
        assert len(blocks) == hit_length // block_size
        real_blocks = [i for i, blk in enumerate(blocks) if not blk.is_null]
        if isinstance(spec, SlidingWindowSpec):
            need = SlidingWindowManager._contiguous_blocks_for_hit(
                spec.sliding_window, block_size, False
            )
            # I1: the reserved tail covers the lookback below the boundary.
            assert need * block_size >= spec.sliding_window - 1
            # The rest of the row is null-padded at absolute positions, so
            # block-table columns still address tokens by position.
            assert real_blocks == list(range(len(blocks) - need, len(blocks)))
            # Replicated groups hold full copies, so their hashes are viewed
            # unsharded.
            assert dcp_seen[group_id] == 1
        else:
            assert real_blocks == list(range(len(blocks)))
            assert dcp_seen[group_id] == _DCP_WORLD_SIZE


def test_find_longest_cache_hit_is_reconciled_by_the_minimum(monkeypatch):
    """B4: one evicted state tail drags the whole request's hit to zero.

    The compressor-state groups free their tails mid-request
    (``get_num_skipped_tokens``), so they reach the free queue long before
    the MLA blocks and are the first eviction candidates. The hybrid
    reconciliation is a monotone MIN fixed point, so losing one of them
    discards a fully cached 200k MLA prefix rather than shortening it.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    coordinator = _build_coordinator(groups)
    hit_tokens = 4 * coordinator.scheduler_block_size
    block_hashes = _block_hashes(hit_tokens // coordinator.hash_block_size)
    c4_state_group_id = next(
        i
        for i, g in enumerate(groups)
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
        and g.kv_cache_spec.sliding_window == _C4_STATE_WINDOW
    )
    _seed_prefix(
        coordinator,
        groups,
        block_hashes,
        hit_tokens,
        evicted_group_ids=(c4_state_group_id,),
    )

    hit_blocks, hit_length, num_uncached = coordinator.find_longest_cache_hit(
        block_hashes, hit_tokens
    )

    assert hit_length == 0
    assert hit_blocks[c4_state_group_id] == []
    # The sharded groups did match the whole prefix; only the reconciliation
    # threw it away. This is the gap the per-group hit-length instrumentation
    # exists to make visible on the rack.
    assert num_uncached == hit_tokens


def test_find_longest_cache_hit_logs_who_shrank_the_hit(monkeypatch, caplog_vllm):
    """B4 is only diagnosable from the log, so the log has to name the group.

    On the rack an evicted state tail and a plain cache miss look identical:
    both report a reconciled hit of 0. What separates them is that the MLA
    groups matched the whole prefix at first sighting and one sliding-window
    group did not.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    coordinator = _build_coordinator(groups)
    hit_tokens = 4 * coordinator.scheduler_block_size
    block_hashes = _block_hashes(hit_tokens // coordinator.hash_block_size)
    c4_state_group_id = next(
        i
        for i, g in enumerate(groups)
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
        and g.kv_cache_spec.sliding_window == _C4_STATE_WINDOW
    )
    _seed_prefix(
        coordinator,
        groups,
        block_hashes,
        hit_tokens,
        evicted_group_ids=(c4_state_group_id,),
    )

    with caplog_vllm.at_level("DEBUG", logger="vllm.v1.core.kv_cache_coordinator"):
        _, hit_length, _ = coordinator.find_longest_cache_hit(block_hashes, hit_tokens)

    assert hit_length == 0
    record = next(
        message
        for message in caplog_vllm.messages
        if message.startswith("Cache hit reconciliation:")
    )
    assert f"reconciled={hit_length}" in record
    assert f"requested={hit_tokens}" in record
    # Only the evicted group shrank the candidate, and the per-group column
    # is the pre-reconciliation view: the MLA groups still report the full
    # prefix even though the reconciled answer is 0.
    assert record.count("shrunk by") == 1
    per_group, _, shrunk = record.partition("shrunk by")
    # Exactly one group pulled the candidate down, and it is named.
    assert shrunk.count(f"{hit_tokens}, 0") == 1
    assert f"[{c4_state_group_id}]" in shrunk
    assert f"[{c4_state_group_id}], 0)" in per_group
    for group_id, group in enumerate(groups):
        if isinstance(group.kv_cache_spec, SlidingWindowSpec):
            continue
        assert f"[{group_id}], {hit_tokens})" in per_group
    # ... and at least one sliding-window group was consulted before the
    # eviction dragged the candidate down, so its own full-length match is
    # still visible. Logging `hit_length_by_group` instead would report 0 for
    # every sliding-window group, since the fixed point re-runs their finders
    # at the reduced candidate.
    num_sharded_groups = len(groups) - _NUM_SLIDING_WINDOW_GROUPS
    assert per_group.count(f", {hit_tokens})") > num_sharded_groups


def test_reachable_block_mask_keeps_only_the_lookback_tails(monkeypatch):
    """The write side caches exactly the blocks a hit can consult.

    ``need`` blocks at the end of every scheduler-block segment, and nothing
    else: at block size 4 that is 2 of every 256 blocks, at 8 it is 16 of
    every 128, at 64 it is 2 of every 16.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    coordinator = _build_coordinator(groups)
    alignment_tokens = coordinator.scheduler_block_size
    num_segments = 3
    checked = 0

    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, SlidingWindowSpec):
            continue
        checked += 1
        block_size = coordinator.single_type_managers[group_id].block_size
        # Replicated: the mask is built at the unsharded block size, which is
        # the one the worker's block table uses.
        assert block_size == spec.block_size
        per_segment = alignment_tokens // block_size
        need = SlidingWindowManager._contiguous_blocks_for_hit(
            spec.sliding_window, block_size, False
        )
        assert need == cdiv(spec.sliding_window - 1, block_size)
        assert need * block_size >= spec.sliding_window - 1
        end_block = num_segments * per_segment

        mask = SlidingWindowManager.reachable_block_mask(
            start_block=0,
            end_block=end_block,
            alignment_tokens=alignment_tokens,
            kv_cache_spec=spec,
            use_eagle=False,
        )

        assert mask == [
            i % per_segment >= per_segment - need for i in range(end_block)
        ]
        assert sum(mask) == num_segments * need

    assert checked == _NUM_SLIDING_WINDOW_GROUPS


def test_cache_blocks_aligns_down_and_still_covers_the_state_tail(monkeypatch):
    """The state tail at [H - (L - 1), H) is inside the cached region.

    ``cache_blocks`` rounds the computed length DOWN to a scheduler block, so
    a request that stopped mid-segment still registers every block a later
    hit at H would need -- which is what closes the researchers' question of
    whether the compressor-state groups get cached at all.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    coordinator = _build_coordinator(groups)
    aligned = 4 * coordinator.scheduler_block_size
    num_computed_tokens = aligned + coordinator.scheduler_block_size // 2
    request = _make_request(
        "cache-blocks", num_computed_tokens, coordinator.hash_block_size
    )
    coordinator.allocate_new_blocks(
        request.request_id, num_computed_tokens, num_computed_tokens
    )

    coordinator.cache_blocks(request, num_computed_tokens)

    checked = 0
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        manager = coordinator.single_type_managers[group_id]
        block_size = manager.block_size
        blocks = manager.req_to_blocks[request.request_id]
        boundary_block = aligned // block_size
        assert manager.num_cached_block[request.request_id] == boundary_block
        # Nothing past the aligned boundary is registered: a hit can only
        # resume where every group agrees.
        assert blocks[boundary_block].block_hash is None
        if not isinstance(spec, SlidingWindowSpec):
            continue
        checked += 1
        need = SlidingWindowManager._contiguous_blocks_for_hit(
            spec.sliding_window, block_size, False
        )
        tail = blocks[boundary_block - need : boundary_block]
        assert all(blk.block_hash is not None for blk in tail)
        # I1 on the write side: the cached tail starts at or below the lowest
        # row the compression kernel's lookback reaches from the boundary.
        tail_start_token = (boundary_block - need) * block_size
        assert tail_start_token <= aligned - (spec.sliding_window - 1)

    assert checked == _NUM_SLIDING_WINDOW_GROUPS


def test_cache_blocks_round_trips_into_find_longest_cache_hit(monkeypatch):
    """The real writer feeds the real reader.

    Every other hit-geometry test seeds the hash map by hand, using the READ
    side's index convention, so a disagreement between what ``cache_blocks``
    registers and what ``find_longest_cache_hit`` looks up would be invisible
    to all of them. This drives one request through both.
    """
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    coordinator = _build_coordinator(groups)
    aligned = 4 * coordinator.scheduler_block_size
    num_computed_tokens = aligned + coordinator.scheduler_block_size // 2
    request = _make_request(
        "round-trip", num_computed_tokens, coordinator.hash_block_size
    )
    coordinator.allocate_new_blocks(
        request.request_id, num_computed_tokens, num_computed_tokens
    )
    coordinator.cache_blocks(request, num_computed_tokens)

    hit_blocks, hit_length, num_uncached = coordinator.find_longest_cache_hit(
        request.block_hashes, aligned
    )

    assert hit_length == aligned
    assert num_uncached == 0
    checked = 0
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        block_size = coordinator.single_type_managers[group_id].block_size
        real_blocks = [blk for blk in hit_blocks[group_id] if not blk.is_null]
        if not isinstance(spec, SlidingWindowSpec):
            assert len(real_blocks) == hit_length // block_size
            continue
        checked += 1
        assert len(real_blocks) == SlidingWindowManager._contiguous_blocks_for_hit(
            spec.sliding_window, block_size, False
        )

    assert checked == _NUM_SLIDING_WINDOW_GROUPS


# --------------------------------------------------------------------------
# Negative tests: the P8 ring guards (step 1) and the profile switch (step 4)
# --------------------------------------------------------------------------


def _ring_state_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=_C4_STATE_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=_C4_STATE_WINDOW,
        state_window=_RING_WINDOW,
    )


def _ring_manager(block_pool: BlockPool) -> SlidingWindowManager:
    spec = _ring_state_spec()
    return SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=_dsv4_scheduler_block_size(),
        max_admission_blocks_per_request=cdiv(_RING_WINDOW, spec.block_size),
    )


def _ring_block_pool() -> BlockPool:
    return BlockPool(
        num_gpu_blocks=_POOL_BLOCKS,
        enable_caching=True,
        hash_block_size=_C4_STATE_BLOCK_SIZE,
    )


def test_ring_reservation_refuses_prefix_cache_hit_blocks():
    """A ring row is exactly ``ring_blocks`` columns wide and cannot grow.

    ``add_local_computed_blocks`` is not overridden for the ring, so accepting
    hit blocks would push ``req_to_blocks`` past that width and overrun into
    the next request's block-table row.
    """
    block_pool = _ring_block_pool()
    manager = _ring_manager(block_pool)
    assert manager.ring_blocks == cdiv(_RING_WINDOW, _C4_STATE_BLOCK_SIZE)

    with pytest.raises(NotImplementedError, match="prefix caching"):
        manager.get_num_blocks_to_allocate(
            request_id="ring",
            num_tokens=_RING_WINDOW,
            new_computed_blocks=[block_pool.blocks[1]],
            total_computed_tokens=_C4_STATE_BLOCK_SIZE,
            num_local_computed_tokens=_C4_STATE_BLOCK_SIZE,
            num_tokens_main_model=_RING_WINDOW,
        )

    # Without hit blocks the reservation is the constant ring width.
    assert (
        manager.get_num_blocks_to_allocate(
            request_id="ring",
            num_tokens=_RING_WINDOW,
            new_computed_blocks=[],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=_RING_WINDOW,
        )
        == manager.ring_blocks
    )


def test_ring_cache_blocks_refuses_to_register_ring_slots():
    """A ring slot is keyed by ``position % W``, so it has no prefix identity.

    The base implementation would derive ``num_full_blocks`` from the token
    count and register the ring's blocks under the hashes of tokens they no
    longer hold once the ring has wrapped.
    """
    block_pool = _ring_block_pool()
    manager = _ring_manager(block_pool)
    request = _make_request("ring", _RING_WINDOW, block_pool.hash_block_size)

    with pytest.raises(NotImplementedError, match="cannot be prefix-cached"):
        manager.cache_blocks(request, _RING_WINDOW)


# Interpolated from the module's own constants rather than repeated as
# literals: the child runs under -O, where `SlidingWindowMLASpec.__post_init__`
# cannot assert its geometry, so a constant that drifted from the fixture
# would build a spec and let this test pass for the wrong reason.
_RING_GUARD_UNDER_O = """
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

if __debug__:
    raise SystemExit("expected to run under python -O")

spec = SlidingWindowMLASpec(
    block_size={block_size},
    num_kv_heads=1,
    head_size=2048,
    dtype=torch.float32,
    sliding_window={sliding_window},
    state_window={state_window},
)
block_pool = BlockPool(
    num_gpu_blocks={num_gpu_blocks},
    enable_caching=True,
    hash_block_size={block_size},
)
manager = SlidingWindowManager(
    spec,
    block_pool=block_pool,
    enable_caching=True,
    kv_cache_group_id=0,
    scheduler_block_size={scheduler_block_size},
    max_admission_blocks_per_request={ring_blocks},
)
try:
    manager.get_num_blocks_to_allocate(
        request_id="ring",
        num_tokens={state_window},
        new_computed_blocks=[block_pool.blocks[1]],
        total_computed_tokens={block_size},
        num_local_computed_tokens={block_size},
        num_tokens_main_model={state_window},
    )
except NotImplementedError:
    raise SystemExit(0)
raise SystemExit("the ring reservation guard did not fire under python -O")
"""


def test_ring_reservation_guard_survives_python_O():
    """The guard must be a raise, not an assert.

    ``python -O`` strips asserts, and what this one prevents is silent memory
    corruption rather than a crash, so it has to survive optimization.
    """
    ring_blocks = cdiv(_RING_WINDOW, _C4_STATE_BLOCK_SIZE)
    script = _RING_GUARD_UNDER_O.format(
        block_size=_C4_STATE_BLOCK_SIZE,
        sliding_window=_C4_STATE_WINDOW,
        state_window=_RING_WINDOW,
        scheduler_block_size=_dsv4_scheduler_block_size(),
        ring_blocks=ring_blocks,
        num_gpu_blocks=ring_blocks // 2,
    )

    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _compressor_config(*, enable_prefix_caching: bool):
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )


def test_compressor_ring_and_prefix_caching_are_a_product_switch(monkeypatch):
    """The conflict is refused by name, never degraded to the default.

    Silently returning ``None`` would swap a constant per-request state
    reservation for one that grows with max_num_batched_tokens -- gigabytes
    of KV footprint, surfacing much later as an admission refusal or an OOM.
    """
    from vllm.models.deepseek_v4.compressor import get_compressor_state_window

    monkeypatch.setenv("VLLM_DSV4_COMPRESSOR_WINDOWED", "1")
    monkeypatch.setenv("VLLM_DSV4_COMPRESSOR_WINDOW", str(_RING_WINDOW))

    with pytest.raises(ValueError) as exc_info:
        get_compressor_state_window(_compressor_config(enable_prefix_caching=True))
    message = str(exc_info.value)
    assert "PROFILE-P8" in message
    assert "PROFILE-CACHE" in message

    # Either profile on its own resolves a placement.
    profile_p8 = _compressor_config(enable_prefix_caching=False)
    assert get_compressor_state_window(profile_p8) == _RING_WINDOW
    monkeypatch.delenv("VLLM_DSV4_COMPRESSOR_WINDOWED")
    profile_cache = _compressor_config(enable_prefix_caching=True)
    assert get_compressor_state_window(profile_cache) is None


# --------------------------------------------------------------------------
# The lookback-coverage validator and its config hook (P11 step 5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dcp_world_size", [1, _DCP_WORLD_SIZE])
def test_validate_lookback_coverage_accepts_the_live_geometry(
    dcp_world_size, monkeypatch
):
    """I1/I2 hold for every sliding-window family at both DCP sizes.

    The three families are (m=4, block 4, window 8) for the C4 compressors
    (attention and indexer alike), (m=128, block 8, window 128) for the C128
    compressors, and the SWA KV group (no m, block 64, window 128). The
    scheduler block size is taken from the resolver rather than written down,
    so it follows the specs at dcp=1 and dcp=4 alike.
    """
    from vllm.models.deepseek_v4.compressor import (
        validate_compressor_lookback_coverage,
    )

    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    specs = [g.kv_cache_spec for g in groups]
    scheduler_block_size, _ = resolve_kv_cache_block_sizes(
        _kv_cache_config(groups),
        _vllm_config(dcp_world_size, enable_prefix_caching=True),
    )

    validate_compressor_lookback_coverage(specs, scheduler_block_size)

    checked = 0
    for spec in specs:
        if not isinstance(spec, SlidingWindowSpec):
            continue
        checked += 1
        need = cdiv(spec.sliding_window - 1, spec.block_size)
        assert need * spec.block_size >= spec.sliding_window - 1
        assert scheduler_block_size % spec.sliding_window == 0
    assert checked == _NUM_SLIDING_WINDOW_GROUPS


def test_validate_lookback_coverage_rejects_a_boundary_inside_a_window(
    monkeypatch,
):
    """The reachable negative: a scheduler block that splits a window.

    A hit boundary H is a multiple of the scheduler block size, so if a
    window does not divide it, H can land inside the window and the first
    recomputed boundary gathers rows below the cached tail.
    """
    from vllm.models.deepseek_v4.compressor import (
        validate_compressor_lookback_coverage,
    )

    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    c128_state = SlidingWindowMLASpec(
        block_size=_C128_STATE_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=_C128_STATE_WINDOW,
    )
    # One block past the real alignment, which is therefore no longer a
    # multiple of the 128-token window.
    misaligned = _dsv4_scheduler_block_size() + _C128_STATE_BLOCK_SIZE
    assert misaligned % c128_state.sliding_window != 0

    with pytest.raises(ValueError, match="multiple of every"):
        validate_compressor_lookback_coverage([c128_state], misaligned)


def test_validate_lookback_coverage_pins_the_cache_hit_reserve_formula(
    monkeypatch,
):
    """The lookback leg is a formula-pin on the cache-hit path, not a check
    on the spec geometry, and cannot fire against today's formula.

    ``_contiguous_blocks_for_hit`` is ``cdiv(window - 1, block_size)``, and
    ``cdiv(w - 1, bs) * bs >= w - 1`` is an identity for every (w, bs) -- so
    no spec can trip this leg while that formula stands. What it catches is
    the cache-hit path reserving LESS than the lookback, which is simulated
    here by shrinking the reserve by one block. NOTE for the designer: step
    5's named negative case, "a C4 spec with block_size 8", does NOT trip the
    validator for exactly this reason (asserted below), so the reachable
    negative is the alignment one in the test above.
    """
    from vllm.models.deepseek_v4.compressor import (
        validate_compressor_lookback_coverage,
    )

    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    scheduler_block_size = _dsv4_scheduler_block_size()
    # The design's named negative case: an overlapped C4 window placed in
    # 8-token blocks. One block already covers the whole 7-token lookback.
    coarse_c4_state = SlidingWindowMLASpec(
        block_size=_C128_STATE_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=_C4_STATE_WINDOW,
    )
    validate_compressor_lookback_coverage([coarse_c4_state], scheduler_block_size)

    monkeypatch.setattr(
        SlidingWindowManager,
        "_contiguous_blocks_for_hit",
        classmethod(
            lambda cls, window_size, block_size, use_eagle: cdiv(
                window_size - 1, block_size
            )
            - 1
        ),
    )

    with pytest.raises(ValueError, match="short of the"):
        validate_compressor_lookback_coverage(
            [g.kv_cache_spec for g in _dsv4_kv_cache_groups()], scheduler_block_size
        )


def test_compressor_state_compress_ratio_identifies_the_state_groups():
    """``m`` is recovered from (dtype, window), and only for state groups.

    The SWA KV group is a SlidingWindowMLASpec too but holds real uint8 KV,
    so it has no compressor behind it and must not be handed a compress
    ratio. The C4 family is the only overlapped one, hence the only window
    that is not its own ``m``.
    """
    from vllm.models.deepseek_v4.compressor import _compressor_state_compress_ratio

    ratios = {
        (spec.block_size, spec.sliding_window): _compressor_state_compress_ratio(spec)
        for spec in (g.kv_cache_spec for g in _dsv4_kv_cache_groups())
        if isinstance(spec, SlidingWindowMLASpec)
    }

    assert ratios == {
        (_SWA_BLOCK_SIZE, _SWA_WINDOW): None,
        (_C4_STATE_BLOCK_SIZE, _C4_STATE_WINDOW): _C4_STATE_WINDOW // 2,
        (_C128_STATE_BLOCK_SIZE, _C128_STATE_WINDOW): _C128_STATE_WINDOW,
    }
    # ``m`` always divides its window, so the compress-ratio leg of the
    # validator is implied by the window leg and can never fire on its own.
    for (_, window), ratio in ratios.items():
        assert ratio is None or window % ratio == 0


@pytest.mark.parametrize(
    ("ring", "enable_prefix_caching", "profile"),
    [
        (False, True, "PROFILE-CACHE"),
        (False, False, "PROFILE-CONTROL"),
        (True, False, "PROFILE-P8"),
    ],
)
def test_check_compressor_kv_cache_config_logs_the_active_profile(
    ring, enable_prefix_caching, profile, monkeypatch, caplog_vllm
):
    """The startup line rack test T0 reads must name the profile and the
    resolved geometry, for all three shipping configurations."""
    from vllm.models.deepseek_v4.compressor import check_compressor_kv_cache_config

    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    groups = _dsv4_kv_cache_groups()
    if ring:
        c4_state_group_id = next(
            i
            for i, g in enumerate(groups)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            and g.kv_cache_spec.sliding_window == _C4_STATE_WINDOW
        )
        groups[c4_state_group_id] = KVCacheGroupSpec(
            groups[c4_state_group_id].layer_names, _ring_state_spec()
        )
    kv_cache_config = _kv_cache_config(groups)
    vllm_config = _vllm_config(_DCP_WORLD_SIZE, enable_prefix_caching)
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )

    with caplog_vllm.at_level("INFO"):
        check_compressor_kv_cache_config(vllm_config, kv_cache_config)

    assert profile in caplog_vllm.text
    assert f"scheduler_block_size={scheduler_block_size}" in caplog_vllm.text
    assert f"hash_block_size={hash_block_size}" in caplog_vllm.text
    assert f"num_gpu_blocks={kv_cache_config.num_blocks}" in caplog_vllm.text
    # A validator that saw no groups would otherwise be silently vacuous.
    assert f"sliding_window_groups={_NUM_SLIDING_WINDOW_GROUPS}" in caplog_vllm.text


def test_check_compressor_kv_cache_config_validates_only_when_caching_is_on(
    monkeypatch,
):
    """The hook runs the lookback validator, and only for PROFILE-CACHE.

    A window that does not divide the scheduler block size is a live hazard
    for a resumed request and nothing at all without cache hits, so the same
    group set must be refused with caching on and accepted with it off.
    """
    from vllm.models.deepseek_v4.compressor import check_compressor_kv_cache_config

    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    # A window coprime with the alignment every other group forces, so the
    # LCM cannot absorb it: cache-hit boundaries land inside this window.
    straddling_window = _C4_STATE_WINDOW + _C4_STATE_BLOCK_SIZE // 2
    groups = _dsv4_kv_cache_groups(
        extra_specs=(
            SlidingWindowMLASpec(
                block_size=_C4_STATE_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=2048,
                dtype=torch.float32,
                sliding_window=straddling_window,
            ),
        )
    )
    kv_cache_config = _kv_cache_config(groups)
    caching_off = _vllm_config(_DCP_WORLD_SIZE, enable_prefix_caching=False)
    scheduler_block_size, _ = resolve_kv_cache_block_sizes(kv_cache_config, caching_off)
    assert scheduler_block_size % straddling_window != 0

    check_compressor_kv_cache_config(caching_off, kv_cache_config)

    with pytest.raises(ValueError, match="multiple of every"):
        check_compressor_kv_cache_config(
            _vllm_config(_DCP_WORLD_SIZE, enable_prefix_caching=True), kv_cache_config
        )
