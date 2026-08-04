# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which KV cache groups record their new blocks for worker-side zeroing.

DeepseekV4 mixes precisions in one shared block pool: the compressed-KV /
indexer groups are uint8 (``fp8_ds_mla``) while the compressor-state groups are
float32, so ``KVCacheConfig.needs_kv_cache_zeroing`` is True. All groups draw
from the single ``BlockPool`` the coordinator owns, and the packed layout
overlays the groups inside one block slab, so a block recycled out of one group
hands its previous tenant's bytes to the next one. Every attention-family group
must therefore hand its freshly allocated block ids to the zeroer -- not just
the full-attention ones.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

pytestmark = pytest.mark.cpu_test

_MAX_MODEL_LEN = 262144
_MAX_IN_FLIGHT_TOKENS = 512
_POOL_BLOCKS = 4096

_MLA_BLOCK_SIZE = 256
_SWA_BLOCK_SIZE = 64
_SWA_WINDOW = 128
_C4_STATE_BLOCK_SIZE = 4
_C4_STATE_WINDOW = 8
_C128_STATE_BLOCK_SIZE = 8
_C128_STATE_WINDOW = 128


def _mla_kv_spec(**kwargs) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=_MLA_BLOCK_SIZE,
        num_kv_heads=1,
        dtype=torch.uint8,
        **kwargs,
    )


def _dsv4_kv_cache_groups() -> list[KVCacheGroupSpec]:
    """The scheduler-side KV cache groups DeepseekV4 produces on this branch.

    ``group_and_unify_kv_cache_specs`` splits the sliding-window layers by
    ``(block_size, sliding_window)``, so the SWA KV window and the two
    compressor-state families are three separate groups even though all three
    are SlidingWindowMLASpec.
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
        # Sparse-indexer k_cache.
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
        # fp32 compressor state, C4 family.
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
    return [KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)]


def _vllm_config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=_MLA_BLOCK_SIZE,
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=_MAX_IN_FLIGHT_TOKENS),
        kv_transfer_config=None,
    )


def _build_coordinator(
    kv_cache_config: KVCacheConfig,
) -> HybridKVCacheCoordinator:
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, _vllm_config()
    )
    return HybridKVCacheCoordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=_MAX_MODEL_LEN,
        max_in_flight_tokens=_MAX_IN_FLIGHT_TOKENS,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )


def test_dsv4_group_set_needs_zeroing():
    """Mixed uint8 KV and fp32 compressor state means zeroing is required."""
    config = KVCacheConfig(
        num_blocks=_POOL_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=_dsv4_kv_cache_groups(),
    )
    assert config.has_mixed_precision_kv_cache
    assert config.needs_kv_cache_zeroing


def test_every_attention_group_records_new_block_ids_for_zeroing():
    """No attention-family group may opt out of zeroing.

    The block pool is shared across groups, so a block recycled into a
    sliding-window group (the SWA KV window, or either fp32 compressor-state
    family) still holds the bytes its previous tenant wrote under a different
    dtype.
    """
    groups = _dsv4_kv_cache_groups()
    config = KVCacheConfig(
        num_blocks=_POOL_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
    )
    assert config.needs_kv_cache_zeroing

    coordinator = _build_coordinator(config)
    managers = coordinator.single_type_managers
    assert len(managers) == len(groups)

    not_recording = [
        type(group.kv_cache_spec).__name__
        for group, manager in zip(groups, managers)
        if isinstance(group.kv_cache_spec, AttentionSpec)
        and not manager.records_new_block_ids
    ]
    assert not not_recording, (
        f"attention groups excluded from KV cache zeroing: {not_recording}"
    )


def test_sliding_window_group_surfaces_its_new_block_ids():
    """The recorded ids actually reach the drain the scheduler reads."""
    groups = _dsv4_kv_cache_groups()
    config = KVCacheConfig(
        num_blocks=_POOL_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
    )
    coordinator = _build_coordinator(config)

    sliding_window_managers = [
        manager
        for group, manager in zip(groups, coordinator.single_type_managers)
        if isinstance(group.kv_cache_spec, SlidingWindowMLASpec)
    ]
    assert sliding_window_managers

    for i, manager in enumerate(sliding_window_managers):
        blocks = manager.allocate_new_blocks(
            f"req-{i}", manager.block_size, manager.block_size
        )
        assert blocks
        assert manager.take_new_block_ids() == [block.block_id for block in blocks]
        # The drain is one-shot.
        assert manager.take_new_block_ids() == []
