# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import KVBlockZeroer, _zero_kv_blocks_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        # Dense per-layer allocation: the block step equals the payload.
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    def largest_power_of_2_divisor(n):
        return n & -n

    blk_size = min(min(largest_power_of_2_divisor(ps) for ps in seg_page_sizes), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        max_ps // blk_size,
        blk_size,
        2,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_every_n_blocks_specialization():
    """After warmup, no launch should trigger a first-request JIT compile.

    ``n_blocks`` is ``do_not_specialize``, so a single warmup launch must
    cover every block count.
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,  # max_chunks
        page_size_el,  # blk_size
        1,  # n_segs
    )

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


# ---------------------------------------------------------------------------
# Packed (DeepseekV4) block slab: the block STEP is not the segment PAYLOAD.
# ---------------------------------------------------------------------------


class _BlocksFirstBackend:
    """Minimal backend stub: blocks outermost, so block_dim == 0."""

    @staticmethod
    def get_kv_cache_block_dim(
        block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ) -> int:
        return 0


class _BoundLayer:
    """Stands in for the forward-context entry the zeroer reads."""

    def __init__(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache


def _packed_group(layer_name: str, block_size: int, head_size: int):
    return SimpleNamespace(
        backend=_BlocksFirstBackend,
        layer_names=[layer_name],
        kv_cache_spec=FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
        ),
        kv_cache_group_id=0,
    )


def _packed_views(num_blocks: int, page_bytes: int, n_layers: int):
    """Rebuild `_reshape_attention_kv_cache`'s packing branch on CPU.

    One shared int8 slab; layer i is
    ``slab.view(-1, block_stride)[:, i*page : (i+1)*page]``, so every layer's
    ``stride(0)`` is the whole block stride and only ``page_bytes`` of each
    block belongs to it.
    """
    block_stride = page_bytes * n_layers
    slab = torch.zeros(num_blocks * block_stride, dtype=torch.int8)
    views = [
        slab.view(-1, block_stride)[:, i * page_bytes : (i + 1) * page_bytes]
        for i in range(n_layers)
    ]
    return slab, views, block_stride


def test_packed_layout_zeroes_the_page_not_the_block_stride():
    """Segment geometry, the assertion that catches an out-of-bounds write.

    Every segment must step by the slab block stride but zero only its own
    page.  Deriving the payload from ``stride(block_dim)`` makes each segment
    at packed offset ``p`` write ``p`` bytes into block_id + 1 -- a block that
    is live for another request -- and past the end of the backing allocation
    on the last block.  Pure geometry, so it runs without CUDA.
    """
    n_layers, page_bytes, num_blocks = 4, 64, 8
    slab, views, block_stride = _packed_views(num_blocks, page_bytes, n_layers)

    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=[
            _packed_group(f"layer.{i}", block_size=1, head_size=page_bytes // 2)
            for i in range(n_layers)
        ],
        kernel_block_sizes=[1],
        cache_dtype="auto",
        static_forward_context={
            f"layer.{i}": _BoundLayer(views[i]) for i in range(n_layers)
        },
    )
    assert zeroer._meta is not None
    seg_addrs, seg_page_sizes, seg_block_strides, _, _, n_segs = zeroer._meta

    assert n_segs == n_layers
    assert [int(x) for x in seg_page_sizes] == [page_bytes // 4] * n_layers
    assert [int(x) for x in seg_block_strides] == [block_stride // 4] * n_layers

    # No segment may reach past the backing allocation on the last block.
    base = slab.data_ptr()
    end = base + slab.numel()
    for addr, page_el, stride_el in zip(seg_addrs, seg_page_sizes, seg_block_strides):
        last = int(addr) + (num_blocks - 1) * int(stride_el) * 4 + int(page_el) * 4
        assert last <= end


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_layout_does_not_clobber_the_neighbouring_block():
    """End to end on the device: zeroing block 1 must leave 0 and 2 intact."""
    device = torch.device("cuda")
    n_layers, page_el, num_blocks = 4, 16, 8  # page_el in int32 elements
    block_stride_el = page_el * n_layers
    slab = torch.ones(num_blocks * block_stride_el, dtype=torch.int32, device=device)

    seg_addrs = [slab.data_ptr() + i * page_el * 4 for i in range(n_layers)]
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(seg_addrs, dtype=torch.uint64, device=device),
        torch.tensor([page_el] * n_layers, dtype=torch.int64, device=device),
        torch.tensor([block_stride_el] * n_layers, dtype=torch.int64, device=device),
        1,  # max_chunks
        page_el,  # blk_size
        n_layers,  # n_segs
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    view = slab.view(num_blocks, block_stride_el)
    assert torch.all(view[0] == 1)
    assert torch.all(view[1] == 0)
    assert torch.all(view[2] == 1)
    assert torch.all(view[num_blocks - 1] == 1)


class _KVFirstBackend:
    """K/V outermost, so block_dim == 1."""

    @staticmethod
    def get_kv_cache_block_dim(
        block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ) -> int:
        return 1


def _kv_first_group(layer_name: str, head_size: int):
    return SimpleNamespace(
        backend=_KVFirstBackend,
        layer_names=[layer_name],
        kv_cache_spec=FullAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.int32,
        ),
        kv_cache_group_id=0,
    )


def _build_zeroer(groups, kv_by_layer):
    return KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=groups,
        kernel_block_sizes=[1],
        cache_dtype="auto",
        static_forward_context={
            name: _BoundLayer(kv) for name, kv in kv_by_layer.items()
        },
    )


def test_kv_first_contiguous_layout_splits_k_and_v_into_two_segments():
    """(2, num_blocks, ...) contiguous: K and V are far apart, one segment each."""
    num_blocks, heads, dim = 8, 2, 4
    h = heads * dim
    kv = torch.zeros(2 * num_blocks * h, dtype=torch.int32).as_strided(
        (2, num_blocks, heads, dim), (num_blocks * h, h, dim, 1)
    )

    zeroer = _build_zeroer([_kv_first_group("layer.0", dim)], {"layer.0": kv})
    _, seg_page_sizes, seg_block_strides, _, _, n_segs = zeroer._meta

    assert n_segs == 2
    assert [int(x) for x in seg_page_sizes] == [h, h]
    assert [int(x) for x in seg_block_strides] == [h, h]


def test_kv_first_restrided_layout_keeps_k_and_v_inside_one_block():
    """The layout `_update_hybrid_attention_mamba_layout` produces.

    Strides become (hidden, 2*hidden, ...): the K/V dim now lives INSIDE a
    block's extent rather than outside it, so there is ONE segment whose
    payload is 2*hidden -- the whole block -- not the `hidden` a
    trailing-shape product would give.
    """
    num_blocks, heads, dim = 8, 2, 4
    h = heads * dim
    kv = torch.zeros(2 * num_blocks * h, dtype=torch.int32).as_strided(
        (2, num_blocks, heads, dim), (h, 2 * h, dim, 1)
    )

    zeroer = _build_zeroer([_kv_first_group("layer.0", dim)], {"layer.0": kv})
    _, seg_page_sizes, seg_block_strides, _, _, n_segs = zeroer._meta

    assert n_segs == 1
    assert [int(x) for x in seg_page_sizes] == [2 * h]
    assert [int(x) for x in seg_block_strides] == [2 * h]


def test_ctor_keeps_aliased_offset_zero_segments_per_group():
    """Packed-slab layouts restart every group's byte offset at 0 inside one
    shared slab, so each group's first layer has the SAME data_ptr. The flat
    table dedups that address globally; each group's OWN table must still
    carry it, or zero_block_groups() silently skips the group and the
    block's previous tenant's bytes survive into the next request."""
    device = torch.device("cpu")
    num_blocks, page = 4, 8
    slab = torch.zeros((num_blocks, page), dtype=torch.int32, device=device)

    def make_group(gid: int) -> SimpleNamespace:
        spec = FullAttentionSpec(
            block_size=1, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        backend = SimpleNamespace(get_kv_cache_block_dim=lambda *a, **k: 0)
        return SimpleNamespace(
            kv_cache_spec=spec,
            backend=backend,
            kv_cache_group_id=gid,
            layer_names=[f"l{gid}"],
        )

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[make_group(0), make_group(1)],
        kernel_block_sizes=[1, 1],
        cache_dtype="auto",
        runner_only_attn_layers=None,
        static_forward_context={
            "l0": SimpleNamespace(kv_cache=slab),
            "l1": SimpleNamespace(kv_cache=slab),
        },
    )
    # BOTH groups must own a segment table containing the shared address.
    assert set(zeroer._metas) == {0, 1}
    for gid in (0, 1):
        assert int(zeroer._metas[gid][0][0]) == slab.data_ptr()
        assert zeroer._metas[gid][5] == 1
    # The flat table keeps the global dedup: one segment, not two.
    assert zeroer._meta is not None
    assert zeroer._meta[5] == 1


def test_cpu_runner_zero_block_ids_accepts_per_group_lists():
    """The CPU override must consume the per-group nested list. Feeding the
    nested list through tensor indexing would return a COPY (advanced
    indexing) and .zero_() would silently never touch the cache."""
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    kv = torch.ones(16, 2, 8, dtype=torch.float32)
    spec = FullAttentionSpec(
        block_size=1, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    stub = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=spec, layer_names=["l0"])
            ]
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={"l0": SimpleNamespace(kv_cache=kv)}
        ),
    )
    CPUModelRunner._zero_block_ids(stub, [[9, 10], []])
    assert kv[9].sum() == 0
    assert kv[10].sum() == 0
    assert kv[8].sum() != 0
