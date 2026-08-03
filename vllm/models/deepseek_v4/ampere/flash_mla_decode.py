# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flash-mla PARTIAL sparse-decode adapter for the SM8x DSV4 DCP path (P9).

Routes the compressed-layer DCP decode attention through the ``flash_mla`` CUDA
op ``fwd_sparse_decode_mla_partial`` instead of the Triton ragged decode kernel,
and feeds the result straight into the EXISTING cross-rank merge
(``common/ops/dcp.py::dcp_merge_flashmla_output``).

Contract match (this is the whole point -- read ``dcp.py`` alongside):

- ``dcp_merge_flashmla_output(local_out, local_lse, attn_sink, output, group)``
  consumes ``local_out`` bf16 ``[tokens, gathered_heads, 512]`` (the rank's
  shard-local, NORMALIZED, PRE-sink output) and ``local_lse`` fp32
  ``[tokens, gathered_heads]`` in the NATURAL log domain.
- The P9 fork op returns exactly that pair. The Triton path returns raw
  ``(o, m, l)`` and needs ``softmax_stats_to_lse`` to fuse ``m + log(l)``; the
  CUDA op fuses it in fp32 registers inside the kernel, so this adapter calls
  the merge DIRECTLY -- one fewer elementwise pass and one fewer rounding of
  the stats through global memory.
- The sink is NEVER passed to the op (it has no ``attn_sink`` argument in
  partial form). ``apply_attn_sink`` inside the merge folds it once, based at
  ``logaddexp(global_lse, sink)`` -- ARCHITECTURE.md section 10 rule 1.
- Empty shards: the kernel writes an exactly-zero output row and the finite
  ``-1e30`` sentinel, bit-identical to ``dcp.py::DCP_LSE_SENTINEL``, so rule 9
  holds without any Python-side patching.

Index spaces:

- Both caches are the REAL paged pools (``[num_blocks, block_size, 584]``
  uint8). The op addresses them with the same arithmetic as the Triton decode
  kernel (``rocm_aiter_mla_sparse.py``: ``data = block + pos*576``,
  ``scale = block + block_size*576 + pos*8``, ``block_stride = stride(0)``), so
  the flat slot ids the existing metadata already carries are passed through
  unchanged. Unlike the PREFILL adapter there is no whole-cache dequant buffer
  here, so passing a paged pool is correct and cheap -- the op's only scratch
  is the ``[T*(swa_width+topk), 512]`` bf16 selection buffer, see the budget
  guard below.
- SWA stream: ``decode_swa_indices`` is already dense ``[T, window]`` flat slot
  ids with a tight ``swa_lens``; passed as-is (the same tensor the Triton path
  ragged-packs).
- Compressed stream: the shared producer emits RAGGED physical slots
  (``topk_ragged_indices`` at ``topk_ragged_indptr[t]``). The op wants a dense
  ``[T, topk]`` row-major block, so this gathers
  ``dense[t, j] = ragged[indptr[t] + j]``. That is exact because the producer's
  rows are prefix-compact: the j-th valid entry of row ``t`` IS column ``j``.

CUDA-graph capture (FULL_DECODE_ONLY captures this path -- P2f):

- ``out`` and ``lse`` are PERSISTENT buffers owned here and handed to the op as
  ``out=`` / ``lse_out=``; the op writes into them instead of allocating, so a
  captured launch and its captured consumer agree on the address forever.
  They must be allocated OUTSIDE capture (a buffer allocated inside graph A's
  private pool and reused by graph B is a dangling read) -- ``ensure_buffers``
  asserts this and the P9 warmup catalog entry forces the allocation at boot.
- The dense compressed-index block is likewise persistent scratch written with
  ``index_select(out=)``.
- The op's internal scratch (``oaccum`` / ``mlse`` / ``counter`` / ``sel_kv``)
  is still allocated per call, from the graph's private pool during capture.
  Their sizes are pure functions of host constants for a captured shape
  (``T``, ``H``, the two index widths, ``sm_count``), so every replay re-uses
  the same pool blocks -- the same property ``dcp_alltoall``'s send/recv
  buffers rely on by design.
- No ``.item()``/``.cpu()``/sync and no data-dependent branch is introduced.
"""

from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_FLASH_MLA_BUILD_HINT = (
    "flash_mla with the SM86 PARTIAL sparse decode op is required for "
    "VLLM_DSV4_FLASH_DECODE. Build the PATCHED fork from source into this "
    "venv with the arch pinned to the 3090s, e.g.:\n"
    "  cd ~/dsv4-dcp/flash-mla-int && git checkout dcp-sm86-patches && \\\n"
    "  FLASH_MLA_CUDA_ARCHS=86 python -m pip install -v "
    "--no-build-isolation .\n"
    "(torch-stable ABI: the wheel works on torch >= 2.9.)"
)

# bf16 bytes per selection-scratch row inside the op (head_dim 512 * 2 B).
_SEL_ROW_BYTES = 512 * 2
# Default ceiling for that scratch: T * (swa_width + topk) * 1 KiB. Decode
# batches are small, but the SWA stream is passed at its FULL window width
# (the tight length is per token and only known on device), so the product is
# worth bounding loudly rather than OOMing inside the op.
_DEFAULT_SCRATCH_MB = 1024

_flash_mla_decode_partial: Any = None


def _get_flash_mla_decode_partial():
    """Lazy ``import flash_mla``; a missing/old build fails with the fix."""
    global _flash_mla_decode_partial
    if _flash_mla_decode_partial is None:
        try:
            from flash_mla import sparse_mla_decode_fp8_partial
        except ImportError as e:
            raise ImportError(f"{_FLASH_MLA_BUILD_HINT}\n(import error: {e})") from e
        if not hasattr(torch.ops.flash_mla, "fwd_sparse_decode_mla_partial"):
            raise ImportError(
                "the installed flash_mla lacks fwd_sparse_decode_mla_partial "
                f"(built from an unpatched fork?).\n{_FLASH_MLA_BUILD_HINT}"
            )
        _flash_mla_decode_partial = sparse_mla_decode_fp8_partial
    return _flash_mla_decode_partial


class FlashMlaDecodeBuffers:
    """Persistent, address-stable destinations for the captured decode op.

    Sized once at the graph-capture bounds: ``max_tokens`` is
    ``scheduler_config.max_num_batched_tokens`` (the row bound every SM86 DCP
    consumer already asserts against), ``num_heads`` is the DCP-GATHERED head
    count (``n_local_heads * dcp_world_size``, an init-time constant), and
    ``topk`` is the compressed row width (``index_topk`` for C4A, the builder's
    pinned per-rank bound for C128A -- both init-time constants, P2f section 1).
    """

    def __init__(
        self,
        max_tokens: int,
        num_heads: int,
        head_dim: int,
        topk: int,
        device: torch.device,
    ) -> None:
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.topk = topk
        self.out = torch.empty(
            max_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        self.lse = torch.empty(
            max_tokens, num_heads, dtype=torch.float32, device=device
        )
        # dense compressed slots + the int64 gather index that fills them
        self.extra_indices = torch.empty(
            max_tokens, topk, dtype=torch.int32, device=device
        )
        self.gather_index = torch.empty(
            max_tokens * topk, dtype=torch.int64, device=device
        )
        self.cols = torch.arange(topk, dtype=torch.int64, device=device)

    def fits(self, num_tokens: int, num_heads: int, topk: int) -> bool:
        # topk must match EXACTLY, not just fit: `extra_indices[:T, :topk]` is
        # only contiguous (hence `.view(-1)`-able as an `index_select` out=)
        # when the column slice is the full width. topk is an init-time
        # constant, so an inexact match is a wiring bug, not a shape we should
        # silently accommodate.
        return (
            num_tokens <= self.max_tokens
            and num_heads == self.num_heads
            and topk == self.topk
        )


def ensure_buffers(
    existing: FlashMlaDecodeBuffers | None,
    *,
    max_tokens: int,
    num_heads: int,
    head_dim: int,
    topk: int,
    device: torch.device,
) -> FlashMlaDecodeBuffers:
    """Return usable buffers, allocating only OUTSIDE a graph capture."""
    if existing is not None and existing.fits(max_tokens, num_heads, topk):
        return existing
    assert not torch.cuda.is_current_stream_capturing(), (
        "VLLM_DSV4_FLASH_DECODE buffers must be allocated before CUDA-graph "
        "capture: a buffer created inside one graph's private pool is freed "
        "when that graph dies, and a second graph would replay against a "
        "dangling address. Keep VLLM_DSV4_WARMUP on (its P9 catalog entry "
        "runs the decode path at boot), or raise max_num_batched_tokens "
        "before capture."
    )
    return FlashMlaDecodeBuffers(max_tokens, num_heads, head_dim, topk, device)


def _scratch_budget_bytes() -> int:
    mb = getattr(envs, "VLLM_DSV4_FLASH_DECODE_SCRATCH_MB", _DEFAULT_SCRATCH_MB)
    return int(mb) * 1024 * 1024


def ragged_to_dense_slots(
    ragged_indices: torch.Tensor,  # [>= T*topk] int32, flat
    ragged_indptr: torch.Tensor,  # [T+1] int32
    lens: torch.Tensor,  # [T] int32 (tight per-token count)
    num_tokens: int,
    topk: int,
    buffers: FlashMlaDecodeBuffers,
) -> torch.Tensor:
    """``dense[t, j] = ragged[indptr[t] + j]`` for ``j < lens[t]``, else 0.

    Exact because the producer rows are prefix-compact (the j-th valid entry of
    row ``t`` is column ``j``); slots at or past ``lens[t]`` are never
    dereferenced by the kernel, and are zeroed anyway so a stale tail can never
    be mistaken for a real slot. Fixed shapes, no host sync -- capture-safe.
    """
    cols = buffers.cols[:topk]
    base = ragged_indptr[:num_tokens].to(torch.int64).unsqueeze(1)
    valid = cols.unsqueeze(0) < lens[:num_tokens].to(torch.int64).unsqueeze(1)
    gather = buffers.gather_index[: num_tokens * topk].view(num_tokens, topk)
    torch.where(valid, base + cols.unsqueeze(0), torch.zeros_like(base), out=gather)
    dense = buffers.extra_indices[:num_tokens, :topk]
    torch.index_select(ragged_indices, 0, gather.view(-1), out=dense.view(-1))
    dense.mul_(valid)
    return dense


def sparse_decode_partial_via_flash_mla(
    q: torch.Tensor,  # [T, H_gathered, 512] bf16 (post DCP all-gather)
    *,
    swa_k_cache: torch.Tensor,  # [nb, bs, 584] uint8 paged pool
    swa_indices: torch.Tensor,  # [T, window] int32 flat slot ids
    swa_lens: torch.Tensor,  # [T] int32, owner-masked, TIGHT
    compressed_k_cache: torch.Tensor,  # [nb, bs, 584] uint8 paged pool
    topk_ragged_indices: torch.Tensor,  # [>= T*topk] int32
    topk_ragged_indptr: torch.Tensor,  # [T+1] int32
    topk_lens: torch.Tensor,  # [T] int32, TIGHT
    topk_width: int,
    scale: float,
    buffers: FlashMlaDecodeBuffers,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One partial decode call; returns ``(out, lse)`` ready for the merge."""
    decode_partial = _get_flash_mla_decode_partial()

    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == 512 and q.dtype == torch.bfloat16, (
        f"flash-mla sparse decode wants q [T, H, 512] bf16, got "
        f"{tuple(q.shape)} {q.dtype}"
    )
    assert swa_k_cache.dtype == torch.uint8 and swa_k_cache.dim() == 3, (
        "swa cache must be the uint8 fp8_ds_mla paged pool "
        f"[blocks, block, 584], got {swa_k_cache.dtype} {tuple(swa_k_cache.shape)}"
    )
    assert compressed_k_cache.dtype == torch.uint8 and compressed_k_cache.dim() == 3
    # the kernel reads q/indices/lens rows with an IMPLIED stride (q row by
    # q.stride(1) then contiguous in d; index rows by size(1); lens by 1), so a
    # non-contiguous view would silently read the wrong bytes.
    assert q.stride(2) == 1, "q head rows must be contiguous in head_dim"
    assert swa_indices.dim() == 2 and swa_indices.is_contiguous(), (
        "swa_indices must be a contiguous [T, window] int32 block"
    )
    assert swa_indices.dtype == torch.int32 and swa_lens.dtype == torch.int32
    assert swa_lens.stride(0) == 1 and topk_lens.stride(0) == 1
    assert buffers.fits(num_tokens, num_heads, topk_width), (
        "flash-decode buffers are too small: "
        f"T={num_tokens} H={num_heads} topk={topk_width} vs "
        f"{buffers.max_tokens}/{buffers.num_heads}/{buffers.topk}"
    )

    swa_width = swa_indices.shape[-1]
    scratch = num_tokens * (swa_width + topk_width) * _SEL_ROW_BYTES
    budget = _scratch_budget_bytes()
    assert scratch <= budget, (
        "the flash-mla decode op's selection scratch would be "
        f"{scratch / 2**20:.0f} MiB (T={num_tokens} x (swa {swa_width} + topk "
        f"{topk_width}) x 1 KiB), over the "
        f"{budget / 2**20:.0f} MiB budget. Raise "
        "VLLM_DSV4_FLASH_DECODE_SCRATCH_MB or lower the decode batch."
    )

    extra_indices = ragged_to_dense_slots(
        topk_ragged_indices,
        topk_ragged_indptr,
        topk_lens,
        num_tokens,
        topk_width,
        buffers,
    )

    out = buffers.out[:num_tokens, :num_heads]
    lse = buffers.lse[:num_tokens, :num_heads]
    decode_partial(
        q=q,
        swa_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        scale=scale,
        extra_cache=compressed_k_cache,
        extra_indices=extra_indices,
        extra_lens=topk_lens[:num_tokens],
        out=out,
        lse_out=lse,
    )
    return out, lse
