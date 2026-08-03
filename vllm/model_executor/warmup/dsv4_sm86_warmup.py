# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P6 JIT warmup catalog for the SM8x DSV4 sparse-MLA path.

One place (not scattered calls) that runs the RESOLVED prefill/decode kernel
family once at engine init, after weights load and cache binding, so no
Triton JIT compile or first-call op load lands inside a measured request.

Catalog (each entry states what it compiles and why it is not covered
elsewhere):

1. Mixed prefill+decode dummy runs at the MIN and MAX chunk sizes
   (``_dummy_run(force_attention=True, create_mixed_batch=True)``). This
   drives the true production code paths against the real bound caches --
   the only way to compile the exact Triton specializations (constexpr
   block sizes AND pointer-alignment/divisibility variants derive from the
   real cache strides and metadata shapes):
     - the sparse prefill/decode attention kernels
       (rocm_aiter_mla_sparse.py; fixed configs, no autotuner),
     - the dequant/gather + combine_topk_swa_indices kernels,
     - under VLLM_SM86_DCP + dcp>1: the DCP pack kernel, the
       ``block_size=1`` dequant specialization over the gathered staging,
       the indexer DCP merge, and the decode pre-sink partial + a2a merge
       (all ranks run this warmup in lockstep, so the collectives are
       symmetric -- same guarantee as real serving),
     - under VLLM_DSV4_FLASH_PREFILL additionally: the SWA window byte-pack
       kernel and the flash-mla sparse prefill op on the production path.
   The MAX-size run is the consumer fork's "long-prefill JIT warm": chunked
   prefill walks shapes upward, and a min-only warmup leaves the large-
   metadata specializations to compile inside the first long request.
2. A direct tiny two-cache ``flash_mla.sparse_mla_prefill`` call (flag on
   only): loads the .so / cuModule and compiles nothing (the op is AOT), but
   surfaces a missing or arch-mismatched build AT BOOT with the build
   command in the error, and covers the ``extra_cache`` stream even when
   the dummy batch is too short to complete a compressed entry.
3. P9: a geometric CHUNK-COUNT ladder over
   ``dequantize_and_gather_k_cache_triton``. That kernel is the leaf under
   every py-spy caller frame of the measured per-bucket JIT tax
   (``ampere_sparse.py`` {148,177,344,623}, ``cache_utils.py`` 377,
   ``amd/rocm.py`` 748): a long prefill walks ``max_entries`` upward chunk by
   chunk, and ``max_entries`` was the kernel's ``max_blocks_per_seq``
   constexpr, so each new context-length bucket compiled a fresh
   specialization inside the first request that reached it. P9 pins that
   argument (``do_not_specialize``, ``cache_utils.py``) because it is only an
   address stride; the ladder stays as the CANARY -- with the pin it costs one
   compile and a handful of microsecond launches, and if the pin is ever
   reverted it moves the whole bucket family back to boot instead of into
   requests. It runs both ``cache_block_size`` shapes that survive as genuine
   constexprs (the paged pool's block size and the DCP/delta stagings'
   ``block_size=1``).
4. P9: a direct tiny ``flash_mla.sparse_mla_decode_fp8_partial`` call
   (``VLLM_DSV4_FLASH_DECODE`` only): same "fail at boot with the build
   command" role as entry 2, and -- more importantly -- it is what forces the
   per-layer persistent decode buffers to be allocated BEFORE CUDA-graph
   capture. See ``flash_mla_decode.py::ensure_buffers``: a buffer allocated
   inside one graph's private pool is dangling for the next graph.

NOT in the catalog (already warmed elsewhere, documented here so the
catalog stays the single map):
  - fp8 MQA logits autotune caches: primed in ``SparseAttnIndexer.__init__``
    (must happen before memory-profiling capture; see
    sparse_attn_indexer.py). The only autotuned kernels on this path live
    there -- everything this module warms is fixed-config, which is what
    makes VLLM_DSV4_WARMUP safe to default ON (warmup cannot change any
    subsequently computed value, only compile caches).
  - metadata-builder kernels: ``sparse_mla_triton_warmup`` under
    ``kernel_config.enable_jit_warmup``.
  - mHC TileLang kernels: ``deepseek_v4_mhc_warmup``.
"""

from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DSV4_SM86_BACKEND = "TRITON_MLA_SPARSE_DSV4"
_MIN_WARMUP_TOKENS = 16


def _has_sm86_dsv4_backend(runner) -> bool:
    from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
        _has_attention_backend,
    )

    return _has_attention_backend(runner, frozenset({_DSV4_SM86_BACKEND}))


def _warmup_flash_mla_prefill_op(device: torch.device) -> None:
    """Direct tiny two-cache op call -- see catalog entry 2.

    Both cache streams are 1-row compact stagings (the in-op whole-cache
    dequant buffer is 2 rows of bf16 -- 2 KiB), lens are tight (1), and the
    output is discarded: no state outside this function is touched.
    """
    from vllm.models.deepseek_v4.ampere.flash_mla_prefill import (
        _get_flash_mla_sparse_prefill,
    )

    sparse_mla_prefill = _get_flash_mla_sparse_prefill()
    q = torch.zeros(1, 1, 512, dtype=torch.bfloat16, device=device)
    swa_cache = torch.zeros(1, 1, 584, dtype=torch.uint8, device=device)
    extra_cache = torch.zeros(1, 1, 584, dtype=torch.uint8, device=device)
    ones = torch.ones(1, dtype=torch.int32, device=device)
    zeros_idx = torch.zeros(1, 1, dtype=torch.int32, device=device)
    sparse_mla_prefill(
        q=q,
        swa_cache=swa_cache,
        swa_indices=zeros_idx,
        swa_lens=ones,
        scale=512.0**-0.5,
        attn_sink=None,
        extra_cache=extra_cache,
        extra_indices=zeros_idx,
        extra_lens=ones,
    )
    torch.cuda.synchronize(device)


def _warmup_flash_mla_decode_partial_op(device: torch.device) -> None:
    """Direct tiny partial-decode op call -- see catalog entry 4."""
    from vllm.models.deepseek_v4.ampere.flash_mla_decode import (
        _get_flash_mla_decode_partial,
    )

    decode_partial = _get_flash_mla_decode_partial()
    q = torch.zeros(1, 1, 512, dtype=torch.bfloat16, device=device)
    swa_cache = torch.zeros(1, 1, 584, dtype=torch.uint8, device=device)
    extra_cache = torch.zeros(1, 1, 584, dtype=torch.uint8, device=device)
    ones = torch.ones(1, dtype=torch.int32, device=device)
    zeros_idx = torch.zeros(1, 1, dtype=torch.int32, device=device)
    decode_partial(
        q=q,
        swa_cache=swa_cache,
        swa_indices=zeros_idx,
        swa_lens=ones,
        scale=512.0**-0.5,
        extra_cache=extra_cache,
        extra_indices=zeros_idx,
        extra_lens=ones,
    )
    torch.cuda.synchronize(device)


# Geometric ladder over the number of PREFILL CHUNKS a request has consumed.
# Bucket b covers a context of b * max_num_batched_tokens tokens, i.e.
# max_entries ~= b * chunk_tokens // compress_ratio compressed entries.
_CHUNK_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _warmup_dequant_gather_ladder(
    device: torch.device,
    max_model_len: int,
    chunk_tokens: int,
) -> None:
    """Catalog entry 3: compile the gather kernel at every chunk bucket.

    Each rung mimics ONE request's compressed-entry gather at that context
    length: a `[1, entries]` block table (the identity/virtual-table shape both
    DCP gather paths use) over a tiny synthetic cache. `gather_lens` is pinned
    to a handful of rows, so the LAUNCH is microseconds regardless of the rung
    -- only the compiled specialization scales with the rung, which is the
    whole point. Runs on freshly allocated synthetic buffers; no model or cache
    state is touched.
    """
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
        sm86_dcp_identity_block_table,
    )

    entry_bytes = 584
    gather_rows = 4  # rows actually dequantized per rung (launch cost only)
    # compress_ratio 4 is the tighter (larger max_entries) of the two
    # compressed-layer kinds; C128A's tables are 32x smaller and hit the same
    # specialization once max_blocks_per_seq stops specializing.
    compress_ratio = 4
    max_entries_cap = max(1, max_model_len // compress_ratio)

    seen: set[int] = set()
    for chunks in _CHUNK_LADDER:
        entries = min(chunks * chunk_tokens // compress_ratio, max_entries_cap)
        entries = max(entries, gather_rows)
        if entries in seen:
            continue
        seen.add(entries)
        # block_size=1 staging layout (DCP all-gather + P7 delta stagings).
        # Only rows [0, gather_rows) are ever dereferenced -- the rung's SIZE
        # lives in the block table's WIDTH, which is what specializes -- so the
        # cache stays a few KiB no matter how long the ladder gets.
        cache = torch.zeros(
            gather_rows, 1, entry_bytes, dtype=torch.uint8, device=device
        )
        out = torch.zeros(1, gather_rows, 512, dtype=torch.bfloat16, device=device)
        lens = torch.full((1,), gather_rows, dtype=torch.int32, device=device)
        dequantize_and_gather_k_cache_triton(
            out,
            cache,
            seq_lens=lens,
            gather_lens=None,
            block_table=sm86_dcp_identity_block_table(entries, device),
            block_size=1,
            offset=0,
            use_fnuz=False,
        )
        if entries >= max_entries_cap:
            break
    torch.cuda.synchronize(device)
    logger.info(
        "DSV4 SM8x JIT warmup: compressed-gather chunk ladder covered "
        "max_blocks_per_seq %s.",
        sorted(seen),
    )


def dsv4_sm86_warmup(worker: "Worker") -> None:
    """Run the P6 warmup catalog (VLLM_DSV4_WARMUP, default on)."""
    if not envs.VLLM_DSV4_WARMUP:
        return
    if not current_platform.is_cuda():
        return
    runner = worker.model_runner
    if runner is None or runner.is_pooling_model:
        return
    if not _has_sm86_dsv4_backend(runner):
        return

    # Catalog entries 2 and 4 first: a broken flash_mla install must fail
    # here, with the build command, not inside the dummy run's stack.
    if envs.VLLM_DSV4_FLASH_PREFILL:
        _warmup_flash_mla_prefill_op(runner.device)
    if envs.VLLM_DSV4_FLASH_DECODE:
        _warmup_flash_mla_decode_partial_op(runner.device)

    # Catalog entry 1: min and max chunk sizes through the real path. The
    # v2-vs-v1 runner split mirrors deepseek_v4_sparse_mla_attention_warmup.
    from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
        _uses_v2_model_runner,
    )

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    if max_tokens <= 0:
        return

    # Catalog entry 3: the chunk-count ladder. Cheap (one compile with the
    # P9 pin in place) and it must run BEFORE the dummy runs so a regression
    # in the pin shows up here, attributed, rather than smeared across them.
    try:
        _warmup_dequant_gather_ladder(
            runner.device,
            worker.model_config.max_model_len,
            max_tokens,
        )
    except Exception:  # pragma: no cover - warmup must never break boot
        logger.warning(
            "DSV4 SM8x JIT warmup: compressed-gather chunk ladder failed; "
            "continuing (the buckets will compile inside the first requests "
            "that reach them).",
            exc_info=True,
        )
    token_sizes = sorted({min(_MIN_WARMUP_TOKENS, max_tokens), max_tokens})
    logger.info(
        "DSV4 SM8x JIT warmup: mixed dummy runs at token sizes %s.", token_sizes
    )
    use_v2 = _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2
    for num_tokens in token_sizes:
        if use_v2:
            from typing import cast

            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as V2GPUModelRunner,
            )
            from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

            run_mixed_prefill_decode_warmup(
                cast("V2GPUModelRunner", runner),
                worker.execute_model,
                worker.sample_tokens,
                num_tokens,
                req_id_prefix="_dsv4_sm86_warmup",
            )
        else:
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_mixed_batch=True,
            )
