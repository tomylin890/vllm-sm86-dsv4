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

    # Catalog entry 2 first: a broken flash_mla install must fail here, with
    # the build command, not inside the dummy run's stack.
    if envs.VLLM_DSV4_FLASH_PREFILL:
        _warmup_flash_mla_prefill_op(runner.device)

    # Catalog entry 1: min and max chunk sizes through the real path. The
    # v2-vs-v1 runner split mirrors deepseek_v4_sparse_mla_attention_warmup.
    from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
        _uses_v2_model_runner,
    )

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    if max_tokens <= 0:
        return
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
