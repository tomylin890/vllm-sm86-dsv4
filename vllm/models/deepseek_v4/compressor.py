# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
    compress_norm_rope_store_two_stage_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import is_cutedsl_supported
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool

logger = init_logger(__name__)


def _prefer_two_stage_compressor() -> bool:
    # Platforms that favor the triton variant of two-stage compressor split.
    # Currently only tested on ROCm
    return current_platform.is_rocm()


def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
    starts = metadata._num_computed_tokens_cpu
    if starts is None:
        return None

    starts_list = starts.tolist()
    query_start_loc = metadata.query_start_loc_cpu.tolist()
    return any(
        start % 128 + query_start_loc[i + 1] - query_start_loc[i] >= 128
        for i, start in enumerate(starts_list)
    )


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    num_decode_tokens: int | None = None
    c128_boundary: bool | None = None


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        num_decode_tokens = None
        if _prefer_two_stage_compressor():
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=1
            )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            num_decode_tokens=num_decode_tokens,
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.block_size == 8
                else None
            ),
        )


def get_compressor_state_window(vllm_config: VllmConfig) -> int | None:
    """P8: the fp32 compressor-state ring capacity in tokens, or None (off).

    ``VLLM_DSV4_COMPRESSOR_WINDOWED`` replaces the default absolute-position
    placement of the fp32 compressor state (one paged row per scheduled
    token, so a step of ``F = max_num_batched_tokens`` tokens forces
    ``F + sliding_window - 1`` live rows and the per-request reservation
    grows linearly with F) by a fixed ``W``-token ring: row ``position`` goes
    to ring slot ``position % W``. Rows are then reused within a step, so the
    reservation collapses to the constant ``cdiv(W, block_size)``.

    Validation (host config, therefore identical on every TP/PP/DCP rank --
    scheduler-visible admission stays rank-invariant):

    * ``W % 128 == 0``: 128 is the largest compressor lookback window
      (C128/indexer) and also ``block_table.get_block_table_width``'s token
      alignment; a multiple of 128 is divisible by both compressor-state
      block sizes (4 and 8) and makes the ring an exact whole number of
      block-table columns for every group.
    * ``W > 128``: the compressor sub-chunk ``G = W - sliding_window + 1``
      must be >= 1 for the widest window.
    * prefix caching must be off: a ring is not prefix-addressable, so a
      cache hit would hand the request rows that were never recomputed.
      The ring and prefix caching are a PRODUCT SWITCH, not a preference:
      PROFILE-P8 (ring, caching off) and PROFILE-CACHE (default
      absolute-position placement, caching on) differ in per-request state
      reservation by hundreds of blocks, so a conflict is refused here
      rather than silently degraded to ``None`` -- degrading it would move
      the KV footprint by gigabytes and only resurface much later as an
      admission refusal or an OOM (P11-DESIGN.md step 4).
    """
    if not envs.VLLM_DSV4_COMPRESSOR_WINDOWED:
        return None
    window = int(envs.VLLM_DSV4_COMPRESSOR_WINDOW)
    if window <= 128 or window % 128 != 0:
        raise ValueError(
            "VLLM_DSV4_COMPRESSOR_WINDOW must be a multiple of 128 and "
            f"greater than 128 (the largest compressor lookback); got {window}"
        )
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "VLLM_DSV4_COMPRESSOR_WINDOWED and prefix caching are mutually "
            "exclusive: the windowed fp32 compressor state is a ring keyed "
            "by absolute position modulo the window, not a prefix-"
            "addressable cache, so a prefix-cache hit would skip recomputing "
            "rows the compression kernel reads. Pick one of the two "
            "supported serving profiles. PROFILE-P8: keep "
            "VLLM_DSV4_COMPRESSOR_WINDOWED=1 and add "
            "--no-enable-prefix-caching; the ring pins the per-request state "
            "reservation at cdiv(W, block_size), so max_num_batched_tokens "
            "can stay at 1024. PROFILE-CACHE: unset "
            "VLLM_DSV4_COMPRESSOR_WINDOWED to fall back to the "
            "prefix-cacheable absolute-position placement and keep prefix "
            "caching on; that reservation grows with max_num_batched_tokens, "
            "so cap it at 768 (the measured ceiling at max_model_len "
            "262144 on 24 GiB cards: 1024 needs 0.55 GiB for admission "
            "against a 0.89 GiB pool that must also hold ~0.6 GiB of "
            "prefill transients). The conflict is never resolved silently -- "
            "either resolution changes the KV footprint by gigabytes."
        )
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        # _promote_local_kv_cache_specs rebuilds SlidingWindowMLASpec as
        # MLAAttentionSpec and cannot carry state_window, so the write side
        # (slot mapping, spec-derived) would fall back to absolute placement
        # while the read side (this module, env-derived) keeps folding
        # position % W -- silent numerically-wrong fp32 state. Note this
        # flag is also set IMPLICITLY (KV connector without HMA support,
        # platforms without support_hybrid_kv_cache()), not only via CLI.
        raise ValueError(
            "VLLM_DSV4_COMPRESSOR_WINDOWED requires the hybrid KV cache "
            "manager (disable_hybrid_kv_cache_manager must be off): spec "
            "promotion drops the ring geometry and would desynchronize the "
            "state writer from the compressor reader."
        )
    return window


def _compressor_state_compress_ratio(spec: SlidingWindowMLASpec) -> int | None:
    """The compressor's ``m`` for an fp32 compressor-state group, else None.

    ``CompressorStateCache`` derives the state window as
    ``coff * compress_ratio`` with ``coff = 1 + (compress_ratio == 4)``, so a
    window of 8 is the overlapped C4 family (m = 4) and every other window
    equals its own compress ratio. The SWA KV group is a SlidingWindowMLASpec
    too, but it holds real KV (uint8 / bfloat16 / fp8) rather than fp32 state
    and has no compressor behind it, so it has no ``m``.

    The inverse is ambiguous in principle (a window of 8 could also be a
    non-overlapped m = 8), but harmlessly so: ``m`` always divides the window,
    so the compress-ratio leg of the validation is implied by its window leg
    whichever ``m`` a window came from.
    """
    if spec.dtype != torch.float32:
        return None
    return 4 if spec.sliding_window == 8 else spec.sliding_window


def validate_compressor_lookback_coverage(
    kv_cache_specs: Iterable[KVCacheSpec],
    scheduler_block_size: int,
    use_eagle: bool = False,
) -> None:
    """P11 I1/I2: a prefix-cache hit must cover the compression lookback.

    The compression kernel gathers state rows ``[p - (1 + OVERLAP) * m + 1,
    p]`` at every boundary position ``p`` with ``(p + 1) % m == 0``
    (``fused_compress_quant_cache.py``), i.e. ``L = sliding_window`` rows, of
    which ``L - m`` fall BELOW the position the request resumes from. Those
    rows are never recomputed by the resumed pass, so reading them without
    having written them is wrong however they were left. A freshly allocated
    state block IS now zeroed -- ``_record_new_block_ids`` admits every
    attention-family group, SlidingWindowMLASpec included
    (``single_type_kv_cache_manager.py``) -- which downgrades the hazard from
    another tenant's uint8 bytes read as fp32 (Inf/NaN) to a lookback of
    zeros, but zeros are still a WRONG lookback, not a safe one, so this
    check still guards the same thing. What makes the resume correct is not a
    trim but the SWA cache-hit geometry: ``SlidingWindowManager`` reserves
    ``_contiguous_blocks_for_hit`` REAL blocks ending at the hit boundary
    ``H`` for every sliding-window group, and ``H`` is a multiple of the
    scheduler block size.

    Checked for every sliding-window group (both compressor-state families
    and the SWA KV group):

    * ``_contiguous_blocks_for_hit(L, block_size, use_eagle) * block_size >=
      L - 1`` -- the reserved tail covers the lookback. ``L - 1`` is the
      strict bound; the ``L - m`` rows actually read below ``H`` are a subset.
      Today's ``cdiv(L - 1, block_size)`` satisfies this by construction, so
      this leg exists to pin that formula: it trips the moment the cache-hit
      path reserves less than the lookback.
    * ``scheduler_block_size % L == 0``.
    * ``scheduler_block_size % m == 0`` -- ``H`` is a multiple of the
      scheduler block size, hence of ``m``, so the first recomputed boundary
      is ``p = H + m - 1`` and its gather starts exactly at ``H - (L - m)``.
      Kept explicit for the error message; ``m`` divides ``L``, so the window
      leg above already implies it.

    ``use_eagle`` defaults to False, which is the strict case: eagle only
    adds one contiguous block (and drops the last matched one), so a geometry
    that satisfies the invariant without eagle satisfies it with eagle.

    Today all three hold by arithmetic coincidence of 1024 / 128 / 8 / 4;
    this is what fails loudly if a block size, window or compress ratio
    moves. Config time only -- nothing here runs per step.
    """
    # Local import: the model module must not pull vllm.v1.core at import
    # time, and this runs once per KV cache config.
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    for spec in kv_cache_specs:
        if not isinstance(spec, SlidingWindowMLASpec):
            continue
        window = spec.sliding_window
        # Deliberately the manager's own helper rather than a copy of its
        # cdiv: the invariant is about what the cache-hit path actually
        # reserves, so a change to that formula must trip this check.
        contiguous_blocks = SlidingWindowManager._contiguous_blocks_for_hit(
            window, spec.block_size, use_eagle
        )
        covered_tokens = contiguous_blocks * spec.block_size
        if covered_tokens < window - 1:
            raise ValueError(
                "DeepseekV4 prefix caching requires every sliding-window "
                "group's cache-hit reservation to cover the compression "
                f"lookback: group (block_size={spec.block_size}, "
                f"sliding_window={window}) reserves {contiguous_blocks} "
                f"contiguous blocks = {covered_tokens} tokens at a hit "
                f"boundary, short of the {window - 1} tokens below it that "
                "the kernel's lookback can reach."
            )
        if scheduler_block_size % window != 0:
            raise ValueError(
                "DeepseekV4 prefix caching requires the scheduler block size "
                f"({scheduler_block_size}) to be a multiple of every "
                "sliding-window group's window; group "
                f"(block_size={spec.block_size}, sliding_window={window}) "
                "would put cache-hit boundaries inside a window."
            )
        compress_ratio = _compressor_state_compress_ratio(spec)
        if compress_ratio is not None and scheduler_block_size % compress_ratio != 0:
            raise ValueError(
                "DeepseekV4 prefix caching requires the scheduler block size "
                f"({scheduler_block_size}) to be a multiple of the compress "
                f"ratio ({compress_ratio}) of the compressor-state group "
                f"(block_size={spec.block_size}, sliding_window={window}); "
                "otherwise a cache-hit boundary lands mid-entry and the "
                "first recomputed boundary gathers rows no pass writes."
            )


def check_compressor_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """P11 post-KV-cache-config hook: log the profile, check the invariants.

    Called from ``GPUModelRunner.may_reinitialize_input_batch`` once the KV
    cache groups are final, which is the earliest point where the scheduler /
    hash block sizes are resolvable. Everything here is a pure function of
    ``vllm_config`` and the KV cache groups -- both rank-invariant, the
    groups differing across ranks only in layer names -- so every rank logs
    and validates the same thing (DCP symmetry).
    """
    # Local import: same reason as in validate_compressor_lookback_coverage.
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes,
    )

    # Worker-side groups carry aggregated UniformTypeKVCacheSpecs; the
    # scheduler view replaces each by a representative per-layer spec (all
    # members of a group share block size, window and dtype), and that view
    # is what the scheduler's own block sizes are resolved from.
    scheduler_view = generate_scheduler_kv_cache_config([kv_cache_config])
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        scheduler_view, vllm_config
    )
    specs = [group.kv_cache_spec for group in scheduler_view.kv_cache_groups]
    sliding_window_specs = [
        spec for spec in specs if isinstance(spec, SlidingWindowMLASpec)
    ]

    enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
    ring_on = any(spec.state_window is not None for spec in sliding_window_specs)
    if ring_on:
        # get_compressor_state_window refuses ring + caching, so this arm is
        # always caching-off.
        profile = "PROFILE-P8"
    elif enable_prefix_caching:
        profile = "PROFILE-CACHE"
    else:
        profile = "PROFILE-CONTROL"
    # Logged so the deployment matrix can be asserted from the startup log
    # (P11 rack test T0); sliding_window_groups makes a validator that saw no
    # groups visible instead of silently vacuous.
    logger.info(
        "DeepseekV4 serving profile %s (compressor-state ring: %s, prefix "
        "caching: %s, max_num_batched_tokens: %d), scheduler_block_size=%d, "
        "hash_block_size=%d, num_gpu_blocks=%d, sliding_window_groups=%d",
        profile,
        "on" if ring_on else "off",
        "on" if enable_prefix_caching else "off",
        vllm_config.scheduler_config.max_num_batched_tokens,
        scheduler_block_size,
        hash_block_size,
        kv_cache_config.num_blocks,
        len(sliding_window_specs),
    )

    if enable_prefix_caching:
        # The invariant only constrains cache hits; PROFILE-P8 and
        # PROFILE-CONTROL never resume from one.
        validate_compressor_lookback_coverage(specs, scheduler_block_size)


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            state_window=get_compressor_state_window(vllm_config),
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache
        self.eager_scratch_pool = eager_scratch_pool

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        # VLLM_SM86_DCP (P2c cache-write path): the compressed-KV groups this
        # module writes are round-robin sharded across DCP ranks, while the
        # fp32 compressor state it reads (self.state_cache) stays replicated
        # (dcp_exempt, P1). Every rank therefore computes every entry
        # identically in fp32 (no collectives needed, capture-safe) and
        # stores only the entries it owns. Gate unset or dcp==1 keeps the
        # original path unchanged.
        self._dcp_world_size = 1
        self._dcp_rank = 0
        if envs.VLLM_SM86_DCP:
            try:
                from vllm.distributed import get_dcp_group

                self._dcp_world_size = get_dcp_group().world_size
                self._dcp_rank = get_dcp_group().rank_in_group
            except AssertionError:
                # DCP group not initialized (single GPU / tests).
                self._dcp_world_size = 1
                self._dcp_rank = 0
        self._dcp_interleave = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self._dcp_enabled = envs.VLLM_SM86_DCP and self._dcp_world_size > 1
        if self._dcp_enabled and use_fp4_cache:
            raise NotImplementedError(
                "VLLM_SM86_DCP: the MXFP4 indexer cache has no DCP "
                "write path (SM8x uses the FP8 indexer cache layout)."
            )

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        # The head=512 cr>=128 no-overlap deep gather uses the two-stage
        # compressor, which needs an fp32 scratch [max_batched, 512] for
        # the intermediate compressed_kv.
        # Currently only tested on ROCm
        self._use_two_stage_fused_compressor = (
            _prefer_two_stage_compressor() and head_dim == 512 and not self.overlap
        )
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._compress_scratch: torch.Tensor | None = None
        if self._use_two_stage_fused_compressor:
            self._compress_scratch = torch.empty(
                self.max_num_batched_tokens,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # ── P8: windowed (ring) compressor-state placement ────────────────
        # `state_window` W is the ring capacity in tokens (None == today's
        # absolute-position placement). `state_chunk` G is the number of FLAT
        # BATCH TOKENS the compressor forward may write before it must run the
        # compression pass over them:
        #
        #     G = W - L + 1,   L = self.state_cache.sliding_window = coff * m
        #
        # L is exactly the reader's span: the fused kernel gathers state rows
        # [p - (1+OVERLAP)*m + 1, p] for every boundary position p. Within one
        # sub-chunk every write happens before every read, so the live set
        # spans at most (G - 1) newly written + (L - 1) looked-back + 1 rows =
        # G + L - 1 distinct positions; the ring is collision-free iff
        # W >= G + L - 1. The bound is tight (a sub-chunk that starts exactly
        # on a boundary position corrupts at G = W - L + 2) -- see
        # P8-NOTES.md section 3 and the sim.
        #
        # Slicing on the FLAT token axis is what makes this per-request safe:
        # each request's tokens are a contiguous ascending run in the batch,
        # so a flat slice of G tokens contributes at most G CONSECUTIVE
        # positions to any single request.
        #
        # The two-stage (ROCm split) and cutedsl (SM90+) fused compressors
        # read the fp32 state with absolute-position addressing and have no
        # ring variant, so the forward's dispatch puts the windowed branch
        # AHEAD of both and routes to the Triton kernels -- which is where
        # SM8x already lands for head_dim 512 and 128 alike. Nothing is
        # rejected at init; the flag just narrows the kernel choice.
        self._state_window = get_compressor_state_window(vllm_config)
        self._state_chunk: int | None = None
        if self._state_window is not None:
            self._state_chunk = self._state_window - self.state_cache.sliding_window + 1
            assert self._state_chunk >= 1

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        # full graph cannot branch on per-step CPU metadata after capture
        skip_compress = (
            current_platform.is_cuda()
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        )

        # P8 sub-chunk plan on the FLAT token axis. `None` == the default
        # single full-batch launch pair with the ORIGINAL (unsliced) tensors,
        # i.e. byte-for-byte the pre-P8 dispatch. Under the flag the plan is a
        # pure function of `num_actual` (a shape), so a captured CUDA graph
        # replays a fixed launch sequence.
        token_ranges: list[tuple[int, int]] | None = None
        if self._state_chunk is not None:
            g = self._state_chunk
            token_ranges = [
                (a, min(a + g, num_actual)) for a in range(0, num_actual, g)
            ]

        if token_ranges is None:
            save_partial_states(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=state_cache,
                slot_mapping=slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=pdl_kwargs,
            )
            if skip_compress:
                return
        elif skip_compress:
            # Windowed, no boundary in this step: still write every row (a
            # LATER step's boundary reads them back through the ring), just
            # never compress here. The loop is still required even without a
            # read: a single launch covering more than W tokens would have two
            # tokens of one request land on the SAME ring slot concurrently
            # (positions p and p + W), and the winner would be
            # nondeterministic. G <= W - L + 1 < W, so within one launch every
            # request contributes at most G < W consecutive positions and no
            # two of them collide; across launches the later (higher) position
            # is written last, which is the correct order.
            for a, b in token_ranges:
                save_partial_states(
                    kv=kv[a:b],
                    score=score[a:b],
                    ape=self.ape,
                    positions=positions[a:b],
                    state_cache=state_cache,
                    slot_mapping=slot_mapping[a:b],
                    block_size=block_size,
                    state_width=state_width,
                    compress_ratio=self.compress_ratio,
                    pdl_kwargs=pdl_kwargs,
                )
            return

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        windowed_kwargs: dict[str, Any] = (
            {} if self._state_window is None else {"state_window": self._state_window}
        )
        if self._dcp_enabled:
            # VLLM_SM86_DCP + dcp>1: the cutedsl and two-stage fused paths
            # have no context-parallel layout support, so route to the
            # Triton kernel (pure Python/Triton constraint -- no csrc
            # edits). It performs the per-entry DCP ownership check and
            # routes owned writes through the P1 sharded block table.
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs: dict[str, Any] = {
                "dcp_world_size": self._dcp_world_size,
                "dcp_rank": self._dcp_rank,
                "cp_kv_cache_interleave_size": self._dcp_interleave,
                **windowed_kwargs,
            }
        elif self._state_window is not None:
            # P8: the cutedsl (SM90+) and two-stage (ROCm) fused paths read
            # the fp32 state with absolute-position addressing and have no
            # ring variant, so windowing routes to the Triton kernels (which
            # is where SM8x already lands for both head_dim 512 and 128).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = dict(windowed_kwargs)
        elif is_cutedsl_supported() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on SM90+ CUDA always uses cutedsl, for both the
            # fp8_ds_mla layout and the plain full-cache layout. The
            # full-cache flags are consumed only here. Pre-Hopper CUDA takes
            # the Triton path below like AMD/XPU.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
            if not self.overlap and self.eager_scratch_pool is not None:
                extra_kwargs["compress_scratch"] = (
                    self.eager_scratch_pool.compressor_scratch(num_actual)
                )
        elif self._use_two_stage_fused_compressor:
            # head=512 cr>=128 (no overlap): two-pass split compressor on the
            # prefill suffix, single-pass on the decode prefix.
            assert state_metadata.num_decode_tokens is not None
            compress_norm_rope_store_fn = compress_norm_rope_store_two_stage_triton
            extra_kwargs = {
                "num_decode_tokens": state_metadata.num_decode_tokens,
                "compress_scratch": self._compress_scratch,
            }
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        if token_ranges is None:
            compress_norm_rope_store_fn(
                state_cache=state_cache,
                num_actual=num_actual,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                block_size=block_size,
                state_width=state_width,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=k_cache_metadata,
                pdl_kwargs=pdl_kwargs,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                use_fp4_cache=self.use_fp4_cache,
                rms_norm_weight=self.norm.weight,
                rms_norm_eps=self.rms_norm_eps,
                quant_block=self._quant_block,
                token_stride=self._token_stride,
                scale_dim=self._scale_dim,
                **extra_kwargs,
            )
            return

        # P8 windowed: write G tokens, compress them, repeat. Interleaving the
        # two launches is what bounds the live row set to G + L - 1 <= W and
        # therefore decouples the reservation from max_num_batched_tokens.
        # Request-indexed tensors (block_table, ape, weights) are passed
        # whole; only the token-indexed ones are sliced, and the compressed-KV
        # slot mapping is sliced through `kv_slot_mapping` (under DCP the
        # head=512 kernel derives its slot from the block table and never
        # reads it).
        kv_slot_mapping = k_cache_metadata.slot_mapping
        for a, b in token_ranges:
            save_partial_states(
                kv=kv[a:b],
                score=score[a:b],
                ape=self.ape,
                positions=positions[a:b],
                state_cache=state_cache,
                slot_mapping=slot_mapping[a:b],
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=pdl_kwargs,
            )
            compress_norm_rope_store_fn(
                state_cache=state_cache,
                num_actual=b - a,
                token_to_req_indices=token_to_req_indices[a:b],
                positions=positions[a:b],
                slot_mapping=slot_mapping[a:b],
                block_table=block_table,
                block_size=block_size,
                state_width=state_width,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=k_cache_metadata,
                kv_slot_mapping=(
                    None if kv_slot_mapping is None else kv_slot_mapping[a:b]
                ),
                pdl_kwargs=pdl_kwargs,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                use_fp4_cache=self.use_fp4_cache,
                rms_norm_weight=self.norm.weight,
                rms_norm_eps=self.rms_norm_eps,
                quant_block=self._quant_block,
                token_stride=self._token_stride,
                scale_dim=self._scale_dim,
                **extra_kwargs,
            )
