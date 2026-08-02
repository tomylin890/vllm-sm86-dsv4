# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank (DCP) merge helpers for DeepSeek V4 sparse attention.

Ported from the Lasimeri DCP reference (``common/ops/dcp.py`` @7fff60c);
the fp32 merge math of ``apply_attn_sink`` / ``dcp_merge_flashmla_output`` /
``dcp_softmax_reduce`` is kept verbatim.  The one adaptation to this base is
``softmax_stats_to_lse``: the SM8x decode kernel
(``rocm_aiter_mla_sparse.rocm_sparse_attn_decode`` with
``return_softmax_stats=True``) emits raw pre-sink partials
``(o, m, l)`` — per-shard normalized output (bf16 ``[tokens, heads, 512]``),
fp32 running row-max ``m`` and fp32 sum-exp ``l`` (both ``[tokens, heads]``,
NATURAL log/exp domain) — instead of a fused LSE, so the layer converts
``(m, l) -> lse = m + log(l)`` here before reusing the in-tree merge
infrastructure (``cp_lse_ag_out_rs`` / ``dcp_a2a_lse_reduce``, both
``is_lse_base_on_e=True``).

Precision rules honored (ARCHITECTURE.md section 10):
- rule 1: the attention sink enters exactly once, at the global merge, based
  at the global running max (``torch.logaddexp`` in ``apply_attn_sink``).
- rule 2: the merge reduces in fp32 with a fixed rank order
  (``tl.static_range`` over ranks inside ``dcp_a2a_lse_reduce``).
- rule 9: empty shards (``l == 0``) carry the finite sentinel ``-1.0e30``
  (matching ``triton_mla_sparse_kernel.py``'s ``NEG_LARGE``), never ``-inf``;
  their merge weight underflows to exactly 0 and they drop out.
"""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Finite empty-shard sentinel for per-shard LSEs.  ``-inf`` is forbidden:
# ``-inf - -inf = NaN`` poisons downstream softmax algebra (ARCHITECTURE.md
# section 10 rule 9); ``exp(-1.0e30 - g)`` underflows to exactly 0.0 in fp32
# for any realistic global max ``g``, so sentinel shards contribute nothing.
# Matches NEG_LARGE in vllm/v1/attention/ops/triton_mla_sparse_kernel.py.
DCP_LSE_SENTINEL = -1.0e30


def softmax_stats_to_lse(
    rowmax: torch.Tensor,
    sumexp: torch.Tensor,
) -> torch.Tensor:
    """Fuse pre-sink softmax stats ``(m, l)`` into a natural-log LSE.

    ``lse = m + log(l)`` where the shard saw at least one valid key
    (``l >= 1`` by construction: the key attaining the row max contributes
    ``exp(0) = 1``), else the finite ``DCP_LSE_SENTINEL``.  Every
    intermediate stays finite: ``l`` is clamped before the log so even the
    discarded lane never produces ``-inf``/NaN.

    Args:
        rowmax: fp32 ``[tokens, heads]`` running max of scaled logits.
        sumexp: fp32 ``[tokens, heads]`` sum of ``exp(logit - rowmax)``.

    Returns:
        fp32 ``[tokens, heads]`` LSE (natural log domain).
    """
    rowmax = rowmax.to(torch.float32)
    sumexp = sumexp.to(torch.float32)
    lse = rowmax + torch.log(torch.clamp(sumexp, min=torch.finfo(torch.float32).tiny))
    return torch.where(
        sumexp > 0,
        lse,
        torch.full_like(lse, DCP_LSE_SENTINEL),
    )


def apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Rescale a sink-less attention output by the attention-sink term.

    ``logaddexp`` internally bases at ``max(lse, sink)`` — the true global
    max including the sink — so this realizes ARCHITECTURE.md section 10
    rule 1 (sink added once, at the global max).  Verbatim from the
    reference.
    """
    sink = attn_sink[: out.shape[1]].to(dtype=lse.dtype)
    output_lse = torch.logaddexp(lse, sink.unsqueeze(0))
    scale = torch.exp(lse - output_lse).to(dtype=out.dtype)
    return out * scale.unsqueeze(-1)


def dcp_merge_flashmla_output(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    group: "GroupCoordinator",
    use_a2a: bool = True,
) -> None:
    """Merge per-rank pre-sink partials and apply the sink exactly once.

    Args:
        local_out: bf16 ``[tokens, gathered_heads, head_dim]`` — this rank's
            shard-local attention over ALL query heads of the DCP group.
        local_lse: fp32 ``[tokens, gathered_heads]`` natural-log LSE
            (``DCP_LSE_SENTINEL`` for empty shards).
        attn_sink: fp32 per-head sink logits of THIS rank's local heads.
        output: destination ``[tokens, >=local_heads, head_dim]``; the first
            ``gathered_heads // world_size`` head slots are written.
        group: the DCP group coordinator.
        use_a2a: fixed-rank-order fp32 a2a merge (default, required by
            ARCHITECTURE.md section 10 rule 2).  The ag+rs fallback is kept
            for parity with the reference but reduce-scatters in the output
            dtype via NCCL — do not use it where bit-stable fp32 merging is
            required.
    """
    if use_a2a:
        out, lse = dcp_a2a_lse_reduce(
            local_out,
            local_lse,
            group,
            return_lse=True,
        )
    else:
        out, lse = cp_lse_ag_out_rs(
            local_out,
            local_lse,
            group,
            return_lse=True,
        )
    output[:, : out.shape[1], :].copy_(apply_attn_sink(out, lse, attn_sink))


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge numerically stable partial softmax statistics across DCP ranks."""
    valid = local_sum > 0
    local_max = torch.where(
        valid,
        local_max,
        torch.full_like(local_max, -float("inf")),
    )
    gathered_max = group.all_gather(local_max, dim=0).reshape(
        (group.world_size,) + local_max.shape
    )
    global_max = gathered_max.max(dim=0).values

    scale = torch.exp(local_max - global_max)
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    reduce_payload = torch.stack(
        (
            torch.where(valid, local_sum * scale, torch.zeros_like(local_sum)),
            torch.where(
                valid,
                local_weighted_value * scale,
                torch.zeros_like(local_weighted_value),
            ),
        )
    )
    global_sum, global_weighted_value = group.all_reduce(reduce_payload).unbind(0)
    return torch.where(
        global_sum > 0,
        global_weighted_value / global_sum,
        torch.zeros_like(global_weighted_value),
    )
