"""No-DP/EP prepare-and-finalize: quantization pass-through, local weighted reduction."""

from __future__ import annotations

from torch import Tensor

from tileops.ops.moe.abc import MoEPrepareAndFinalize, PrepareResult, WeightedReduce

__all__ = ["MoEPrepareAndFinalizeNoDPEP"]


class MoEPrepareAndFinalizeNoDPEP(MoEPrepareAndFinalize):
    """No Data-Parallel / Expert-Parallel prepare and finalize.

    prepare(): bf16/fp16 pass-through (no quantization, no dispatch).
    finalize(): delegates to weight_and_reduce (typically WeightedReduceNoOp).

    For EP, replace with a subclass that adds All-to-All dispatch/gather.
    """

    def prepare(
        self,
        hidden: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        num_experts: int,
        expert_map: Tensor | None,
    ) -> PrepareResult:
        return PrepareResult(
            hidden_q=hidden,
            scale=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    def finalize(
        self,
        output: Tensor,
        expert_out: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        weight_and_reduce: WeightedReduce,
    ) -> None:
        weight_and_reduce.apply(output, expert_out, topk_weights, topk_ids)
