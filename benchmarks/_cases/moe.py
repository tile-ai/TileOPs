"""Case factories of the moe family."""

import torch

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from workloads.moe import (
    FusedMoESharedExpertWorkload,
    FusedMoEWorkload,
    FusedTopKWorkload,
    IndexedExpertMLPWorkload,
    MoEExpertMLPWorkload,
    MoEExpertsWorkload,
    MoEGroupedGemmWorkload,
    MoEPermuteAlignWorkload,
    MoEPostPermuteWorkload,
    MoEPrePermuteWorkload,
    SharedExpertMLPWorkload,
)


def _private_output(op, case) -> Implementation:
    """The op writing a private output buffer, which it overwrites whole, so no reset."""
    output = torch.empty_like(case.inputs[0])

    def run(*args):
        op(output, *args)
        return output

    return Implementation(run=run, args=case.inputs[1:])


def _topk_inputs(workload) -> tuple:
    """The router logits, and the correction bias only where the call passes one."""
    gating_output, correction_bias = workload.gen_inputs()
    return (gating_output,) if correction_bias is None else (gating_output, correction_bias)


ENTRIES = {
    "FusedTopKFwdOp": Entry(FusedTopKWorkload, inputs=_topk_inputs),
    "MoEPermuteAlignFwdOp": Entry(MoEPermuteAlignWorkload),
    "MoEPrePermuteFwdOp": Entry(MoEPrePermuteWorkload),
    "MoEPostPermuteFwdOp": Entry(MoEPostPermuteWorkload),
    "MoEGroupedGemmFwdOp": Entry(MoEGroupedGemmWorkload),
    "MoEExpertMLPFwdOp": Entry(MoEExpertMLPWorkload),
    "FusedMoEExpertsFwdOp": Entry(MoEExpertsWorkload, count_copies=True, binder=_private_output),
    "IndexedExpertMLPFwdOp": Entry(
        IndexedExpertMLPWorkload, count_copies=True, binder=_private_output
    ),
    "FusedMoEFwdOp": Entry(FusedMoEWorkload, count_copies=True),
    "FusedMoESharedExpertFwdOp": Entry(FusedMoESharedExpertWorkload, count_copies=True),
    "SharedExpertMLPFwdOp": Entry(SharedExpertMLPWorkload, count_copies=True),
}
