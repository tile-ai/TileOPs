"""How the moe family's benchmarks call their ops."""

import torch

from benchmarks._cases import Entry
from benchmarks.api import Implementation


def _private_output(op, case) -> Implementation:
    """The op writing a private ``out`` buffer, which it overwrites whole, so no reset."""
    output = torch.empty_like(case.inputs[0])

    def run(*args):
        return op(*args, out=output)

    return Implementation(run=run, args=case.inputs)


def _topk_inputs(workload) -> tuple:
    """The router logits, and the correction bias only where the call passes one."""
    gating_output, correction_bias = workload.gen_inputs()
    return (gating_output,) if correction_bias is None else (gating_output, correction_bias)


ENTRIES = {
    "FusedTopKFwdOp": Entry(inputs=_topk_inputs),
    "FusedMoEExpertsFwdOp": Entry(count_copies=True, binder=_private_output),
    "IndexedExpertMLPFwdOp": Entry(count_copies=True, binder=_private_output),
    "FusedMoEFwdOp": Entry(count_copies=True),
    "FusedMoESharedExpertFwdOp": Entry(count_copies=True),
    "SharedExpertMLPFwdOp": Entry(count_copies=True),
}
