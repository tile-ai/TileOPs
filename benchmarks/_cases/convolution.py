"""Case factories of the convolution family."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from workloads.convolution import Conv1dWorkload, Conv2dWorkload, Conv3dWorkload


def _static_weight(op, case) -> Implementation:
    """The op called on ``x`` alone, the weight and bias bound across calls."""
    x, weight, bias = case.inputs

    def run(x_i):
        if bias is None:
            return op(x_i, weight)
        return op(x_i, weight, bias)

    return Implementation(run=run, args=(x,))


ENTRIES = {
    "Conv1dFwdOp": Entry(Conv1dWorkload.from_call, binder=_static_weight),
    "Conv2dFwdOp": Entry(Conv2dWorkload.from_call),
    "Conv3dFwdOp": Entry(Conv3dWorkload.from_call),
}
