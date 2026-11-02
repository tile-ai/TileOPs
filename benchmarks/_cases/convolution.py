"""How the convolution family's benchmarks call their ops."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation


def _static_weight(op, case) -> Implementation:
    """The op called on ``x`` alone, the weight and bias bound across calls."""
    x, weight, bias = case.inputs

    def run(x_i):
        if bias is None:
            return op(x_i, weight)
        return op(x_i, weight, bias)

    return Implementation(run=run, args=(x,))


ENTRIES = {
    "Conv1dFwdOp": Entry(binder=_static_weight),
}
