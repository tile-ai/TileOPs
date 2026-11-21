"""How the convolution family's benchmarks call their ops."""

from collections.abc import Callable

from benchmarks._cases import Entry
from benchmarks.api import Implementation


def static_weight(fn: Callable, case) -> Implementation:
    """*fn* called on ``x`` alone, the weight and bias bound across calls; *fn* is the op or a
    baseline taking ``(x, weight, bias)``."""
    x, weight, bias = case.inputs

    def run(x_i):
        return fn(x_i, weight, bias)

    return Implementation(run=run, args=(x,))


ENTRIES = {
    "Conv1dFwdOp": Entry(binder=static_weight),
}
