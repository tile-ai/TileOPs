"""Where the pool benchmark picks its baseline.

The two ways torch is the right answer are worth pinning: everything else must reach the
library, and a library that is missing raises rather than reporting torch under a tag that
claims it.
"""


def test_pool_baseline_routes_or_says_so():
    """The pool baseline is torch only where it is meant to be.

    Two ways torch is the right answer: the op is not in the table, or the library cannot
    express the workload. Anything else must reach the library — a missing one raises rather
    than reporting torch under a case that claims a library baseline.
    """
    from benchmarks.ops.bench_pool import _BASELINE, pool_baseline

    class _Case:
        kernel_size, stride, padding, ceil_mode = 3, 2, 0, False
        dilation, return_indices = 1, False

        def ref_program(self, x):
            return x

    case = _Case()
    assert pool_baseline("AdaptiveAvgPool2dFwdOp", case)[0] == "torch-ref", "not in the table"

    case.dilation = 2  # cuDNN Resample has no dilation
    assert pool_baseline("MaxPool3dFwdOp", case)[0] == "torch-ref", "library cannot express it"

    assert {choice for choice, _, _ in _BASELINE.values()} == {"flaggems", "cudnn"}
