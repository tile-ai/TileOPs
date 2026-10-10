"""Benchmarks for the MHC pre/post ops.

Workload shapes and the pre-op scaling params come from the ops manifest; roofline
FLOP and byte counts come from each op's ``eval_roofline()``.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.ops import MHCPostFwdOp, MHCPreFwdOp

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("case", bench.cases(MHCPreFwdOp), ids=lambda case: case.id)
def test_mhc_pre_bench(case) -> None:
    op = MHCPreFwdOp(**case.arguments)
    if _TUNE:
        op.autotune()
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(MHCPostFwdOp), ids=lambda case: case.id)
def test_mhc_post_bench(case) -> None:
    op = MHCPostFwdOp(**case.arguments)
    if _TUNE:
        op.autotune()
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
