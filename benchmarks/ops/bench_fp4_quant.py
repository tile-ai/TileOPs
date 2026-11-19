"""Benchmark for the FP4 e2m1 quantization op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts come from the op's
``eval_roofline()``.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference, private_inputs
from tileops.quantization import FP4QuantPerGroupFwdOp


@pytest.mark.parametrize("case", bench.cases(FP4QuantPerGroupFwdOp), ids=lambda case: case.id)
def test_fp4_quant_bench(case) -> None:
    op = FP4QuantPerGroupFwdOp(**case.arguments)
    implementations = {
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    if case.arguments["inplace"]:
        implementations = {
            name: private_inputs(fn, case.inputs, 0) for name, fn in implementations.items()
        }
    bench.Runner(op, case).compare(implementations)
