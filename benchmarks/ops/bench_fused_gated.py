"""Benchmarks for the fused gated elementwise ops, one case per manifest call.

Every row times the reference in torch eager and through inductor, and flashinfer's
kernel, which takes the concatenated input the reference splits.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flashinfer_op,
)
from tileops.ops.elementwise import (
    GeluAndMulFwdOp,
    GeluTanhAndMulFwdOp,
    SiluAndMulFwdOp,
)


# flashinfer names its fused gated kernels after the same three activations and
# takes the same concatenated input, so a key here doubles as its entry-point name.
def _profile_fused_gated(op_cls, case: bench.Case, library: str) -> None:
    op = op_cls(**case.arguments)
    flashinfer_fn = flashinfer_op(library)
    bench.Runner(op, case).compare(
        {
            FLASHINFER_TAG: flashinfer_fn,
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(SiluAndMulFwdOp), ids=lambda case: case.id)
def test_silu_and_mul_bench(case) -> None:
    _profile_fused_gated(SiluAndMulFwdOp, case, "silu_and_mul")


@pytest.mark.parametrize("case", bench.cases(GeluAndMulFwdOp), ids=lambda case: case.id)
def test_gelu_and_mul_bench(case) -> None:
    _profile_fused_gated(GeluAndMulFwdOp, case, "gelu_and_mul")


@pytest.mark.parametrize("case", bench.cases(GeluTanhAndMulFwdOp), ids=lambda case: case.id)
def test_gelu_tanh_and_mul_bench(case) -> None:
    _profile_fused_gated(GeluTanhAndMulFwdOp, case, "gelu_tanh_and_mul")
