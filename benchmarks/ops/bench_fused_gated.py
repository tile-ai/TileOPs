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
from tileops._ops.elementwise import (
    GELUAndMulFwdOp,
    SiLUAndMulFwdOp,
)


# flashinfer names its fused gated kernels after the same three activations and
# takes the same concatenated input, so a key here doubles as its entry-point name.
def _profile_fused_gated(op_cls, case: bench.Case, library: str) -> None:
    op = op_cls(**case.arguments)
    flashinfer_fn = flashinfer_op(library)
    implementations = {}
    if case.arguments.get("swiglu_limit", 0.0) > 0 or any(t is not None for t in case.inputs[1:]):
        print(f"  [skip] {FLASHINFER_TAG}: {library} clamps no input and scales no row")
    else:
        implementations[FLASHINFER_TAG] = lambda x, *_absent: flashinfer_fn(x)
    implementations["torch-ref"] = case.torch
    implementations[TORCH_COMPILE_TAG] = compiled_reference(case.torch)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SiLUAndMulFwdOp), ids=lambda case: case.id)
def test_silu_and_mul_bench(case) -> None:
    _profile_fused_gated(SiLUAndMulFwdOp, case, "silu_and_mul")


@pytest.mark.parametrize("case", bench.cases(GELUAndMulFwdOp), ids=lambda case: case.id)
def test_gelu_and_mul_bench(case) -> None:
    tanh = case.arguments.get("approximate") == "tanh"
    _profile_fused_gated(GELUAndMulFwdOp, case, "gelu_tanh_and_mul" if tanh else "gelu_and_mul")
