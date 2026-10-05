"""Tests for fused gated elementwise ops (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul).

Covers L1 smoke correctness, multi-dtype coverage, and strategy selection.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.elementwise import GeluAndMulFwdOp, GeluTanhAndMulFwdOp, SiluAndMulFwdOp
from workloads.device import run_device
from workloads.elementwise import (
    GeluAndMulCase,
    GeluTanhAndMulCase,
    SiluAndMulCase,
)


class SiluAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
                pytest.param(2048, 2048, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class SiluAndMulTest(SiluAndMulCase, TestBase):
    pass


@SiluAndMulFixture
def test_silu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = SiluAndMulTest(m, n, dtype)
    op = SiluAndMulFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_lazy_op_rebinds_shape() -> None:
    """Lazy construction should not lock the op to the first runtime shape."""
    op = SiluAndMulFwdOp()
    for m, n in [(32, 64), (16, 128)]:
        test = SiluAndMulTest(m, n, torch.float16)
        test.check(op, *test.gen_inputs())


class GeluAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class GeluAndMulTest(GeluAndMulCase, TestBase):
    pass


@GeluAndMulFixture
def test_gelu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype)
    op = GeluAndMulFwdOp()
    test.check(op, *test.gen_inputs())


class GeluTanhAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class GeluTanhAndMulTest(GeluTanhAndMulCase, TestBase):
    pass


@GeluTanhAndMulFixture
def test_gelu_tanh_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluTanhAndMulTest(m, n, dtype)
    op = GeluTanhAndMulFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_fused_gated_rejects_integer_dtype() -> None:
    """Fused gated ops are float-only; the rejection follows the tensor."""
    op = GeluAndMulFwdOp()
    x = torch.zeros(16, 32, device=run_device(), dtype=torch.int32)
    # The manifest dtype union rejects it before any kernel is asked for.
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fused_gated_serves_two_dtypes_from_one_instance() -> None:
    """The element type comes from the tensor, so both are valid on one op."""
    op = SiluAndMulFwdOp()
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(16, 16, device=run_device(), dtype=dtype)
        assert op(x).dtype == dtype


# Strategy selection tests
