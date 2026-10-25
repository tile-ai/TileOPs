"""Tests for fused gated elementwise ops (silu_and_mul, gelu_and_mul in both GELU forms).

Covers L1 smoke correctness, multi-dtype coverage, and strategy selection.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import GELUAndMulFwdOp, SiLUAndMulFwdOp
from workloads.device import run_device
from workloads.elementwise import (
    GELUAndMulCase,
    SiLUAndMulCase,
    SiLUAndMulGateSweepWorkload,
)


class SiLUAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                # Tile boundary: the second 1024-column block is half past N, so the evict-last
                # path masks its upper threads with the col < N guard.
                pytest.param(1024, 1536, torch.bfloat16, marks=pytest.mark.smoke),
                # A bfloat16 row of whole vectors that is not whole pairs of them.
                pytest.param(1024, 1032, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class SiLUAndMulTest(SiLUAndMulCase, TestBase):
    pass


@SiLUAndMulFixture
def test_silu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = SiLUAndMulTest(m, n, dtype)
    op = SiLUAndMulFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_tuned() -> None:
    """Tuned mode picks a launch whose result matches the reference."""
    test = SiLUAndMulTest(1024, 1536, torch.bfloat16)
    op = SiLUAndMulFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_lazy_op_rebinds_shape() -> None:
    """Lazy construction should not lock the op to the first runtime shape."""
    op = SiLUAndMulFwdOp()
    for m, n in [(32, 64), (16, 128)]:
        test = SiLUAndMulTest(m, n, torch.float16)
        test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_matches_torch_at_every_bfloat16_gate() -> None:
    """Edge: a bfloat16 gate's fast float32 divide rounds to torch's result at every gate.

    Near gate -88 the divisor passes 2**126, where a bare ``__fdividef`` returns zero
    for a result torch keeps at about -5e-37.
    """
    workload = SiLUAndMulGateSweepWorkload()
    TestBase.check(workload, SiLUAndMulFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_input_off_a_vector_boundary() -> None:
    """A contiguous input starting one element past a 16-byte boundary matches the reference."""
    test = SiLUAndMulTest(64, 1024, torch.bfloat16)
    (aligned,) = test.gen_inputs()
    x = torch.empty(aligned.numel() + 1, device=run_device(), dtype=aligned.dtype)[1:]
    x = x.view(aligned.shape).copy_(aligned)
    test.check(SiLUAndMulFwdOp(), x)


class GELUAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, approximate",
            [
                pytest.param(1024, 1024, torch.float16, "none", marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, "none", marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, "none", marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float16, "tanh", marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, "tanh", marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, "tanh", marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, "none", marks=pytest.mark.full),
                pytest.param(2048, 2048, torch.float16, "tanh", marks=pytest.mark.full),
            ],
        ),
    ]


class GELUAndMulTest(GELUAndMulCase, TestBase):
    pass


@GELUAndMulFixture
def test_gelu_and_mul_op(m: int, n: int, dtype: torch.dtype, approximate: str) -> None:
    test = GELUAndMulTest(m, n, dtype, approximate)
    op = GELUAndMulFwdOp(approximate=approximate)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_fused_gated_rejects_integer_dtype() -> None:
    """Fused gated ops are float-only; the rejection follows the tensor."""
    op = GELUAndMulFwdOp()
    x = torch.zeros(16, 32, device=run_device(), dtype=torch.int32)
    # The manifest dtype union rejects it before any kernel is asked for.
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fused_gated_serves_two_dtypes_from_one_instance() -> None:
    """The element type comes from the tensor, so both are valid on one op."""
    op = SiLUAndMulFwdOp()
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(16, 16, device=run_device(), dtype=dtype)
        assert op(x).dtype == dtype


# Strategy selection tests
