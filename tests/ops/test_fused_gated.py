"""Tests for fused gated elementwise ops (silu_and_mul, gelu_and_mul in both GELU forms).

Covers L1 smoke correctness, multi-dtype coverage, and strategy selection.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import GELUAndMulFwdOp, SiLUAndMulFwdOp
from workloads.device import run_device
from workloads.elementwise import (
    ElementwiseWorkload,
    GELUAndMulCase,
    GELUAndMulTailWorkload,
    SiLUAndMulCase,
    SiLUAndMulClampEdgeWorkload,
    SiLUAndMulEdgeWorkload,
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
                # A row ending one 16-byte vector past a power-of-two width.
                pytest.param(1024, 1032, torch.bfloat16, marks=pytest.mark.smoke),
                # Tile boundary: the second 1024-column block is half past N.
                pytest.param(1024, 1536, torch.bfloat16, marks=pytest.mark.smoke),
                # More rows than a grid's y dimension holds.
                pytest.param(65537, 128, torch.bfloat16, marks=pytest.mark.smoke),
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_multiplies_the_unrounded_activation(dtype: torch.dtype) -> None:
    """Edge: zeros, subnormals, extremes and nonfinite gates times the same values."""
    workload = SiLUAndMulEdgeWorkload(dtype)
    TestBase.check(workload, SiLUAndMulFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_input_off_a_vector_boundary() -> None:
    """A contiguous input starting one element past a 16-byte boundary matches the reference."""
    test = SiLUAndMulTest(64, 1024, torch.bfloat16)
    (aligned,) = test.gen_inputs()
    x = torch.empty(aligned.numel() + 1, device=run_device(), dtype=aligned.dtype)[1:]
    x = x.view(aligned.shape).copy_(aligned)
    test.check(SiLUAndMulFwdOp(), x)


class SiLUAndMulClampedFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, swiglu_limit, scaled",
            [
                # DeepSeek-V4-Flash's routed rows of one token at top 6.
                pytest.param(6, 2048, torch.bfloat16, 10.0, True, marks=pytest.mark.smoke),
                # Its shared expert: clamped, not scaled.
                pytest.param(64, 2048, torch.bfloat16, 10.0, False, marks=pytest.mark.smoke),
                # Scaled, not clamped, on a row off the vector width.
                pytest.param(37, 1001, torch.bfloat16, 0.0, True, marks=pytest.mark.smoke),
                pytest.param(64, 1024, torch.float16, 10.0, True, marks=pytest.mark.smoke),
                pytest.param(64, 1024, torch.float32, 10.0, True, marks=pytest.mark.smoke),
                # More rows than a grid's y dimension holds.
                pytest.param(65537, 128, torch.bfloat16, 10.0, True, marks=pytest.mark.full),
                # A limit bfloat16 does not hold.
                pytest.param(64, 1024, torch.bfloat16, 7.3, True, marks=pytest.mark.full),
                # DeepSeek-V4-Flash's routed rows of 4096 and 32768 tokens.
                pytest.param(24576, 2048, torch.bfloat16, 10.0, True, marks=pytest.mark.full),
                pytest.param(196608, 2048, torch.bfloat16, 10.0, True, marks=pytest.mark.full),
            ],
        ),
    ]


@SiLUAndMulClampedFixture
def test_silu_and_mul_clamps_and_scales_rows(
    m: int, n: int, dtype: torch.dtype, swiglu_limit: float, scaled: bool
) -> None:
    """Gates and values spread past the limit are clamped, and each row's product scaled, in
    float32 before the one rounding."""
    x = (torch.randn(m, 2 * n, device=run_device()) * 8).to(dtype)
    scale = torch.rand(m, device=run_device()) * 1.5 if scaled else None
    workload = ElementwiseWorkload("SiLUAndMulFwdOp", (x, scale), swiglu_limit=swiglu_limit)
    TestBase.check(workload, SiLUAndMulFwdOp(swiglu_limit=swiglu_limit), x, scale)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_silu_and_mul_clamp_keeps_nan(dtype: torch.dtype) -> None:
    """Edge: a NaN gate or value passes the clamp, an infinite value clamps to the limit, and
    a gate of -inf stays -inf."""
    workload = SiLUAndMulClampEdgeWorkload(dtype)
    x, scale = workload.gen_inputs()
    TestBase.check(workload, SiLUAndMulFwdOp(swiglu_limit=10.0), x, scale)


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
                # Degenerate row off the vector width: the last active thread holds
                # elements on both sides of N, and the value half starts misaligned.
                pytest.param(1, 1001, torch.bfloat16, "none", marks=pytest.mark.smoke),
                # Latency-bound row wide enough to double its block width; the last block
                # holds one element and the value half starts off the vector width.
                pytest.param(1, 8193, torch.bfloat16, "none", marks=pytest.mark.smoke),
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
def test_gelu_and_mul_tuned() -> None:
    """Tuned mode picks a launch whose result matches the reference on a row off the vector width."""
    test = GELUAndMulTest(1, 1001, torch.bfloat16)
    op = GELUAndMulFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_and_mul_tails_are_exact(dtype: torch.dtype) -> None:
    """Edge: a gate past GELU's tails gives exactly 0 or exactly gate * value.

    The value scales GELU's residual from its saturated tail, so a tail off by eps is off
    by |gate * value| * eps / 2. Infinite and NaN gates ride on the same saturation.
    """
    workload = GELUAndMulTailWorkload(dtype)
    TestBase.check(workload, GELUAndMulFwdOp(), *workload.gen_inputs())


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
