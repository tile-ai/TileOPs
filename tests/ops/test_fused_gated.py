"""Tests for fused gated elementwise ops (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul).

Covers L1 smoke correctness, multi-dtype coverage, and strategy selection.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.elementwise import (
    FusedGatedKernel,
    SiluAndMulFwdKernel,
)
from tileops.ops.elementwise import GeluAndMulFwdOp, GeluTanhAndMulFwdOp, SiluAndMulFwdOp

# ---------------------------------------------------------------------------
# SiluAndMul
# ---------------------------------------------------------------------------


class SiluAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
            pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
            pytest.param(2048, 2048, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


class SiluAndMulTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, 2 * self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate = x_f32[:, : self.n]
        value = x_f32[:, self.n :]
        return (F.silu(gate) * value).to(x.dtype)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@SiluAndMulFixture
def test_silu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = SiluAndMulTest(m, n, dtype)
    op = SiluAndMulFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# GeluAndMul
# ---------------------------------------------------------------------------


class GeluAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
            pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class GeluAndMulTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, 2 * self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate = x_f32[:, : self.n]
        value = x_f32[:, self.n :]
        return (F.gelu(gate) * value).to(x.dtype)


@GeluAndMulFixture
def test_gelu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype)
    op = GeluAndMulFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# GeluTanhAndMul
# ---------------------------------------------------------------------------


class GeluTanhAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
            pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class GeluTanhAndMulTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, 2 * self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate = x_f32[:, : self.n]
        value = x_f32[:, self.n :]
        return (F.gelu(gate, approximate="tanh") * value).to(x.dtype)


@GeluTanhAndMulFixture
def test_gelu_tanh_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluTanhAndMulTest(m, n, dtype)
    op = GeluTanhAndMulFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.full
def test_fused_gated_rejects_integer_dtype() -> None:
    """Fused gated ops are float-only and must reject integer dtypes early."""
    with pytest.raises(ValueError, match="does not support dtype"):
        GeluAndMulFwdOp(M=16, N=16, dtype=torch.int32)


@pytest.mark.full
def test_fused_gated_rejects_runtime_dtype_mismatch() -> None:
    """Runtime inputs should match the construction-time dtype contract."""
    op = SiluAndMulFwdOp(M=16, N=8, dtype=torch.float16)
    x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected x.dtype"):
        op(x)


# ---------------------------------------------------------------------------
# Strategy selection tests
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_fused_gated_kernel_has_strategies() -> None:
    """FusedGatedKernel must expose STRATEGIES and DEFAULT_STRATEGY class attrs."""
    assert hasattr(FusedGatedKernel, "STRATEGIES")
    assert hasattr(FusedGatedKernel, "DEFAULT_STRATEGY")
    assert "direct" in FusedGatedKernel.STRATEGIES
    assert "explicit_parallel" in FusedGatedKernel.STRATEGIES
    assert FusedGatedKernel.DEFAULT_STRATEGY in FusedGatedKernel.STRATEGIES


@pytest.mark.full
def test_fused_gated_kernel_rejects_unknown_strategy() -> None:
    """FusedGatedKernel must reject unknown strategy names."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        SiluAndMulFwdKernel(M=16, N=16, dtype=torch.float16, strategy="nonexistent")


class FusedGatedDirectStrategyFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@FusedGatedDirectStrategyFixture
def test_silu_and_mul_direct_strategy(m: int, n: int, dtype: torch.dtype) -> None:
    """SiluAndMul with strategy='direct' produces correct results."""
    test = SiluAndMulTest(m, n, dtype)
    op = SiluAndMulFwdOp(M=m, N=n, dtype=dtype, strategy="direct")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@FusedGatedDirectStrategyFixture
def test_gelu_and_mul_direct_strategy(m: int, n: int, dtype: torch.dtype) -> None:
    """GeluAndMul with strategy='direct' produces correct results."""
    test = GeluAndMulTest(m, n, dtype)
    op = GeluAndMulFwdOp(M=m, N=n, dtype=dtype, strategy="direct")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@FusedGatedDirectStrategyFixture
def test_gelu_tanh_and_mul_direct_strategy(m: int, n: int, dtype: torch.dtype) -> None:
    """GeluTanhAndMul with strategy='direct' produces correct results."""
    test = GeluTanhAndMulTest(m, n, dtype)
    op = GeluTanhAndMulFwdOp(M=m, N=n, dtype=dtype, strategy="direct")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.full
def test_fused_gated_default_strategy_is_explicit_parallel() -> None:
    """Default strategy for FusedGatedKernel should be explicit_parallel."""
    assert FusedGatedKernel.DEFAULT_STRATEGY == "explicit_parallel"


@pytest.mark.full
def test_fused_gated_kernel_stores_strategy() -> None:
    """FusedGatedKernel.strategy should record the chosen strategy."""
    k = SiluAndMulFwdKernel(M=16, N=16, dtype=torch.float16, strategy="direct")
    assert k.strategy == "direct"
    k2 = SiluAndMulFwdKernel(M=16, N=16, dtype=torch.float16)
    assert k2.strategy == FusedGatedKernel.DEFAULT_STRATEGY


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
