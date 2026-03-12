"""Tests for fused gated elementwise ops (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul).

Covers L1 smoke correctness and multi-dtype coverage.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import GeluAndMulOp, GeluTanhAndMulOp, SiluAndMulOp

# ---------------------------------------------------------------------------
# SiluAndMul
# ---------------------------------------------------------------------------


class SiluAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.full),
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
    op = SiluAndMulOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# GeluAndMul
# ---------------------------------------------------------------------------


class GeluAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.full),
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
    op = GeluAndMulOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# GeluTanhAndMul
# ---------------------------------------------------------------------------


class GeluTanhAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1024, 1024, torch.float32, marks=pytest.mark.full),
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
    op = GeluTanhAndMulOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
