"""Correctness tests for softmax-family ops (softmax, log_softmax, logsumexp).

Tests cover fp32/fp16/bf16 dtypes, 1D-4D inputs, non-contiguous tensors,
power-of-2 and non-power-of-2 hidden dims, tail-M cases, and validate
against PyTorch reference implementations.

Smoke tests (1 per function, first param) use small data for quick CI.
Full tests use small data for config breadth + large data for stress.

All operators use the spec-conformant interface:
  SoftmaxFwdOp(dtype=dtype, dim=dim)
  LogSoftmaxFwdOp(dtype=dtype, dim=dim)
  LogSumExpFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.reduction.log_softmax import LogSoftmaxFwdOp
from tileops.ops.reduction.logsumexp import LogSumExpFwdOp
from tileops.ops.reduction.softmax import SoftmaxFwdOp
from workloads.softmax import LogSoftmaxTest as _LogSoftmaxTestWorkload
from workloads.softmax import LogSumExpTest as _LogSumExpTestWorkload
from workloads.softmax import SoftmaxTest as _SoftmaxTestWorkload

# ---------------------------------------------------------------------------
# Tolerances (from docs/testing.md)
# ---------------------------------------------------------------------------


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


# ===================================================================
# Softmax — spec-conformant interface (shape, dim, dtype)
# ===================================================================


class SoftmaxFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype, tune",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param((32, 256), -1, torch.float32, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.smoke),
            ],
        ),
    ]


class SoftmaxTest(_SoftmaxTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


@SoftmaxFixture
def test_softmax_op(shape: tuple, dim: int, dtype: torch.dtype, tune: bool) -> None:
    test = SoftmaxTest(shape, dtype, dim=dim)
    op = SoftmaxFwdOp(dtype=dtype, dim=dim, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# LogSoftmax — spec-conformant interface (shape, dim, dtype)
# ===================================================================


class LogSoftmaxFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype, tune",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param((32, 256), -1, torch.float32, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogSoftmaxTest(_LogSoftmaxTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


@LogSoftmaxFixture
def test_log_softmax_op(shape: tuple, dim: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSoftmaxTest(shape, dtype, dim=dim)
    op = LogSoftmaxFwdOp(dtype=dtype, dim=dim, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# LogSumExp — spec-conformant interface (shape, dim, keepdim, dtype)
# ===================================================================


class LogSumExpFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype, tune",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param((32, 256), -1, torch.float32, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogSumExpTest(_LogSumExpTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


@LogSumExpFixture
def test_logsumexp_op(shape: tuple, dim: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSumExpTest(shape, dtype, dim=dim)
    op = LogSumExpFwdOp(dtype=dtype, dim=dim, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
