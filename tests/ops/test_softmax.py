"""Correctness tests for softmax-family ops (softmax, log_softmax, logsumexp).

Tests cover fp32/fp16/bf16 dtypes, 1D-4D inputs, non-contiguous tensors,
power-of-2 and non-power-of-2 hidden dims, tail-M cases, and validate
against PyTorch reference implementations.

Smoke tests (1 per function, first param) use small data for quick CI.
Full tests use small data for config breadth + large data for stress.

All operators use the spec-conformant interface:
  SoftmaxOp(dtype=dtype, dim=dim)
  LogSoftmaxOp(dtype=dtype, dim=dim)
  LogSumExpOp(dtype=dtype, dim=dim, keepdim=keepdim)
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.reduction.log_softmax import LogSoftmaxOp
from tileops.ops.reduction.logsumexp import LogSumExpOp
from tileops.ops.reduction.softmax import SoftmaxOp
from workloads.ops.softmax import LogSoftmaxTest as _LogSoftmaxTestWorkload
from workloads.ops.softmax import LogSumExpTest as _LogSumExpTestWorkload
from workloads.ops.softmax import SoftmaxTest as _SoftmaxTestWorkload

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
                # dim=-1 (default path): dtypes x pow2/non-pow2
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, tail-M (non-aligned M)
                pytest.param((33, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, 3D input
                pytest.param((2, 16, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, 4D input
                pytest.param((2, 4, 8, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=0 (reduce along first dim — different M/N split)
                pytest.param((256, 32), 0, torch.float32, False, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.float16, False, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=1 (middle dim for 3D)
                pytest.param((2, 256, 16), 1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.bfloat16, False, marks=pytest.mark.full),
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
    op = SoftmaxOp(dtype=dtype, dim=dim, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# Softmax — non-contiguous input (spec interface)
# ===================================================================


class SoftmaxNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float32, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@SoftmaxNonContigFixture
def test_softmax_non_contiguous(shape: tuple, dtype: torch.dtype) -> None:
    """Test softmax with non-contiguous input (sliced tensor)."""
    m, n = shape
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice

    op = SoftmaxOp(dtype=dtype, dim=-1)

    y_ref = F.softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous softmax failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# Softmax — 1D input (spec interface)
# ===================================================================


class Softmax1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.full),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@Softmax1DFixture
def test_softmax_1d(n: int, dtype: torch.dtype) -> None:
    """Test softmax with 1D input (single row)."""
    x = torch.randn(n, dtype=dtype, device="cuda")
    op = SoftmaxOp(dtype=dtype, dim=-1)

    y_ref = F.softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D softmax failed, max err: {(y - y_ref).abs().max()}"
    )


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
                # dim=-1 (default path): dtypes x pow2/non-pow2
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, tail-M
                pytest.param((33, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, 3D input
                pytest.param((2, 16, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, 4D input
                pytest.param((2, 4, 8, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=0 (reduce along first dim)
                pytest.param((256, 32), 0, torch.float32, False, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.float16, False, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=1 (middle dim for 3D)
                pytest.param((2, 256, 16), 1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.bfloat16, False, marks=pytest.mark.full),
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
    op = LogSoftmaxOp(dtype=dtype, dim=dim, tune=tune)
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
                # dim=-1: dtypes x pow2/non-pow2
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((32, 300), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, tail-M
                pytest.param((33, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((33, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, 3D input
                pytest.param((2, 16, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, 4D input
                pytest.param((2, 4, 8, 256), -1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=0
                pytest.param((256, 32), 0, torch.float32, False, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.float16, False, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=1 (middle dim for 3D)
                pytest.param((2, 256, 16), 1, torch.float32, False, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((2, 256, 16), 1, torch.bfloat16, False, marks=pytest.mark.full),
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
    op = LogSumExpOp(dtype=dtype, dim=dim, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# LogSumExp — keepdim=True (exercises _reshape_output keepdim path)
# ===================================================================


class LogSumExpKeepdimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=-1 (last dim, no transpose)
                pytest.param((32, 256), -1, torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 16, 256), -1, torch.float32, marks=pytest.mark.full),
                # dim=0 (non-last dim, exercises transpose + keepdim)
                pytest.param((256, 32), 0, torch.float32, marks=pytest.mark.full),
                pytest.param((256, 32), 0, torch.float16, marks=pytest.mark.full),
                # dim=1 (middle dim, 3D)
                pytest.param((2, 256, 16), 1, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpKeepdimFixture
def test_logsumexp_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    """Test logsumexp with keepdim=True — output retains reduced dim as size 1."""
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = LogSumExpOp(dtype=dtype, dim=dim, keepdim=True)

    y_ref = torch.logsumexp(x.float(), dim=dim, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"keepdim logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# Non-contiguous input tests (spec interface)
# ===================================================================


class LogSoftmaxNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float32, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmaxNonContigFixture
def test_log_softmax_non_contiguous(shape: tuple, dtype: torch.dtype) -> None:
    """Test log_softmax with non-contiguous input (sliced tensor)."""
    m, n = shape
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]

    op = LogSoftmaxOp(dtype=dtype, dim=-1)

    y_ref = F.log_softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


class LogSumExpNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float32, marks=pytest.mark.full),
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpNonContigFixture
def test_logsumexp_non_contiguous(shape: tuple, dtype: torch.dtype) -> None:
    """Test logsumexp with non-contiguous input."""
    m, n = shape
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]

    op = LogSumExpOp(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# 1D input tests (spec interface)
# ===================================================================


class LogSoftmax1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.full),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmax1DFixture
def test_log_softmax_1d(n: int, dtype: torch.dtype) -> None:
    """Test log_softmax with 1D input."""
    x = torch.randn(n, dtype=dtype, device="cuda")
    op = LogSoftmaxOp(dtype=dtype, dim=-1)

    y_ref = F.log_softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


class LogSumExp1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.full),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExp1DFixture
def test_logsumexp_1d(n: int, dtype: torch.dtype) -> None:
    """Test logsumexp with 1D input -- output should be a scalar."""
    x = torch.randn(n, dtype=dtype, device="cuda")
    op = LogSumExpOp(dtype=dtype, dim=-1)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
