"""Correctness tests for softmax-family ops (softmax, log_softmax, logsumexp).

Tests cover fp32/fp16/bf16 dtypes, 1D-4D inputs, non-contiguous tensors,
power-of-2 and non-power-of-2 hidden dims, tail-M cases, and validate
against PyTorch reference implementations.

Smoke tests (1 per function, first param) use small data for quick CI.
Full tests use small data for config breadth + large data for stress.

All operators use the spec-conformant interface:
  SoftmaxFwdOp(dim=dim)
  LogSoftmaxFwdOp(dim=dim)
  LogSumExpFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs

# Tolerances (from docs/design/testing.md)
# Softmax — spec-conformant interface (shape, dim, dtype)
from workloads.reduction import (
    LogSoftmaxCase,
    LogSumExpCase,
    SoftmaxCase,
    softmax_verification,
)


class SoftmaxFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype, tune",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param(
                    (32, 256),
                    -1,
                    torch.float32,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="reduction")],
                ),
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.smoke),
                # tune=True regression: kernel must be built before autotune runs
                pytest.param((32, 256), -1, torch.float16, True, marks=pytest.mark.full),
                # dim=-1 (default path): dtypes x pow2/non-pow2
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
                # dim=-1, a few long rows (split kernel)
                pytest.param((4, 32768), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((4, 32768), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned (single-tile path)
                pytest.param((33, 300), -1, torch.float32, False, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned, long rows (split kernel)
                pytest.param((33, 33000), -1, torch.float16, False, marks=pytest.mark.full),
                # dim=-1, non-aligned M, long aligned rows (split kernel)
                pytest.param((33, 32768), -1, torch.float16, False, marks=pytest.mark.full),
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


class SoftmaxTest(SoftmaxCase, TestBase):
    pass


@SoftmaxFixture
def test_softmax_op(shape: tuple, dim: int, dtype: torch.dtype, tune: bool) -> None:
    test = SoftmaxTest(shape, dtype, dim=dim)
    op = SoftmaxFwdOp(dim=dim, tune=tune)
    test.check(op, *test.gen_inputs())


# Softmax — non-contiguous input (spec interface)


class SoftmaxNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.smoke),
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
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice

    op = SoftmaxFwdOp(dim=-1)

    y_ref = F.softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=False))


# Softmax — 1D input (spec interface)


class Softmax1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@Softmax1DFixture
def test_softmax_1d(n: int, dtype: torch.dtype) -> None:
    """Test softmax with 1D input (single row)."""
    x = torch.randn(n, dtype=dtype, device=run_device())
    op = SoftmaxFwdOp(dim=-1)

    y_ref = F.softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=False))


# LogSoftmax — spec-conformant interface (shape, dim, dtype)


class LogSoftmaxFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype, tune",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param(
                    (32, 256),
                    -1,
                    torch.float32,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="reduction")],
                ),
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.smoke),
                # tune=True regression: kernel must be built before autotune runs
                pytest.param((32, 256), -1, torch.float16, True, marks=pytest.mark.full),
                # dim=-1 (default path): dtypes x pow2/non-pow2
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
                # dim=-1, a few long rows (split kernel)
                pytest.param((4, 32768), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((4, 32768), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned (single-tile path)
                pytest.param((33, 300), -1, torch.float32, False, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned, long rows (split kernel)
                pytest.param((33, 33000), -1, torch.float16, False, marks=pytest.mark.full),
                # dim=-1, non-aligned M, long aligned rows (split kernel)
                pytest.param((33, 32768), -1, torch.float16, False, marks=pytest.mark.full),
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


class LogSoftmaxTest(LogSoftmaxCase, TestBase):
    pass


@LogSoftmaxFixture
def test_log_softmax_op(shape: tuple, dim: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSoftmaxTest(shape, dtype, dim=dim)
    op = LogSoftmaxFwdOp(dim=dim, tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "op_cls, ref_fn, shape",
    [
        pytest.param(SoftmaxFwdOp, F.softmax, (8, 1000), marks=pytest.mark.smoke, id="single"),
        pytest.param(
            LogSoftmaxFwdOp, F.log_softmax, (512, 40000), marks=pytest.mark.full, id="streaming"
        ),
        # An odd row: no access wider than one element starts every row.
        pytest.param(
            SoftmaxFwdOp, F.softmax, (512, 40001), marks=pytest.mark.full, id="streaming-odd-width"
        ),
        pytest.param(
            LogSoftmaxFwdOp, F.log_softmax, (64, 40000), marks=pytest.mark.full, id="split"
        ),
        pytest.param(
            SoftmaxFwdOp, F.softmax, (4, 128256), marks=pytest.mark.full, id="fused-split"
        ),
        pytest.param(SoftmaxFwdOp, F.softmax, (264, 32768), marks=pytest.mark.full, id="on-chip"),
    ],
)
def test_softmax_dtype_widens_in_kernel(op_cls, ref_fn, shape: tuple) -> None:
    """A float32 ``dtype`` on a bfloat16 input matches torch on every kernel path."""
    x = torch.randn(shape, device=run_device()).to(torch.bfloat16)
    y = op_cls(dim=-1, dtype=torch.float32)(x)
    assert y.dtype == torch.float32
    # Relative only: softmax values sit below any absolute floor that would let a
    # bfloat16-rounded output pass.
    compare_outputs(
        y,
        ref_fn(x, dim=-1, dtype=torch.float32),
        softmax_verification(y.dtype, input_dtype=x.dtype, logarithmic=op_cls is LogSoftmaxFwdOp),
    )


@pytest.mark.parametrize(
    "case_cls, op_cls, dim, shape",
    [
        pytest.param(
            SoftmaxCase, SoftmaxFwdOp, -1, (8, 1000), marks=pytest.mark.smoke, id="softmax"
        ),
        pytest.param(
            LogSumExpCase, LogSumExpFwdOp, -1, (32, 1000), marks=pytest.mark.full, id="logsumexp"
        ),
    ],
)
def test_contiguous_input_off_a_vector_boundary(case_cls, op_cls, dim, shape: tuple) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    x = torch.randn(math.prod(shape) + 1, device=run_device(), dtype=torch.float16)[1:]
    x = x.view(shape)
    case = case_cls(shape, x.dtype, dim=dim)
    compare_outputs(op_cls(dim=dim)(x), case.ref_program(x), case.verification(x))


# LogSumExp — spec-conformant interface (shape, dim, keepdim, dtype)


class LogSumExpFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype, tune",
            [
                # Smoke: 2D, dim=-1, fp32, pow2
                pytest.param(
                    (32, 256),
                    -1,
                    torch.float32,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="reduction")],
                ),
                pytest.param((32, 256), -1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.bfloat16, False, marks=pytest.mark.smoke),
                # tune=True regression: kernel must be built before autotune runs
                pytest.param((32, 256), -1, torch.float16, True, marks=pytest.mark.full),
                # dim=-1: dtypes x pow2/non-pow2
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
                # dim=-1, large-N (triggers N-tiling path)
                pytest.param((4, 32768), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((4, 32768), -1, torch.bfloat16, False, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned (single-tile path)
                pytest.param((33, 300), -1, torch.float32, False, marks=pytest.mark.full),
                # dim=-1, M×N both non-aligned (multi-tile, masked loads)
                pytest.param((33, 33000), -1, torch.float16, False, marks=pytest.mark.full),
                # dim=-1, non-aligned M + large-N tiled path
                pytest.param((33, 32768), -1, torch.float16, False, marks=pytest.mark.full),
                # dim=-1, long rows on a filled grid (streaming kernel)
                pytest.param((256, 16384), -1, torch.float16, False, marks=pytest.mark.full),
                pytest.param((256, 16384), -1, torch.bfloat16, False, marks=pytest.mark.full),
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


class LogSumExpTest(LogSumExpCase, TestBase):
    pass


@LogSumExpFixture
def test_logsumexp_op(shape: tuple, dim: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSumExpTest(shape, dtype, dim=dim)
    op = LogSumExpFwdOp(dim=dim, tune=tune)
    test.check(op, *test.gen_inputs())


# LogSumExp — keepdim=True (exercises _reshape_output keepdim path)


class LogSumExpKeepdimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=-1 (last dim, no transpose)
                pytest.param((32, 256), -1, torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), -1, torch.float16, marks=pytest.mark.smoke),
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
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = LogSumExpFwdOp(dim=dim, keepdim=True)

    y_ref = torch.logsumexp(x.float(), dim=dim, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape, dim, dtype",
    [
        pytest.param((256, 16384), -1, torch.bfloat16, id="streaming"),
        pytest.param((64, 4096), -1, torch.float16, id="single-tile"),
        pytest.param((300, 100000), -1, torch.float32, id="tiled"),
        pytest.param((8, 102400), -1, torch.float32, id="split"),
        pytest.param((4, 128, 4096), [0, 2], torch.float16, id="edge-axes"),
    ],
)
def test_logsumexp_special_values(shape: tuple, dim, dtype: torch.dtype) -> None:
    """Every kernel path keeps torch's -inf, +inf, and NaN row semantics."""
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    # Row r of the reduced output is x[r] for a trailing dim, x[:, r] for edge axes.
    rows = x if dim == -1 else x.transpose(0, 1)
    rows[0] = float("-inf")
    rows[1] = float("-inf")
    rows[1][..., 7] = 2.0
    rows[2][..., ::2] = float("-inf")
    rows[3][..., 100] = float("nan")
    rows[4][..., 200] = float("inf")
    rows[5][..., 200] = float("inf")
    rows[5][..., 300] = float("nan")
    # A NaN in a run of -inf, far from the row's one finite or +inf value.
    first, final = (0,) * rows[6].dim(), (-1,) * rows[6].dim()
    rows[6] = float("-inf")
    rows[6][first] = float("nan")
    rows[6][final] = -10.0
    rows[7] = float("-inf")
    rows[7][first] = float("nan")
    rows[7][final] = float("inf")

    y = LogSumExpFwdOp(dim=dim)(x).float()
    y_ref = torch.logsumexp(x.float(), dim=dim)
    assert y[0].item() == float("-inf")
    assert torch.isnan(y[3])
    assert y[4].item() == float("inf")
    assert torch.isnan(y[5])
    assert torch.isnan(y[6])
    assert torch.isnan(y[7])
    finite = torch.isfinite(y_ref)
    compare_outputs(y[finite], y_ref[finite], softmax_verification(x.dtype, logarithmic=True))


# Non-contiguous input tests (spec interface)


class LogSoftmaxNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.smoke),
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
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]

    op = LogSoftmaxFwdOp(dim=-1)

    y_ref = F.log_softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


class LogSumExpNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((32, 256), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.smoke),
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
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]

    op = LogSumExpFwdOp(dim=-1)

    y_ref = torch.logsumexp(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


# 1D input tests (spec interface)


class LogSoftmax1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmax1DFixture
def test_log_softmax_1d(n: int, dtype: torch.dtype) -> None:
    """Test log_softmax with 1D input."""
    x = torch.randn(n, dtype=dtype, device=run_device())
    op = LogSoftmaxFwdOp(dim=-1)

    y_ref = F.log_softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


class LogSumExp1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(256, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(300, torch.float32, marks=pytest.mark.full),
                pytest.param(300, torch.float16, marks=pytest.mark.full),
                pytest.param(300, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExp1DFixture
def test_logsumexp_1d(n: int, dtype: torch.dtype) -> None:
    """Test logsumexp with 1D input -- output should be a scalar."""
    x = torch.randn(n, dtype=dtype, device=run_device())
    op = LogSumExpFwdOp(dim=-1)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [SoftmaxFwdOp, LogSoftmaxFwdOp])
def test_softmax_rejects_a_sequence_dim_at_construction(op_cls) -> None:
    """``dim`` is one axis, as in torch."""
    with pytest.raises(ValueError, match="dim = "):
        op_cls(dim=[-1, 0])


@pytest.mark.smoke
def test_logsumexp_accepts_multidim() -> None:
    """LogSumExpFwdOp must accept list dim without error (multi-dim is supported)."""
    x = torch.randn(4, 8, device=run_device(), dtype=torch.float32)
    op = LogSumExpFwdOp(dim=[0, 1])
    y = op(x)
    y_ref = torch.logsumexp(x.float(), dim=[0, 1])
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


class SoftmaxImplicitDimFixture(FixtureBase):
    # Smoke covers each ndim branch (1D, 2D, 3D) and each dtype at least once.
    PARAMS = [
        (
            "shape, dtype",
            [
                pytest.param((256,), torch.float32, marks=pytest.mark.smoke),
                pytest.param((32, 256), torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 16, 32), torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@SoftmaxImplicitDimFixture
def test_softmax_dim_none_implicit_axis(shape: tuple, dtype: torch.dtype) -> None:
    """SoftmaxFwdOp(dim=None) must match F.softmax(x, dim=None) and warn."""
    import warnings as _warnings

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = SoftmaxFwdOp(dim=None)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        y = op(x)
        assert any(
            issubclass(w.category, UserWarning) and "Implicit dimension choice" in str(w.message)
            for w in caught
        ), f"Expected implicit-dim UserWarning, got {[str(w.message) for w in caught]}"

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        y_ref = F.softmax(x.float(), dim=None).to(dtype)

    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=False))


@SoftmaxImplicitDimFixture
def test_log_softmax_dim_none_implicit_axis(shape: tuple, dtype: torch.dtype) -> None:
    """LogSoftmaxFwdOp(dim=None) must match F.log_softmax(x, dim=None) and warn."""
    import warnings as _warnings

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = LogSoftmaxFwdOp(dim=None)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        y = op(x)
        assert any(
            issubclass(w.category, UserWarning) and "Implicit dimension choice" in str(w.message)
            for w in caught
        ), f"Expected implicit-dim UserWarning, got {[str(w.message) for w in caught]}"

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        y_ref = F.log_softmax(x.float(), dim=None).to(dtype)

    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


@pytest.mark.smoke
def test_softmax_dim_none_reused_across_ranks() -> None:
    """SoftmaxFwdOp(dim=None) must re-resolve per call across input ranks."""
    import warnings as _warnings

    op = SoftmaxFwdOp(dim=None)

    x1 = torch.randn(4, dtype=torch.float32, device=run_device())
    x2 = torch.randn(2, 4, dtype=torch.float32, device=run_device())
    x3 = torch.randn(4, 3, 5, dtype=torch.float32, device=run_device())

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        y1 = op(x1)
        y2 = op(x2)
        y3 = op(x3)
        y1_ref = F.softmax(x1.float(), dim=None)
        y2_ref = F.softmax(x2.float(), dim=None)
        y3_ref = F.softmax(x3.float(), dim=None)

    assert op.dim is None, f"op.dim was mutated to {op.dim!r}; expected None"

    compare_outputs(y1, y1_ref, softmax_verification(y1.dtype, logarithmic=False))
    compare_outputs(y2, y2_ref, softmax_verification(y2.dtype, logarithmic=False))
    compare_outputs(y3, y3_ref, softmax_verification(y3.dtype, logarithmic=False))


@pytest.mark.smoke
def test_log_softmax_dim_none_reused_across_ranks() -> None:
    """LogSoftmaxFwdOp(dim=None) must re-resolve per call (no self.dim mutation)."""
    import warnings as _warnings

    op = LogSoftmaxFwdOp(dim=None)

    x1 = torch.randn(4, dtype=torch.float32, device=run_device())
    x2 = torch.randn(2, 4, dtype=torch.float32, device=run_device())
    x3 = torch.randn(4, 3, 5, dtype=torch.float32, device=run_device())

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        y1 = op(x1)
        y2 = op(x2)
        y3 = op(x3)
        y1_ref = F.log_softmax(x1.float(), dim=None)
        y2_ref = F.log_softmax(x2.float(), dim=None)
        y3_ref = F.log_softmax(x3.float(), dim=None)

    assert op.dim is None, f"op.dim was mutated to {op.dim!r}; expected None"

    compare_outputs(y1, y1_ref, softmax_verification(y1.dtype, logarithmic=True))
    compare_outputs(y2, y2_ref, softmax_verification(y2.dtype, logarithmic=True))
    compare_outputs(y3, y3_ref, softmax_verification(y3.dtype, logarithmic=True))


# Roofline regression: LogSoftmax FLOPs must equal 5 * M * N (not 6 * M * N).
# Direct construction — no manifest-string indirection.


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_log_softmax_eval_roofline_flops_5mn() -> None:
    """LogSoftmaxFwdOp.eval_roofline() must report flops == 5 * M * N."""
    M, N = 64, 256
    dtype = torch.float16
    op = LogSoftmaxFwdOp(dim=-1)
    x = torch.randn(M, N, dtype=dtype, device=run_device())
    op(x)  # bind dynamic shape
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 5 * M * N, f"LogSoftmax flops {flops} != 5 * M * N = {5 * M * N}"
    assert mem_bytes == 2 * M * N * elem_bytes, (
        f"LogSoftmax bytes {mem_bytes} != 2 * M * N * elem_bytes = {2 * M * N * elem_bytes}"
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_softmax_few_long_rows_split_across_blocks(dtype: torch.dtype) -> None:
    """A handful of long rows runs as segments plus a fold, not one block per row.

    102400 is not a segment multiple, so the tail segment's masked lanes
    contribute ``exp(-inf) = 0`` to the fold.
    """
    x = torch.randn(4, 102400, dtype=dtype, device=run_device())

    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=False),
    )
    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x),
        F.log_softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_logsumexp_few_long_rows_split_across_blocks() -> None:
    """logsumexp folds the shared segment statistics without re-reading the input."""
    x = torch.randn(4, 102400, dtype=torch.float32, device=run_device())
    compare_outputs(
        LogSumExpFwdOp(dim=-1)(x),
        torch.logsumexp(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_split_rows_survive_fully_masked_segments() -> None:
    """A segment of only ``-inf`` contributes zero to the fold, not NaN.

    Masked inputs make whole segments ``-inf``; the segment-local max is then
    ``-inf`` and an unguarded ``exp(-inf - -inf)`` would poison rows the
    single-block path computes fine. An all--inf row stays torch: NaN for
    softmax, ``-inf`` for logsumexp.
    """
    x = torch.randn(4, 102400, dtype=torch.float32, device=run_device())
    x[:, :4096] = float("-inf")  # first segments fully masked, rest finite
    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=False),
    )
    compare_outputs(
        LogSumExpFwdOp(dim=-1)(x),
        torch.logsumexp(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )

    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x),
        F.log_softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )

    x[0, :] = float("-inf")
    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=False),
    )
    compare_outputs(
        LogSumExpFwdOp(dim=-1)(x),
        torch.logsumexp(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
def test_large_row_shifts_its_maximum_to_exactly_one() -> None:
    """A row of large values still shifts its maximum to exactly 1 before the sum.

    The shift is applied in base 2. Scaling the value and the shift separately and
    subtracting inside the product lets the pair contract into one ``FFMA``, which keeps
    the value's product exact against an already-rounded shift: equal operands stop
    cancelling, and the error grows with the row's magnitude. Every manifest row draws
    from ``randn``, which holds ``|x|`` under about 5 and cannot show it, so the rows
    here are scaled up and the second one repeats its maximum.

    ``log_softmax`` is the whole guard for the shared ``exp_shifted`` helper. The other
    two ops cannot show the fault at any tolerance: ``softmax`` divides every term by
    their sum, which takes the error out with it, and ``logsumexp`` returns a value of
    the row's own magnitude, whose fp32 spacing against a float64 reference is two
    orders larger than the fault and identical with or without it.
    """
    device = run_device()
    torch.manual_seed(1235)
    x = torch.randn((2, 4096), dtype=torch.float32, device=device) * 1e5
    x[1, ::512] = x[1].max()

    expected = F.log_softmax(x.double(), dim=-1).to(torch.float32)
    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x), expected, softmax_verification(x.dtype, logarithmic=True)
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op, reference",
    [
        pytest.param(SoftmaxFwdOp, lambda t: F.softmax(t, dim=-1), id="softmax"),
        pytest.param(LogSoftmaxFwdOp, lambda t: F.log_softmax(t, dim=-1), id="log_softmax"),
    ],
)
@pytest.mark.parametrize(
    "width, prefix",
    [
        # 16384-element tiles on the streaming kernel: the first tile and most of the
        # second are the -inf prefix.
        pytest.param(40000, 28672, id="streaming"),
        # Eight CTAs a row on the on-chip kernel: the first CTA's share is the prefix.
        pytest.param(40960, 5120, id="on-chip"),
    ],
)
def test_leading_tiles_of_only_neg_inf_do_not_poison_a_row(
    op: type, reference, width: int, prefix: int
) -> None:
    """A row whose leading part holds only ``-inf`` still reduces its finite tail.

    The streaming path folds one tile at a time and rescales the running sum by the change
    in the row maximum; the on-chip path folds the partials of a row's CTAs. While a
    maximum is still ``-inf`` the rescale subtracts one infinity from another, and the
    ``NaN`` reaches the rest of the row. The second row is ``-inf`` throughout and must
    still come back as torch returns it.
    """
    device = run_device()
    torch.manual_seed(1235)
    # Too few rows dispatch to the split kernel instead.
    x = torch.randn((264, width), dtype=torch.float32, device=device)
    x[0, :prefix] = float("-inf")
    x[1, :] = float("-inf")

    compare_outputs(
        op(dim=-1)(x),
        reference(x),
        softmax_verification(x.dtype, logarithmic=op is LogSoftmaxFwdOp),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op, reference, shape, dtype",
    [
        # 32 KB shares: 16 KB in registers, the rest staged, four CTAs an SM.
        pytest.param(SoftmaxFwdOp, F.softmax, (264, 32768), torch.bfloat16, id="share-32k"),
        # 64 KB shares: 32 KB in registers, the rest staged, three CTAs an SM.
        pytest.param(LogSoftmaxFwdOp, F.log_softmax, (264, 65536), torch.float16, id="share-64k"),
        # Eight CTAs a row, each past the share a cluster is aimed at.
        pytest.param(SoftmaxFwdOp, F.softmax, (264, 262144), torch.float32, id="widest"),
    ],
)
def test_rows_read_once_across_a_cluster(op, reference, shape: tuple, dtype) -> None:
    """Rows the on-chip kernel splits across a cluster match torch."""
    torch.manual_seed(1235)
    x = torch.randn(shape, dtype=dtype, device=run_device()) * 4
    compare_outputs(
        op(dim=-1)(x),
        reference(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=op is LogSoftmaxFwdOp),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, low",
    [
        pytest.param(torch.float32, -100.0, id="float32"),
        pytest.param(torch.bfloat16, -88.0, id="bfloat16"),
    ],
)
def test_subnormal_probabilities_survive(dtype, low: float) -> None:
    """A probability below the smallest normal float comes back as torch returns it.

    Every row is one zero and the rest ``low``, so all but one probability is subnormal
    in float32. A tolerance check cannot see them flushed to zero, so they are compared
    exactly against torch.
    """
    x = torch.full((264, 65536), low, dtype=dtype, device=run_device())
    x[:, 0] = 0
    y = SoftmaxFwdOp(dim=-1)(x)
    expected = F.softmax(x, dim=-1)
    assert torch.equal(y[:, 1:] != 0, expected[:, 1:] != 0)
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=0)


_H200 = {"arch": 90, "sm_count": 132, "smem_budget": 232448}
