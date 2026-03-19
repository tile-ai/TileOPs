"""Correctness tests for softmax-family ops (softmax, log_softmax, logsumexp).

Tests cover fp32/fp16/bf16 dtypes, 1D-4D inputs, non-contiguous tensors,
power-of-2 and non-power-of-2 hidden dims, tail-M cases, and validate
against PyTorch reference implementations.

Smoke tests (1 per function, first param) use small data for quick CI.
Full tests use small data for config breadth + large data for stress.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.reduction.log_softmax import LogSoftmaxOp
from tileops.ops.reduction.logsumexp import LogSumExpOp
from tileops.ops.reduction.softmax import SoftmaxOp

# ---------------------------------------------------------------------------
# Tolerances (from DEVELOPMENT.md)
# ---------------------------------------------------------------------------


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


# ===================================================================
# Softmax — 2D
# ===================================================================


class SoftmaxFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Smoke: first param — small data, fp32, pow2 dim
                pytest.param(32, 256, torch.float32, False, marks=pytest.mark.smoke),
                # Full: all dtypes x pow2/non-pow2 x normal-M/tail-M (small data)
                pytest.param(32, 256, torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 256, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float32, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.float32, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.float16, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.float32, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.float16, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.bfloat16, False, marks=pytest.mark.full),
                # Full: larger configs
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


class SoftmaxTest(TestBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x.float(), dim=-1).to(x.dtype)


@SoftmaxFixture
def test_softmax_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = SoftmaxTest(m, n, dtype)
    op = SoftmaxOp(M=m, N=n, dtype=dtype, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# LogSoftmax — 2D
# ===================================================================


class LogSoftmaxFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                pytest.param(32, 256, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(32, 256, torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 256, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float32, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.float32, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.float16, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.float32, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.float16, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


class LogSoftmaxTest(TestBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x.float(), dim=-1).to(x.dtype)


@LogSoftmaxFixture
def test_log_softmax_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSoftmaxTest(m, n, dtype)
    op = LogSoftmaxOp(M=m, N=n, dtype=dtype, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# LogSumExp — 2D
# ===================================================================


class LogSumExpFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                pytest.param(32, 256, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(32, 256, torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 256, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float32, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 300, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.float32, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.float16, False, marks=pytest.mark.full),
                pytest.param(33, 256, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.float32, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.float16, False, marks=pytest.mark.full),
                pytest.param(33, 300, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


class LogSumExpTest(TestBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


@LogSumExpFixture
def test_logsumexp_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSumExpTest(m, n, dtype)
    op = LogSumExpOp(M=m, N=n, dtype=dtype, tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===================================================================
# Non-contiguous input tests — all 3 ops x all 3 dtypes
# ===================================================================


class NonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(32, 256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(32, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(32, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(32, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(32, 300, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@NonContigFixture
def test_softmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test softmax with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice

    op = SoftmaxOp(M=m, N=n, dtype=dtype)

    y_ref = F.softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@NonContigFixture
def test_log_softmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test log_softmax with non-contiguous input."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]

    op = LogSoftmaxOp(M=m, N=n, dtype=dtype)

    y_ref = F.log_softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@NonContigFixture
def test_logsumexp_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test logsumexp with non-contiguous input."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]

    op = LogSumExpOp(M=m, N=n, dtype=dtype)

    y_ref = torch.logsumexp(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# 1D input tests — all 3 ops x all 3 dtypes x pow2/non-pow2
# ===================================================================


class Input1DFixture(FixtureBase):
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


@Input1DFixture
def test_softmax_1d(n: int, dtype: torch.dtype) -> None:
    """Test softmax with 1D input (single row)."""
    x = torch.randn(n, dtype=dtype, device="cuda")
    op = SoftmaxOp(M=1, N=n, dtype=dtype)

    y_ref = F.softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@Input1DFixture
def test_log_softmax_1d(n: int, dtype: torch.dtype) -> None:
    """Test log_softmax with 1D input."""
    x = torch.randn(n, dtype=dtype, device="cuda")
    op = LogSoftmaxOp(M=1, N=n, dtype=dtype)

    y_ref = F.log_softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@Input1DFixture
def test_logsumexp_1d(n: int, dtype: torch.dtype) -> None:
    """Test logsumexp with 1D input -- output should be a scalar."""
    x = torch.randn(n, dtype=dtype, device="cuda")
    op = LogSumExpOp(M=1, N=n, dtype=dtype)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"1D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# 3D input tests — all 3 ops x all 3 dtypes x pow2/non-pow2
# ===================================================================


class Input3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 16, 256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 16, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 16, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 16, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(2, 16, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 16, 300, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@Input3DFixture
def test_softmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test softmax with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = SoftmaxOp(M=M, N=hidden, dtype=dtype)

    y_ref = F.softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"3D softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@Input3DFixture
def test_log_softmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test log_softmax with 3D input."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = LogSoftmaxOp(M=M, N=hidden, dtype=dtype)

    y_ref = F.log_softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"3D log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@Input3DFixture
def test_logsumexp_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test logsumexp with 3D input -- output shape should be (batch, seq)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = LogSumExpOp(M=M, N=hidden, dtype=dtype)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"3D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# 4D input tests — all 3 ops x all 3 dtypes x pow2/non-pow2
# ===================================================================


class Input4DFixture(FixtureBase):
    PARAMS = [
        (
            "b, h, s, d, dtype",
            [
                pytest.param(2, 4, 8, 256, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 300, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 256, torch.float32, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 256, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@Input4DFixture
def test_softmax_4d(b: int, h: int, s: int, d: int, dtype: torch.dtype) -> None:
    """Test softmax with 4D input (batch, heads, seq, dim)."""
    x = torch.randn(b, h, s, d, dtype=dtype, device="cuda")
    M = b * h * s
    op = SoftmaxOp(M=M, N=d, dtype=dtype)

    y_ref = F.softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"4D softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@Input4DFixture
def test_log_softmax_4d(b: int, h: int, s: int, d: int, dtype: torch.dtype) -> None:
    """Test log_softmax with 4D input (batch, heads, seq, dim)."""
    x = torch.randn(b, h, s, d, dtype=dtype, device="cuda")
    M = b * h * s
    op = LogSoftmaxOp(M=M, N=d, dtype=dtype)

    y_ref = F.log_softmax(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"4D log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


@Input4DFixture
def test_logsumexp_4d(b: int, h: int, s: int, d: int, dtype: torch.dtype) -> None:
    """Test logsumexp with 4D input -- output shape should be (b, h, s)."""
    x = torch.randn(b, h, s, d, dtype=dtype, device="cuda")
    M = b * h * s
    op = LogSumExpOp(M=M, N=d, dtype=dtype)

    y_ref = torch.logsumexp(x.float(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert y.shape == y_ref.shape, f"Shape mismatch: {y.shape} vs {y_ref.shape}"
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"4D logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
