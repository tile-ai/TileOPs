"""Correctness tests for softmax-family ops (softmax, log_softmax, logsumexp).

Tests cover fp16/bf16 dtypes, 2D-4D inputs, non-contiguous tensors,
multi-dim reduction, power-of-2 and non-power-of-2 hidden dims, and validate
against PyTorch reference implementations.

Softmax, LogSoftmax, and LogSumExp tests target the manifest-declared interface:
  SoftmaxOp(dim=-1, dtype=...) / LogSoftmaxOp(dim=-1, dtype=...)
  LogSumExpOp(dim=-1, keepdim=False, dtype=...)
  with arbitrary-rank input tensors.

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
# Softmax — spec interface: SoftmaxOp(dim, dtype)
# ===================================================================


class SoftmaxFixture2D(FixtureBase):
    """2D softmax: pow2/non-pow2 hidden dim x fp16/bf16."""

    PARAMS = [
        (
            "shape, dtype",
            [
                # Smoke: small 2D, fp16, pow2
                pytest.param((32, 256), torch.float16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                # Full: dtype coverage
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.full),
                # Full: non-pow2 hidden dim
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
                # Full: tail-M (odd batch)
                pytest.param((33, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((33, 256), torch.bfloat16, marks=pytest.mark.full),
                # Full: larger LLM-scale shapes
                pytest.param((1024, 4096), torch.float16, marks=pytest.mark.full),
                pytest.param((1024, 4096), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((1024, 3000), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class SoftmaxTest(TestBase):
    """TestBase adapter for spec-interface SoftmaxOp."""

    def __init__(self, shape: tuple[int, ...], dim: int, dtype: torch.dtype):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x.float(), dim=self.dim).to(x.dtype)


@SoftmaxFixture2D
def test_softmax_2d(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Softmax on 2D input with dim=-1 (default)."""
    dim = -1
    test = SoftmaxTest(shape, dim, dtype)
    op = SoftmaxOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class SoftmaxFixtureND(FixtureBase):
    """3D/4D softmax: multi-rank inputs x fp16/bf16."""

    PARAMS = [
        (
            "shape, dtype",
            [
                # 3D smoke
                pytest.param((2, 16, 256), torch.float16, marks=pytest.mark.smoke),
                # 3D full
                pytest.param((2, 16, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((2, 16, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 512, 4096), torch.float16, marks=pytest.mark.full),
                # 4D: attention-score shape (batch, heads, seq, seq)
                pytest.param((2, 4, 8, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 128, 256), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@SoftmaxFixtureND
def test_softmax_nd(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Softmax on 3D/4D input with dim=-1 (default)."""
    dim = -1
    test = SoftmaxTest(shape, dim, dtype)
    op = SoftmaxOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class SoftmaxFixtureNonLastDim(FixtureBase):
    """Softmax along non-last dimensions (dim=0, dim=1)."""

    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=0: reduce along batch
                pytest.param((16, 256), 0, torch.float16, marks=pytest.mark.smoke),
                pytest.param((16, 256), 0, torch.bfloat16, marks=pytest.mark.full),
                # dim=1: reduce along middle axis of 3D
                pytest.param((2, 16, 256), 1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 16, 256), 1, torch.bfloat16, marks=pytest.mark.full),
                # dim=0 on 3D
                pytest.param((4, 16, 256), 0, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@SoftmaxFixtureNonLastDim
def test_softmax_non_last_dim(shape: tuple[int, ...], dim: int, dtype: torch.dtype) -> None:
    """Softmax along a non-last dimension — tests dim param generality."""
    test = SoftmaxTest(shape, dim, dtype)
    op = SoftmaxOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class SoftmaxFixtureNonContig(FixtureBase):
    """Non-contiguous input coverage for softmax."""

    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(32, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@SoftmaxFixtureNonContig
def test_softmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Softmax with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice

    op = SoftmaxOp(dim=-1, dtype=dtype)

    y_ref = F.softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous softmax failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# LogSoftmax — spec interface: LogSoftmaxOp(dim, dtype)
# ===================================================================


class LogSoftmaxTest(TestBase):
    """TestBase adapter for spec-interface LogSoftmaxOp."""

    def __init__(self, shape: tuple[int, ...], dim: int, dtype: torch.dtype):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x.float(), dim=self.dim).to(x.dtype)


class LogSoftmaxFixture2D(FixtureBase):
    """2D log_softmax: pow2/non-pow2 hidden dim x fp16/bf16."""

    PARAMS = [
        (
            "shape, dtype",
            [
                # Smoke: small 2D, fp16, pow2
                pytest.param((32, 256), torch.float16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                # Full: dtype coverage
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.full),
                # Full: non-pow2 hidden dim
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
                # Full: tail-M (odd batch)
                pytest.param((33, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((33, 256), torch.bfloat16, marks=pytest.mark.full),
                # Full: larger LLM-scale shapes
                pytest.param((1024, 4096), torch.float16, marks=pytest.mark.full),
                pytest.param((1024, 4096), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((1024, 3000), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmaxFixture2D
def test_log_softmax_2d(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """LogSoftmax on 2D input with dim=-1 (default)."""
    dim = -1
    test = LogSoftmaxTest(shape, dim, dtype)
    op = LogSoftmaxOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSoftmaxFixtureND(FixtureBase):
    """3D/4D log_softmax: multi-rank inputs x fp16/bf16."""

    PARAMS = [
        (
            "shape, dtype",
            [
                # 3D smoke
                pytest.param((2, 16, 256), torch.float16, marks=pytest.mark.smoke),
                # 3D full
                pytest.param((2, 16, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((2, 16, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 512, 4096), torch.float16, marks=pytest.mark.full),
                # 4D: attention-score shape (batch, heads, seq, seq)
                pytest.param((2, 4, 8, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 128, 256), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmaxFixtureND
def test_log_softmax_nd(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """LogSoftmax on 3D/4D input with dim=-1 (default)."""
    dim = -1
    test = LogSoftmaxTest(shape, dim, dtype)
    op = LogSoftmaxOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSoftmaxFixtureNonLastDim(FixtureBase):
    """LogSoftmax along non-last dimensions (dim=0, dim=1)."""

    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=0: reduce along batch
                pytest.param((16, 256), 0, torch.float16, marks=pytest.mark.smoke),
                pytest.param((16, 256), 0, torch.bfloat16, marks=pytest.mark.full),
                # dim=1: reduce along middle axis of 3D
                pytest.param((2, 16, 256), 1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 16, 256), 1, torch.bfloat16, marks=pytest.mark.full),
                # dim=0 on 3D
                pytest.param((4, 16, 256), 0, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmaxFixtureNonLastDim
def test_log_softmax_non_last_dim(shape: tuple[int, ...], dim: int, dtype: torch.dtype) -> None:
    """LogSoftmax along a non-last dimension -- tests dim param generality."""
    test = LogSoftmaxTest(shape, dim, dtype)
    op = LogSoftmaxOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSoftmaxFixtureNonContig(FixtureBase):
    """Non-contiguous input coverage for log_softmax."""

    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(32, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSoftmaxFixtureNonContig
def test_log_softmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """LogSoftmax with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice

    op = LogSoftmaxOp(dim=-1, dtype=dtype)

    y_ref = F.log_softmax(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous log_softmax failed, max err: {(y - y_ref).abs().max()}"
    )


# ===================================================================
# LogSumExp — spec interface: LogSumExpOp(dim, keepdim, dtype)
# ===================================================================


class LogSumExpTest(TestBase):
    """TestBase adapter for spec-interface LogSumExpOp."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        dtype: torch.dtype,
        keepdim: bool = False,
    ):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype
        self.keepdim = keepdim

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim, keepdim=self.keepdim).to(
            x.dtype
        )


class LogSumExpFixture2D(FixtureBase):
    """2D logsumexp: pow2/non-pow2 hidden dim x fp16/bf16."""

    PARAMS = [
        (
            "shape, dtype",
            [
                # Smoke: small 2D, fp16, pow2
                pytest.param((32, 256), torch.float16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                # Full: dtype coverage
                pytest.param((32, 256), torch.bfloat16, marks=pytest.mark.full),
                # Full: non-pow2 hidden dim
                pytest.param((32, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((32, 300), torch.bfloat16, marks=pytest.mark.full),
                # Full: tail-M (odd batch)
                pytest.param((33, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((33, 256), torch.bfloat16, marks=pytest.mark.full),
                # Full: larger LLM-scale shapes
                pytest.param((1024, 4096), torch.float16, marks=pytest.mark.full),
                pytest.param((1024, 4096), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((1024, 3000), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpFixture2D
def test_logsumexp_2d(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """LogSumExp on 2D input with dim=-1 (default)."""
    dim = -1
    test = LogSumExpTest(shape, dim, dtype)
    op = LogSumExpOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSumExpFixtureND(FixtureBase):
    """3D/4D logsumexp: multi-rank inputs x fp16/bf16."""

    PARAMS = [
        (
            "shape, dtype",
            [
                # 3D smoke
                pytest.param((2, 16, 256), torch.float16, marks=pytest.mark.smoke),
                # 3D full
                pytest.param((2, 16, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((2, 16, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 512, 4096), torch.float16, marks=pytest.mark.full),
                # 4D: attention-score shape (batch, heads, seq, seq)
                pytest.param((2, 4, 8, 256), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), torch.bfloat16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 300), torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 128, 256), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpFixtureND
def test_logsumexp_nd(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """LogSumExp on 3D/4D input with dim=-1 (default)."""
    dim = -1
    test = LogSumExpTest(shape, dim, dtype)
    op = LogSumExpOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSumExpFixtureNonLastDim(FixtureBase):
    """LogSumExp along non-last dimensions (dim=0, dim=1)."""

    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=0: reduce along batch
                pytest.param((16, 256), 0, torch.float16, marks=pytest.mark.smoke),
                pytest.param((16, 256), 0, torch.bfloat16, marks=pytest.mark.full),
                # dim=1: reduce along middle axis of 3D
                pytest.param((2, 16, 256), 1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 16, 256), 1, torch.bfloat16, marks=pytest.mark.full),
                # dim=0 on 3D
                pytest.param((4, 16, 256), 0, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpFixtureNonLastDim
def test_logsumexp_non_last_dim(shape: tuple[int, ...], dim: int, dtype: torch.dtype) -> None:
    """LogSumExp along a non-last dimension -- tests dim param generality."""
    test = LogSumExpTest(shape, dim, dtype)
    op = LogSumExpOp(dim=dim, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSumExpFixtureKeepdim(FixtureBase):
    """LogSumExp with keepdim=True."""

    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # 2D keepdim smoke
                pytest.param((32, 256), -1, torch.float16, marks=pytest.mark.smoke),
                # 2D keepdim full
                pytest.param((32, 256), -1, torch.bfloat16, marks=pytest.mark.full),
                # 3D keepdim
                pytest.param((2, 16, 256), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 16, 256), 1, torch.float16, marks=pytest.mark.full),
                # 4D keepdim
                pytest.param((2, 4, 8, 256), -1, torch.float16, marks=pytest.mark.full),
                pytest.param((2, 4, 8, 256), -1, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpFixtureKeepdim
def test_logsumexp_keepdim(shape: tuple[int, ...], dim: int, dtype: torch.dtype) -> None:
    """LogSumExp with keepdim=True -- output retains reduced dimension."""
    test = LogSumExpTest(shape, dim, dtype, keepdim=True)
    op = LogSumExpOp(dim=dim, keepdim=True, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LogSumExpFixtureNonContig(FixtureBase):
    """Non-contiguous input coverage for logsumexp."""

    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(32, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@LogSumExpFixtureNonContig
def test_logsumexp_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """LogSumExp with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice

    op = LogSumExpOp(dim=-1, dtype=dtype)

    y_ref = torch.logsumexp(x.float().contiguous(), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
        f"Non-contiguous logsumexp failed, max err: {(y - y_ref).abs().max()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
