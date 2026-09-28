"""Correctness tests for logical reduce ops (any, all, count_nonzero).

Covers: AnyFwdOp, AllFwdOp, CountNonzeroFwdOp.
any/all reduce along the configured dim and return bool dtype.
count_nonzero reduces along the configured dim and returns int64 dtype.
Uses exact match (torch.equal) for comparison.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, served_in_tree
from tileops.backend import BUILTIN
from tileops.kernels.reduction.call_spec import LogicalReduceCall
from tileops.kernels.reduction.logical_reduce import (
    LogicalReduceEdgeFusedKernel,
    LogicalReduceEdgeTwoPassKernel,
    LogicalReduceKernel,
)
from workloads.device import run_device, run_device_available
from workloads.reduction import AnyWorkload


class LogicalReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.int32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.int64, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.complex64, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.complex128, marks=pytest.mark.smoke),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-pow2 last dim
                pytest.param(128, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bool, marks=pytest.mark.full),
                pytest.param(128, 300, torch.complex64, marks=pytest.mark.full),
                # Row count that is not a power of two
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogicalReduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogicalReduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogicalReduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(512, torch.bool, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogicalReduceDimFixture(FixtureBase):
    """Fixture for testing dim=0, dim=1, and keepdim variants."""

    PARAMS = [
        (
            "shape, dim, dtype",
            [
                # dim=0 reduction on 2D
                pytest.param((64, 512), 0, torch.float16, marks=pytest.mark.smoke),
                pytest.param((64, 512), 0, torch.float32, marks=pytest.mark.smoke),
                # dim=1 reduction on 3D (reduces middle dim)
                pytest.param((4, 64, 512), 1, torch.float16, marks=pytest.mark.full),
                # dim=0 reduction on 3D
                pytest.param((4, 64, 512), 0, torch.float16, marks=pytest.mark.full),
                # negative dim on 3D (dim=-2 = middle)
                pytest.param((4, 64, 512), -2, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduceKeepdimFixture(FixtureBase):
    """Fixture for keepdim=True tests (AllFwdOp, AnyFwdOp only)."""

    PARAMS = [
        (
            "shape, dim, dtype",
            [
                pytest.param((64, 512), -1, torch.float16, marks=pytest.mark.smoke),
                pytest.param((64, 512), 0, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 64, 512), 1, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduceTest(AnyWorkload, TestBase):
    """Parameterized test helper for logical reduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "any":
            return x.bool().any(dim=-1)
        elif self.op_kind == "all":
            return x.bool().all(dim=-1)
        elif self.op_kind == "count_nonzero":
            return torch.count_nonzero(x, dim=-1).to(torch.int64)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


def _exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison using torch.equal."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert output_ref.dtype == torch.bool, f"Expected ref bool dtype, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


def _exact_compare_int64(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison for int64 count_nonzero outputs."""
    assert output.dtype == torch.int64, f"Expected int64 dtype, got {output.dtype}"
    assert output_ref.dtype == torch.int64, f"Expected ref int64 dtype, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Int64 mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


def _make_noncontig_input(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a non-contiguous 2D tensor of shape (m, n*2) for slicing tests."""
    if dtype == torch.bool:
        return torch.randint(0, 2, (m, n * 2), dtype=torch.bool, device=run_device())
    return torch.randn(m, n * 2, dtype=dtype, device=run_device())


def _make_1d_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D tensor of shape (n,) for 1D tests."""
    if dtype == torch.bool:
        return torch.randint(0, 2, (n,), dtype=torch.bool, device=run_device())
    return torch.randn(n, dtype=dtype, device=run_device())


def _make_nd_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create an N-D tensor for dim/keepdim tests."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=run_device())
    return torch.randn(shape, dtype=dtype, device=run_device())


@LogicalReduceBasicFixture
def test_any_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@LogicalReduceNonContigFixture
def test_any_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = AnyFwdOp(dim=-1)
    ref = x.contiguous().bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"non-contig any mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_any_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = AnyFwdOp(dim=-1)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"3D any mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_any_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = AnyFwdOp(dim=-1)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"4D any mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_any_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_1d_input(n, dtype)
    op = AnyFwdOp(dim=-1)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y.view_as(ref), ref), "1D any mismatch"


@LogicalReduceDimFixture
def test_any_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_nd_input(shape, dtype)
    op = AnyFwdOp(dim=dim)
    ref = x.bool().any(dim=dim)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"any dim={dim} mismatch: {(y != ref).sum().item()}"


@LogicalReduceKeepdimFixture
def test_any_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_nd_input(shape, dtype)
    op = AnyFwdOp(dim=dim, keepdim=True)
    ref = x.bool().any(dim=dim, keepdim=True)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"any keepdim dim={dim} mismatch: {(y != ref).sum().item()}"


@LogicalReduceBasicFixture
def test_all_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@LogicalReduceNonContigFixture
def test_all_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = AllFwdOp(dim=-1)
    ref = x.contiguous().bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"non-contig all mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_all_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = AllFwdOp(dim=-1)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"3D all mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_all_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = AllFwdOp(dim=-1)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"4D all mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_all_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_1d_input(n, dtype)
    op = AllFwdOp(dim=-1)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y.view_as(ref), ref), "1D all mismatch"


@LogicalReduceDimFixture
def test_all_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_nd_input(shape, dtype)
    op = AllFwdOp(dim=dim)
    ref = x.bool().all(dim=dim)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"all dim={dim} mismatch: {(y != ref).sum().item()}"


@LogicalReduceKeepdimFixture
def test_all_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_nd_input(shape, dtype)
    op = AllFwdOp(dim=dim, keepdim=True)
    ref = x.bool().all(dim=dim, keepdim=True)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"all keepdim dim={dim} mismatch: {(y != ref).sum().item()}"


@LogicalReduceBasicFixture
def test_count_nonzero_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@LogicalReduceNonContigFixture
def test_count_nonzero_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x.contiguous(), dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig count_nonzero mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_count_nonzero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D count_nonzero mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_count_nonzero_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D count_nonzero mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_count_nonzero_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x = _make_1d_input(n, dtype)
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D count_nonzero mismatch"


@pytest.mark.smoke
def test_count_nonzero_past_fp32_integer_range() -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    # 2^24 + 1 nonzeros: the first count fp32 cannot hold.
    x = torch.zeros(1 << 25, dtype=torch.bool, device=run_device())
    x[: 1 << 24] = True
    x[-1] = True
    y = CountNonzeroFwdOp(dim=None)(x)
    assert torch.equal(y, torch.count_nonzero(x))


@LogicalReduceDimFixture
def test_count_nonzero_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x = _make_nd_input(shape, dtype)
    op = CountNonzeroFwdOp(dim=dim)
    ref = torch.count_nonzero(x, dim=dim).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"count_nonzero dim={dim} mismatch: {(y != ref).sum().item()}"


# Dtype smoke tests: ensure all 6 supported dtypes are covered at smoke tier.
# Each uses a single-param fixture so the framework's "exactly 1 smoke per
# test function" constraint is satisfied while giving broad dtype coverage.

_DTYPE_SMOKE_M, _DTYPE_SMOKE_N = 64, 512


def _make_dtype_smoke_fixture(dt: torch.dtype) -> type:
    """Create a single-param smoke fixture for the given dtype."""
    dt_name = str(dt).split(".")[-1]

    class _Fixture(FixtureBase):
        PARAMS = [
            (
                "m, n, dtype",
                [pytest.param(_DTYPE_SMOKE_M, _DTYPE_SMOKE_N, dt, marks=pytest.mark.smoke)],
            )
        ]

    _Fixture.__name__ = f"_DtypeSmoke_{dt_name}"
    _Fixture.__qualname__ = _Fixture.__name__
    return _Fixture


_DtypeSmoke_float16 = _make_dtype_smoke_fixture(torch.float16)
_DtypeSmoke_bfloat16 = _make_dtype_smoke_fixture(torch.bfloat16)
_DtypeSmoke_float32 = _make_dtype_smoke_fixture(torch.float32)
_DtypeSmoke_int32 = _make_dtype_smoke_fixture(torch.int32)
_DtypeSmoke_int64 = _make_dtype_smoke_fixture(torch.int64)
_DtypeSmoke_bool = _make_dtype_smoke_fixture(torch.bool)


@_DtypeSmoke_float16
def test_any_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bfloat16
def test_any_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int32
def test_any_smoke_int32(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int64
def test_any_smoke_int64(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bool
def test_any_smoke_bool(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_float16
def test_all_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bfloat16
def test_all_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int32
def test_all_smoke_int32(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int64
def test_all_smoke_int64(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bool
def test_all_smoke_bool(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_float16
def test_count_nonzero_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_bfloat16
def test_count_nonzero_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_int32
def test_count_nonzero_smoke_int32(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_int64
def test_count_nonzero_smoke_int64(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_bool
def test_count_nonzero_smoke_bool(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, dtype",
    [
        ("any", torch.bool),
        ("all", torch.bool),
        ("count_nonzero", torch.float16),
    ],
)
def test_logical_reduce_long_sequence(op_kind: str, dtype: torch.dtype) -> None:
    """A long row whose last step only part of the block reaches."""
    from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp

    op_map = {
        "any": AnyFwdOp,
        "all": AllFwdOp,
        "count_nonzero": CountNonzeroFwdOp,
    }
    test = LogicalReduceTest(3, 33024, dtype, op_kind)
    op = op_map[op_kind](dim=-1)
    compare = _exact_compare_int64 if op_kind == "count_nonzero" else _exact_compare
    test.check(op, *test.gen_inputs(), compare=compare)


@pytest.mark.smoke
def test_logical_reduce_autotune() -> None:
    """``tune=True`` must build and time every candidate width."""
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    m, n, dtype = 4, 40000, torch.bool
    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1, tune=True)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)

    if served_in_tree(op):
        (kernel,) = op.built_kernels("reduce").values()
        assert kernel.config in kernel.autotune_configs


# Manifest dtype contract: bool input + int64 / bool output dtypes.

_M = 64
_N = 256


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("op_name", ["AllFwdOp", "AnyFwdOp"])
def test_logical_reduce_accepts_bool(op_name: str) -> None:
    """All / Any must accept bool inputs (manifest dtype contract)."""
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    op = cls(dim=-1)
    x = torch.randint(0, 2, (_M, _N), device=run_device()).bool()
    out = op(x)
    assert out.dtype == torch.bool
    assert out.shape == (_M,)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_count_nonzero_returns_int64() -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    op = CountNonzeroFwdOp(dim=-1)
    x = torch.randn(_M, _N, dtype=torch.float16, device=run_device())
    out = op(x)
    assert out.dtype == torch.int64, f"CountNonzero output dtype {out.dtype} != int64"


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("op_name", ["AllFwdOp", "AnyFwdOp"])
def test_logical_reduce_returns_bool(op_name: str) -> None:
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    op = cls(dim=-1)
    x = torch.randn(_M, _N, dtype=torch.float16, device=run_device())
    out = op(x)
    assert out.dtype == torch.bool


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, dtype",
    [
        ("any", torch.bool),
        ("all", torch.bool),
        ("count_nonzero", torch.float16),
    ],
)
def test_logical_reduce_edge_axes_in_own_layout(op_kind: str, dtype: torch.dtype) -> None:
    """``dim=[0, 2]`` reduces without a permute: 0/1 (or count) partials, then a fold."""
    from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp

    op_map = {"any": AnyFwdOp, "all": AllFwdOp, "count_nonzero": CountNonzeroFwdOp}
    op = op_map[op_kind](dim=[0, 2])
    if dtype == torch.bool:
        x = torch.rand(4, 24, 4096, device=run_device()) > 0.999
        if op_kind == "all":
            x = ~x
    else:
        x = torch.randn(4, 24, 4096, dtype=dtype, device=run_device())
    ref = {
        "any": lambda: x.any(0).any(-1),
        "all": lambda: x.all(0).all(-1),
        "count_nonzero": lambda: torch.count_nonzero(x, (0, 2)),
    }[op_kind]()
    assert torch.equal(op(x), ref)


@pytest.mark.cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_kind, dtype, tune",
    [
        pytest.param("all", torch.bool, False, marks=pytest.mark.smoke),
        pytest.param("count_nonzero", torch.float16, False, marks=pytest.mark.smoke),
        pytest.param("any", torch.bool, True, marks=pytest.mark.full),
    ],
)
def test_logical_reduce_edge_axes_fused_dispatch(
    op_kind: str, dtype: torch.dtype, tune: bool
) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
    from tileops.utils import device_calibration

    if device_calibration() is None:
        pytest.skip("fused edge logical reduce is selected only on a calibrated board")

    op_map = {"any": AnyFwdOp, "all": AllFwdOp, "count_nonzero": CountNonzeroFwdOp}
    op = op_map[op_kind](dim=[0, 2], tune=tune, target=BUILTIN)
    if dtype == torch.bool:
        x = torch.rand(4, 128, 4096, device="cuda") > 0.999
        if op_kind == "all":
            x = ~x
    else:
        x = torch.randn(4, 128, 4096, dtype=dtype, device="cuda")
    ref = {
        "any": lambda: x.any(0).any(-1),
        "all": lambda: x.all(0).all(-1),
        "count_nonzero": lambda: torch.count_nonzero(x, (0, 2)),
    }[op_kind]()
    assert torch.equal(op(x), ref)
    # The role is the op's one memoization bucket; which implementation served the call
    # is the entry that was built under it.
    (built,) = op.built_kernels("reduce").values()
    assert isinstance(built, LogicalReduceEdgeFusedKernel)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, shape, axes, calibration, expected",
    [
        pytest.param("any", (4, 24, 4096), (1,), "h200", LogicalReduceKernel, id="inner-axis"),
        pytest.param(
            "all", (4, 128, 4096), (0, 2), None, LogicalReduceEdgeTwoPassKernel, id="uncalibrated"
        ),
        pytest.param(
            "any", (4, 31, 4096), (0, 2), "h200", LogicalReduceEdgeTwoPassKernel, id="few-kept"
        ),
        pytest.param(
            "any", (4, 32, 4096), (0, 2), "h200", LogicalReduceEdgeFusedKernel, id="many-kept"
        ),
        pytest.param(
            "count_nonzero",
            (2, 4, 1 << 23),
            (0, 2),
            "h200",
            LogicalReduceEdgeTwoPassKernel,
            id="count-at-fp32-limit",
        ),
        pytest.param(
            "count_nonzero",
            (2, 4, (1 << 23) + 1),
            (0, 2),
            "h200",
            LogicalReduceKernel,
            id="count-past-fp32",
        ),
    ],
)
def test_logical_reduce_selection(
    op_kind: str, shape: tuple, axes: tuple, calibration: "str | None", expected: type
) -> None:
    """Each edge candidate serves its side of the kept-column threshold; the row fold the rest."""
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    call = LogicalReduceCall(
        arch=90,
        calibration=calibration,
        sm_count=132,
        shape=shape,
        axes=axes,
        op_kind=op_kind,
        dtype=torch.bool,
    )
    assert AnyFwdOp(dim=list(axes)).select_kernel(call) is expected


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "dtype, value",
    [
        pytest.param(torch.float32, -0.0, id="negative-zero"),
        pytest.param(torch.complex128, 1e-300j, id="float64-only-imag"),
    ],
)
def test_logical_reduce_truth_at_own_width(dtype: torch.dtype, value: complex) -> None:
    """Truth is decided in the input's own dtype, and a conjugated input reads the same."""
    from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp

    x = torch.ones(4, 1000, dtype=dtype, device=run_device())
    x[1, 3] = value
    x[2] = 0
    x[2, 999] = value
    inputs = [x, x.conj()] if dtype.is_complex else [x]
    for t in inputs:
        assert torch.equal(AnyFwdOp(dim=-1)(t), t.any(-1))
        assert torch.equal(AllFwdOp(dim=-1)(t), t.all(-1))
        assert torch.equal(CountNonzeroFwdOp(dim=-1)(t), torch.count_nonzero(t, dim=-1))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_count_nonzero_exact_past_fp32_integers() -> None:
    """A row counting past 2^24 stays exact: fp32 cannot hold 2^24 + 1."""
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x = torch.ones(2, (1 << 24) + 4, dtype=torch.bool, device=run_device())
    x[1, 7] = False
    assert torch.equal(CountNonzeroFwdOp(dim=-1)(x), torch.count_nonzero(x, dim=-1))


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_logical_reduce_rejects_width_its_reduction_cannot_fold() -> None:
    """A block width that is not a power of two would drop warps from the reduction."""
    kernel = LogicalReduceKernel(
        (4, 4096), (1,), "count_nonzero", torch.float16, config={"threads": 96}
    )
    with pytest.raises(ValueError, match="power of two"):
        kernel(torch.ones(4, 4096, dtype=torch.float16, device="cuda"))
