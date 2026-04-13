"""Correctness tests for logical reduce ops (any, all, count_nonzero).

Covers: AnyFwdOp, AllFwdOp, CountNonzeroFwdOp.
any/all reduce along the configured dim and return bool dtype.
count_nonzero reduces along the configured dim and returns int64 dtype.
Uses exact match (torch.equal) for comparison.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.logical_reduce import AnyTest as _AnyWorkload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LogicalReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.full),
                pytest.param(128, 512, torch.int32, marks=pytest.mark.full),
                pytest.param(128, 512, torch.int64, marks=pytest.mark.full),
                pytest.param(128, 512, torch.complex64, marks=pytest.mark.full),
                pytest.param(128, 512, torch.complex128, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-pow2 last dim
                pytest.param(128, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bool, marks=pytest.mark.full),
                pytest.param(128, 300, torch.complex64, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
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
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(512, torch.bool, marks=pytest.mark.full),
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
                pytest.param((64, 512), 0, torch.float32, marks=pytest.mark.full),
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


# ---------------------------------------------------------------------------
# TestBase helpers — inherit gen_inputs() from workload classes
# ---------------------------------------------------------------------------


class LogicalReduceTest(_AnyWorkload, TestBase):
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
        return torch.randint(0, 2, (m, n * 2), dtype=torch.bool, device="cuda")
    return torch.randn(m, n * 2, dtype=dtype, device="cuda")


def _make_1d_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D tensor of shape (n,) for 1D tests."""
    if dtype == torch.bool:
        return torch.randint(0, 2, (n,), dtype=torch.bool, device="cuda")
    return torch.randn(n, dtype=dtype, device="cuda")


def _make_nd_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create an N-D tensor for dim/keepdim tests."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    return torch.randn(shape, dtype=dtype, device="cuda")


# ---------------------------------------------------------------------------
# AnyFwdOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_any_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@LogicalReduceNonContigFixture
def test_any_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = AnyFwdOp(dtype=dtype)
    ref = x.contiguous().bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"non-contig any mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_any_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = AnyFwdOp(dtype=dtype)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"3D any mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_any_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = AnyFwdOp(dtype=dtype)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"4D any mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_any_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_1d_input(n, dtype)
    op = AnyFwdOp(dtype=dtype)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y.view_as(ref), ref), "1D any mismatch"


@LogicalReduceDimFixture
def test_any_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_nd_input(shape, dtype)
    op = AnyFwdOp(dtype=dtype, dim=dim)
    ref = x.bool().any(dim=dim)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"any dim={dim} mismatch: {(y != ref).sum().item()}"


@LogicalReduceKeepdimFixture
def test_any_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_nd_input(shape, dtype)
    op = AnyFwdOp(dtype=dtype, dim=dim, keepdim=True)
    ref = x.bool().any(dim=dim, keepdim=True)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"any keepdim dim={dim} mismatch: {(y != ref).sum().item()}"


# ---------------------------------------------------------------------------
# AllFwdOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_all_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@LogicalReduceNonContigFixture
def test_all_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = AllFwdOp(dtype=dtype)
    ref = x.contiguous().bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"non-contig all mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_all_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = AllFwdOp(dtype=dtype)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"3D all mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_all_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = AllFwdOp(dtype=dtype)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"4D all mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_all_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_1d_input(n, dtype)
    op = AllFwdOp(dtype=dtype)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y.view_as(ref), ref), "1D all mismatch"


@LogicalReduceDimFixture
def test_all_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_nd_input(shape, dtype)
    op = AllFwdOp(dtype=dtype, dim=dim)
    ref = x.bool().all(dim=dim)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"all dim={dim} mismatch: {(y != ref).sum().item()}"


@LogicalReduceKeepdimFixture
def test_all_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_nd_input(shape, dtype)
    op = AllFwdOp(dtype=dtype, dim=dim, keepdim=True)
    ref = x.bool().all(dim=dim, keepdim=True)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"all keepdim dim={dim} mismatch: {(y != ref).sum().item()}"


# ---------------------------------------------------------------------------
# CountNonzeroFwdOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_count_nonzero_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@LogicalReduceNonContigFixture
def test_count_nonzero_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = CountNonzeroFwdOp(dtype=dtype)
    ref = torch.count_nonzero(x.contiguous(), dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig count_nonzero mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_count_nonzero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = CountNonzeroFwdOp(dtype=dtype)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D count_nonzero mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_count_nonzero_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = CountNonzeroFwdOp(dtype=dtype)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D count_nonzero mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_count_nonzero_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = _make_1d_input(n, dtype)
    op = CountNonzeroFwdOp(dtype=dtype)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D count_nonzero mismatch"


@LogicalReduceDimFixture
def test_count_nonzero_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = _make_nd_input(shape, dtype)
    op = CountNonzeroFwdOp(dtype=dtype, dim=dim)
    ref = torch.count_nonzero(x, dim=dim).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"count_nonzero dim={dim} mismatch: {(y != ref).sum().item()}"


# ---------------------------------------------------------------------------
# Dtype smoke tests: ensure all 6 supported dtypes are covered at smoke tier.
# Each uses a single-param fixture so the framework's "exactly 1 smoke per
# test function" constraint is satisfied while giving broad dtype coverage.
# ---------------------------------------------------------------------------

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
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bfloat16
def test_any_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int32
def test_any_smoke_int32(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int64
def test_any_smoke_int64(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bool
def test_any_smoke_bool(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_float16
def test_all_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bfloat16
def test_all_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int32
def test_all_smoke_int32(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_int64
def test_all_smoke_int64(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_bool
def test_all_smoke_bool(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@_DtypeSmoke_float16
def test_count_nonzero_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_bfloat16
def test_count_nonzero_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_int32
def test_count_nonzero_smoke_int32(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_int64
def test_count_nonzero_smoke_int64(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


@_DtypeSmoke_bool
def test_count_nonzero_smoke_bool(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
