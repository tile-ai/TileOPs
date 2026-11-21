"""Correctness tests for logical reduce ops (any, all, count_nonzero).

Covers: AnyFwdOp, AllFwdOp, CountNonzeroFwdOp.
any/all reduce along the configured dim and return bool dtype.
count_nonzero reduces along the configured dim and returns int64 dtype.
Uses exact match (torch.equal) for comparison.
"""

from collections.abc import Callable

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.reduction import (
    AllFwdOp,
    AnyFwdOp,
    CountNonzeroFwdOp,
)
from tileops.backend import BUILTIN
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import LogicalReduceCase, reduction_verification


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


class LogicalReduceTest(LogicalReduceCase, TestBase):
    pass


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
    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@LogicalReduceNonContigFixture
def test_any_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = AnyFwdOp(dim=-1)
    ref = x.contiguous().bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce3DFixture
def test_any_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = AnyFwdOp(dim=-1)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce4DFixture
def test_any_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = AnyFwdOp(dim=-1)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce1DFixture
def test_any_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = AnyFwdOp(dim=-1)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y.view_as(ref), ref, reduction_verification(ref.dtype))


@LogicalReduceDimFixture
def test_any_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    x = _make_nd_input(shape, dtype)
    op = AnyFwdOp(dim=dim)
    ref = x.bool().any(dim=dim)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduceKeepdimFixture
def test_any_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    x = _make_nd_input(shape, dtype)
    op = AnyFwdOp(dim=dim, keepdim=True)
    ref = x.bool().any(dim=dim, keepdim=True)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduceBasicFixture
def test_all_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@LogicalReduceNonContigFixture
def test_all_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = AllFwdOp(dim=-1)
    ref = x.contiguous().bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce3DFixture
def test_all_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = AllFwdOp(dim=-1)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce4DFixture
def test_all_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = AllFwdOp(dim=-1)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce1DFixture
def test_all_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = AllFwdOp(dim=-1)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    compare_outputs(y.view_as(ref), ref, reduction_verification(ref.dtype))


@LogicalReduceDimFixture
def test_all_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    x = _make_nd_input(shape, dtype)
    op = AllFwdOp(dim=dim)
    ref = x.bool().all(dim=dim)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduceKeepdimFixture
def test_all_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    x = _make_nd_input(shape, dtype)
    op = AllFwdOp(dim=dim, keepdim=True)
    ref = x.bool().all(dim=dim, keepdim=True)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduceBasicFixture
def test_count_nonzero_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@LogicalReduceNonContigFixture
def test_count_nonzero_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x.contiguous(), dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce3DFixture
def test_count_nonzero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce4DFixture
def test_count_nonzero_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@LogicalReduce1DFixture
def test_count_nonzero_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = CountNonzeroFwdOp(dim=-1)
    ref = torch.count_nonzero(x, dim=-1).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    compare_outputs(y.view_as(ref), ref, reduction_verification(ref.dtype))


@pytest.mark.smoke
def test_count_nonzero_past_fp32_integer_range() -> None:
    # 2^24 + 1 nonzeros: the first count fp32 cannot hold.
    x = torch.zeros(1 << 25, dtype=torch.bool, device=run_device())
    x[: 1 << 24] = True
    x[-1] = True
    y = CountNonzeroFwdOp(dim=None)(x)
    compare_outputs(y, torch.count_nonzero(x), reduction_verification(torch.int64))


@LogicalReduceDimFixture
def test_count_nonzero_dim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    x = _make_nd_input(shape, dtype)
    op = CountNonzeroFwdOp(dim=dim)
    ref = torch.count_nonzero(x, dim=dim).to(torch.int64)
    y = op(x)
    assert y.dtype == torch.int64
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, dtype", [("any", torch.bool), ("all", torch.bool), ("count_nonzero", torch.float16)]
)
def test_logical_reduce_long_sequence(op_kind: str, dtype: torch.dtype) -> None:
    """A long row whose last step only part of the block reaches."""

    op_map = {"any": AnyFwdOp, "all": AllFwdOp, "count_nonzero": CountNonzeroFwdOp}
    test = LogicalReduceTest(3, 33024, dtype, op_kind)
    op = op_map[op_kind](dim=-1)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_logical_reduce_autotune() -> None:
    """``request_tune()`` must build and time every candidate width."""

    m, n, dtype = 4, 40000, torch.bool
    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dim=-1)
    op.request_tune()
    test.check(op, *test.gen_inputs())


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
    compare_outputs(op(x), ref, reduction_verification(ref.dtype))


@pytest.mark.in_tree_kernels
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
    from tileops._utils import device_calibration

    if device_calibration() is None:
        pytest.skip("fused edge logical reduce is selected only on a calibrated board")

    op_map = {"any": AnyFwdOp, "all": AllFwdOp, "count_nonzero": CountNonzeroFwdOp}
    op = op_map[op_kind](dim=[0, 2], target=BUILTIN)
    if tune:
        op.request_tune()
    if dtype == torch.bool:
        x = torch.rand(4, 128, 4096, device=run_device()) > 0.999
        if op_kind == "all":
            x = ~x
    else:
        x = torch.randn(4, 128, 4096, dtype=dtype, device=run_device())
    ref = {
        "any": lambda: x.any(0).any(-1),
        "all": lambda: x.all(0).all(-1),
        "count_nonzero": lambda: torch.count_nonzero(x, (0, 2)),
    }[op_kind]()
    compare_outputs(op(x), ref, reduction_verification(ref.dtype))
    # The role is the op's one memoization bucket; which implementation served the call
    # is the entry that was built under it.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, value",
    [
        pytest.param(torch.float32, -0.0, id="negative-zero"),
        # 16-bit floats go two to a 32-bit word: the sign mask must clear -0.0 in either half.
        pytest.param(torch.float16, -0.0, id="float16-negative-zero"),
        pytest.param(torch.bfloat16, -0.0, id="bfloat16-negative-zero"),
        # The sign mask must leave NaN nonzero.
        pytest.param(torch.bfloat16, float("nan"), id="bfloat16-nan"),
        pytest.param(torch.complex128, 1e-300j, id="float64-only-imag"),
    ],
)
def test_logical_reduce_truth_at_own_width(dtype: torch.dtype, value: complex) -> None:
    """Truth is decided in the input's own dtype, and a conjugated input reads the same.

    Rows of 1000 elements fill whole 32-bit words for 16- and 32-bit floats, so the fold tests
    them a word at a time. *value* sits at an even column (a word's low half) and an odd one
    (its high half): among ones in rows 0 and 1, which ``all`` sees, and among zeros in rows 2
    and 3, which ``any`` sees.
    """

    x = torch.ones(4, 1000, dtype=dtype, device=run_device())
    x[0, 2] = value
    x[1, 3] = value
    x[2:] = 0
    x[2, 998] = value
    x[3, 999] = value
    inputs = [x, x.conj()] if dtype.is_complex else [x]
    for t in inputs:
        compare_outputs(AnyFwdOp(dim=-1)(t), t.any(-1), reduction_verification(torch.bool))
        compare_outputs(AllFwdOp(dim=-1)(t), t.all(-1), reduction_verification(torch.bool))
        compare_outputs(
            CountNonzeroFwdOp(dim=-1)(t),
            torch.count_nonzero(t, dim=-1),
            reduction_verification(torch.int64),
        )


@pytest.mark.smoke
def test_count_nonzero_exact_past_fp32_integers() -> None:
    """A row counting past 2^24 stays exact: fp32 cannot hold 2^24 + 1."""

    x = torch.ones(2, (1 << 24) + 4, dtype=torch.bool, device=run_device())
    x[1, 7] = False
    compare_outputs(
        CountNonzeroFwdOp(dim=-1)(x),
        torch.count_nonzero(x, dim=-1),
        reduction_verification(torch.int64),
    )


# PyTorch conformance over the manifest's dim forms (int, tuple, None), shape, dtype and values;
# All and Any also over both keepdim forms.


# (op_cls, torch_fn) pairs for ops sharing the (dim, keepdim) signature.
_ALL_ANY: list[tuple[type, Callable]] = [
    (AllFwdOp, torch.all),
    (AnyFwdOp, torch.any),
]


_DIMS = [(-1, "dim=int"), ((0, 2), "dim=tuple"), (None, "dim=None")]
_FLOAT_DTYPES = [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")]


def _conformance_cases() -> list:
    """Every (op, dim, keepdim) cell once, the dtypes taken in turn; the first three cells,
    one a dtype, are smoke."""
    cells = [(op, dim, keepdim) for op in _ALL_ANY for dim in _DIMS for keepdim in (False, True)]
    cases = []
    for i, ((op_cls, torch_fn), (dim, dim_id), keepdim) in enumerate(cells):
        dtype, dtype_id = _FLOAT_DTYPES[i % len(_FLOAT_DTYPES)]
        cases.append(
            pytest.param(
                op_cls,
                torch_fn,
                dim,
                keepdim,
                dtype,
                id=f"{op_cls.__name__}-{dim_id}-keepdim={keepdim}-{dtype_id}",
                marks=pytest.mark.smoke if i < len(_FLOAT_DTYPES) else pytest.mark.full,
            )
        )
    return cases


@pytest.mark.parametrize("op_cls, torch_fn, dim, keepdim, dtype", _conformance_cases())
def test_logical_reduce_conformance(
    op_cls: type,
    torch_fn: Callable,
    dim,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    """Each supported logical-reduction cell must match PyTorch."""
    shape = (4, 8, 256)
    torch.manual_seed(0)
    # Mix in exact zeros so All/Any actually see a False contribution.
    raw = torch.randn(*shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = op_cls(dim=dim, keepdim=keepdim)
    y = op(x)
    if dim is None:
        ref = torch_fn(x)
        if keepdim:
            ref = ref.reshape([1] * x.ndim)
    else:
        ref = torch_fn(x, dim=dim, keepdim=keepdim)

    assert y.dtype == torch.bool, f"{op_cls.__name__} output dtype {y.dtype}, expected torch.bool"
    assert ref.dtype == torch.bool
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} keepdim={keepdim} dtype={dtype}: "
        f"shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn",
    _ALL_ANY,
    ids=[c[0].__name__ for c in _ALL_ANY],
)
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_logical_reduce_unaligned_innermost(
    op_cls: type,
    torch_fn: Callable,
    dim,
) -> None:
    """Unaligned innermost dim must still match PyTorch.

    An aligned innermost extent of 256 (a kernel-tile multiple) bypasses
    the logical-reduce kernel's masked-load boundary path. Use 255 to flush
    the pad branch on every (op, dim-mode) cell.
    """
    unaligned_shape = (4, 8, 255)
    torch.manual_seed(0)
    dtype = torch.float16
    raw = torch.randn(*unaligned_shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(unaligned_shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = op_cls(dim=dim, keepdim=False)
    y = op(x)
    ref = torch_fn(x) if dim is None else torch_fn(x, dim=dim, keepdim=False)

    assert y.dtype == torch.bool
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification(ref.dtype))


# CountNonzero: separate matrix because the op does not accept ``keepdim``.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim, dtype",
    [
        pytest.param(dim, dtype, id=f"{dim_id}-{dtype_id}")
        for (dim, dim_id), (dtype, dtype_id) in zip(_DIMS, _FLOAT_DTYPES, strict=True)
    ],
)
def test_count_nonzero_conformance(dim, dtype: torch.dtype) -> None:
    """``CountNonzeroFwdOp`` must match ``torch.count_nonzero`` and emit int64.

    All input dtypes here differ from the int64 output dtype, satisfying the
    spec-conformance requirement that at least one input dtype differ from
    the output dtype.
    """
    shape = (4, 8, 256)
    torch.manual_seed(0)
    raw = torch.randn(*shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = CountNonzeroFwdOp(dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim)

    assert y.dtype == torch.int64, f"CountNonzeroFwdOp output dtype {y.dtype}, expected torch.int64"
    assert ref.dtype == torch.int64
    assert y.shape == ref.shape, (
        f"CountNonzeroFwdOp dim={dim} dtype={dtype}: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_count_nonzero_unaligned_innermost(dim) -> None:
    """Unaligned innermost dim must still match ``torch.count_nonzero``."""
    unaligned_shape = (4, 8, 255)
    torch.manual_seed(0)
    dtype = torch.float16
    raw = torch.randn(*unaligned_shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(unaligned_shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = CountNonzeroFwdOp(dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim)

    assert y.dtype == torch.int64
    assert y.shape == ref.shape, (
        f"CountNonzeroFwdOp dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification(ref.dtype))
