"""Exact-match tests for AnyFwdOp and AllFwdOp (bool out) and CountNonzeroFwdOp (int64 out)."""

import itertools

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
                # Non-pow2 last dim
                pytest.param(128, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bool, marks=pytest.mark.full),
                pytest.param(128, 300, torch.complex64, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduceTest(LogicalReduceCase, TestBase):
    pass


_OPS = {"any": AnyFwdOp, "all": AllFwdOp, "count_nonzero": CountNonzeroFwdOp}


def _reference(op_kind: str, x: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
    """PyTorch's answer for *op_kind* over *dim*; ``dim=None`` with *keepdim* keeps every axis."""
    if op_kind == "count_nonzero":
        return torch.count_nonzero(x, dim=dim)
    fn = torch.any if op_kind == "any" else torch.all
    if dim is None:
        return fn(x).reshape([1] * x.ndim) if keepdim else fn(x)
    return fn(x, dim=dim, keepdim=keepdim)


def _mixed(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Random values with about one exact zero in ten, so all and any see both truths."""
    torch.manual_seed(0)
    if dtype == torch.bool:
        return torch.rand(shape, device=run_device()) < 0.9
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    return x.masked_fill(torch.rand(shape, device=run_device()) < 0.1, 0)


def _check(op_kind: str, x: torch.Tensor, dim, keepdim: bool = False) -> None:
    kwargs = {} if op_kind == "count_nonzero" else {"keepdim": keepdim}
    y = _OPS[op_kind](dim=dim, **kwargs)(x)
    ref = _reference(op_kind, x, dim, keepdim)
    assert y.dtype == (torch.int64 if op_kind == "count_nonzero" else torch.bool)
    assert y.shape == ref.shape, f"shape {y.shape} vs ref {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@pytest.mark.parametrize("op_kind", list(_OPS))
@LogicalReduceBasicFixture
def test_logical_reduce_rows(op_kind: str, m: int, n: int, dtype: torch.dtype) -> None:
    test = LogicalReduceTest(m, n, dtype, op_kind)
    test.check(_OPS[op_kind](dim=-1), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_OPS))
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param("strided", id="strided"),
        pytest.param("1d", id="1d"),
        pytest.param("4d", id="4d"),
        pytest.param("leading-dim", id="leading-dim"),
        pytest.param("middle-dim", id="middle-dim"),
    ],
)
def test_logical_reduce_layouts(op_kind: str, layout: str) -> None:
    """A strided view, a 1-D row, leading axes folded into rows, and one non-last axis."""
    shape, dim = {
        "strided": ((128, 1024), -1),
        "1d": ((512,), -1),
        "4d": ((2, 4, 8, 512), -1),
        "leading-dim": ((64, 512), 0),
        "middle-dim": ((4, 64, 512), 1),
    }[layout]
    x = _mixed(shape, torch.float16)
    if layout == "strided":
        x = x[:, :512]
    _check(op_kind, x, dim)


def _edge_input(op_kind: str, dtype: torch.dtype, rows: int) -> torch.Tensor:
    """A ``(4, rows, 4096)`` input whose rare nonzeros (zeros for all) decide most outputs."""
    if dtype != torch.bool:
        return torch.randn(4, rows, 4096, dtype=dtype, device=run_device())
    x = torch.rand(4, rows, 4096, device=run_device()) > 0.999
    return ~x if op_kind == "all" else x


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, dtype", [("any", torch.bool), ("all", torch.bool), ("count_nonzero", torch.float16)]
)
def test_logical_reduce_long_sequence(op_kind: str, dtype: torch.dtype) -> None:
    """A long row whose last step only part of the block reaches."""
    test = LogicalReduceTest(3, 33024, dtype, op_kind)
    test.check(_OPS[op_kind](dim=-1), *test.gen_inputs())


@pytest.mark.smoke
def test_logical_reduce_autotune() -> None:
    """``request_tune()`` must build and time every candidate width."""
    test = LogicalReduceTest(4, 40000, torch.bool, "any")
    op = AnyFwdOp(dim=-1)
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, dtype", [("any", torch.bool), ("all", torch.bool), ("count_nonzero", torch.float16)]
)
def test_logical_reduce_edge_axes_in_own_layout(op_kind: str, dtype: torch.dtype) -> None:
    """``dim=[0, 2]`` reduces without a permute: 0/1 (or count) partials, then a fold."""
    x = _edge_input(op_kind, dtype, 24)
    y = _OPS[op_kind](dim=[0, 2])(x)
    compare_outputs(y, _reference(op_kind, x, (0, 2)), reduction_verification(y.dtype))


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

    op = _OPS[op_kind](dim=[0, 2], target=BUILTIN)
    if tune:
        op.request_tune()
    x = _edge_input(op_kind, dtype, 128)
    y = op(x)
    compare_outputs(y, _reference(op_kind, x, (0, 2)), reduction_verification(y.dtype))


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
    for t, op_kind in itertools.product(inputs, _OPS):
        y = _OPS[op_kind](dim=-1)(t)
        compare_outputs(y, _reference(op_kind, t, -1), reduction_verification(y.dtype))


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


_DIMS = [(-1, "dim=int"), ((0, 2), "dim=tuple"), (None, "dim=None")]
_FLOAT_DTYPES = [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")]


def _conformance_cases() -> list:
    """Every (op, dim, keepdim) cell once, the dtypes taken in turn; count_nonzero has no
    keepdim. The first three cells, one a dtype, are smoke."""
    cells = [
        (op_kind, dim, keepdim)
        for op_kind in _OPS
        for dim in _DIMS
        for keepdim in ((False,) if op_kind == "count_nonzero" else (False, True))
    ]
    cases = []
    for i, (op_kind, (dim, dim_id), keepdim) in enumerate(cells):
        dtype, dtype_id = _FLOAT_DTYPES[i % len(_FLOAT_DTYPES)]
        cases.append(
            pytest.param(
                op_kind,
                dim,
                keepdim,
                dtype,
                id=f"{op_kind}-{dim_id}-keepdim={keepdim}-{dtype_id}",
                marks=pytest.mark.smoke if i < len(_FLOAT_DTYPES) else pytest.mark.full,
            )
        )
    return cases


@pytest.mark.parametrize("op_kind, dim, keepdim, dtype", _conformance_cases())
def test_logical_reduce_conformance(op_kind: str, dim, keepdim: bool, dtype: torch.dtype) -> None:
    """PyTorch conformance over the manifest's dim forms (int, tuple, None) and keepdim."""
    _check(op_kind, _mixed((4, 8, 256), dtype), dim, keepdim)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_OPS))
@pytest.mark.parametrize("dim", [d for d, _ in _DIMS], ids=[i for _, i in _DIMS])
def test_logical_reduce_unaligned_innermost(op_kind: str, dim) -> None:
    """An innermost extent of 255, off every tile multiple, takes the masked-load boundary."""
    _check(op_kind, _mixed((4, 8, 255), torch.float16), dim)
