"""ArgmaxFwdOp and ArgminFwdOp: int64 indices, matched exactly against torch."""

import math

import pytest
import torch

from tests.cuda_graph_contract import check_replay
from tests.workload_test_base import TestBase
from tileops._ops.reduction import ArgmaxFwdOp, ArgminFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import ArgreduceCase, reduction_verification

_OPS = {"argmax": ArgmaxFwdOp, "argmin": ArgminFwdOp}


class ArgreduceTest(ArgreduceCase, TestBase):
    pass


def _check(op_kind: str, x: torch.Tensor, dim, keepdim: bool = False) -> None:
    y = _OPS[op_kind](dim=dim, keepdim=keepdim)(x)
    ref = getattr(torch, op_kind)(x, dim=dim, keepdim=keepdim)
    assert y.dtype == torch.int64
    assert y.shape == ref.shape, f"shape {y.shape} vs ref {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


def _row_cases() -> list:
    cases = [
        pytest.param(k, 128, 512, dtype, marks=pytest.mark.smoke)
        for k in _OPS
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
    ]
    for k in _OPS:
        cases += [
            pytest.param(k, 2048, 4096, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(k, 128, 300, torch.float16, marks=pytest.mark.full),
        ]
    return cases


@pytest.mark.parametrize("op_kind, m, n, dtype", _row_cases())
def test_argreduce_rows(op_kind: str, m: int, n: int, dtype: torch.dtype) -> None:
    test = ArgreduceTest(m, n, dtype, op_kind)
    test.check(_OPS[op_kind](dim=-1), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_OPS))
@pytest.mark.parametrize("view", ["strided", "1d", "4d"])
def test_argreduce_views(op_kind: str, view: str) -> None:
    """A strided view, a 1-D row, and leading axes folded into rows."""
    shape = {"strided": (128, 1024), "1d": (512,), "4d": (2, 4, 8, 512)}[view]
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())
    _check(op_kind, x[:, :512] if view == "strided" else x, -1)


def _dim_cases() -> list:
    """The strided kernel at each dtype, then a long and a middle axis and keepdim."""
    cases = [
        pytest.param(k, (4, 8, 256), 0, False, dtype, marks=pytest.mark.smoke)
        for k in _OPS
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ]
    cases += [
        pytest.param(k, (2, 4, 8, 256), 0, True, torch.float16, marks=pytest.mark.smoke)
        for k in _OPS
    ]
    for k in _OPS:
        cases += [
            pytest.param(k, (512, 4, 32), 0, False, torch.float16, marks=pytest.mark.full),
            pytest.param(k, (4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
            pytest.param(k, (4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
        ]
    return cases


@pytest.mark.parametrize("op_kind, shape, dim, keepdim, dtype", _dim_cases())
def test_argreduce_dim_and_keepdim(
    op_kind: str, shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype
) -> None:
    _check(op_kind, torch.randn(*shape, dtype=dtype, device=run_device()), dim, keepdim)


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="ArgmaxFwdOp")
@pytest.mark.smoke
def test_argmax_graph_replay() -> None:
    """DeepSeek-V4-Flash's greedy pick over one token's float32 logits."""
    check_replay(ArgreduceTest(1, 129280, torch.float32, "argmax"), ArgmaxFwdOp(dim=-1))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("op_name", ["argmax", "argmin"])
def test_argreduce_signed_zero_breaks_to_lower_index(op_name: str, dtype: torch.dtype) -> None:
    """A row of alternating -0.0 and +0.0 is one tie, which index 0 has to win."""
    x = torch.zeros(8, 4096, dtype=dtype, device=run_device())
    x[:, ::2] = -0.0
    y = _OPS[op_name](dim=-1)(x)
    compare_outputs(y, torch.zeros_like(y), reduction_verification(y.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_OPS))
@pytest.mark.parametrize("dim", [[0, 1], (0, 1)], ids=["list", "tuple"])
def test_argreduce_rejects_multidim(op_kind: str, dim) -> None:
    """Argreduce ops take one int dim or None."""
    with pytest.raises((TypeError, ValueError)):
        _OPS[op_kind](dim=dim)


@pytest.mark.parametrize(
    "op_kind, shape, dtype",
    [
        # dtype dispatch on one 2D shape; the flatten of more axes is dtype-agnostic.
        *[
            pytest.param(k, (16, 64), dtype, marks=pytest.mark.smoke)
            for k in _OPS
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ],
        *[pytest.param(k, (4, 8, 32), torch.float16, marks=pytest.mark.smoke) for k in _OPS],
        # A flat size off the vector width.
        *[pytest.param(k, (10, 30), torch.float16, marks=pytest.mark.full) for k in _OPS],
    ],
)
def test_argreduce_dim_none(op_kind: str, shape: tuple, dtype: torch.dtype) -> None:
    """``dim=None`` indexes the flattened input; keepdim keeps every axis at size 1."""
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    ref = getattr(torch, op_kind)(x)
    for keepdim in (False, True):
        y = _OPS[op_kind](dim=None, keepdim=keepdim)(x)
        assert y.dtype == torch.int64
        assert y.shape == ((1,) * x.ndim if keepdim else ()), f"shape {y.shape}"
        compare_outputs(y.reshape(()), ref, reduction_verification(ref.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_kind", "dtype"),
    [("argmax", torch.float16), ("argmin", torch.bfloat16)],
)
def test_argreduce_large_n(op_kind: str, dtype: torch.dtype) -> None:
    """The multi-CTA path supports the LM-head workload without tiling skips."""
    _check(op_kind, torch.randn(4, 102400, dtype=dtype, device=run_device()), -1)


@pytest.mark.smoke
def test_strided_stage_with_a_partial_last_step() -> None:
    """Three axis positions leave the last staging step partial."""
    x = torch.randn(3, 8, 256, dtype=torch.float16, device=run_device())
    for op_kind in _OPS:
        _check(op_kind, x, 0)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["argmax", "argmin"])
def test_argreduce_first_index_and_nan_semantics(op_kind: str) -> None:
    """Pair reduction preserves PyTorch's first-index and NaN behavior."""
    x = torch.tensor(
        [
            [1.0, float("nan"), 3.0, float("nan"), 3.0],
            [2.0, 2.0, -1.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
        device=run_device(),
    )
    _check(op_kind, x, -1)


@pytest.mark.parametrize("op_kind", ["argmax", "argmin"])
@pytest.mark.parametrize(
    "n",
    [
        pytest.param(8, marks=pytest.mark.smoke),
        pytest.param(4096, marks=pytest.mark.full),
        pytest.param(40960, marks=pytest.mark.full),
    ],
)
def test_argreduce_ties_between_two_nans(op_kind: str, n: int) -> None:
    """Two NaNs tie, so the lower index wins — on every reduction layout.

    No float comparison can express this: every comparison between two NaNs is
    false, so an ordering that leans on them returns whichever lane merged
    first. The three row lengths select the warp, CTA and multi-CTA paths, and
    the NaNs sit far enough apart to land in different lanes.
    """
    x = torch.arange(1, n + 1, dtype=torch.float32, device=run_device())
    first, second = n // 4, n // 2
    x[first] = float("nan")
    x[second] = float("nan")

    op = _OPS[op_kind](dim=-1)
    ref = getattr(torch, op_kind)(x)
    got = op(x)
    assert got.item() == ref.item() == first, (
        f"{op_kind} n={n}: got {got.item()}, torch {ref.item()}, expected {first}"
    )


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_argmax_wide_rows_under_tuning(tune: bool) -> None:
    """Rows wide enough to split across CTAs; the extremes sit in late chunks."""
    x = torch.randn(3, 65536, dtype=torch.float16, device=run_device())
    x[0, 65000] = 100.0
    x[1, 40000] = 100.0
    x[2, 64000] = float("nan")
    op = ArgmaxFwdOp(dim=-1)
    if tune:
        op.request_tune()
    compare_outputs(
        op(x),
        torch.argmax(x, dim=-1),
        reduction_verification(torch.int64),
    )


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((64, 8192), -1, marks=pytest.mark.smoke, id="rows"),
        pytest.param((4, 128, 4096), 0, marks=pytest.mark.full, id="strided"),
    ],
)
def test_argmax_contiguous_input_off_a_vector_boundary(shape: tuple, dim: int) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    x = torch.randn(math.prod(shape) + 1, dtype=torch.float16, device=run_device())[1:]
    x = x.view(shape)
    compare_outputs(
        ArgmaxFwdOp(dim=dim)(x),
        torch.argmax(x, dim=dim),
        reduction_verification(torch.int64),
    )
