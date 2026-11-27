"""VectorNormFwdOp at ord 1, 2 and inf against ``torch.linalg.vector_norm``."""

import math

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.reduction.vector_norm import VectorNormFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import VectorNormCase, vector_norm_verification

_ORD_MAP = {"l1": 1, "l2": 2, "inf": math.inf}


class VectorNormTest(VectorNormCase, TestBase):
    pass


def _make_op(op_kind: str, dim=-1, keepdim: bool = False) -> VectorNormFwdOp:
    return VectorNormFwdOp(_ORD_MAP[op_kind], dim=dim, keepdim=keepdim)


def _check(op_kind: str, x: torch.Tensor, dim=-1, keepdim: bool = False) -> None:
    y = _make_op(op_kind, dim, keepdim)(x)
    ref = torch.linalg.vector_norm(x.float(), _ORD_MAP[op_kind], dim, keepdim).to(x.dtype)
    assert y.shape == ref.shape, f"shape {y.shape} vs ref {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))


def _row_cases() -> list:
    cases = [
        pytest.param(k, 128, 512, dtype, marks=pytest.mark.smoke)
        for k in _ORD_MAP
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
    ]
    for k in _ORD_MAP:
        cases += [
            pytest.param(k, 256, 4096, torch.float16, marks=pytest.mark.full),
            pytest.param(k, 128, 300, torch.float32, marks=pytest.mark.full),
            pytest.param(k, 128, 300, torch.float16, marks=pytest.mark.full),
        ]
    return cases


@pytest.mark.parametrize("op_kind, m, n, dtype", _row_cases())
def test_vector_norm_rows(op_kind: str, m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, op_kind)
    test.check(_make_op(op_kind), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_ORD_MAP))
@pytest.mark.parametrize("view", ["strided", "1d", "4d"])
def test_vector_norm_views(op_kind: str, view: str) -> None:
    """A strided view, a 1-D row, and leading axes folded into rows."""
    shape = {"strided": (128, 1024), "1d": (512,), "4d": (2, 4, 8, 512)}[view]
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())
    _check(op_kind, x[:, :512] if view == "strided" else x)


class VectorNormNaNFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(4, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(4, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 300, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@VectorNormNaNFixture
def test_inf_nan_propagation(m: int, n: int, dtype: torch.dtype) -> None:
    """``ord=inf`` must return NaN for rows containing NaN, matching PyTorch."""
    x = torch.randn(m, n, dtype=dtype, device=run_device())
    x[0, 0] = float("nan")
    x[1, -1] = float("nan")

    y = _make_op("inf")(x)
    ref = torch.linalg.vector_norm(x.float(), ord=math.inf, dim=-1).to(dtype)

    assert y[0].isnan().item(), f"Row 0 should be NaN, got {y[0]}"
    assert y[1].isnan().item(), f"Row 1 should be NaN, got {y[1]}"
    compare_outputs(y[2:], ref[2:], vector_norm_verification(dtype))


@pytest.mark.parametrize(
    "op_kind, dtype, shape, dim, keepdim",
    [
        pytest.param(k, dtype, shape, dim, keepdim, marks=mark, id=f"{k}-{form}")
        for k, dtype, mark in (
            ("l1", torch.float16, pytest.mark.smoke),
            ("l2", torch.float32, pytest.mark.smoke),
            ("inf", torch.float16, pytest.mark.full),
        )
        for form, shape, dim, keepdim in (
            ("dim0", (64, 512), 0, False),
            ("dim1-3d", (4, 64, 512), 1, False),
            ("keepdim", (32, 512), -1, True),
            ("dim0-keepdim", (64, 512), 0, True),
        )
    ],
)
def test_vector_norm_dim_and_keepdim(
    op_kind: str, dtype: torch.dtype, shape: tuple, dim: int, keepdim: bool
) -> None:
    _check(op_kind, torch.randn(*shape, dtype=dtype, device=run_device()), dim, keepdim)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, keepdim, dtype",
    [("l1", False, torch.float16), ("l2", True, torch.bfloat16), ("inf", False, torch.float32)],
)
def test_empty_dim_full_reduction(op_kind: str, keepdim: bool, dtype: torch.dtype) -> None:
    _check(op_kind, torch.randn(2, 16, 128, dtype=dtype, device=run_device()), [], keepdim)


@pytest.mark.smoke
def test_vector_norm_tiled_autotune() -> None:
    """``request_tune()`` must build and time every tiled candidate.

    N is not a power of two: a power-of-two N_padded lets ``compute_tile_n``
    fall back on an exact divisor, which hides a mis-derived tile_n.
    """
    m, n, dtype = 4, 39999, torch.float16
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op("l2")
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
def test_vector_norm_edge_axes_in_own_layout(op_kind: str) -> None:
    """``dim=[0, 2]`` reduces without a permute: a rows pass, then a fold.

    ``l2`` takes its square root only at the fold; a sqrt applied per partial
    would compound instead of finishing.
    """
    _check(op_kind, torch.randn(17, 24, 2048, dtype=torch.float16, device=run_device()), [0, 2])


@pytest.mark.smoke
@pytest.mark.parametrize("lead", [4, 17], ids=["one-launch-fold", "rows-pass-then-fold"])
def test_inf_norm_edge_axes_carries_nan(lead: int) -> None:
    """A NaN survives every pass: the folds compare bit patterns, not floats."""
    x = torch.randn(lead, 24, 2048, dtype=torch.float16, device=run_device())
    x[lead - 1, 7, 2047] = float("nan")
    y = _make_op("inf", dim=[0, 2])(x)
    ref = torch.linalg.vector_norm(x, torch.inf, (0, 2))
    assert torch.equal(y.isnan(), ref.isnan())
