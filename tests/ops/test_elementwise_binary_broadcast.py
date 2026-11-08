"""Bidirectional-broadcast behavior tests for ``elementwise_binary`` ops.

L1 signature/parity (forward arg names, ``__init__`` defaults, manifest
entry resolution, construction smoke) is specified by
``scripts/validate_manifest.py`` strict-parity gates C3/C4/C5. This
file covers the load-bearing external behavior: bidirectional
broadcast against a PyTorch reference.
"""

from __future__ import annotations

import pytest
import torch

import tileops._ops.elementwise as elementwise_mod
from tests.workload_test_base import TestBase
from workloads.device import run_device, run_device_available
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs


def _randn(s, d):
    return torch.randn(*s, dtype=d, device=run_device())


def _rand_pos(s, d):
    return torch.rand(*s, dtype=d, device=run_device()) + 0.1


def _rand_bool(s, d):
    return (torch.randn(*s, dtype=d, device=run_device()) > 0).to(d)


def _randint(s, d):
    return torch.randint(-1000, 1000, s, dtype=d, device=run_device())


def _pow_base(s, d):
    return torch.rand(*s, dtype=d, device=run_device()) + 0.5


def _pow_exp(s, d):
    return torch.rand(*s, dtype=d, device=run_device()) * 2.0


# (op_name, dtype, gen_a, gen_b).
_F16 = torch.float16
_I32 = torch.int32

_BROADCAST_OPS = [
    ("AddFwdOp", _F16, _randn, _randn),
    ("SubFwdOp", _F16, _randn, _randn),
    ("MulFwdOp", _F16, _randn, _randn),
    ("DivFwdOp", _F16, _rand_pos, _rand_pos),
    ("RemainderFwdOp", _F16, _rand_pos, _rand_pos),
    ("PowFwdOp", _F16, _pow_base, _pow_exp),
    (
        "FloorDivideFwdOp",
        _F16,
        _rand_pos,
        _rand_pos,
    ),
    ("LerpScalarFwdOp", _F16, _randn, _randn),
    ("MaximumFwdOp", _F16, _randn, _randn),
    ("MinimumFwdOp", _F16, _randn, _randn),
    ("EqFwdOp", _F16, _rand_bool, _rand_bool),
    ("NeFwdOp", _F16, _rand_bool, _rand_bool),
    ("GtFwdOp", _F16, _randn, _randn),
    ("LtFwdOp", _F16, _randn, _randn),
    ("GeFwdOp", _F16, _randn, _randn),
    ("LeFwdOp", _F16, _randn, _randn),
    ("LogicalAndFwdOp", _F16, _rand_bool, _rand_bool),
    ("LogicalOrFwdOp", _F16, _rand_bool, _rand_bool),
    ("BitwiseAndFwdOp", _I32, _randint, _randint),
    ("BitwiseOrFwdOp", _I32, _randint, _randint),
    ("BitwiseXorFwdOp", _I32, _randint, _randint),
]


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "op_name, dtype, gen_a, gen_b",
    _BROADCAST_OPS,
    ids=[entry[0] for entry in _BROADCAST_OPS],
)
def test_binary_op_bidirectional_broadcast(
    op_name: str,
    dtype: torch.dtype,
    gen_a,
    gen_b,
) -> None:
    """Bidirectional broadcast: (3,1) x (1,4) -> (3,4)."""
    cls = getattr(elementwise_mod, op_name)
    a_shape = (3, 1)
    b_shape = (1, 4)
    a = gen_a(a_shape, dtype)
    b = gen_b(b_shape, dtype)
    op = cls()
    workload = ElementwiseWorkload(op_name, (a, b))
    TestBase.check(workload, op, a, b)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("op_name", ["MaximumFwdOp", "DivFwdOp"])
def test_channel_broadcast_with_ragged_inner_dim(op_name: str) -> None:
    """A per-channel operand over a non-tile-multiple inner dim.

    The row-broadcast body splits at trace time into full blocks and one
    guarded tail block; 300 columns force the tail. ``div`` is ordered, so a
    swapped operand would not cancel out.
    """
    cls = getattr(elementwise_mod, op_name)
    a = torch.randn(2, 3, 10, 30, dtype=torch.float16, device=run_device())
    b = torch.rand(3, 1, 1, dtype=torch.float16, device=run_device()) + 0.5
    ref = torch.maximum(a, b) if op_name == "MaximumFwdOp" else a / b
    out = cls()(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(op_name, (a, b)).verification(a, b))


# Rows that are not a whole number of a thread's vectors, against a full operand; each
# ``short`` case ends short of a vector. A five-column row is shorter than the float16
# vector of a launch this long, so it keeps the row grid.
_RAGGED_ROW_CASES = [
    pytest.param(
        "SubFwdOp",
        (3, 5, 13),
        (5, 1),
        torch.bfloat16,
        id="sub-short-of-a-vector",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        "RemainderFwdOp",
        (6, 5, 9),
        (5, 1),
        torch.float16,
        id="remainder-short-b",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        "RemainderFwdOp",
        (7, 1),
        (7, 5),
        torch.float32,
        id="remainder-short-a",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        "GtFwdOp",
        (3, 13),
        (3, 1),
        torch.float32,
        id="gt-bool-short-of-a-vector",
        marks=pytest.mark.full,
    ),
    pytest.param(
        "SubFwdOp",
        (65535, 5),
        (65535, 1),
        torch.float16,
        id="sub-row-shorter-than-a-vector",
        marks=pytest.mark.full,
    ),
]


@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("op_name, a_shape, b_shape, dtype", _RAGGED_ROW_CASES)
def test_ragged_rows_match_torch(op_name, a_shape, b_shape, dtype) -> None:
    """Each lane of a vector that straddles two rows reads its own row's operand.

    The ops are ordered, so a swapped operand shows. Zeros and infinities send some
    remainder threads to the exact fallback.
    """
    gen = torch.Generator(device=run_device()).manual_seed(0)
    a = torch.randn(a_shape, generator=gen, device=run_device()).mul(8).to(dtype)
    b = torch.rand(b_shape, generator=gen, device=run_device()).add(0.5).to(dtype)
    specials = torch.tensor([0.0, float("inf"), -3.0], device=run_device(), dtype=dtype)
    a.view(-1)[-3:] = specials
    b.view(-1)[-3:] = specials
    workload = ElementwiseWorkload(op_name, (a, b))
    TestBase.check(workload, getattr(elementwise_mod, op_name)(), a, b)


# Staged row blocks short enough for two 16-byte float32 vectors a thread, with the
# stride-0 operand as a for ``maximum`` and as b for the truncating divide.
_TWO_VECTOR_ROW_CASES = [
    pytest.param(
        "MaximumFwdOp", {}, (64, 1), (64, 2048), id="maximum-row-a", marks=pytest.mark.smoke
    ),
    pytest.param(
        "DivFwdOp",
        {"rounding_mode": "trunc"},
        (64, 2048),
        (64, 1),
        id="div-trunc-row-b",
        marks=pytest.mark.full,
    ),
]


@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("op_name, kwargs, a_shape, b_shape", _TWO_VECTOR_ROW_CASES)
def test_two_vector_row_blocks_match_torch(op_name, kwargs, a_shape, b_shape) -> None:
    """Both of a thread's vectors land where torch puts them.

    ``maximum`` sees NaNs in both operands; the divide sees zero divisors.
    """
    gen = torch.Generator(device=run_device()).manual_seed(0)
    a = torch.randn(a_shape, generator=gen, device=run_device()).mul(8)
    b = torch.randn(b_shape, generator=gen, device=run_device())
    a.view(-1)[::37] = float("nan") if op_name == "MaximumFwdOp" else 0.0
    b.view(-1)[::29] = float("nan") if op_name == "MaximumFwdOp" else 0.0
    workload = ElementwiseWorkload(op_name, (a, b), **kwargs)
    TestBase.check(workload, getattr(elementwise_mod, op_name)(**kwargs), a, b)


_STAGED_SHAPES = [
    pytest.param((4, 1), (4, 1000), torch.float16, id="inner-stride-0-and-1"),
    pytest.param((4, 1000), (1, 1000), torch.float16, id="inner-stride-1-and-1"),
    pytest.param((3, 5000), (3, 1), torch.float16, id="inner-spans-several-blocks"),
    pytest.param((64, 4096), (1, 4096), torch.float32, id="hinted-whole-blocks"),
    pytest.param((64, 4096), (64, 1), torch.float32, id="hinted-whole-blocks-channel-b"),
    pytest.param((64, 1), (64, 4096), torch.float32, id="hinted-whole-blocks-channel-a"),
    pytest.param((64, 4104), (1, 4104), torch.float32, id="hinted-blocks-and-tail"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("a_shape, b_shape, dtype", _STAGED_SHAPES)
def test_staged_row_broadcast_matches_torch(a_shape, b_shape, dtype):
    """A staged predicate broadcast agrees with torch on every stride pair."""
    from tileops._ops.elementwise import GtFwdOp

    a = torch.randn(a_shape, device=run_device(), dtype=dtype)
    b = torch.randn(b_shape, device=run_device(), dtype=dtype)
    out = GtFwdOp()(a, b)
    assert out.dtype == torch.bool
    compare_outputs(out, torch.gt(a, b), ElementwiseWorkload("GtFwdOp", (a, b)).verification(a, b))


_TAIL_SHAPES = [
    pytest.param((4, 1088), (4, 1), id="packed-tail-inner-stride-0"),
    pytest.param((5, 1088), (1, 1088), id="packed-tail-inner-stride-1"),
    pytest.param((3, 5000), (3, 1), id="guarded-tail"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("a_shape, b_shape", _TAIL_SHAPES)
def test_row_broadcast_tail_matches_torch(a_shape, b_shape):
    """Both endings of a ragged row write every element torch writes."""
    from tileops._ops.elementwise import AddFwdOp

    a = torch.randn(a_shape, device=run_device(), dtype=torch.float32)
    b = torch.randn(b_shape, device=run_device(), dtype=torch.float32)
    workload = ElementwiseWorkload("AddFwdOp", (a, b))
    TestBase.check(workload, AddFwdOp(), a, b)


def _with_nans(a_shape, b_shape, dtype):
    """Random operands with NaNs in both, at strides that cross the broadcast."""
    a = torch.randn(a_shape, device=run_device(), dtype=dtype)
    b = torch.randn(b_shape, device=run_device(), dtype=dtype)
    a.view(-1)[::997] = float("nan")
    b.view(-1)[::331] = float("nan")
    return a, b


# Each grid is 128 blocks of 128 threads, under every supported device's resident bound,
# so each case takes the bias-row grid wherever the suite runs.
_BIAS_ROW_CASES = [
    pytest.param("SubFwdOp", (64, 4096), (1, 4096), torch.float32, id="sub-bias-b-float32"),
    pytest.param("SubFwdOp", (4096,), (64, 4096), torch.float32, id="sub-bias-a-float32"),
    pytest.param("MaximumFwdOp", (64, 8192), (8192,), torch.float16, id="maximum-bias-b-float16"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("op_name, a_shape, b_shape, dtype", _BIAS_ROW_CASES)
def test_bias_row_grid_matches_torch(op_name, a_shape, b_shape, dtype):
    """Four rows a thread against one vector of the bias row, with the bias on either side.

    ``sub`` is ordered, so a swapped operand would not cancel out. The float16 case runs
    eight lanes a vector, and its NaNs in both operands reach the maximum body's NaN test.
    """
    a, b = _with_nans(a_shape, b_shape, dtype)
    TestBase.check(ElementwiseWorkload(op_name, (a, b)), getattr(elementwise_mod, op_name)(), a, b)


_ROUNDS_CASES = [
    pytest.param(
        "NeFwdOp",
        (-1, 4096),
        (1, 4096),
        torch.float16,
        id="ne-bias-b-float16",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        "GtFwdOp",
        (1, 4096),
        (-1, 4096),
        torch.bfloat16,
        id="gt-bias-a-bfloat16",
        marks=pytest.mark.smoke,
    ),
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("op_name, a_shape, b_shape, dtype", _ROUNDS_CASES)
def test_row_broadcast_rounds_match_torch(op_name, a_shape, b_shape, dtype):
    """Half the hinted blocks, each walking two of them in turn, with the bias on either side.

    The row count, ``-1`` in the case, makes the grid of 128-thread hinted blocks, two a
    4096-column row, three quarters of the device's resident threads. ``gt`` is ordered,
    so a swapped operand would not cancel out. Equal pairs and NaNs in both operands reach
    each comparison's true and false answers.
    """
    props = torch.cuda.get_device_properties(run_device())
    rows = 3 * props.multi_processor_count * props.max_threads_per_multi_processor // (4 * 128 * 2)
    a_shape = tuple(rows if d == -1 else d for d in a_shape)
    b_shape = tuple(rows if d == -1 else d for d in b_shape)
    a, b = _with_nans(a_shape, b_shape, dtype)
    full, bias = (a, b) if a.numel() > b.numel() else (b, a)
    full[..., 5::13] = bias[..., 5::13]
    TestBase.check(ElementwiseWorkload(op_name, (a, b)), getattr(elementwise_mod, op_name)(), a, b)
