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

import tileops.ops.elementwise as elementwise_mod
from tests.test_base import TestBase
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


_STAGED_SHAPES = [
    pytest.param((4, 1), (4, 1000), id="inner-stride-0-and-1"),
    pytest.param((4, 1000), (1, 1000), id="inner-stride-1-and-1"),
    pytest.param((3, 5000), (3, 1), id="inner-spans-several-blocks"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("a_shape, b_shape", _STAGED_SHAPES)
def test_staged_row_broadcast_matches_torch(a_shape, b_shape):
    """A staged predicate broadcast agrees with torch on every stride pair."""
    from tileops.ops.elementwise import GtFwdOp

    a = torch.randn(a_shape, device=run_device(), dtype=torch.float16)
    b = torch.randn(b_shape, device=run_device(), dtype=torch.float16)
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
    from tileops.ops.elementwise import AddFwdOp

    a = torch.randn(a_shape, device=run_device(), dtype=torch.float32)
    b = torch.randn(b_shape, device=run_device(), dtype=torch.float32)
    workload = ElementwiseWorkload("AddFwdOp", (a, b))
    TestBase.check(workload, AddFwdOp(), a, b)
