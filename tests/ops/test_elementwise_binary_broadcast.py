"""Bidirectional-broadcast behavior tests for ``elementwise_binary`` ops.

L1 signature/parity (forward arg names, ``__init__`` defaults, manifest
entry resolution, construction smoke) is specified by
``scripts/validate_manifest.py`` strict-parity gates C3/C4/C5. This
file covers the load-bearing external behavior: bidirectional
broadcast against a PyTorch reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import tileops.ops.elementwise as elementwise_mod
from tileops.perf import formulas


def _randn(s, d):
    return torch.randn(*s, dtype=d, device="cuda")


def _rand_pos(s, d):
    return torch.rand(*s, dtype=d, device="cuda") + 0.1


def _rand_bool(s, d):
    return (torch.randn(*s, dtype=d, device="cuda") > 0).to(d)


def _randint(s, d):
    return torch.randint(-1000, 1000, s, dtype=d, device="cuda")


def _pow_base(s, d):
    return torch.rand(*s, dtype=d, device="cuda") + 0.5


def _pow_exp(s, d):
    return torch.rand(*s, dtype=d, device="cuda") * 2.0


# (op_name, dtype, gen_a, gen_b, ref_fn).
_F16 = torch.float16
_I32 = torch.int32

_BROADCAST_OPS = [
    ("AddFwdOp", _F16, _randn, _randn, lambda a, b: a + b),
    ("SubFwdOp", _F16, _randn, _randn, lambda a, b: a - b),
    ("MulFwdOp", _F16, _randn, _randn, lambda a, b: a * b),
    ("DivFwdOp", _F16, _rand_pos, _rand_pos, lambda a, b: a / b),
    ("RemainderFwdOp", _F16, _rand_pos, _rand_pos, lambda a, b: torch.remainder(a, b)),
    ("PowFwdOp", _F16, _pow_base, _pow_exp, lambda a, b: torch.pow(a, b)),
    # floor_divide reference computed in fp32 to match the kernel's internal
    # path; tolerance widened to 1.0 because rounding boundaries flip the
    # quotient by ±1 around exact integer ratios — see test_binary_arith.py.
    (
        "FloorDivideFwdOp",
        _F16,
        _rand_pos,
        _rand_pos,
        lambda a, b: torch.floor(a.float() / b.float()).to(a.dtype),
    ),
    ("LerpFwdOp", _F16, _randn, _randn, lambda a, b: torch.lerp(a, b, 0.5)),
    ("MaximumFwdOp", _F16, _randn, _randn, lambda a, b: torch.maximum(a, b)),
    ("MinimumFwdOp", _F16, _randn, _randn, lambda a, b: torch.minimum(a, b)),
    ("EqFwdOp", _F16, _rand_bool, _rand_bool, lambda a, b: a == b),
    ("NeFwdOp", _F16, _rand_bool, _rand_bool, lambda a, b: a != b),
    ("GtFwdOp", _F16, _randn, _randn, lambda a, b: a > b),
    ("LtFwdOp", _F16, _randn, _randn, lambda a, b: a < b),
    ("GeFwdOp", _F16, _randn, _randn, lambda a, b: a >= b),
    ("LeFwdOp", _F16, _randn, _randn, lambda a, b: a <= b),
    ("LogicalAndFwdOp", _F16, _rand_bool, _rand_bool, lambda a, b: torch.logical_and(a, b)),
    ("LogicalOrFwdOp", _F16, _rand_bool, _rand_bool, lambda a, b: torch.logical_or(a, b)),
    ("BitwiseAndFwdOp", _I32, _randint, _randint, lambda a, b: torch.bitwise_and(a, b)),
    ("BitwiseOrFwdOp", _I32, _randint, _randint, lambda a, b: torch.bitwise_or(a, b)),
    ("BitwiseXorFwdOp", _I32, _randint, _randint, lambda a, b: torch.bitwise_xor(a, b)),
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, dtype, gen_a, gen_b, ref_fn",
    _BROADCAST_OPS,
    ids=[entry[0] for entry in _BROADCAST_OPS],
)
def test_binary_op_bidirectional_broadcast(
    op_name: str,
    dtype: torch.dtype,
    gen_a,
    gen_b,
    ref_fn,
) -> None:
    """Bidirectional broadcast: (3,1) x (1,4) -> (3,4)."""
    cls = getattr(elementwise_mod, op_name)
    a_shape = (3, 1)
    b_shape = (1, 4)
    a = gen_a(a_shape, dtype)
    b = gen_b(b_shape, dtype)
    op = cls()
    out = op(a, b)
    ref = ref_fn(a, b)
    assert tuple(out.shape) == (3, 4), (
        f"{op_name}: expected output shape (3, 4), got {tuple(out.shape)}"
    )
    if op_name == "FloorDivideFwdOp":
        atol, rtol = 1.0, 1e-2  # boundary rounding flips quotient by ±1
    elif out.dtype.is_floating_point:
        atol, rtol = 1e-2, 1e-2
    else:
        torch.testing.assert_close(out, ref.to(out.dtype))
        return
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name", ["MaximumFwdOp", "DivFwdOp"])
def test_channel_broadcast_with_ragged_inner_dim(op_name: str) -> None:
    """A per-channel operand over a non-tile-multiple inner dim.

    The row-broadcast body splits at trace time into full blocks and one
    guarded tail block; 300 columns force the tail. ``div`` is ordered, so a
    swapped operand would not cancel out.
    """
    cls = getattr(elementwise_mod, op_name)
    a = torch.randn(2, 3, 10, 30, dtype=torch.float16, device="cuda")
    b = torch.rand(3, 1, 1, dtype=torch.float16, device="cuda") + 0.5
    ref = torch.maximum(a, b) if op_name == "MaximumFwdOp" else a / b
    out = cls()(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# ----------------------------------------------------------------------
# Spec pins for the broadcast-binary roofline helpers (no CUDA build).
# ----------------------------------------------------------------------


@dataclass
class _StubBinaryOp:
    a_numel: int
    b_numel: int
    N_total: int
    dtype: torch.dtype


@pytest.mark.smoke
def test_broadcast_binary_helper_no_broadcast():
    """When inputs share the output shape, a_numel == b_numel == N_total."""
    op = _StubBinaryOp(a_numel=1024, b_numel=1024, N_total=1024, dtype=torch.float32)
    flops, nbytes = formulas.add_fwd_roofline(op)
    assert flops == 2 * 1024
    # 2 reads (4 bytes each) + 1 write (4 bytes) per element
    assert nbytes == (1024 + 1024 + 1024) * 4


@pytest.mark.smoke
def test_broadcast_binary_helper_bool_output_byte_accounting():
    """Comparison ops emit a 1-byte output regardless of input dtype."""
    op = _StubBinaryOp(a_numel=1024, b_numel=1024, N_total=1024, dtype=torch.float32)
    flops, nbytes = formulas.eq_fwd_roofline(op)
    assert flops == 1024
    # 2 fp32 reads + 1 bool write
    assert nbytes == (1024 + 1024) * 4 + 1024


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

    a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
    b = torch.randn(b_shape, device="cuda", dtype=torch.float16)
    out = GtFwdOp()(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, torch.gt(a, b))


# A row extent of 1088 leaves 64 columns past two full blocks of 512, which is
# within the share the builder packs across rows; 5000 leaves 392, which is not.
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

    a = torch.randn(a_shape, device="cuda", dtype=torch.float32)
    b = torch.randn(b_shape, device="cuda", dtype=torch.float32)
    assert torch.equal(AddFwdOp()(a, b), a + b)
