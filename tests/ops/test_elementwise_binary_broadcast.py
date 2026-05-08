"""Bidirectional-broadcast behavior tests for ``elementwise_binary`` ops.

L1 signature/parity — forward arg names (C4), ``__init__`` defaults
(C3), and structural ``__init__`` shape with ``kernel_map`` kwarg + a
``self.dispatch_kernel(...)`` call (C5) — is enforced by
``scripts/validate_manifest.py`` strict-parity gates; per-op pytest
fanouts mirroring the manifest are redundant. The runtime invariant
that ``kernel_map=`` is actually *forwarded* into ``dispatch_kernel``
(which C5 does not verify) lives in
``tests/ops/test_op_kernel_map_override.py``. What remains here is the
load-bearing external behavior: bidirectional broadcast against a
PyTorch reference.
"""

from __future__ import annotations

import pytest
import torch

import tileops.ops.elementwise as elementwise_mod


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
    ("AddFwdOp",         _F16, _randn,     _randn,     lambda a, b: a + b),
    ("SubFwdOp",         _F16, _randn,     _randn,     lambda a, b: a - b),
    ("MulFwdOp",         _F16, _randn,     _randn,     lambda a, b: a * b),
    ("DivFwdOp",         _F16, _rand_pos,  _rand_pos,  lambda a, b: a / b),
    ("RemainderFwdOp",   _F16, _rand_pos,  _rand_pos,  lambda a, b: torch.remainder(a, b)),
    ("PowFwdOp",         _F16, _pow_base,  _pow_exp,   lambda a, b: torch.pow(a, b)),
    # floor_divide reference computed in fp32 to match the kernel's internal
    # path; tolerance widened to 1.0 because rounding boundaries flip the
    # quotient by ±1 around exact integer ratios — see test_binary_arith.py.
    ("FloorDivideFwdOp", _F16, _rand_pos,  _rand_pos,
     lambda a, b: torch.floor(a.float() / b.float()).to(a.dtype)),
    ("LerpFwdOp",        _F16, _randn,     _randn,     lambda a, b: torch.lerp(a, b, 0.5)),
    ("MaximumFwdOp",     _F16, _randn,     _randn,     lambda a, b: torch.maximum(a, b)),
    ("MinimumFwdOp",     _F16, _randn,     _randn,     lambda a, b: torch.minimum(a, b)),
    ("EqFwdOp",          _F16, _rand_bool, _rand_bool, lambda a, b: a == b),
    ("NeFwdOp",          _F16, _rand_bool, _rand_bool, lambda a, b: a != b),
    ("GtFwdOp",          _F16, _randn,     _randn,     lambda a, b: a > b),
    ("LtFwdOp",          _F16, _randn,     _randn,     lambda a, b: a < b),
    ("GeFwdOp",          _F16, _randn,     _randn,     lambda a, b: a >= b),
    ("LeFwdOp",          _F16, _randn,     _randn,     lambda a, b: a <= b),
    ("LogicalAndFwdOp",  _F16, _rand_bool, _rand_bool, lambda a, b: torch.logical_and(a, b)),
    ("LogicalOrFwdOp",   _F16, _rand_bool, _rand_bool, lambda a, b: torch.logical_or(a, b)),
    ("BitwiseAndFwdOp",  _I32, _randint,   _randint,   lambda a, b: torch.bitwise_and(a, b)),
    ("BitwiseOrFwdOp",   _I32, _randint,   _randint,   lambda a, b: torch.bitwise_or(a, b)),
    ("BitwiseXorFwdOp",  _I32, _randint,   _randint,   lambda a, b: torch.bitwise_xor(a, b)),
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, dtype, gen_a, gen_b, ref_fn",
    _BROADCAST_OPS,
    ids=[entry[0] for entry in _BROADCAST_OPS],
)
def test_binary_op_bidirectional_broadcast(
    op_name: str, dtype: torch.dtype, gen_a, gen_b, ref_fn,
) -> None:
    """Bidirectional broadcast: (3,1) x (1,4) -> (3,4)."""
    cls = getattr(elementwise_mod, op_name)
    a_shape = (3, 1)
    b_shape = (1, 4)
    a = gen_a(a_shape, dtype)
    b = gen_b(b_shape, dtype)
    op = cls(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
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
