"""Manifest-alignment conformance tests for ``elementwise_binary``.

Covers the 24 ops in ``tileops/manifest/elementwise_binary.yaml``:

- ``PreluFwdOp`` (channel-broadcast PReLU)
- ``MaskedFillFwdOp`` / ``MaskedFillScalarFwdOp``
- 9 binary arithmetic ops (Add/Sub/Mul/Div/Remainder/Pow/FloorDivide/Lerp/
  Maximum/Minimum)
- 6 binary comparison ops (Eq/Ne/Gt/Lt/Ge/Le)
- 2 binary logical ops (LogicalAnd/LogicalOr)
- 3 binary bitwise ops (BitwiseAnd/BitwiseOr/BitwiseXor)

Each test asserts the live Op class signature satisfies the manifest L1
contract: every manifest input appears in ``forward()`` in declaration
order, and every manifest param is reachable through either
``__init__()`` or ``forward()``.

A separate group of tests exercises bidirectional broadcast: every
broadcast-capable op must accept ``input.shape != other.shape`` and
produce the bidirectional broadcast output shape. The ``MaskedFillScalar``
dtype-widening contract (bool / uint8 / int8 / int16 / int32 / int64 /
float16 / bfloat16 / float32) is also covered here.
"""

import inspect

import pytest
import torch

# 24 ops in scope. (op_class_name, manifest_inputs (forward order),
# manifest_params (any field that lives on __init__ or forward).)
_BINARY_OPS = [
    # PReLU and MaskedFill are independent custom-signature ops.
    ("PreluFwdOp", ["input", "weight"], []),
    ("MaskedFillFwdOp", ["input", "mask", "value"], []),
    ("MaskedFillScalarFwdOp", ["input", "mask"], ["value"]),
    # Binary arithmetic ops.
    ("AddFwdOp", ["input", "other"], ["alpha"]),
    ("SubFwdOp", ["input", "other"], ["alpha"]),
    ("MulFwdOp", ["input", "other"], []),
    ("DivFwdOp", ["input", "other"], ["rounding_mode"]),
    ("RemainderFwdOp", ["input", "other"], []),
    ("PowFwdOp", ["input", "exponent"], []),
    ("FloorDivideFwdOp", ["input", "other"], []),
    ("LerpFwdOp", ["input", "end"], ["weight"]),
    ("MaximumFwdOp", ["input", "other"], []),
    ("MinimumFwdOp", ["input", "other"], []),
    # Binary comparison ops (output bool).
    ("EqFwdOp", ["input", "other"], []),
    ("NeFwdOp", ["input", "other"], []),
    ("GtFwdOp", ["input", "other"], []),
    ("LtFwdOp", ["input", "other"], []),
    ("GeFwdOp", ["input", "other"], []),
    ("LeFwdOp", ["input", "other"], []),
    # Binary logical ops (output bool).
    ("LogicalAndFwdOp", ["input", "other"], []),
    ("LogicalOrFwdOp", ["input", "other"], []),
    # Binary bitwise ops.
    ("BitwiseAndFwdOp", ["input", "other"], []),
    ("BitwiseOrFwdOp", ["input", "other"], []),
    ("BitwiseXorFwdOp", ["input", "other"], []),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, manifest_inputs, manifest_params",
    _BINARY_OPS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_binary_signature_matches_manifest(
    op_name: str, manifest_inputs: list[str], manifest_params: list[str],
) -> None:
    """Op class signatures must satisfy the manifest L1 contract."""
    import tileops.ops.elementwise as mod
    from scripts.validate_manifest import (
        _get_forward_params,
        _get_init_params,
        check_l1_signature,
    )

    cls = getattr(mod, op_name)
    forward_params = _get_forward_params(cls)
    assert forward_params is not None, (
        f"Cannot extract forward() params for {op_name}"
    )
    init_params = _get_init_params(cls)
    inputs_dict = {n: {} for n in manifest_inputs}
    params_dict = {n: {} for n in manifest_params}
    errors = check_l1_signature(
        op_name, inputs_dict, params_dict, forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"


# ---------------------------------------------------------------------------
# Bidirectional broadcast coverage
# ---------------------------------------------------------------------------
#
# Every broadcast-capable binary op must accept input.shape != other.shape.
# A representative bidirectional case ((3,1) x (1,4) -> (3,4)) exercises
# both directions: input contributes the leading axis, other contributes
# the trailing axis, neither operand alone is the output shape.

# (op_name, dtype, gen_a, gen_b, ref_fn)
_BROADCAST_FLOAT_OPS = [
    ("AddFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a + b),
    ("SubFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a - b),
    ("MulFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a * b),
    ("DivFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda a, b: a / b),
    ("RemainderFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda a, b: torch.remainder(a, b)),
    ("PowFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.5,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") * 2.0,
     lambda a, b: torch.pow(a, b)),
    ("FloorDivideFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda a, b: torch.floor_divide(a, b)),
    ("LerpFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: torch.lerp(a, b, 0.5)),
    ("MaximumFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: torch.maximum(a, b)),
    ("MinimumFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: torch.minimum(a, b)),
    ("EqFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: a == b),
    ("NeFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: a != b),
    ("GtFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a > b),
    ("LtFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a < b),
    ("GeFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a >= b),
    ("LeFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a <= b),
    ("LogicalAndFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: torch.logical_and(a, b)),
    ("LogicalOrFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: torch.logical_or(a, b)),
]

_BROADCAST_INT_OPS = [
    ("BitwiseAndFwdOp",
     torch.int32,
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda a, b: torch.bitwise_and(a, b)),
    ("BitwiseOrFwdOp",
     torch.int32,
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda a, b: torch.bitwise_or(a, b)),
    ("BitwiseXorFwdOp",
     torch.int32,
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda a, b: torch.bitwise_xor(a, b)),
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, dtype, gen_a, gen_b, ref_fn",
    _BROADCAST_FLOAT_OPS + _BROADCAST_INT_OPS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_binary_op_bidirectional_broadcast(
    op_name: str, dtype: torch.dtype, gen_a, gen_b, ref_fn,
) -> None:
    """Bidirectional broadcast: (3,1) x (1,4) -> (3,4)."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
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
    if out.dtype.is_floating_point:
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref.to(out.dtype))


# ---------------------------------------------------------------------------
# MaskedFillScalar dtype widening
# ---------------------------------------------------------------------------

# Manifest declares: bool | uint8 | int8 | int16 | int32 | int64 |
# float16 | bfloat16 | float32. Construction must succeed for every dtype
# at the impl level even when the underlying float-only kernel cannot run
# integer dtypes — the op layer routes integer inputs through a torch
# fallback so the manifest contract is honored end-to-end.
_MASKED_FILL_DTYPE_UNION = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
]


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _MASKED_FILL_DTYPE_UNION)
def test_masked_fill_scalar_accepts_manifest_dtype_union(
    dtype: torch.dtype,
) -> None:
    """MaskedFillScalarFwdOp must accept the full manifest dtype union."""
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    # Construction must succeed for every manifest-declared dtype.
    op = MaskedFillScalarFwdOp(
        input=(8,), mask=(8,), value=0, dtype=dtype,
    )
    assert op.dtype is dtype


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_masked_fill_scalar_int_fallback_matches_torch(
    dtype: torch.dtype,
) -> None:
    """Integer-dtype MaskedFillScalar must match torch.masked_fill."""
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    n = 16
    if dtype is torch.bool:
        x = torch.zeros(n, dtype=dtype, device="cuda")
        x[::2] = True
        fill_value = True
    else:
        x = torch.arange(n, dtype=dtype, device="cuda")
        fill_value = 7
    mask = torch.zeros(n, dtype=torch.bool, device="cuda")
    mask[1::2] = True
    op = MaskedFillScalarFwdOp(
        input=(n,), mask=(n,), value=fill_value, dtype=dtype,
    )
    out = op(x, mask)
    ref = x.masked_fill(mask, fill_value)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# Param plumbing for ops with extra manifest params
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_add_alpha_param_plumbed() -> None:
    """AddFwdOp must accept manifest-declared ``alpha`` param.

    PyTorch ``torch.add(input, other, *, alpha=1)`` defaults alpha to 1.
    """
    from tileops.ops.elementwise import AddFwdOp

    init_sig = inspect.signature(AddFwdOp.__init__)
    forward_sig = inspect.signature(AddFwdOp.forward)
    keys = set(init_sig.parameters) | set(forward_sig.parameters)
    assert "alpha" in keys, (
        f"AddFwdOp missing manifest 'alpha' param; init keys "
        f"{list(init_sig.parameters)}, forward keys "
        f"{list(forward_sig.parameters)}"
    )


@pytest.mark.smoke
def test_sub_alpha_param_plumbed() -> None:
    """SubFwdOp must accept manifest-declared ``alpha`` param."""
    from tileops.ops.elementwise import SubFwdOp

    init_sig = inspect.signature(SubFwdOp.__init__)
    forward_sig = inspect.signature(SubFwdOp.forward)
    keys = set(init_sig.parameters) | set(forward_sig.parameters)
    assert "alpha" in keys, f"SubFwdOp missing 'alpha'; got {keys}"


@pytest.mark.smoke
def test_div_rounding_mode_param_plumbed() -> None:
    """DivFwdOp must accept manifest-declared ``rounding_mode`` param."""
    from tileops.ops.elementwise import DivFwdOp

    init_sig = inspect.signature(DivFwdOp.__init__)
    forward_sig = inspect.signature(DivFwdOp.forward)
    keys = set(init_sig.parameters) | set(forward_sig.parameters)
    assert "rounding_mode" in keys, (
        f"DivFwdOp missing 'rounding_mode'; got {keys}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
