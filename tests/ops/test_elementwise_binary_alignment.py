"""Manifest-alignment conformance tests for ``elementwise_binary``.

Covers the 24 ops in ``tileops/manifest/elementwise_binary.yaml``:

- ``PreluFwdOp`` (channel-broadcast PReLU)
- ``MaskedFillFwdOp`` / ``MaskedFillScalarFwdOp``
- 10 binary arithmetic ops (Add/Sub/Mul/Div/Remainder/Pow/FloorDivide/Lerp/
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


# ---------------------------------------------------------------------------
# Functional correctness for non-default alpha / rounding_mode
# ---------------------------------------------------------------------------
#
# The Op layer routes non-default ``alpha`` / ``rounding_mode`` through a
# torch eager fallback (the kernel does not bake these in). These tests
# guard the runtime semantics on the fallback path so the manifest
# contract for non-default values is honored end-to-end, not just at the
# signature level.


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_add_alpha_nondefault_matches_torch() -> None:
    """``AddFwdOp(alpha=2)`` must agree with ``torch.add(..., alpha=2)``."""
    from tileops.ops.elementwise import AddFwdOp

    a = torch.randn(8, dtype=torch.float16, device="cuda")
    b = torch.randn(8, dtype=torch.float16, device="cuda")
    op = AddFwdOp(a_shape=(8,), b_shape=(8,), dtype=torch.float16, alpha=2)
    out = op(a, b)
    ref = torch.add(a, b, alpha=2)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sub_alpha_nondefault_matches_torch() -> None:
    """``SubFwdOp(alpha=3)`` must agree with ``torch.sub(..., alpha=3)``."""
    from tileops.ops.elementwise import SubFwdOp

    a = torch.randn(8, dtype=torch.float16, device="cuda")
    b = torch.randn(8, dtype=torch.float16, device="cuda")
    op = SubFwdOp(a_shape=(8,), b_shape=(8,), dtype=torch.float16, alpha=3)
    out = op(a, b)
    ref = torch.sub(a, b, alpha=3)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_div_rounding_mode_nondefault_matches_torch(rounding_mode: str) -> None:
    """``DivFwdOp(rounding_mode=...)`` must agree with ``torch.div``."""
    from tileops.ops.elementwise import DivFwdOp

    a = torch.randn(8, dtype=torch.float32, device="cuda") * 4.0
    b = torch.rand(8, dtype=torch.float32, device="cuda") + 0.5
    op = DivFwdOp(
        a_shape=(8,), b_shape=(8,), dtype=torch.float32,
        rounding_mode=rounding_mode,
    )
    out = op(a, b)
    ref = torch.div(a, b, rounding_mode=rounding_mode)
    torch.testing.assert_close(out, ref)


@pytest.mark.smoke
def test_alpha_and_rounding_mode_are_keyword_only() -> None:
    """``alpha`` / ``rounding_mode`` must be keyword-only.

    Inserting them positionally before ``strategy/kernel_map/tune`` would
    silently re-bind any existing positional ``strategy`` arguments to
    the new param.
    """
    from tileops.ops.elementwise import AddFwdOp, DivFwdOp, SubFwdOp

    for cls, name in [(AddFwdOp, "alpha"), (SubFwdOp, "alpha"), (DivFwdOp, "rounding_mode")]:
        sig = inspect.signature(cls.__init__)
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{cls.__name__}.__init__({name}) must be keyword-only, "
            f"got kind={param.kind}"
        )


# ---------------------------------------------------------------------------
# Manifest dtype-union coverage
# ---------------------------------------------------------------------------
#
# The manifest declares the full
# ``bool | uint8 | int8 | int16 | int32 | int64 | float16 | bfloat16 |
# float32`` dtype union for arithmetic Add/Sub/Mul/Maximum/Minimum and
# every comparison op (Eq/Ne/Gt/Lt/Ge/Le). The underlying TileLang
# kernel only supports float dtypes, so the op layer routes integer /
# bool inputs through a torch eager fallback. These tests:
#
#   1. ``test_*_construction_*`` -- construction must succeed for every
#      manifest dtype (was previously ``ValueError`` for non-float).
#   2. ``test_*_int_fallback_matches_torch`` -- the fallback path must
#      match ``torch`` runtime semantics.

_BINARY_FULL_UNION = [
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

# Bool is excluded for ops where torch itself rejects bool-in-bool-out
# arithmetic (sub does not accept bool).
_ARITH_DTYPES_NO_BOOL = [d for d in _BINARY_FULL_UNION if d is not torch.bool]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name", ["AddFwdOp", "MulFwdOp", "MaximumFwdOp", "MinimumFwdOp"],
)
@pytest.mark.parametrize("dtype", _BINARY_FULL_UNION)
def test_arith_construction_accepts_manifest_dtype_union(
    op_name: str, dtype: torch.dtype,
) -> None:
    """Add/Mul/Maximum/Minimum must construct for every manifest dtype."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    op = cls(a_shape=(8,), b_shape=(8,), dtype=dtype)
    assert op.dtype is dtype


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _ARITH_DTYPES_NO_BOOL)
def test_sub_construction_accepts_manifest_dtype_union(
    dtype: torch.dtype,
) -> None:
    """SubFwdOp must construct for every manifest dtype except bool."""
    from tileops.ops.elementwise import SubFwdOp

    op = SubFwdOp(a_shape=(8,), b_shape=(8,), dtype=dtype)
    assert op.dtype is dtype


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name",
    ["EqFwdOp", "NeFwdOp", "GtFwdOp", "LtFwdOp", "GeFwdOp", "LeFwdOp"],
)
@pytest.mark.parametrize("dtype", _BINARY_FULL_UNION)
def test_compare_construction_accepts_manifest_dtype_union(
    op_name: str, dtype: torch.dtype,
) -> None:
    """Comparison ops must construct for every manifest dtype."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    op = cls(a_shape=(8,), b_shape=(8,), dtype=dtype)
    assert op.dtype is dtype


# (op_name, torch ref) for arithmetic ops that share the full union.
_ARITH_INT_FALLBACK = [
    ("AddFwdOp", lambda a, b: a + b),
    ("MulFwdOp", lambda a, b: a * b),
    ("MaximumFwdOp", torch.maximum),
    ("MinimumFwdOp", torch.minimum),
]

_COMPARE_INT_FALLBACK = [
    ("EqFwdOp", torch.eq),
    ("NeFwdOp", torch.ne),
    ("GtFwdOp", torch.gt),
    ("LtFwdOp", torch.lt),
    ("GeFwdOp", torch.ge),
    ("LeFwdOp", torch.le),
]

_INT_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name, ref_fn", _ARITH_INT_FALLBACK)
@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_arith_int_fallback_matches_torch(
    op_name: str, ref_fn, dtype: torch.dtype,
) -> None:
    """Integer arithmetic ops must match torch via the op-layer fallback."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    a = torch.randint(0, 8, (8,), dtype=dtype, device="cuda")
    b = torch.randint(1, 8, (8,), dtype=dtype, device="cuda")
    op = cls(a_shape=(8,), b_shape=(8,), dtype=dtype)
    out = op(a, b)
    ref = ref_fn(a, b)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name, ref_fn", _COMPARE_INT_FALLBACK)
@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_compare_int_fallback_matches_torch(
    op_name: str, ref_fn, dtype: torch.dtype,
) -> None:
    """Integer comparison ops must match torch via the op-layer fallback."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    a = torch.randint(0, 4, (8,), dtype=dtype, device="cuda")
    b = torch.randint(0, 4, (8,), dtype=dtype, device="cuda")
    op = cls(a_shape=(8,), b_shape=(8,), dtype=dtype)
    out = op(a, b)
    ref = ref_fn(a, b)
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, ref)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, ref_fn",
    [
        ("AddFwdOp", torch.add),
        ("MulFwdOp", torch.mul),
        ("MaximumFwdOp", torch.maximum),
        ("MinimumFwdOp", torch.minimum),
    ],
)
def test_arith_bool_fallback_matches_torch(op_name: str, ref_fn) -> None:
    """Bool arithmetic ops in the manifest dtype union must match torch."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    a = torch.tensor([True, False, True, False, True, False, True, False],
                     dtype=torch.bool, device="cuda")
    b = torch.tensor([True, True, False, False, True, True, False, False],
                     dtype=torch.bool, device="cuda")
    op = cls(a_shape=(8,), b_shape=(8,), dtype=torch.bool)
    out = op(a, b)
    ref = ref_fn(a, b)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
