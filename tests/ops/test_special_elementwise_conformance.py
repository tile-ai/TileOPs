"""Conformance tests for elementwise multi-input ops.

Covers PyTorch-aligned signatures, broadcasting semantics, and split
variants (Tensor-bound clamp / masked_fill, single-bound clamp_min /
clamp_max). Once these tests pass, the corresponding manifest entries
can flip from ``status: spec-only`` to ``status: implemented`` per the
manifest trust model (.claude/rules/manifest-trust-model.md).
"""

import inspect

import pytest
import torch

# ---------------------------------------------------------------------------
# WhereFwdOp full broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cond_shape, inp_shape, other_shape, dtype",
    [
        ((4, 8), (4, 8), (4, 8), torch.float16),  # same shape
        ((1, 8), (4, 1), (1, 1), torch.float32),  # full 3-way broadcast
        ((4, 8), (1, 8), (4, 1), torch.bfloat16),  # mixed broadcast
        ((), (4, 8), (4, 8), torch.float16),  # 0-dim condition
    ],
)
def test_where_broadcast_parity(cond_shape, inp_shape, other_shape, dtype):
    from tileops.ops.elementwise import WhereFwdOp

    cond = torch.randint(0, 2, cond_shape, device="cuda").bool() if cond_shape else \
        torch.tensor(True, device="cuda")
    inp = torch.randn(inp_shape, device="cuda", dtype=dtype)
    other = torch.randn(other_shape, device="cuda", dtype=dtype)
    ref = torch.where(cond, inp, other)

    op = WhereFwdOp(condition=tuple(cond.shape), input=tuple(inp.shape),
                    other=tuple(other.shape), dtype=dtype)
    out = op(cond, inp, other)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.smoke
def test_where_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import WhereFwdOp
    init_params = list(inspect.signature(WhereFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(WhereFwdOp.forward).parameters.keys())
    # __init__: self, condition, input, other, dtype, ...
    assert init_params[1:5] == ["condition", "input", "other", "dtype"], init_params
    # forward: self, condition, input, other
    assert fwd_params[1:] == ["condition", "input", "other"], fwd_params


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float32, torch.int32],
)
def test_where_rejects_non_bool_condition(bad_dtype):
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.zeros(shape, device="cuda", dtype=bad_dtype)
    inp = torch.randn(shape, device="cuda", dtype=torch.float16)
    other = torch.randn(shape, device="cuda", dtype=torch.float16)
    op = WhereFwdOp(
        condition=shape, input=shape, other=shape, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="condition.dtype torch.bool"):
        op(cond, inp, other)


# ---------------------------------------------------------------------------
# ClampFwdOp Tensor min/max
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, min_shape, max_shape, dtype",
    [
        ((4, 8), (4, 8), (4, 8), torch.float16),
        ((4, 8), (1, 8), (4, 1), torch.float32),
        ((4, 8), (), (), torch.bfloat16),  # 0-dim Tensor bounds
    ],
)
def test_clamp_tensor_bounds_parity(input_shape, min_shape, max_shape, dtype):
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=dtype)
    mn = torch.randn(min_shape, device="cuda", dtype=dtype) - 0.5
    mx = torch.randn(max_shape, device="cuda", dtype=dtype) + 0.5
    # Make max >= min where tested ranges overlap; PyTorch clamp tolerates
    # mismatch but we want a meaningful ref.
    ref = torch.clamp(inp, mn, mx)

    op = ClampFwdOp(input=tuple(inp.shape), min=tuple(mn.shape),
                    max=tuple(mx.shape), dtype=dtype)
    out = op(inp, mn, mx)
    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1.6e-2, 1.6e-2
    else:
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_clamp_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import ClampFwdOp
    init_params = list(inspect.signature(ClampFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(ClampFwdOp.forward).parameters.keys())
    assert init_params[1:5] == ["input", "min", "max", "dtype"], init_params
    assert fwd_params[1:] == ["input", "min", "max"], fwd_params


# ClampFwdOp must accept Tensor min with max=None and
# Tensor max with min=None, matching torch.clamp(input, min=tensor, max=None)
# and torch.clamp(input, min=None, max=tensor) on CUDA. Single-bound shape
# matrix is covered by test_clamp_min_tensor / test_clamp_max_tensor on the
# dedicated ClampMin/Max ops; here we only verify ClampFwdOp's None routing.
@pytest.mark.smoke
def test_clamp_min_only_tensor_parity():
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    mn = torch.randn((4, 8), device="cuda", dtype=torch.float32) - 0.5
    ref = torch.clamp(inp, mn, None)
    op = ClampFwdOp(input=(4, 8), min=(4, 8), max=None, dtype=torch.float32)
    out = op(inp, mn, None)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_max_only_tensor_parity():
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    mx = torch.randn((4, 8), device="cuda", dtype=torch.float32) + 0.5
    ref = torch.clamp(inp, None, mx)
    op = ClampFwdOp(input=(4, 8), min=None, max=(4, 8), dtype=torch.float32)
    out = op(inp, None, mx)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_both_none_rejected():
    """ClampFwdOp must reject min=None and max=None (no-op clamp is invalid)."""
    from tileops.ops.elementwise import ClampFwdOp
    with pytest.raises(ValueError, match="at least one of"):
        ClampFwdOp(input=(4,), min=None, max=None, dtype=torch.float32)


@pytest.mark.smoke
def test_clamp_scalar_both_none_rejected():
    """ClampScalarFwdOp must reject min=None and max=None.

    Mirrors torch.clamp(input, None, None), which raises
    RuntimeError("At least one of min or max must not be None").
    """
    from tileops.ops.elementwise import ClampScalarFwdOp
    with pytest.raises(ValueError, match="at least one of"):
        ClampScalarFwdOp(input=(4,), min=None, max=None, dtype=torch.float32)


@pytest.mark.smoke
def test_clamp_scalar_rejects_same_numel_wrong_shape():
    """ClampScalarFwdOp.forward must validate full input.shape, not just numel."""
    from tileops.ops.elementwise import ClampScalarFwdOp

    op = ClampScalarFwdOp(input=(2, 3), min=0.0, max=1.0, dtype=torch.float32)
    bad = torch.randn(6, device="cuda", dtype=torch.float32)  # same numel, wrong shape
    with pytest.raises(ValueError, match=r"input\.shape"):
        op(bad)


@pytest.mark.smoke
def test_clamp_runtime_tensor_none_must_match_init():
    """Forward-time None / Tensor presence must agree with __init__ config."""
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(4, device="cuda", dtype=torch.float32)
    mn = torch.zeros(4, device="cuda", dtype=torch.float32)

    # Configured for min-only at __init__, then passed a Tensor for max:
    op = ClampFwdOp(input=(4,), min=(4,), max=None, dtype=torch.float32)
    with pytest.raises(ValueError, match="max"):
        op(inp, mn, mn)

    # Configured for max-only at __init__, then passed a Tensor for min:
    op2 = ClampFwdOp(input=(4,), min=None, max=(4,), dtype=torch.float32)
    with pytest.raises(ValueError, match="min"):
        op2(inp, mn, mn)


# ---------------------------------------------------------------------------
# ClampScalarFwdOp / ClampMinFwdOp / ClampMaxFwdOp
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "min_val, max_val",
    [(-0.5, 0.5), (None, 0.5), (-0.5, None)],
)
def test_clamp_scalar_param_names(min_val, max_val):
    from tileops.ops.elementwise import ClampScalarFwdOp

    inp = torch.randn(1024, device="cuda", dtype=torch.float32)
    ref = torch.clamp(inp, min_val, max_val)
    op = ClampScalarFwdOp(input=(1024,), min=min_val, max=max_val, dtype=torch.float32)
    out = op(inp)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_scalar_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import ClampScalarFwdOp
    init_params = list(inspect.signature(ClampScalarFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(ClampScalarFwdOp.forward).parameters.keys())
    # __init__ exposes manifest params (min, max) and the input
    assert "input" in init_params
    assert "min" in init_params
    assert "max" in init_params
    assert "dtype" in init_params
    # forward only takes input (manifest params are bound at __init__)
    assert fwd_params[1:] == ["input"], fwd_params


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, min_shape",
    [((4, 8), (4, 8)), ((4, 8), (1, 8)), ((4, 8), ())],
)
def test_clamp_min_tensor(input_shape, min_shape):
    from tileops.ops.elementwise import ClampMinFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    mn = torch.randn(min_shape, device="cuda", dtype=torch.float32)
    ref = torch.clamp_min(inp, mn) if min_shape else torch.clamp(inp, min=mn.item())

    op = ClampMinFwdOp(input=tuple(inp.shape), min=tuple(mn.shape), dtype=torch.float32)
    out = op(inp, mn)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_min_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import ClampMinFwdOp
    init_params = list(inspect.signature(ClampMinFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(ClampMinFwdOp.forward).parameters.keys())
    assert init_params[1:4] == ["input", "min", "dtype"], init_params
    assert fwd_params[1:] == ["input", "min"], fwd_params


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, max_shape",
    [((4, 8), (4, 8)), ((4, 8), (4, 1)), ((4, 8), ())],
)
def test_clamp_max_tensor(input_shape, max_shape):
    from tileops.ops.elementwise import ClampMaxFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    mx = torch.randn(max_shape, device="cuda", dtype=torch.float32)
    ref = torch.clamp_max(inp, mx) if max_shape else torch.clamp(inp, max=mx.item())

    op = ClampMaxFwdOp(input=tuple(inp.shape), max=tuple(mx.shape), dtype=torch.float32)
    out = op(inp, mx)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_max_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import ClampMaxFwdOp
    init_params = list(inspect.signature(ClampMaxFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(ClampMaxFwdOp.forward).parameters.keys())
    assert init_params[1:4] == ["input", "max", "dtype"], init_params
    assert fwd_params[1:] == ["input", "max"], fwd_params


# ---------------------------------------------------------------------------
# Regression: NaN propagation for Tensor-bound clamp variants.
#
# torch.clamp / torch.clamp_min / torch.clamp_max propagate NaN: if any of
# input / min / max is NaN at position i, the output at i is NaN. CUDA's
# fmax / fmin (used by T.max / T.min) drop NaN by returning the non-NaN
# operand, so the kernel adds explicit isnan guards. These tests pin the
# semantics so a future refactor cannot regress to non-IEEE behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_tensor_nan_propagation(dtype):
    """ClampFwdOp must match torch.clamp NaN semantics (Tensor min + max)."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device="cuda", dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device="cuda", dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device="cuda", dtype=dtype)

    ref = torch.clamp(x, mn, mx)
    op = ClampFwdOp(input=(4,), min=(4,), max=(4,), dtype=dtype)
    out = op(x, mn, mx)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


# Single-bound NaN behaviour is covered by test_clamp_min_nan_propagation /
# test_clamp_max_nan_propagation below — same ClampTensorFwdKernel branch
# (has_min only / has_max only). ClampFwdOp(min=Tensor, max=None) /
# (min=None, max=Tensor) dispatch is verified separately.


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_min_nan_propagation(dtype):
    """ClampMinFwdOp must match torch.clamp_min NaN semantics."""
    from tileops.ops.elementwise import ClampMinFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device="cuda", dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device="cuda", dtype=dtype)

    ref = torch.clamp_min(x, mn)
    op = ClampMinFwdOp(input=(4,), min=(4,), dtype=dtype)
    out = op(x, mn)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_max_nan_propagation(dtype):
    """ClampMaxFwdOp must match torch.clamp_max NaN semantics."""
    from tileops.ops.elementwise import ClampMaxFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device="cuda", dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device="cuda", dtype=dtype)

    ref = torch.clamp_max(x, mx)
    op = ClampMaxFwdOp(input=(4,), max=(4,), dtype=dtype)
    out = op(x, mx)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# MaskedFillFwdOp (0-dim Tensor) / MaskedFillScalarFwdOp (Number)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, mask_shape",
    [((4, 8), (4, 8)), ((1, 8), (4, 8)), ((4, 8), (1, 8)), ((2, 1), (2, 3))],
)
def test_masked_fill_tensor_value(input_shape, mask_shape):
    from tileops.ops.elementwise import MaskedFillFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    mask = torch.randint(0, 2, mask_shape, device="cuda").bool()
    value = torch.tensor(-1.5, device="cuda", dtype=torch.float32)

    out_shape = torch.broadcast_shapes(input_shape, mask_shape)
    ref = inp.expand(out_shape).clone().masked_fill(mask.expand(out_shape), value.item())

    op = MaskedFillFwdOp(input=tuple(inp.shape), mask=tuple(mask.shape),
                        value=tuple(value.shape), dtype=torch.float32)
    out = op(inp, mask, value)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_masked_fill_tensor_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import MaskedFillFwdOp
    init_params = list(inspect.signature(MaskedFillFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(MaskedFillFwdOp.forward).parameters.keys())
    assert init_params[1:5] == ["input", "mask", "value", "dtype"], init_params
    assert fwd_params[1:] == ["input", "mask", "value"], fwd_params


@pytest.mark.smoke
def test_masked_fill_scalar_param_names():
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    inp = torch.randn(1024, device="cuda", dtype=torch.float32)
    mask = torch.randint(0, 2, (1024,), device="cuda").bool()
    ref = inp.masked_fill(mask, -1.0)

    op = MaskedFillScalarFwdOp(input=(1024,), mask=(1024,), value=-1.0,
                               dtype=torch.float32)
    out = op(inp, mask)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_masked_fill_scalar_init_signature_pytorch_aligned():
    from tileops.ops.elementwise import MaskedFillScalarFwdOp
    init_params = list(inspect.signature(MaskedFillScalarFwdOp.__init__).parameters.keys())
    fwd_params = list(inspect.signature(MaskedFillScalarFwdOp.forward).parameters.keys())
    assert "input" in init_params
    assert "mask" in init_params
    assert "value" in init_params
    assert "dtype" in init_params
    assert fwd_params[1:] == ["input", "mask"], fwd_params


# ---------------------------------------------------------------------------
# Validator passes: this test exercises the L1 signature check directly
# so it doesn't depend on the manifest YAML ``status`` value.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, expected_inputs, expected_params",
    [
        ("WhereFwdOp", ["condition", "input", "other"], []),
        ("ClampFwdOp", ["input", "min", "max"], []),
        ("ClampScalarFwdOp", ["input"], ["min", "max"]),
        ("ClampMinFwdOp", ["input", "min"], []),
        ("ClampMaxFwdOp", ["input", "max"], []),
        ("MaskedFillFwdOp", ["input", "mask", "value"], []),
        ("MaskedFillScalarFwdOp", ["input", "mask"], ["value"]),
    ],
)
def test_l1_signature_conformance(op_name, expected_inputs, expected_params):
    """L1 signature check (validator parity) for each conformed op class.

    Mirrors ``scripts.validate_manifest.check_l1_signature``: forward()
    must list manifest inputs in order, and every manifest param must
    appear in either ``__init__()`` or ``forward()``.
    """
    import tileops.ops.elementwise as mod
    from scripts.validate_manifest import (
        _get_forward_params,
        _get_init_params,
        check_l1_signature,
    )

    cls = getattr(mod, op_name)
    forward_params = _get_forward_params(cls)
    assert forward_params is not None, f"Cannot extract forward params for {op_name}"
    init_params = _get_init_params(cls)
    manifest_inputs = {n: {} for n in expected_inputs}
    manifest_params = {n: {} for n in expected_params}
    errors = check_l1_signature(
        op_name, manifest_inputs, manifest_params, forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"
