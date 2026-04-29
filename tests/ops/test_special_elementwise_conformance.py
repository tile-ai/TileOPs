"""Conformance tests for elementwise multi-input ops (issue #1107).

Covers PyTorch-aligned signatures, broadcasting semantics, and split
variants (Tensor-bound clamp / masked_fill, single-bound clamp_min /
clamp_max) for the entries that landed as ``status: spec-only`` in
PR #1109. After this PR lands, these ops conform to the manifest
signatures and the manifest entries can flip to ``status: implemented``
in a follow-up manifest-only PR per the manifest trust model
(.claude/rules/manifest-trust-model.md).
"""

import inspect

import pytest
import torch

# ---------------------------------------------------------------------------
# AC-1: WhereFwdOp full broadcasting
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


# ---------------------------------------------------------------------------
# AC-2: ClampFwdOp Tensor min/max
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


# ---------------------------------------------------------------------------
# AC-3: ClampScalarFwdOp / ClampMinFwdOp / ClampMaxFwdOp
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
# AC-4: MaskedFillFwdOp (0-dim Tensor) / MaskedFillScalarFwdOp (Number)
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
# AC-5: validator passes (the entries can flip to ``implemented`` in a
# follow-up manifest-only PR — this test exercises the L1 signature check
# directly so we don't depend on the YAML status flip.)
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
