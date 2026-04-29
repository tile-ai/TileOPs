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


# AC-2 (mixed): ClampFwdOp must accept Tensor min with max=None and
# Tensor max with min=None, matching torch.clamp(input, min=tensor, max=None)
# and torch.clamp(input, min=None, max=tensor) on CUDA.
@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, bound_shape, dtype",
    [
        ((4, 8), (4, 8), torch.float16),
        ((4, 8), (1, 8), torch.float32),
        ((4, 8), (), torch.bfloat16),
    ],
)
def test_clamp_min_only_tensor_parity(input_shape, bound_shape, dtype):
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=dtype)
    mn = torch.randn(bound_shape, device="cuda", dtype=dtype) - 0.5
    ref = torch.clamp(inp, mn, None)
    op = ClampFwdOp(input=tuple(inp.shape), min=tuple(mn.shape), max=None, dtype=dtype)
    out = op(inp, mn, None)
    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1.6e-2, 1.6e-2
    else:
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, bound_shape, dtype",
    [
        ((4, 8), (4, 8), torch.float16),
        ((4, 8), (1, 8), torch.float32),
        ((4, 8), (), torch.bfloat16),
    ],
)
def test_clamp_max_only_tensor_parity(input_shape, bound_shape, dtype):
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=dtype)
    mx = torch.randn(bound_shape, device="cuda", dtype=dtype) + 0.5
    ref = torch.clamp(inp, None, mx)
    op = ClampFwdOp(input=tuple(inp.shape), min=None, max=tuple(mx.shape), dtype=dtype)
    out = op(inp, None, mx)
    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1.6e-2, 1.6e-2
    else:
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "fp8_dtype",
    [
        pytest.param(torch.float8_e4m3fn, id="e4m3fn"),
        pytest.param(torch.float8_e5m2, id="e5m2"),
    ],
)
@pytest.mark.parametrize(
    "which", ["min_only", "max_only"],
)
def test_clamp_mixed_tensor_none_fp8(which, fp8_dtype):
    """fp8 routing for mixed Tensor/None bounds (matches both-Tensor case)."""
    from tileops.ops.elementwise import ClampFwdOp

    n = 256
    inp_fp16 = torch.randn(n, device="cuda", dtype=torch.float16)
    bound_fp16 = (torch.randn(n, device="cuda", dtype=torch.float16)
                  - (0.5 if which == "min_only" else -0.5))
    inp = inp_fp16.to(fp8_dtype)
    bound = bound_fp16.to(fp8_dtype)
    if which == "min_only":
        op = ClampFwdOp(input=(n,), min=(n,), max=None, dtype=fp8_dtype)
        out = op(inp, bound, None)
        ref = torch.clamp(inp.to(torch.float16), bound.to(torch.float16), None).to(fp8_dtype)
    else:
        op = ClampFwdOp(input=(n,), min=None, max=(n,), dtype=fp8_dtype)
        out = op(inp, None, bound)
        ref = torch.clamp(inp.to(torch.float16), None, bound.to(torch.float16)).to(fp8_dtype)
    # fp8 has no torch.testing.assert_close support; compare via fp16 view.
    torch.testing.assert_close(out.to(torch.float16), ref.to(torch.float16),
                               atol=1e-2, rtol=1e-2)


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


# ---------------------------------------------------------------------------
# fp8 coverage: the manifest declares ``float8_e4m3fn | float8_e5m2`` for the
# Tensor-bound clamp variants and both masked_fill variants. PyTorch's CUDA
# backend does not implement clamp/maximum/minimum/masked_fill on Float8, so
# the eager fallback paths must upcast to fp16 internally and cast back. These
# tests exercise both fp8 dtypes against a fp16 reference computation.
# ---------------------------------------------------------------------------


_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def _fp8_to_fp16_via_pytorch(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_clamp_tensor_bounds_fp8(dtype):
    from tileops.ops.elementwise import ClampFwdOp

    # Use moderate magnitudes so values fit in e4m3fn finite range (±448).
    inp = (torch.randn((4, 8), device="cuda") * 4).to(dtype)
    mn = (torch.full((1, 8), -1.0, device="cuda")).to(dtype)
    mx = (torch.full((4, 1), 2.0, device="cuda")).to(dtype)

    ref = torch.clamp(_fp8_to_fp16_via_pytorch(inp),
                      _fp8_to_fp16_via_pytorch(mn),
                      _fp8_to_fp16_via_pytorch(mx)).to(dtype)
    op = ClampFwdOp(input=tuple(inp.shape), min=tuple(mn.shape),
                    max=tuple(mx.shape), dtype=dtype)
    out = op(inp, mn, mx)
    assert out.dtype == dtype
    torch.testing.assert_close(out.to(torch.float16), ref.to(torch.float16),
                               atol=0, rtol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_clamp_min_tensor_fp8(dtype):
    from tileops.ops.elementwise import ClampMinFwdOp

    inp = (torch.randn((4, 8), device="cuda") * 4).to(dtype)
    mn = (torch.full((1, 8), -0.5, device="cuda")).to(dtype)
    ref = torch.maximum(_fp8_to_fp16_via_pytorch(inp),
                        _fp8_to_fp16_via_pytorch(mn)).to(dtype)
    op = ClampMinFwdOp(input=tuple(inp.shape), min=tuple(mn.shape), dtype=dtype)
    out = op(inp, mn)
    assert out.dtype == dtype
    torch.testing.assert_close(out.to(torch.float16), ref.to(torch.float16),
                               atol=0, rtol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_clamp_max_tensor_fp8(dtype):
    from tileops.ops.elementwise import ClampMaxFwdOp

    inp = (torch.randn((4, 8), device="cuda") * 4).to(dtype)
    mx = (torch.full((4, 1), 1.5, device="cuda")).to(dtype)
    ref = torch.minimum(_fp8_to_fp16_via_pytorch(inp),
                        _fp8_to_fp16_via_pytorch(mx)).to(dtype)
    op = ClampMaxFwdOp(input=tuple(inp.shape), max=tuple(mx.shape), dtype=dtype)
    out = op(inp, mx)
    assert out.dtype == dtype
    torch.testing.assert_close(out.to(torch.float16), ref.to(torch.float16),
                               atol=0, rtol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_masked_fill_tensor_value_fp8(dtype):
    from tileops.ops.elementwise import MaskedFillFwdOp

    inp = (torch.randn((4, 8), device="cuda") * 2).to(dtype)
    mask = torch.randint(0, 2, (1, 8), device="cuda").bool()
    value = torch.tensor(-1.0, device="cuda").to(dtype)

    out_shape = torch.broadcast_shapes(inp.shape, mask.shape)
    ref = (
        _fp8_to_fp16_via_pytorch(inp).expand(out_shape).clone()
        .masked_fill(mask.expand(out_shape), float(value.to(torch.float16).item()))
        .to(dtype)
    )

    op = MaskedFillFwdOp(input=tuple(inp.shape), mask=tuple(mask.shape),
                         value=tuple(value.shape), dtype=dtype)
    out = op(inp, mask, value)
    assert out.dtype == dtype
    torch.testing.assert_close(out.to(torch.float16), ref.to(torch.float16),
                               atol=0, rtol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
@pytest.mark.parametrize(
    "input_shape, mask_shape",
    [
        ((4, 8), (4, 8)),  # exact-shape (kernel path)
        ((1, 8), (4, 8)),  # broadcast (eager fallback path)
    ],
)
def test_masked_fill_scalar_fp8(dtype, input_shape, mask_shape):
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    inp = (torch.randn(input_shape, device="cuda") * 2).to(dtype)
    mask = torch.randint(0, 2, mask_shape, device="cuda").bool()

    out_shape = torch.broadcast_shapes(input_shape, mask_shape)
    ref = (
        _fp8_to_fp16_via_pytorch(inp).expand(out_shape).clone()
        .masked_fill(mask.expand(out_shape), -1.0)
        .to(dtype)
    )

    op = MaskedFillScalarFwdOp(input=tuple(inp.shape), mask=tuple(mask.shape),
                               value=-1.0, dtype=dtype)
    out = op(inp, mask)
    assert out.dtype == dtype
    torch.testing.assert_close(out.to(torch.float16), ref.to(torch.float16),
                               atol=0, rtol=0)
