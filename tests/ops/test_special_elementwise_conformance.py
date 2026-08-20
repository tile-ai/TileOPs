"""Conformance tests for elementwise multi-input ops.

Covers PyTorch-aligned signatures, broadcasting semantics, and split
variants (Tensor-bound clamp / masked_fill, single-bound clamp_min /
clamp_max). Once these tests pass, the corresponding manifest entries
can flip from ``status: spec-only`` to ``status: implemented`` per the
manifest spec rules (.claude/domain-rules/manifest-spec.md).
"""

import inspect

import pytest
import torch

import tileops.ops.elementwise as elementwise_mod
from tileops.manifest import load_manifest

# Construction and call signatures, for every op in the family
#
# Replaces the five per-op signature tests this file used to carry: the rule is the
# same for all of them, and it is the one the manifest states.

_ELEMENTWISE_OPS = sorted(n for n in elementwise_mod.__all__ if n.endswith("FwdOp"))

#: Construction arguments every op takes whatever its manifest says: which target
#: serves it, which kernels to use, and whether to autotune.
_OP_LAYER_ARGS = {"target", "kernel_map", "tune"}


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _ELEMENTWISE_OPS)
def test_the_signature_is_the_manifest_signature(op_name: str) -> None:
    """``__init__`` takes the manifest's params; ``forward`` takes its inputs.

    Nothing about shape or element type is a construction argument: both arrive with
    the tensors. And every construction argument is keyword-only, so the manifest's
    declaration order is the only order anyone has to know.
    """
    cls = getattr(elementwise_mod, op_name)
    signature = load_manifest()[op_name]["signature"]

    init = inspect.signature(cls.__init__).parameters
    declared = set(signature.get("params", {}))
    taken = {name for name in init if name != "self"}
    assert taken <= declared | _OP_LAYER_ARGS, (
        f"{op_name}.__init__ takes {sorted(taken - declared - _OP_LAYER_ARGS)}, "
        "which the manifest does not declare"
    )
    positional = [
        name
        for name, p in init.items()
        if name != "self" and p.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert not positional, f"{op_name}.__init__ takes {positional} positionally"

    forward = [name for name in inspect.signature(cls.forward).parameters if name != "self"]
    params_in_forward = [name for name in signature.get("params", {}) if name in forward]
    assert forward == list(signature["inputs"]) + params_in_forward, (
        f"{op_name}.forward takes {forward}, not the manifest's inputs"
    )

    missing = declared - taken - set(forward)
    assert not missing, (
        f"{op_name} declares manifest param(s) {sorted(missing)} that neither "
        "__init__ nor forward accepts"
    )


# WhereFwdOp full broadcasting


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

    cond = (
        torch.randint(0, 2, cond_shape, device="cuda").bool()
        if cond_shape
        else torch.tensor(True, device="cuda")
    )
    inp = torch.randn(inp_shape, device="cuda", dtype=dtype)
    other = torch.randn(other_shape, device="cuda", dtype=dtype)
    ref = torch.where(cond, inp, other)

    op = WhereFwdOp()
    out = op(cond, inp, other)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


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
    op = WhereFwdOp()
    with pytest.raises(ValueError, match="condition.dtype torch.bool"):
        op(cond, inp, other)


# ClampFwdOp Tensor min/max


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

    op = ClampFwdOp()
    out = op(inp, mn, mx)
    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1.6e-2, 1.6e-2
    else:
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ClampFwdOp must accept Tensor min with max=None and
# Tensor max with min=None, matching torch.clamp(input, min=tensor, max=None)
# and torch.clamp(input, min=None, max=tensor) on CUDA. The single-bound shape
# matrix is covered by test_clamp_min_only_tensor / test_clamp_max_only_tensor;
# these two verify the None routing at one shape.
@pytest.mark.smoke
def test_clamp_min_only_none_routing():
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    mn = torch.randn((4, 8), device="cuda", dtype=torch.float32) - 0.5
    ref = torch.clamp(inp, mn, None)
    op = ClampFwdOp()
    out = op(inp, mn, None)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_max_only_none_routing():
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    mx = torch.randn((4, 8), device="cuda", dtype=torch.float32) + 0.5
    ref = torch.clamp(inp, None, mx)
    op = ClampFwdOp()
    out = op(inp, None, mx)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_clamp_both_none_rejected():
    """ClampFwdOp must reject a call with neither bound (a no-op clamp is invalid).

    Which bounds it serves is a fact of the call, so the refusal is too.
    """
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(4, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="at least one of"):
        ClampFwdOp()(inp)


@pytest.mark.smoke
def test_clamp_scalar_both_none_rejected():
    """ClampScalarFwdOp must reject min=None and max=None.

    Mirrors torch.clamp(input, None, None), which raises
    RuntimeError("At least one of min or max must not be None").
    """
    from tileops.ops.elementwise import ClampScalarFwdOp

    with pytest.raises(ValueError, match="at least one of"):
        ClampScalarFwdOp(min=None, max=None)


@pytest.mark.smoke
def test_one_clamp_instance_serves_clamp_and_both_one_sided_forms():
    """Which bounds a call passes is read off the call, so one instance serves all three.

    Each presence pattern needs its own kernel — ``has_min`` / ``has_max`` change what
    gets built — so this also pins that the three do not share one.
    """
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(4, device="cuda", dtype=torch.float32)
    mn = torch.zeros(4, device="cuda", dtype=torch.float32)
    mx = torch.ones(4, device="cuda", dtype=torch.float32)

    op = ClampFwdOp()
    torch.testing.assert_close(op(inp, mn, mx), torch.clamp(inp, mn, mx))
    torch.testing.assert_close(op(inp, mn, None), torch.clamp(inp, min=mn))
    torch.testing.assert_close(op(inp, None, mx), torch.clamp(inp, max=mx))

    assert len(op.built_kernels(op._slot)) == 3, "one kernel per presence pattern"


# ClampScalarFwdOp, and ClampFwdOp with one bound withheld


@pytest.mark.smoke
@pytest.mark.parametrize(
    "min_val, max_val",
    [(-0.5, 0.5), (None, 0.5), (-0.5, None)],
)
def test_clamp_scalar_param_names(min_val, max_val):
    from tileops.ops.elementwise import ClampScalarFwdOp

    inp = torch.randn(1024, device="cuda", dtype=torch.float32)
    ref = torch.clamp(inp, min_val, max_val)
    op = ClampScalarFwdOp(min=min_val, max=max_val)
    out = op(inp)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, min_shape",
    [((4, 8), (4, 8)), ((4, 8), (1, 8)), ((4, 8), ())],
)
def test_clamp_min_only_tensor(input_shape, min_shape):
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    mn = torch.randn(min_shape, device="cuda", dtype=torch.float32)
    ref = torch.clamp_min(inp, mn) if min_shape else torch.clamp(inp, min=mn.item())

    op = ClampFwdOp()
    out = op(inp, mn)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, max_shape",
    [((4, 8), (4, 8)), ((4, 8), (4, 1)), ((4, 8), ())],
)
def test_clamp_max_only_tensor(input_shape, max_shape):
    from tileops.ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device="cuda", dtype=torch.float32)
    mx = torch.randn(max_shape, device="cuda", dtype=torch.float32)
    ref = torch.clamp_max(inp, mx) if max_shape else torch.clamp(inp, max=mx.item())

    op = ClampFwdOp()
    out = op(inp, None, mx)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# Regression: NaN propagation for Tensor-bound clamp variants.
#
# torch.clamp / torch.clamp_min / torch.clamp_max propagate NaN: if any of
# input / min / max is NaN at position i, the output at i is NaN. CUDA's
# fmax / fmin (used by T.max / T.min) drop NaN by returning the non-NaN
# operand, so the kernel adds explicit isnan guards. These tests pin the
# semantics so a future refactor cannot regress to non-IEEE behaviour.


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_tensor_nan_propagation(dtype):
    """ClampFwdOp must match torch.clamp NaN semantics (Tensor min + max)."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device="cuda", dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device="cuda", dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device="cuda", dtype=dtype)

    ref = torch.clamp(x, mn, mx)
    op = ClampFwdOp()
    out = op(x, mn, mx)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


# Single-bound NaN behaviour is covered by test_clamp_min_nan_propagation /
# test_clamp_max_only_nan_propagation below exercises the same
# ClampTensorFwdKernel branches (has_min only / has_max only).


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_min_only_nan_propagation(dtype):
    """A min-only clamp must match torch.clamp_min NaN semantics."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device="cuda", dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device="cuda", dtype=dtype)

    ref = torch.clamp_min(x, mn)
    op = ClampFwdOp()
    out = op(x, mn)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_max_only_nan_propagation(dtype):
    """A max-only clamp must match torch.clamp_max NaN semantics."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device="cuda", dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device="cuda", dtype=dtype)

    ref = torch.clamp_max(x, mx)
    op = ClampFwdOp()
    out = op(x, None, mx)
    torch.testing.assert_close(out, ref, equal_nan=True, atol=0.0, rtol=0.0)


# MaskedFillFwdOp (0-dim Tensor) / MaskedFillScalarFwdOp (Number)


_MASKED_FILL_TENSOR_VALUE_FLOAT_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]
_MASKED_FILL_TENSOR_VALUE_INT_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


def _masked_fill_tensor_value_inputs(input_shape, mask_shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, input_shape, device="cuda").bool()
        value = torch.tensor(True, device="cuda", dtype=torch.bool)
    elif dtype in _MASKED_FILL_TENSOR_VALUE_INT_DTYPES:
        iinfo = torch.iinfo(dtype)
        lo = max(iinfo.min, -1000)
        hi = min(iinfo.max, 1000) + 1
        inp = torch.randint(lo, hi, input_shape, device="cuda", dtype=dtype)
        value = torch.tensor(7, device="cuda", dtype=dtype)
    else:
        inp = torch.randn(input_shape, device="cuda", dtype=dtype)
        value = torch.tensor(-1.5, device="cuda", dtype=dtype)
    mask = torch.randint(0, 2, mask_shape, device="cuda").bool()
    return inp, mask, value


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, mask_shape",
    [((4, 8), (4, 8)), ((1, 8), (4, 8)), ((4, 8), (1, 8)), ((2, 1), (2, 3))],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bool, *_MASKED_FILL_TENSOR_VALUE_INT_DTYPES, *_MASKED_FILL_TENSOR_VALUE_FLOAT_DTYPES],
)
def test_masked_fill_tensor_value(input_shape, mask_shape, dtype):
    from tileops.ops.elementwise import MaskedFillFwdOp

    inp, mask, value = _masked_fill_tensor_value_inputs(input_shape, mask_shape, dtype)

    out_shape = torch.broadcast_shapes(input_shape, mask_shape)
    ref = inp.expand(out_shape).clone().masked_fill(mask.expand(out_shape), value.item())

    op = MaskedFillFwdOp()
    out = op(inp, mask, value)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    elif dtype == torch.float32:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    else:
        tol = {"atol": 0, "rtol": 0}
    torch.testing.assert_close(out, ref, **tol)


@pytest.mark.smoke
def test_masked_fill_scalar_param_names():
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    inp = torch.randn(1024, device="cuda", dtype=torch.float32)
    mask = torch.randint(0, 2, (1024,), device="cuda").bool()
    ref = inp.masked_fill(mask, -1.0)

    op = MaskedFillScalarFwdOp(value=-1.0)
    out = op(inp, mask)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# Validator passes: this test exercises the L1 signature check directly
# so it doesn't depend on the manifest YAML ``status`` value.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, expected_inputs, expected_params",
    [
        ("WhereFwdOp", ["condition", "input", "other"], []),
        ("ClampFwdOp", ["input", "min", "max"], []),
        ("ClampScalarFwdOp", ["input"], ["min", "max"]),
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
        op_name,
        manifest_inputs,
        manifest_params,
        forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"
