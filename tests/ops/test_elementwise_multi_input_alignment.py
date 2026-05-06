"""Manifest-alignment conformance tests for ``elementwise_multi_input``.

Covers the 2 ops in ``tileops/manifest/elementwise_multi_input.yaml``:

- ``WhereFwdOp`` — fp16 / bf16 / fp32 only (manifest contract).
- ``LerpTensorFwdOp`` — Tensor-weight ``torch.lerp`` overload.

Each test asserts the live Op class signature satisfies the manifest L1
contract (every manifest input appears in ``forward()`` in declaration
order, every manifest param is reachable through ``__init__`` or
``forward``) and exercises the runtime contract end-to-end.
"""

import pytest
import torch

# (op_class_name, manifest_inputs (forward order), manifest_params)
_MULTI_INPUT_OPS = [
    ("WhereFwdOp", ["condition", "input", "other"], []),
    ("LerpTensorFwdOp", ["input", "end", "weight"], []),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, manifest_inputs, manifest_params",
    _MULTI_INPUT_OPS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_multi_input_signature_matches_manifest(
    op_name: str, manifest_inputs: list, manifest_params: list,
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
# WhereFwdOp dtype contract: manifest declares fp16 | bf16 | fp32.
# fp8 dtypes must be rejected at the op-layer signature.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_where_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """WhereFwdOp must reject fp8 dtypes at construction (manifest contract)."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    with pytest.raises((ValueError, TypeError)):
        WhereFwdOp(
            condition=shape, input=shape, other=shape, dtype=bad_dtype,
        )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_where_accepts_manifest_dtypes(dtype: torch.dtype) -> None:
    """WhereFwdOp constructs and runs for every manifest-declared dtype."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.randint(0, 2, shape, device="cuda").bool()
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    other = torch.randn(shape, device="cuda", dtype=dtype)
    op = WhereFwdOp(condition=shape, input=shape, other=shape, dtype=dtype)
    out = op(cond, inp, other)
    ref = torch.where(cond, inp, other)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# LerpTensorFwdOp construction + execution (Tensor-weight overload).
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_lerp_tensor_same_shape(dtype: torch.dtype) -> None:
    """LerpTensorFwdOp matches torch.lerp on same-shape inputs."""
    from tileops.ops.elementwise import LerpTensorFwdOp

    shape = (4, 8)
    a = torch.randn(shape, device="cuda", dtype=dtype)
    b = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.rand(shape, device="cuda", dtype=dtype)
    op = LerpTensorFwdOp(input=shape, end=shape, weight=shape, dtype=dtype)
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tol = {"atol": 1e-6, "rtol": 1e-6}
    torch.testing.assert_close(out, ref, **tol)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lerp_tensor_broadcast() -> None:
    """LerpTensorFwdOp supports the manifest's 3-way broadcast rule."""
    from tileops.ops.elementwise import LerpTensorFwdOp

    a_shape, b_shape, w_shape = (3, 1), (1, 4), (3, 4)
    dtype = torch.float32
    a = torch.randn(a_shape, device="cuda", dtype=dtype)
    b = torch.randn(b_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)
    op = LerpTensorFwdOp(
        input=a_shape, end=b_shape, weight=w_shape, dtype=dtype,
    )
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    assert tuple(out.shape) == (3, 4)


@pytest.mark.smoke
def test_lerp_tensor_init_signature() -> None:
    """__init__ takes the manifest input names plus dtype."""
    import inspect

    from tileops.ops.elementwise import LerpTensorFwdOp

    init_params = list(
        inspect.signature(LerpTensorFwdOp.__init__).parameters.keys()
    )
    fwd_params = list(
        inspect.signature(LerpTensorFwdOp.forward).parameters.keys()
    )
    assert init_params[1:5] == ["input", "end", "weight", "dtype"], init_params
    assert fwd_params[1:] == ["input", "end", "weight"], fwd_params


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lerp_tensor_dtype_mismatch_rejected() -> None:
    """forward() must reject inputs whose dtype disagrees with __init__."""
    from tileops.ops.elementwise import LerpTensorFwdOp

    shape = (4, 8)
    op = LerpTensorFwdOp(
        input=shape, end=shape, weight=shape, dtype=torch.float32,
    )
    a = torch.randn(shape, device="cuda", dtype=torch.float32)
    b = torch.randn(shape, device="cuda", dtype=torch.float32)
    w_bad = torch.rand(shape, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="weight.dtype"):
        op(a, b, w_bad)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_lerp_tensor_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """LerpTensorFwdOp must reject fp8 dtypes (manifest declares no fp8)."""
    from tileops.ops.elementwise import LerpTensorFwdOp

    shape = (4, 8)
    with pytest.raises((ValueError, TypeError)):
        LerpTensorFwdOp(
            input=shape, end=shape, weight=shape, dtype=bad_dtype,
        )
