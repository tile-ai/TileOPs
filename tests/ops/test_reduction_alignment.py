"""Manifest-alignment conformance tests for the ``reduction`` family.

Covers the 12 spec-only ops plus LogSoftmax/Softmax/LogSumExp roofline parity.

Each test asserts the live Op class signature satisfies the manifest L1
contract (every manifest input appears in ``forward()`` in declaration
order, every manifest param is reachable through ``__init__`` or
``forward``) and exercises the runtime roofline contract end-to-end.

Test data is read from ``tileops.manifest.load_manifest()`` rather than
duplicated here, so manifest changes flow into the assertions
automatically. Only the op allowlist (spec-only selection) is hardcoded.
"""

from __future__ import annotations

import pytest
import torch

from tileops.manifest import load_manifest

# Spec-only ops covered by this PR. Manifest provides every other field.
_SPEC_ONLY_OPS = (
    "SumFwdOp",
    "MeanFwdOp",
    "AmaxFwdOp",
    "AminFwdOp",
    "ProdFwdOp",
    "LogSumExpFwdOp",
    "VarFwdOp",
    "StdFwdOp",
    "VarMeanFwdOp",
    "AllFwdOp",
    "AnyFwdOp",
    "CountNonzeroFwdOp",
)

# Roofline parity is also exercised on the already-implemented softmax
# family so the LogSoftmax FLOP fix is regression-tested against
# Softmax / LogSumExp directly.
_ROOFLINE_OPS = (*_SPEC_ONLY_OPS, "LogSoftmaxFwdOp", "SoftmaxFwdOp")

# Representative shape used for every roofline / runtime case.
_M = 64
_N = 256

_MANIFEST = load_manifest()


def _signature_case(op_name: str):
    entry = _MANIFEST[op_name]["signature"]
    return (op_name, entry.get("inputs", {}), entry.get("params", {}))


def _roofline_case(op_name: str):
    """Return (op_name, op_kwargs, flops_expr, bytes_expr) from the manifest."""
    entry = _MANIFEST[op_name]
    rf = entry.get("roofline", {})
    params = entry["signature"].get("params", {})
    op_kwargs: dict = {}
    if "correction" in params:
        op_kwargs["correction"] = 1
    return (op_name, op_kwargs, rf["flops"], rf["bytes"])


def _eval_roofline_expr(expr: str, m: int, n: int, elem_bytes: int) -> int:
    """Evaluate a manifest roofline expression at fixed (M, N, elem_bytes)."""
    return int(eval(expr, {"M": m, "N": n, "elem_bytes": elem_bytes}))


_SIGNATURE_CASES = [_signature_case(n) for n in _SPEC_ONLY_OPS]
_ROOFLINE_CASES = [_roofline_case(n) for n in _ROOFLINE_OPS]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, manifest_inputs, manifest_params",
    _SIGNATURE_CASES,
    ids=[c[0] for c in _SIGNATURE_CASES],
)
def test_reduction_signature_matches_manifest(
    op_name: str, manifest_inputs: dict, manifest_params: dict,
) -> None:
    """Op class signatures must satisfy the manifest L1 contract."""
    import tileops.ops.reduction as mod
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
    errors = check_l1_signature(
        op_name, manifest_inputs, manifest_params, forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"


# ---------------------------------------------------------------------------
# Roofline FLOP/byte parity: op.eval_roofline() must agree with the
# manifest formulas evaluated at the representative (M=64, N=256) shape.
# Drives both the 12 spec-only ops and LogSoftmax/Softmax (regression
# coverage for the LogSoftmax FLOP fix).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, op_kwargs, flops_expr, bytes_expr",
    _ROOFLINE_CASES,
    ids=[c[0] for c in _ROOFLINE_CASES],
)
def test_reduction_eval_roofline_matches_manifest(
    op_name: str,
    op_kwargs: dict,
    flops_expr: str,
    bytes_expr: str,
) -> None:
    """eval_roofline() must match the manifest formulas evaluated at (M, N)."""
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    dtype = torch.float16
    # LogSoftmax/Softmax expose N at __init__; spec-only reduce ops don't.
    ctor_kwargs = dict(op_kwargs)
    if op_name in {"LogSoftmaxFwdOp", "SoftmaxFwdOp"}:
        ctor_kwargs["N"] = _N
    op = cls(dtype=dtype, dim=-1, **ctor_kwargs)
    x = torch.randn(_M, _N, dtype=dtype, device="cuda")
    op(x)  # bind dynamic shape
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    expected_flops = _eval_roofline_expr(flops_expr, _M, _N, elem_bytes)
    expected_bytes = _eval_roofline_expr(bytes_expr, _M, _N, elem_bytes)
    assert flops == expected_flops, (
        f"{op_name} flops {flops} != manifest "
        f"'{flops_expr}' = {expected_flops}"
    )
    assert mem_bytes == expected_bytes, (
        f"{op_name} bytes {mem_bytes} != manifest "
        f"'{bytes_expr}' = {expected_bytes}"
    )


# ---------------------------------------------------------------------------
# Construction smoke: every op constructs over its manifest dtype contract.
# Per-op dtype list comes from ``manifest[op].signature.inputs.x.dtype``
# so e.g. LogSumExp (fp16 | bf16) is not exercised with fp32.
# ---------------------------------------------------------------------------

_DTYPE_NAME_TO_TORCH = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _float_dtypes_for(op_name: str) -> list[torch.dtype]:
    """Manifest-declared float dtypes for the op's ``x`` input."""
    raw = _MANIFEST[op_name]["signature"]["inputs"]["x"]["dtype"]
    names = [t.strip() for t in raw.split("|")]
    return [_DTYPE_NAME_TO_TORCH[n] for n in names if n in _DTYPE_NAME_TO_TORCH]


_DTYPE_CASES = [
    (op_name, dtype)
    for op_name in _SPEC_ONLY_OPS
    for dtype in _float_dtypes_for(op_name)
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, dtype",
    _DTYPE_CASES,
    ids=[f"{n}-{d}".replace("torch.", "") for n, d in _DTYPE_CASES],
)
def test_reduction_constructs_for_manifest_dtypes(
    op_name: str, dtype: torch.dtype,
) -> None:
    """Every op must construct + run for each manifest-declared dtype."""
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    params = _MANIFEST[op_name]["signature"].get("params", {})
    op_kwargs: dict = {}
    if "correction" in params:
        op_kwargs["correction"] = 1
    op = cls(dtype=dtype, dim=-1, **op_kwargs)
    x = torch.randn(_M, _N, dtype=dtype, device="cuda")
    out = op(x)
    if op_name == "VarMeanFwdOp":
        var, mean = out
        assert var.shape == (_M,)
        assert mean.shape == (_M,)
    else:
        assert out.shape == (_M,)


# ---------------------------------------------------------------------------
# AllFwdOp / AnyFwdOp also accept bool inputs per manifest.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name", ["AllFwdOp", "AnyFwdOp"])
def test_logical_reduce_accepts_bool(op_name: str) -> None:
    """All / Any must accept bool inputs (manifest dtype contract)."""
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    op = cls(dtype=torch.bool, dim=-1)
    x = torch.randint(0, 2, (_M, _N), device="cuda").bool()
    out = op(x)
    assert out.dtype == torch.bool
    assert out.shape == (_M,)


# ---------------------------------------------------------------------------
# Output dtype contract: count_nonzero must return int64; all/any must return
# bool. Per manifest signature.outputs.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_count_nonzero_returns_int64() -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    op = CountNonzeroFwdOp(dtype=torch.float16, dim=-1)
    x = torch.randn(_M, _N, dtype=torch.float16, device="cuda")
    out = op(x)
    assert out.dtype == torch.int64, (
        f"CountNonzero output dtype {out.dtype} != int64"
    )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name", ["AllFwdOp", "AnyFwdOp"])
def test_logical_reduce_returns_bool(op_name: str) -> None:
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    op = cls(dtype=torch.float16, dim=-1)
    x = torch.randn(_M, _N, dtype=torch.float16, device="cuda")
    out = op(x)
    assert out.dtype == torch.bool
