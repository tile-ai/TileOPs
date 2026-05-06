"""Manifest-alignment conformance tests for the ``reduction`` family.

Covers the 12 spec-only ops plus LogSoftmax/Softmax/LogSumExp roofline parity.

Each test asserts the live Op class signature satisfies the manifest L1
contract (every manifest input appears in ``forward()`` in declaration
order, every manifest param is reachable through ``__init__`` or
``forward``) and exercises the runtime roofline contract end-to-end.
"""

from __future__ import annotations

import pytest
import torch

# (op_class_name, manifest_inputs (forward order), manifest_params)
_REDUCTION_ALIGNMENT_OPS = [
    ("SumFwdOp", ["x"], ["dim", "keepdim"]),
    ("MeanFwdOp", ["x"], ["dim", "keepdim"]),
    ("AmaxFwdOp", ["x"], ["dim", "keepdim"]),
    ("AminFwdOp", ["x"], ["dim", "keepdim"]),
    ("ProdFwdOp", ["x"], ["dim", "keepdim"]),
    ("LogSumExpFwdOp", ["x"], ["dim", "keepdim"]),
    ("VarFwdOp", ["x"], ["dim", "correction", "keepdim"]),
    ("StdFwdOp", ["x"], ["dim", "correction", "keepdim"]),
    ("VarMeanFwdOp", ["x"], ["dim", "correction", "keepdim"]),
    ("AllFwdOp", ["x"], ["dim", "keepdim"]),
    ("AnyFwdOp", ["x"], ["dim", "keepdim"]),
    ("CountNonzeroFwdOp", ["x"], ["dim"]),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, manifest_inputs, manifest_params",
    _REDUCTION_ALIGNMENT_OPS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_reduction_signature_matches_manifest(
    op_name: str, manifest_inputs: list, manifest_params: list,
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
    inputs_dict = {n: {} for n in manifest_inputs}
    params_dict = {n: {} for n in manifest_params}
    errors = check_l1_signature(
        op_name, inputs_dict, params_dict, forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"


# ---------------------------------------------------------------------------
# Roofline FLOP parity: op.eval_roofline() must agree with the manifest
# ``flops`` formula for a representative (M, N).
# ---------------------------------------------------------------------------

# (op_name, op_kwargs, expected_flops_fn(M,N,elem_bytes), expected_bytes_fn(M,N,elem_bytes))
# All formulas are taken straight from ``tileops/manifest/reduction.yaml``.
_M = 64
_N = 256


def _softmax_flops(M: int, N: int, _eb: int) -> int:
    return 5 * M * N


def _softmax_bytes(M: int, N: int, eb: int) -> int:
    return 2 * M * N * eb


def _log_softmax_flops(M: int, N: int, _eb: int) -> int:
    return 5 * M * N


def _log_softmax_bytes(M: int, N: int, eb: int) -> int:
    return 2 * M * N * eb


def _logsumexp_flops(M: int, N: int, _eb: int) -> int:
    return 4 * M * N


def _logsumexp_bytes(M: int, N: int, eb: int) -> int:
    return (M * N + M) * eb


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_log_softmax_eval_roofline_matches_manifest() -> None:
    """LogSoftmaxFwdOp.eval_roofline must return manifest's 5*M*N FLOPs."""
    from tileops.ops.reduction.log_softmax import LogSoftmaxFwdOp

    dtype = torch.float16
    op = LogSoftmaxFwdOp(N=_N, dtype=dtype, dim=-1)
    x = torch.randn(_M, _N, dtype=dtype, device="cuda")
    op(x)  # bind dynamic shape
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == _log_softmax_flops(_M, _N, elem_bytes), (
        f"LogSoftmax flops {flops} != manifest 5*M*N = "
        f"{_log_softmax_flops(_M, _N, elem_bytes)}"
    )
    assert mem_bytes == _log_softmax_bytes(_M, _N, elem_bytes), (
        f"LogSoftmax bytes {mem_bytes} != manifest = "
        f"{_log_softmax_bytes(_M, _N, elem_bytes)}"
    )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_eval_roofline_matches_manifest() -> None:
    """SoftmaxFwdOp.eval_roofline must return manifest's 5*M*N FLOPs."""
    from tileops.ops.reduction.softmax import SoftmaxFwdOp

    dtype = torch.float16
    op = SoftmaxFwdOp(N=_N, dtype=dtype, dim=-1)
    x = torch.randn(_M, _N, dtype=dtype, device="cuda")
    op(x)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == _softmax_flops(_M, _N, elem_bytes)
    assert mem_bytes == _softmax_bytes(_M, _N, elem_bytes)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_logsumexp_eval_roofline_matches_manifest() -> None:
    """LogSumExpFwdOp.eval_roofline must return manifest's 4*M*N FLOPs."""
    from tileops.ops.reduction.logsumexp import LogSumExpFwdOp

    dtype = torch.float16
    op = LogSumExpFwdOp(dtype=dtype, dim=-1)
    x = torch.randn(_M, _N, dtype=dtype, device="cuda")
    op(x)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == _logsumexp_flops(_M, _N, elem_bytes)
    assert mem_bytes == _logsumexp_bytes(_M, _N, elem_bytes)


# ---------------------------------------------------------------------------
# Roofline parity for the simple-reduce / Welford / logical-reduce families.
# Each op's eval_roofline(M, N, elem_bytes) must match its manifest formula.
# ---------------------------------------------------------------------------

# Manifest formulas, per reduction.yaml:
#   sum: flops = M * N            bytes = (M*N + M) * eb
#   mean: flops = M * (N + 1)     bytes = (M*N + M) * eb
#   amax / amin / prod: flops = M * N   bytes = (M*N + M) * eb
#   var: flops = 5*M*N            bytes = (M*N + M) * eb
#   std: flops = 5*M*N + M        bytes = (M*N + M) * eb
#   var_mean: flops = 5*M*N       bytes = (M*N + 2*M) * eb
#   all / any: flops = M*N        bytes = M*N*eb + M
#   count_nonzero: flops = 2*M*N  bytes = M*N*eb + M*8

_REDUCE_ROOFLINE_CASES = [
    # (op_name, op_kwargs, flops_fn, bytes_fn)
    ("SumFwdOp", {}, lambda M, N, eb: M * N, lambda M, N, eb: (M * N + M) * eb),
    ("MeanFwdOp", {}, lambda M, N, eb: M * (N + 1), lambda M, N, eb: (M * N + M) * eb),
    ("AmaxFwdOp", {}, lambda M, N, eb: M * N, lambda M, N, eb: (M * N + M) * eb),
    ("AminFwdOp", {}, lambda M, N, eb: M * N, lambda M, N, eb: (M * N + M) * eb),
    ("ProdFwdOp", {}, lambda M, N, eb: M * N, lambda M, N, eb: (M * N + M) * eb),
    ("VarFwdOp", {"correction": 1}, lambda M, N, eb: 5 * M * N, lambda M, N, eb: (M * N + M) * eb),
    ("StdFwdOp", {"correction": 1}, lambda M, N, eb: 5 * M * N + M, lambda M, N, eb: (M * N + M) * eb),
    ("VarMeanFwdOp", {"correction": 1}, lambda M, N, eb: 5 * M * N, lambda M, N, eb: (M * N + 2 * M) * eb),
    ("AllFwdOp", {}, lambda M, N, eb: M * N, lambda M, N, eb: M * N * eb + M),
    ("AnyFwdOp", {}, lambda M, N, eb: M * N, lambda M, N, eb: M * N * eb + M),
    # CountNonzeroFwdOp has no keepdim param.
    ("CountNonzeroFwdOp", {}, lambda M, N, eb: 2 * M * N, lambda M, N, eb: M * N * eb + M * 8),
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, op_kwargs, flops_fn, bytes_fn",
    _REDUCE_ROOFLINE_CASES,
    ids=[c[0] for c in _REDUCE_ROOFLINE_CASES],
)
def test_reduce_eval_roofline_matches_manifest(
    op_name: str,
    op_kwargs: dict,
    flops_fn,
    bytes_fn,
) -> None:
    """eval_roofline() output must match the manifest roofline formula."""
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
    dtype = torch.float16
    op = cls(dtype=dtype, dim=-1, **op_kwargs)
    x = torch.randn(_M, _N, dtype=dtype, device="cuda")
    op(x)
    flops, mem_bytes = op.eval_roofline()
    eb = dtype.itemsize
    assert flops == flops_fn(_M, _N, eb), (
        f"{op_name} flops {flops} != manifest {flops_fn(_M, _N, eb)}"
    )
    assert mem_bytes == bytes_fn(_M, _N, eb), (
        f"{op_name} bytes {mem_bytes} != manifest {bytes_fn(_M, _N, eb)}"
    )


# ---------------------------------------------------------------------------
# Construction smoke: every op constructs over its manifest dtype contract.
# Manifest declares ``float16 | bfloat16 | float32`` for every op except
# AllFwdOp / AnyFwdOp which additionally accept ``bool``.
# ---------------------------------------------------------------------------

_FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize(
    "op_name, op_kwargs",
    [
        ("SumFwdOp", {}),
        ("MeanFwdOp", {}),
        ("AmaxFwdOp", {}),
        ("AminFwdOp", {}),
        ("ProdFwdOp", {}),
        ("LogSumExpFwdOp", {}),
        ("VarFwdOp", {"correction": 1}),
        ("StdFwdOp", {"correction": 1}),
        ("VarMeanFwdOp", {"correction": 1}),
        ("AllFwdOp", {}),
        ("AnyFwdOp", {}),
        ("CountNonzeroFwdOp", {}),
    ],
)
def test_reduction_constructs_for_manifest_dtypes(
    op_name: str, op_kwargs: dict, dtype: torch.dtype,
) -> None:
    """Every op must construct + run for each manifest-declared dtype."""
    import tileops.ops.reduction as mod

    cls = getattr(mod, op_name)
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
