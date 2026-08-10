"""Benchmarks for RMSNorm / LayerNorm and their fused-add variants."""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.normalization import (
    FusedAddLayerNormWorkload,
    FusedAddRMSNormWorkload,
    LayerNormWorkload,
    RMSNormWorkload,
)

_RMS_OP_NAME = "RMSNormFwdOp"


def _rms_params():
    """Convert manifest workloads to pytest params: (m, n, dtype, tune)."""
    params = []
    for w in load_workloads(_RMS_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _rms_params())
def test_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = RMSNormFwdOp(normalized_shape=(n,), tune=tune)
    bm = ManifestBenchmark(_RMS_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


_FUSED_RMS_OP_NAME = "FusedAddRMSNormFwdOp"


def _fused_rms_params():
    params = []
    # Autotune has no valid configs for N=16384 (Llama-405B hidden_dim).
    _XFAIL_LABELS = {"llama-3.1-405b-prefill", "llama-3.1-405b-decode"}
    for w in load_workloads(_FUSED_RMS_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            marks = ()
            if label in _XFAIL_LABELS:
                marks = pytest.mark.xfail(
                    reason="autotune has no valid configs for N=16384",
                    strict=False)
            params.append(pytest.param(m, n, dtype, True,
                                       id=f"{label}-{dtype_str}",
                                       marks=marks))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _fused_rms_params())
def test_fused_add_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRMSNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = FusedAddRMSNormFwdOp(tune=tune)
    bm = ManifestBenchmark(_FUSED_RMS_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: add + manual rmsnorm (separate ops)
    def baseline_fn(x, residual, weight):
        add_result = (x.float() + residual.float()).to(x.dtype)
        rms = torch.sqrt(add_result.float().pow(2).mean(dim=-1, keepdim=True) + test.eps)
        y = ((add_result.float() / rms) * weight.float()).to(x.dtype)
        return y, add_result

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


_LN_OP_NAME = "LayerNormFwdOp"


def _ln_params():
    params = []
    for w in load_workloads(_LN_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _ln_params())
def test_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = LayerNormFwdOp(normalized_shape=(n,), tune=tune)
    bm = ManifestBenchmark(_LN_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline uses torch.nn.functional.layer_norm
    def baseline_fn(x, weight, bias):
        return F.layer_norm(x, (n,), weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


_FUSED_LN_OP_NAME = "FusedAddLayerNormFwdOp"


def _fused_ln_params():
    params = []
    for w in load_workloads(_FUSED_LN_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _fused_ln_params())
def test_fused_add_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = FusedAddLayerNormFwdOp(tune=tune)
    bm = ManifestBenchmark(_FUSED_LN_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: add + F.layer_norm (separate ops)
    def baseline_fn(x, residual, weight, bias):
        add_result = (x.float() + residual.float()).to(x.dtype)
        return F.layer_norm(add_result, (n,), weight=weight, bias=bias, eps=test.eps), add_result

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
