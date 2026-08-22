"""Benchmarks for RMSNorm / LayerNorm and their fused-add variants.

Normalization is where the serving stacks ship hand-written kernels, so every row
takes the ones that cover it, plus torch eager and inductor: RMSNorm all three of
flag_gems, flashinfer and vllm; LayerNorm the two with a kernel for it, flag_gems
and flashinfer; fused-add RMSNorm the two fused kernels, flashinfer's and vllm's;
fused-add LayerNorm none, which nobody fuses.
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
    flashinfer_op,
    reference_tolerance,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
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


def _flaggems_rms_norm(n: int, eps: float):
    """flag_gems' aten-level ``rms_norm(x, normalized_shape, weight, eps)``."""
    fn = flaggems_op("rms_norm")

    def baseline_fn(x, weight):
        return fn(x, [n], weight, eps)

    return baseline_fn


def _flashinfer_rms_norm(eps: float):
    fn = flashinfer_op("rmsnorm")

    def baseline_fn(x, weight):
        return fn(x, weight, eps)

    return baseline_fn


def _vllm_rms_norm(x: torch.Tensor, eps: float):
    """vllm's kernel writes into a caller-allocated tensor, so allocate it here.

    Allocating inside the callable would charge the tag for an ``empty_like`` the
    other tags never pay.
    """
    fn = vllm_op("rms_norm")
    out = torch.empty_like(x)

    def baseline_fn(x_i, weight):
        fn(out, x_i, weight, eps)
        return out

    return baseline_fn


def _in_place_fused_add(fn, args: tuple, eps: float):
    """Bind an in-place fused-add norm to its own copies of ``(x, residual)``.

    Both kernels overwrite input and residual, and sharing them would hand every
    later tag a different tensor than the reference read.

    The residual therefore grows across iterations, by at most ``max|weight|`` each
    — under 10 for a standard-normal weight, against an fp16 range of 65504 over a
    few hundred iterations. The kernel reads and writes the same bytes regardless,
    which is what the row reports.
    """
    x, residual, weight = args
    private = (x.clone(), residual.clone(), weight)

    def baseline_fn(x_i, residual_i, weight_i):
        fn(x_i, residual_i, weight_i, eps)
        return x_i, residual_i

    return baseline_fn, private


def _assert_fused_add_matches(fn, reference, x, residual, weight, eps, **tolerance) -> None:
    """Check an in-place fused-add kernel on throwaway copies of its inputs."""
    x_copy, residual_copy = x.clone(), residual.clone()
    fn(x_copy, residual_copy, weight, eps)
    expected_y, expected_add = reference(x, residual, weight)
    torch.testing.assert_close(x_copy, expected_y, **tolerance)
    torch.testing.assert_close(residual_copy, expected_add, **tolerance)


def _norm_args(w: dict, dtype: torch.dtype) -> tuple:
    m, n = w["x_shape"]
    return (m, n, dtype, True)


_RMS_OP_NAME = "RMSNormFwdOp"


@pytest.mark.parametrize(
    "m, n, dtype, tune", workload_params(load_workloads(_RMS_OP_NAME), _norm_args)
)
def test_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = RMSNormFwdOp(normalized_shape=(n,), tune=tune)
    bm = ManifestBenchmark(_RMS_OP_NAME, op, test)

    tolerance = reference_tolerance(dtype)
    library = {
        FLAGGEMS_TAG: _flaggems_rms_norm(n, test.eps),
        FLASHINFER_TAG: _flashinfer_rms_norm(test.eps),
        VLLM_TAG: _vllm_rms_norm(inputs[0], test.eps),
    }
    for baseline_fn in library.values():
        assert_matches_reference(baseline_fn, test.ref_program, *inputs, **tolerance)

    bm.compare(
        {
            "tileops": op,
            **library,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


_FUSED_RMS_OP_NAME = "FusedAddRMSNormFwdOp"


@pytest.mark.parametrize(
    "m, n, dtype, tune", workload_params(load_workloads(_FUSED_RMS_OP_NAME), _norm_args)
)
def test_fused_add_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRMSNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = FusedAddRMSNormFwdOp(tune=tune)
    bm = ManifestBenchmark(_FUSED_RMS_OP_NAME, op, test)

    # Baseline: add + manual rmsnorm (separate ops)
    def baseline_fn(x, residual, weight):
        add_result = (x.float() + residual.float()).to(x.dtype)
        rms = torch.sqrt(add_result.float().pow(2).mean(dim=-1, keepdim=True) + test.eps)
        y = ((add_result.float() / rms) * weight.float()).to(x.dtype)
        return y, add_result

    tolerance = reference_tolerance(dtype)
    fused_kernels = {
        FLASHINFER_TAG: flashinfer_op("fused_add_rmsnorm"),
        VLLM_TAG: vllm_op("fused_add_rms_norm"),
    }
    functors = {"tileops": op}
    for tag, fn in fused_kernels.items():
        _assert_fused_add_matches(fn, baseline_fn, *inputs, eps=test.eps, **tolerance)
        functors[tag] = _in_place_fused_add(fn, inputs, test.eps)
    functors["torch-ref"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)

    bm.compare(functors, *inputs, record_as=op, params=locals())


_LN_OP_NAME = "LayerNormFwdOp"


@pytest.mark.parametrize(
    "m, n, dtype, tune", workload_params(load_workloads(_LN_OP_NAME), _norm_args)
)
def test_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = LayerNormFwdOp(normalized_shape=(n,), tune=tune)
    bm = ManifestBenchmark(_LN_OP_NAME, op, test)

    # Baseline uses torch.nn.functional.layer_norm
    def baseline_fn(x, weight, bias):
        return F.layer_norm(x, (n,), weight=weight, bias=bias, eps=1e-5)

    flaggems_layer_norm = flaggems_op("layer_norm")
    flashinfer_layer_norm = flashinfer_op("layernorm")

    def flaggems_fn(x, weight, bias):
        # Returns (output, mean, rstd); the row reports the output.
        return flaggems_layer_norm(x, [n], weight, bias, 1e-5)[0]

    def flashinfer_fn(x, weight, bias):
        return flashinfer_layer_norm(x, weight, bias, 1e-5)

    tolerance = reference_tolerance(dtype)
    for library_fn in (flaggems_fn, flashinfer_fn):
        assert_matches_reference(library_fn, baseline_fn, *inputs, **tolerance)

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            FLASHINFER_TAG: flashinfer_fn,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


_FUSED_LN_OP_NAME = "FusedAddLayerNormFwdOp"


@pytest.mark.parametrize(
    "m, n, dtype, tune", workload_params(load_workloads(_FUSED_LN_OP_NAME), _norm_args)
)
def test_fused_add_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = FusedAddLayerNormFwdOp(tune=tune)
    bm = ManifestBenchmark(_FUSED_LN_OP_NAME, op, test)

    # Baseline: add + F.layer_norm (separate ops)
    def baseline_fn(x, residual, weight, bias):
        add_result = (x.float() + residual.float()).to(x.dtype)
        return F.layer_norm(add_result, (n,), weight=weight, bias=bias, eps=test.eps), add_result

    # flashinfer's and vllm's fused-add kernels are RMSNorm only, so this row is
    # torch against itself, eager and compiled.
    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
