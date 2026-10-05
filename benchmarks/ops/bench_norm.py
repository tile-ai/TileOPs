"""Normalization benchmarks against vendor, QuACK and compiled PyTorch kernels."""

import math

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    FLASHINFER_TAG,
    QUACK_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flaggems_op,
    flashinfer_op,
    quack_op,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.norm import NormCall


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
    """Bind vLLM's RMSNorm to a reusable output buffer."""
    fn = vllm_op("rms_norm")
    out = torch.empty_like(x)

    def baseline_fn(x_i, weight):
        fn(out, x_i, weight, eps)
        return out

    return baseline_fn


def _in_place_fused_add(fn, args: tuple, eps: float):
    """Reset private inputs per call, excluding reset copies from kernel timing."""
    x, residual, weight = args
    private = (x.clone(), residual.clone(), weight)

    def baseline_fn(x_i, residual_i, weight_i):
        x_i.copy_(x)
        residual_i.copy_(residual)
        fn(x_i, residual_i, weight_i, eps)
        return x_i, residual_i

    return baseline_fn, private


@pytest.mark.parametrize("call", manifest_calls(RMSNormFwdOp))
def test_rms_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    x, weight = inputs
    op = RMSNormFwdOp(**workload.arguments(), tune=True)
    shape = tuple(call.params["normalized_shape"])
    eps = call.params["eps"]
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    reference = workload.ref_program
    library = {}
    if weight is not None and x.ndim == 2:
        library = {
            FLAGGEMS_TAG: _flaggems_rms_norm(shape[-1], eps),
            FLASHINFER_TAG: _flashinfer_rms_norm(eps),
            VLLM_TAG: _vllm_rms_norm(x, eps),
        }
    quack_rms = quack_op("rmsnorm")
    width = math.prod(shape)

    def quack_fn(x, weight):
        return quack_rms(
            x.reshape(-1, width), None if weight is None else weight.reshape(-1), eps=eps
        ).reshape_as(x)

    library[QUACK_TAG] = quack_fn
    functors = {
        "tileops": op,
        **library,
        "torch-ref": reference,
        TORCH_COMPILE_TAG: compiled_reference(reference),
    }
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(FusedAddRMSNormFwdOp))
def test_fused_add_rms_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    op = FusedAddRMSNormFwdOp(**workload.arguments(), tune=True)
    eps = call.params["eps"]
    baseline_fn = workload.ref_program
    fused_kernels = {
        FLASHINFER_TAG: flashinfer_op("fused_add_rmsnorm"),
        VLLM_TAG: vllm_op("fused_add_rms_norm"),
    }
    functors = {"tileops": op}
    for tag, fn in fused_kernels.items():
        functors[tag] = _in_place_fused_add(fn, inputs, eps)
    functors["torch-ref"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(LayerNormFwdOp))
def test_layer_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    x, weight, bias = inputs
    op = LayerNormFwdOp(**workload.arguments(), tune=True)
    shape, eps = (tuple(call.params["normalized_shape"]), call.params["eps"])
    baseline_fn = workload.ref_program
    library = {}
    if weight is not None and bias is not None:
        flaggems_layer_norm = flaggems_op("layer_norm")
        flashinfer_layer_norm = flashinfer_op("layernorm")

        def flaggems_fn(x, weight, bias):
            return flaggems_layer_norm(x, list(shape), weight, bias, eps)[0]

        def flashinfer_fn(x, weight, bias):
            return flashinfer_layer_norm(x, weight, bias, eps)

        library = {FLAGGEMS_TAG: flaggems_fn, FLASHINFER_TAG: flashinfer_fn}
    quack_ln = quack_op("layernorm_fwd", "quack.rmsnorm")
    unit_weight = (
        torch.ones(math.prod(shape), dtype=torch.float32, device=x.device)
        if weight is None
        else None
    )

    def quack_fn(x, weight, bias):
        out = quack_ln(
            x.reshape(-1, math.prod(shape)),
            unit_weight if weight is None else weight.float().reshape(-1),
            None if bias is None else bias.float().reshape(-1),
            eps=eps,
        )
        return out.reshape_as(x)

    library[QUACK_TAG] = quack_fn
    functors = {
        "tileops": op,
        **library,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(FusedAddLayerNormFwdOp))
def test_fused_add_layer_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    op = FusedAddLayerNormFwdOp(**workload.arguments(), tune=True)
    eps = call.params["eps"]

    # Baseline: add + F.layer_norm (separate ops)
    baseline_fn = workload.ref_program

    from flash_attn.ops.triton.layer_norm import layer_norm_fn

    def flash_attention_fn(x, residual, weight, bias):
        # The contract rounds the residual sum to the input dtype before normalization.
        added = x + residual
        return layer_norm_fn(added, weight, bias, eps=eps), added

    quack_ln = quack_op("layernorm_fwd", "quack.rmsnorm")

    def quack_fn(x, residual, weight, bias):
        added = x + residual
        return quack_ln(
            added.reshape(-1, added.shape[-1]), weight.float(), bias.float(), eps=eps
        ).reshape_as(x), added

    functors = {
        "flash-attn": flash_attention_fn,
        QUACK_TAG: quack_fn,
        "tileops": op,
        "torch-ref": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }

    ManifestBenchmark(op, workload).compare(
        functors,
        *inputs,
    )


# AdaLayerNorm and AdaLayerNormZero: no library ships either kernel.
def _bench(op_cls: type, call) -> None:
    workload = NormCall(call)
    baseline_fn = workload.ref_program
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    functors = {
        "tileops": op,
        "torch-ref": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }

    ManifestBenchmark(op, workload).compare(
        functors,
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(AdaLayerNormFwdOp))
def test_ada_layer_norm_bench(call) -> None:
    _bench(AdaLayerNormFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(AdaLayerNormZeroFwdOp))
def test_ada_layer_norm_zero_bench(call) -> None:
    _bench(AdaLayerNormZeroFwdOp, call)
