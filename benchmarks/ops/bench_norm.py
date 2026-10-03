"""Normalization benchmarks against vendor, QuACK and compiled PyTorch kernels."""

import pytest
import torch
import torch.nn.functional as F

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
    reference_tolerance,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Exact
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

    Restore both buffers inside the callable; the timer excludes these reset copies.
    """
    x, residual, weight = args
    private = (x.clone(), residual.clone(), weight)

    def baseline_fn(x_i, residual_i, weight_i):
        x_i.copy_(x)
        residual_i.copy_(residual)
        fn(x_i, residual_i, weight_i, eps)
        return x_i, residual_i

    return baseline_fn, private


def _eps(call) -> float:
    """The row's ``eps``; ``None`` is float32's machine epsilon, torch's accumulation dtype."""
    eps = call.params["eps"]
    return torch.finfo(torch.float32).eps if eps is None else eps


@pytest.mark.parametrize("call", manifest_calls(RMSNormFwdOp))
def test_rms_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    x, weight = inputs
    op = RMSNormFwdOp(**workload.arguments(), tune=True)
    shape, eps = tuple(call.params["normalized_shape"]), _eps(call)

    def reference(x, weight):
        return F.rms_norm(x.float(), shape, None if weight is None else weight.float(), eps).to(
            x.dtype
        )

    tolerance = reference_tolerance(x.dtype)
    # The library kernels take a 2-D input and a weight; a row without one has no tag.
    library = {}
    if weight is not None and x.ndim == 2:
        library = {
            FLAGGEMS_TAG: _flaggems_rms_norm(shape[-1], eps),
            FLASHINFER_TAG: _flashinfer_rms_norm(eps),
            VLLM_TAG: _vllm_rms_norm(x, eps),
        }

    if len(shape) == 1:
        quack_rms = quack_op("rmsnorm")
        library[QUACK_TAG] = lambda x, weight: quack_rms(x, weight, eps=eps)

    functors = {
        "tileops": op,
        **library,
        "torch-ref": reference,
        TORCH_COMPILE_TAG: compiled_reference(reference),
    }

    ManifestBenchmark(op, workload).compare(
        functors,
        *inputs,
        evidence=dict.fromkeys(functors, Exact(reference=reference, **tolerance)),
    )


@pytest.mark.parametrize("call", manifest_calls(FusedAddRMSNormFwdOp))
def test_fused_add_rms_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    op = FusedAddRMSNormFwdOp(**workload.arguments(), tune=True)
    eps = call.params["eps"]

    # Baseline: add + manual rmsnorm (separate ops)
    def baseline_fn(x, residual, weight):
        add_result = (x.float() + residual.float()).to(x.dtype)
        rms = torch.sqrt(add_result.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        y = ((add_result.float() / rms) * weight.float()).to(x.dtype)
        return y, add_result

    tolerance = reference_tolerance(inputs[0].dtype)
    fused_kernels = {
        FLASHINFER_TAG: flashinfer_op("fused_add_rmsnorm"),
        VLLM_TAG: vllm_op("fused_add_rms_norm"),
    }
    functors = {"tileops": op}
    for tag, fn in fused_kernels.items():
        functors[tag] = _in_place_fused_add(fn, inputs, eps)
    functors["torch-ref"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)

    ManifestBenchmark(op, workload).compare(
        functors,
        *inputs,
        evidence=dict.fromkeys(functors, Exact(reference=baseline_fn, **tolerance)),
    )


@pytest.mark.parametrize("call", manifest_calls(LayerNormFwdOp))
def test_layer_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    x, weight, bias = inputs
    op = LayerNormFwdOp(**workload.arguments(), tune=True)
    shape, eps = tuple(call.params["normalized_shape"]), call.params["eps"]

    def baseline_fn(x, weight, bias):
        return F.layer_norm(x, shape, weight=weight, bias=bias, eps=eps)

    tolerance = reference_tolerance(x.dtype)
    # The library kernels take both affine tensors; a row without them has no tag.
    library = {}
    if weight is not None and bias is not None:
        flaggems_layer_norm = flaggems_op("layer_norm")
        flashinfer_layer_norm = flashinfer_op("layernorm")

        def flaggems_fn(x, weight, bias):
            # Returns (output, mean, rstd); the row reports the output.
            return flaggems_layer_norm(x, list(shape), weight, bias, eps)[0]

        def flashinfer_fn(x, weight, bias):
            return flashinfer_layer_norm(x, weight, bias, eps)

        library = {FLAGGEMS_TAG: flaggems_fn, FLASHINFER_TAG: flashinfer_fn}
    if len(shape) == 1 and weight is not None:
        quack_ln = quack_op("layernorm_fwd", "quack.rmsnorm")

        def quack_fn(x, weight, bias):
            out = quack_ln(
                x.reshape(-1, x.shape[-1]),
                weight.float(),
                None if bias is None else bias.float(),
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

    ManifestBenchmark(op, workload).compare(
        functors,
        *inputs,
        evidence=dict.fromkeys(functors, Exact(reference=baseline_fn, **tolerance)),
    )


@pytest.mark.parametrize("call", manifest_calls(FusedAddLayerNormFwdOp))
def test_fused_add_layer_norm_bench(call) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    op = FusedAddLayerNormFwdOp(**workload.arguments(), tune=True)
    eps = call.params["eps"]

    # Baseline: add + F.layer_norm (separate ops)
    def baseline_fn(x, residual, weight, bias):
        add_result = (x.float() + residual.float()).to(x.dtype)
        n = x.shape[-1]
        return F.layer_norm(add_result, (n,), weight=weight, bias=bias, eps=eps), add_result

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
        evidence=dict.fromkeys(
            functors, Exact(reference=baseline_fn, **reference_tolerance(inputs[0].dtype))
        ),
    )


# AdaLayerNorm and AdaLayerNormZero: no library ships either kernel.
def _bench(op_cls: type, call, baseline_fn) -> None:
    workload = NormCall(call)
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
        evidence=dict.fromkeys(
            functors, Exact(reference=baseline_fn, **reference_tolerance(inputs[0].dtype))
        ),
    )


@pytest.mark.parametrize("call", manifest_calls(AdaLayerNormFwdOp))
def test_ada_layer_norm_bench(call) -> None:
    eps = call.params["eps"]

    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
        return (scale.float() * normed + shift.float()).to(x.dtype)

    _bench(AdaLayerNormFwdOp, call, baseline_fn)


@pytest.mark.parametrize("call", manifest_calls(AdaLayerNormZeroFwdOp))
def test_ada_layer_norm_zero_bench(call) -> None:
    eps = call.params["eps"]

    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
        return (gate.float() * (scale.float() * normed + shift.float())).to(x.dtype)

    _bench(AdaLayerNormZeroFwdOp, call, baseline_fn)
