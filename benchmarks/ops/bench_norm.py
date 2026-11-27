"""Normalization benchmarks against vendor, QuACK and compiled PyTorch kernels."""

import math

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLA_TAG,
    FLAGGEMS_TAG,
    FLASHINFER_TAG,
    QUACK_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    fla_op,
    flaggems_op,
    flashinfer_op,
    private_inputs,
    quack_op,
    vllm_op,
)
from tileops._ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops._ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from tileops._ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops._ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops._ops.norm.fused_add_rms_norm_per_token_quant import FusedAddRMSNormPerTokenQuantFwdOp
from tileops._ops.norm.layer_norm import LayerNormFwdOp
from tileops._ops.norm.rms_norm import RMSNormFwdOp
from tileops._ops.norm.rms_norm_gated import RMSNormGatedFwdOp


def _vllm_rms_norm(x: torch.Tensor, eps: float):
    """Bind vLLM's RMSNorm to a reusable output buffer."""
    fn = vllm_op("rms_norm")
    out = torch.empty_like(x)

    def baseline_fn(x_i, weight):
        fn(out, x_i, weight, eps)
        return out

    return baseline_fn


def _vllm_gemma_rms_norm(weight: torch.Tensor, eps: float):
    """vLLM's ``GemmaRMSNorm`` holding *weight*, compiled as vLLM runs a model: its C kernels
    take no float32 ``1 + weight``, so the layer runs its native form."""
    config = baseline_module("vllm.config", VLLM_TAG)
    layernorm = baseline_module("vllm.model_executor.layers.layernorm", VLLM_TAG)
    with config.set_current_vllm_config(config.VllmConfig()):
        layer = layernorm.GemmaRMSNorm(weight.numel(), eps=eps).to(weight.device)
    layer.weight = torch.nn.Parameter(weight.reshape(-1).clone(), requires_grad=False)
    return compiled_reference(layer.forward_native)


def _hf_qwen3_5_rms_norm(weight: torch.Tensor, eps: float):
    """Hugging Face's ``Qwen3_5RMSNorm`` holding *weight*, compiled: it forms ``1 + weight`` in
    float32 and rounds once."""
    layer = (
        baseline_module("transformers.models.qwen3_5.modeling_qwen3_5", "transformers")
        .Qwen3_5RMSNorm(weight.numel(), eps=eps)
        .to(weight.device)
        .requires_grad_(False)
    )
    layer.weight.data = weight.reshape(-1).clone()
    return compiled_reference(layer)


def _in_place_fused_add(fn, inputs: tuple, eps: float) -> bench.Implementation:
    """A kernel writing the normalized row into ``x`` and the sum into ``residual``."""

    def run(x, residual, weight):
        fn(x, residual, weight, eps)
        return x, residual

    return private_inputs(run, inputs, 0, 1)


@pytest.mark.parametrize("case", bench.cases(RMSNormFwdOp))
def test_rms_norm_bench(case) -> None:
    x, weight = case.inputs
    op = RMSNormFwdOp(**case.arguments)
    op.request_tune()
    shape = tuple(case.params["normalized_shape"])
    eps = case.params["eps"]
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    reference = case.torch
    library = {}
    # The libraries normalize whole rows only.
    if case.params["zero_centered"] and case.params["group_size"] is None:
        n = weight.numel()
        # QuACK and flag_gems multiply a float32 1 + weight in float32 before the one rounding.
        scale = weight.reshape(n).float() + 1
        gemma = flashinfer_op("gemma_rmsnorm")
        flaggems = flaggems_op("rms_norm")
        quack = quack_op("rmsnorm")
        vllm = _vllm_gemma_rms_norm(weight, eps)
        hf = _hf_qwen3_5_rms_norm(weight, eps)
        library = {
            FLAGGEMS_TAG: lambda x, w: flaggems(x, [n], scale, eps),
            FLASHINFER_TAG: lambda x, w: gemma(x.reshape(-1, n), w.reshape(n), eps).reshape_as(x),
            QUACK_TAG: lambda x, w: quack(x.reshape(-1, n), scale, eps=eps).reshape_as(x),
            VLLM_TAG: lambda x, w: vllm(x),
            "transformers": lambda x, w: hf(x),
        }
    elif not case.params["zero_centered"]:
        if weight is not None and x.ndim == 2:
            flaggems = flaggems_op("rms_norm")
            flashinfer = flashinfer_op("rmsnorm")
            library = {
                FLAGGEMS_TAG: lambda x, w: flaggems(x, [shape[-1]], w, eps),
                FLASHINFER_TAG: lambda x, w: flashinfer(x, w, eps),
                VLLM_TAG: _vllm_rms_norm(x, eps),
            }
        quack_rms = quack_op("rmsnorm")
        width = math.prod(shape)

        def quack_fn(x, weight):
            return quack_rms(
                x.reshape(-1, width), None if weight is None else weight.reshape(-1), eps=eps
            ).reshape_as(x)

        library[QUACK_TAG] = quack_fn
    bench.Runner(op, case).compare(
        {
            **library,
            "torch-ref": reference,
            TORCH_COMPILE_TAG: compiled_reference(reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(FusedAddRMSNormFwdOp))
def test_fused_add_rms_norm_bench(case) -> None:
    op = FusedAddRMSNormFwdOp(**case.arguments)
    op.request_tune()
    eps = case.params["eps"]
    if case.params["zero_centered"]:
        weight = case.inputs[2]
        scale = weight.float() + 1
        quack = quack_op("rmsnorm")
        vllm = _vllm_gemma_rms_norm(weight, eps)
        library = {
            FLASHINFER_TAG: _in_place_fused_add(
                flashinfer_op("gemma_fused_add_rmsnorm"), case.inputs, eps
            ),
            # QuACK multiplies a float32 1 + weight in float32 before the one rounding.
            QUACK_TAG: lambda x, residual, w: quack(
                x, scale, residual=residual, eps=eps, prenorm=True
            ),
            VLLM_TAG: lambda x, residual, w: vllm(x, residual),
        }
    else:
        library = {
            FLASHINFER_TAG: _in_place_fused_add(
                flashinfer_op("fused_add_rmsnorm"), case.inputs, eps
            ),
            VLLM_TAG: _in_place_fused_add(vllm_op("fused_add_rms_norm"), case.inputs, eps),
        }
    bench.Runner(op, case).compare(
        {
            **library,
            "torch-ref": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(RMSNormGatedFwdOp))
def test_rms_norm_gated_bench(case) -> None:
    """vLLM's ``layernorm_guard.rmsnorm_fn`` takes every call; fla's ``rms_norm_gated`` only a
    gate after one norm over the row; Hugging Face's ``Qwen3_5RMSNormGated`` only a silu gate
    after one norm over the row, without bias."""
    p = case.arguments
    x, _, bias, z = case.inputs
    one_group = p["group_size"] in (None, x.shape[-1])
    vllm_rmsnorm = baseline_module(
        "vllm.third_party.flash_linear_attention.ops.layernorm_guard", "vllm"
    ).rmsnorm_fn

    def vllm_fn(x, weight, bias, z):
        return vllm_rmsnorm(
            x,
            weight,
            bias,
            z=z,
            eps=p["eps"],
            group_size=p["group_size"],
            norm_before_gate=p["norm_before_gate"],
            activation=p["activation"],
        )

    library = {VLLM_TAG: vllm_fn}
    if z is not None and p["norm_before_gate"] and one_group:
        fla_rmsnorm = fla_op("modules.fused_norm_gate.rms_norm_gated")

        def fla_fn(x, weight, bias, z):
            return fla_rmsnorm(x, z, weight, bias, activation=p["activation"], eps=p["eps"])

        library[FLA_TAG] = fla_fn
    if (
        z is not None
        and bias is None
        and p["activation"] == "silu"
        and p["norm_before_gate"]
        and one_group
    ):
        qwen = (
            baseline_module("transformers.models.qwen3_5.modeling_qwen3_5", "transformers")
            .Qwen3_5RMSNormGated(x.shape[-1], eps=p["eps"])
            .to(x.device)
            .requires_grad_(False)
        )

        def hf_fn(x, weight, bias, z):
            qwen.weight.data = weight
            return qwen(x, z)

        library["transformers"] = bench.Implementation(
            hf_fn,
            noncomparable_reason=(
                "rounds norm(x) and w * norm(x) to float16 before the gate"
                if x.dtype == torch.float16
                else None
            ),
        )
    bench.Runner(RMSNormGatedFwdOp(**p), case).compare(
        {**library, "torch-ref": case.torch, TORCH_COMPILE_TAG: compiled_reference(case.torch)}
    )


def _vllm_per_token_quant(inputs: tuple, quant_dtype, eps: float, scale_ub) -> bench.Implementation:
    """vLLM's kernel, into outputs allocated outside timing; it adds into ``residual``."""
    vllm_op("rms_norm_dynamic_per_token_quant")  # loads the extension behind torch.ops._C
    fn = torch.ops._C.rms_norm_dynamic_per_token_quant
    x = inputs[0]
    q = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    scale = torch.empty(x.numel() // x.shape[-1], 1, dtype=torch.float32, device=x.device)
    ub = None if scale_ub is None else torch.tensor([scale_ub], device=x.device)

    def run(x, residual, weight):
        fn(q, x, weight, scale, eps, ub, residual)
        return q, scale.view(*x.shape[:-1], 1), residual

    return private_inputs(run, inputs, 1)


def _flashinfer_static_quant(inputs: tuple, eps: float) -> bench.Implementation:
    """FlashInfer's fused add, RMSNorm and FP8 quantization, which adds into ``residual``."""
    fn = flashinfer_op("fused_add_rmsnorm_quant", "norm")
    x = inputs[0]
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.ones(1, dtype=torch.float32, device=x.device)

    def run(x, residual, weight):
        fn(q, x, residual, weight, scale, eps)
        return q, scale, residual

    return private_inputs(
        run,
        inputs,
        1,
        noncomparable_reason="quantizes with one static per-tensor scale, not a per-token one",
    )


@pytest.mark.parametrize("case", bench.cases(FusedAddRMSNormPerTokenQuantFwdOp))
def test_fused_add_rms_norm_per_token_quant_bench(case) -> None:
    """FlashInfer quantizes only to FP8, so an INT8 case drops it.

    No torch-compile tag: inductor drops the two roundings to the input dtype the contract
    places around the weight multiply, which moves the scale by up to 0.7% and an INT8 code
    by two steps.
    """
    p = case.arguments
    implementations = {
        VLLM_TAG: _vllm_per_token_quant(case.inputs, p["quant_dtype"], p["eps"], p["scale_ub"]),
    }
    if p["quant_dtype"] == torch.float8_e4m3fn:
        implementations[FLASHINFER_TAG] = _flashinfer_static_quant(case.inputs, p["eps"])
    op = FusedAddRMSNormPerTokenQuantFwdOp(**case.arguments)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(LayerNormFwdOp))
def test_layer_norm_bench(case) -> None:
    x, weight, bias = case.inputs
    op = LayerNormFwdOp(**case.arguments)
    op.request_tune()
    shape, eps = (tuple(case.params["normalized_shape"]), case.params["eps"])
    baseline_fn = case.torch
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
    bench.Runner(op, case).compare(
        {
            **library,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        }
    )


@pytest.mark.parametrize("case", bench.cases(FusedAddLayerNormFwdOp))
def test_fused_add_layer_norm_bench(case) -> None:
    op = FusedAddLayerNormFwdOp(**case.arguments)
    op.request_tune()
    eps = case.params["eps"]
    baseline_fn = case.torch

    layer_norm_fn = baseline_module("flash_attn.ops.triton.layer_norm", "flash-attn").layer_norm_fn

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

    bench.Runner(op, case).compare(
        {
            "flash-attn": flash_attention_fn,
            QUACK_TAG: quack_fn,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        }
    )


# AdaLayerNorm and AdaLayerNormZero: no library ships either kernel.
def _bench(op_cls: type, case: bench.Case) -> None:
    op = op_cls(**case.arguments)
    torch_fn, compiled = case.torch, compiled_reference(case.torch)
    if case.inputs[0].dtype != torch.float32:
        reason = "torch rounds the normalized row to the input dtype before scale and shift"
        torch_fn = bench.Implementation(torch_fn, noncomparable_reason=reason)
        compiled = bench.Implementation(compiled, noncomparable_reason=reason)
    bench.Runner(op, case).compare({"torch-ref": torch_fn, TORCH_COMPILE_TAG: compiled})


@pytest.mark.parametrize("case", bench.cases(AdaLayerNormFwdOp))
def test_ada_layer_norm_bench(case) -> None:
    _bench(AdaLayerNormFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AdaLayerNormZeroFwdOp))
def test_ada_layer_norm_zero_bench(case) -> None:
    _bench(AdaLayerNormZeroFwdOp, case)
