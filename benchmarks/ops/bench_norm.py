"""Normalization benchmarks against vendor, QuACK and compiled PyTorch kernels."""

import math

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    FLASHINFER_TAG,
    QUACK_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flaggems_op,
    flashinfer_op,
    private_inputs,
    quack_op,
    vllm_op,
)
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp


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


def _in_place_fused_add(fn, inputs: tuple, eps: float) -> bench.Implementation:
    """A kernel writing the normalized row into ``x`` and the sum into ``residual``."""

    def run(x, residual, weight):
        fn(x, residual, weight, eps)
        return x, residual

    return private_inputs(run, inputs, 0, 1)


@pytest.mark.parametrize("case", bench.cases(RMSNormFwdOp), ids=lambda case: case.id)
def test_rms_norm_bench(case) -> None:
    x, weight = case.inputs
    op = RMSNormFwdOp(**case.arguments)
    op.autotune()
    shape = tuple(case.params["normalized_shape"])
    eps = case.params["eps"]
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    reference = case.reference
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
    bench.Runner(op, case).compare(
        {
            **library,
            "torch-ref": reference,
            TORCH_COMPILE_TAG: compiled_reference(reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(FusedAddRMSNormFwdOp), ids=lambda case: case.id)
def test_fused_add_rms_norm_bench(case) -> None:
    op = FusedAddRMSNormFwdOp(**case.arguments)
    op.autotune()
    eps = case.params["eps"]
    bench.Runner(op, case).compare(
        {
            FLASHINFER_TAG: _in_place_fused_add(
                flashinfer_op("fused_add_rmsnorm"), case.inputs, eps
            ),
            VLLM_TAG: _in_place_fused_add(vllm_op("fused_add_rms_norm"), case.inputs, eps),
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(LayerNormFwdOp), ids=lambda case: case.id)
def test_layer_norm_bench(case) -> None:
    x, weight, bias = case.inputs
    op = LayerNormFwdOp(**case.arguments)
    op.autotune()
    shape, eps = (tuple(case.params["normalized_shape"]), case.params["eps"])
    baseline_fn = case.reference
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


@pytest.mark.parametrize("case", bench.cases(FusedAddLayerNormFwdOp), ids=lambda case: case.id)
def test_fused_add_layer_norm_bench(case) -> None:
    op = FusedAddLayerNormFwdOp(**case.arguments)
    op.autotune()
    eps = case.params["eps"]
    baseline_fn = case.reference

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
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(AdaLayerNormFwdOp), ids=lambda case: case.id)
def test_ada_layer_norm_bench(case) -> None:
    _bench(AdaLayerNormFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AdaLayerNormZeroFwdOp), ids=lambda case: case.id)
def test_ada_layer_norm_zero_bench(case) -> None:
    _bench(AdaLayerNormZeroFwdOp, case)
