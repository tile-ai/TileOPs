"""Benchmark FusedQKNormRoPEFwdOp against vLLM's fused kernel and torch-compile of the reference."""

import dataclasses

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    private_inputs,
    vllm_op,
)
from tileops.norm import FusedQKNormRoPEFwdOp
from workloads.norm import fused_qk_norm_rope_heads


def _vllm(op, inputs):
    """vLLM's ``fused_qk_norm_rope``, the TensorRT-LLM kernel it ports, on a private qkv.

    It takes the same packed qkv, weights, ``cos ‖ sin`` cache and int64 positions. The
    heads a warp holds are left to its own choice (``-1``), as its fusion pass calls it.
    The library is a selected baseline: an image without it fails the row.
    """
    fused = vllm_op("fused_qk_norm_rope")
    dim, rot = inputs[1].shape[0], inputs[3].shape[1]
    neox = op.rope_layout == "neox"

    def run(qkv, q_weight, k_weight, cos_sin_cache, positions):
        fused(
            qkv,
            op.num_heads,
            op.num_kv_heads,
            op.num_kv_heads,
            dim,
            op.eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            neox,
            positions,
            -1,
        )
        return qkv

    implementation = private_inputs(run, inputs, 0)
    if neox and rot < dim:
        # On large calls vLLM holds several heads a warp, and those kernels rotate the wrong
        # columns under NeoX with R < D.
        return dataclasses.replace(
            implementation,
            noncomparable_reason="vLLM's multi-head-per-warp kernels miscompute NeoX with R < D",
        )
    return implementation


def _vllm_mrope(op, inputs):
    """The multimodal path of vLLM's Qwen3.5 attention: ``GemmaRMSNorm`` on each q and k head,
    compiled as vLLM runs a model, then ``MRotaryEmbedding``'s Triton kernel on the case's cache;
    q and k go on to attention, so nothing is written back into qkv. vLLM's
    ``fused_qk_norm_rope`` reads one position a token."""
    config = baseline_module("vllm.config", VLLM_TAG)
    layernorm = baseline_module("vllm.model_executor.layers.layernorm", VLLM_TAG)
    rotary = baseline_module("vllm.model_executor.layers.rotary_embedding", VLLM_TAG)
    qkv, q_weight, k_weight, cache, _ = inputs
    dim, rot = q_weight.shape[0], cache.shape[1]
    q_cols, kv_cols = op.num_heads * dim, op.num_kv_heads * dim
    with config.set_current_vllm_config(config.VllmConfig()):
        q_norm, k_norm = (layernorm.GemmaRMSNorm(dim, eps=op.eps).to(qkv.device) for _ in range(2))
        rope = rotary.MRotaryEmbedding(
            dim,
            rot,
            cache.shape[0],
            10000,
            op.rope_layout == "neox",
            qkv.dtype,
            mrope_section=list(op.mrope_section),
            mrope_interleaved=op.mrope_interleaved,
        ).to(qkv.device)
    for layer, weight in ((q_norm, q_weight), (k_norm, k_weight)):
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    # The layer computes its own cache from a base; the case's cache replaces it, in the
    # query's dtype, which vLLM converts it to.
    rope.cos_sin_cache = cache.to(qkv.dtype)
    norms = compiled_reference(lambda q, k: (q_norm.forward_native(q), k_norm.forward_native(k)))

    def run(qkv, q_weight, k_weight, cos_sin_cache, positions):
        q, k, _ = qkv.split([q_cols, kv_cols, kv_cols], -1)
        q, k = norms(q.reshape(-1, op.num_heads, dim), k.reshape(-1, op.num_kv_heads, dim))
        rope.forward_cuda(positions, q.view(-1, q_cols), k.view(-1, kv_cols))
        return qkv

    return bench.Implementation(
        run=run,
        noncomparable_reason=(
            "vLLM's GemmaRMSNorm lands the normalized heads in the qkv dtype before "
            "MRotaryEmbedding rotates them"
        ),
    )


def _compiled(op, inputs):
    """The workload's reference arithmetic compiled, writing the q and k columns in place."""
    columns = (op.num_heads + op.num_kv_heads) * inputs[1].shape[0]

    def run(qkv, q_weight, k_weight, cos_sin_cache, positions):
        qkv[:, :columns] = fused_qk_norm_rope_heads(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            op.num_heads,
            op.num_kv_heads,
            op.eps,
            op.rope_layout,
            op.zero_centered,
            op.mrope_section,
            op.mrope_interleaved,
        )
        return qkv

    return private_inputs(compiled_reference(run), inputs, 0)


@pytest.mark.parametrize("case", bench.cases(FusedQKNormRoPEFwdOp))
def test_fused_qk_norm_rope_bench(case) -> None:
    op = FusedQKNormRoPEFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: (_vllm if op.mrope_section is None else _vllm_mrope)(op, case.inputs),
            TORCH_COMPILE_TAG: _compiled(op, case.inputs),
        }
    )
