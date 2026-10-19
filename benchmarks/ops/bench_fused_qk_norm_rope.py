"""Benchmark FusedQKNormRoPEFwdOp against vLLM's fused kernel and torch-compile of the reference."""

import dataclasses

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
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
        )
        return qkv

    return private_inputs(compiled_reference(run), inputs, 0)


@pytest.mark.parametrize("case", bench.cases(FusedQKNormRoPEFwdOp), ids=lambda case: case.id)
def test_fused_qk_norm_rope_bench(case) -> None:
    op = FusedQKNormRoPEFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: _vllm(op, case.inputs),
            TORCH_COMPILE_TAG: _compiled(op, case.inputs),
        }
    )
