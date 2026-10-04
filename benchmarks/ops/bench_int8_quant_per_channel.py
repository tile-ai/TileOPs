"""Benchmark for the per-channel INT8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import INT8QuantPerChannelFwdOp
from workloads.quantization.quantize import INT8QuantPerChannelWorkload


@pytest.mark.parametrize("call", manifest_calls(INT8QuantPerChannelFwdOp))
def test_int8_quant_per_channel_bench(call) -> None:
    workload = INT8QuantPerChannelWorkload.from_call(call)
    inputs = workload.gen_inputs()

    # Autotuning is a bench-run policy, not a workload property.
    op = INT8QuantPerChannelFwdOp(**call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)

    # vllm's Triton ``per_token_quant_int8`` and its CUDA ``scaled_int8_quant`` without a
    # scale, both dynamic with one scale per row; each returns ``scale`` as ``[N, 1]``.
    per_token = vllm_op(
        "per_token_quant_int8", "model_executor.layers.quantization.utils.int8_utils"
    )
    scaled = vllm_op("scaled_int8_quant")

    def vllm_triton(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, scale = per_token(w)
        return q, scale.view(-1)

    def vllm_cuda(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, scale, _ = scaled(w)
        return q, scale.view(-1)

    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: vllm_triton,
            "vllm-cuda": vllm_cuda,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        noncomparable={
            VLLM_TAG: "vendor multiplies by 127 / amax; reciprocal rounding can change an INT8 code",
            "vllm-cuda": "vendor multiplies by 127 / amax; reciprocal rounding can change an INT8 code",
            TORCH_COMPILE_TAG: "Inductor lowering does not preserve the reference's exact INT8 codes",
        },
    )
