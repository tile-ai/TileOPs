"""Benchmark for the per-channel INT8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops.quantization import INT8QuantPerChannelFwdOp


@pytest.mark.parametrize("case", bench.cases(INT8QuantPerChannelFwdOp), ids=lambda case: case.id)
def test_int8_quant_per_channel_bench(case) -> None:
    # Autotuning is a bench-run policy, not a workload property.
    op = INT8QuantPerChannelFwdOp(**case.arguments)
    op.autotune()

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

    reciprocal = "vendor multiplies by 127 / amax; reciprocal rounding can change an INT8 code"
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: bench.Implementation(run=vllm_triton, noncomparable_reason=reciprocal),
            "vllm-cuda": bench.Implementation(run=vllm_cuda, noncomparable_reason=reciprocal),
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: bench.Implementation(
                run=compiled_reference(case.reference),
                noncomparable_reason="Inductor lowering does not preserve the reference's exact INT8 codes",
            ),
        }
    )
