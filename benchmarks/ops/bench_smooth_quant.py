"""Benchmark for the SmoothQuant activation quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops.quantization import SmoothQuantFwdOp


@pytest.mark.parametrize("case", bench.cases(SmoothQuantFwdOp), ids=lambda case: case.id)
def test_smooth_quant_bench(case) -> None:
    # Autotuning is a bench-run policy, not a workload property.
    op = SmoothQuantFwdOp(**case.arguments)
    op.request_tune()

    # The unfused path: torch's float32 divide, then vllm's CUDA dynamic per-token quantize.
    scaled = vllm_op("scaled_int8_quant")

    def vllm_unfused(x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, scale, _ = scaled(x / smooth)
        return q, scale.view(-1)

    compiled = compiled_reference(case.reference)
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: bench.Implementation(
                run=vllm_unfused,
                noncomparable_reason="vendor multiplies by 127 / amax; reciprocal rounding can change an INT8 code",
            ),
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: bench.Implementation(
                run=compiled,
                noncomparable_reason="Inductor lowering does not preserve the reference's exact INT8 codes",
            ),
        }
    )
