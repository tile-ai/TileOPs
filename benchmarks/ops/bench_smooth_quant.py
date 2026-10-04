"""Benchmark for the SmoothQuant activation quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import SmoothQuantFwdOp
from workloads.quantization.quantize import SmoothQuantWorkload


@pytest.mark.parametrize("call", manifest_calls(SmoothQuantFwdOp))
def test_smooth_quant_bench(call) -> None:
    workload = SmoothQuantWorkload.from_call(call)
    inputs = workload.gen_inputs()

    # Autotuning is a bench-run policy, not a workload property.
    op = SmoothQuantFwdOp(**call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)

    # The unfused path: torch's float32 divide, then vllm's CUDA dynamic per-token quantize.
    scaled = vllm_op("scaled_int8_quant")

    def vllm_unfused(x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, scale, _ = scaled(x / smooth)
        return q, scale.view(-1)

    compiled = compiled_reference(workload.ref_program)
    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: vllm_unfused,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled,
        },
        *inputs,
        noncomparable={
            VLLM_TAG: "vendor multiplies by 127 / amax; reciprocal rounding can change an INT8 code",
            TORCH_COMPILE_TAG: "Inductor lowering does not preserve the reference's exact INT8 codes",
        },
    )
