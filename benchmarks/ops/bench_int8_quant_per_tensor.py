"""Benchmark for the per-tensor INT8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import INT8QuantPerTensorFwdOp
from workloads.quantization.quantize import INT8QuantPerTensorWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


def _vllm_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vllm's static per-tensor ``scaled_int8_quant``, given the scale the reference derives.

    vllm quantizes per tensor only against a scale it is handed, so the scale comes from
    torch and both launches are timed.
    """
    amax = x.abs().amax().float()
    scale = torch.where(amax > 0, amax / 127, torch.ones_like(amax)).reshape(1)
    q, scale, _ = vllm_op("scaled_int8_quant")(x, scale)
    return q, scale


@pytest.mark.parametrize("call", manifest_calls(INT8QuantPerTensorFwdOp))
def test_int8_quant_per_tensor_bench(call) -> None:
    workload = INT8QuantPerTensorWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = INT8QuantPerTensorFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: _vllm_quant,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        noncomparable={
            TORCH_COMPILE_TAG: "Inductor lowering does not preserve the reference's exact INT8 codes",
        },
    )
