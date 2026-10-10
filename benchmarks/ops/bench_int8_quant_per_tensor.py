"""Benchmark for the per-tensor INT8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    vllm_op,
)
from tileops.quantization import INT8QuantPerTensorFwdOp

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


@pytest.mark.parametrize("case", bench.cases(INT8QuantPerTensorFwdOp), ids=lambda case: case.id)
def test_int8_quant_per_tensor_bench(case) -> None:
    op = INT8QuantPerTensorFwdOp(**case.arguments)
    if _TUNE:
        op.autotune()
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: _vllm_quant,
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: bench.Implementation(
                run=compiled_reference(case.reference),
                noncomparable_reason="Inductor lowering does not preserve the reference's exact INT8 codes",
            ),
        }
    )
