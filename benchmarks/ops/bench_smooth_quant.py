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


def _assert_within_one_code(fn, workload, *inputs: torch.Tensor) -> None:
    """vllm multiplies by ``127 / amax`` and inductor by ``1 / scale`` instead of dividing by
    the scale, so a value near a rounding tie can take the neighbouring code; the scales
    agree to float32 rounding."""
    q, scale = fn(*inputs)
    q_ref, scale_ref = workload.ref_program(*inputs)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-6, atol=0.0)
    assert (q.int() - q_ref.int()).abs().max().item() <= 1


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
    _assert_within_one_code(vllm_unfused, workload, *inputs)
    _assert_within_one_code(compiled, workload, *inputs)
    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: vllm_unfused,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled,
        },
        *inputs,
    )
