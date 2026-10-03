"""Benchmark for the FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Custom, assert_quantized
from tileops.ops import FP8QuantFwdOp
from workloads.quantization.fp8_quant import FP8QuantWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("call", manifest_calls(FP8QuantFwdOp))
def test_fp8_quant_bench(call) -> None:
    workload = FP8QuantWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = FP8QuantFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        evidence={
            tag: Custom(assert_quantized, "scales checked; FP8 rounding within one code")
            for tag in ("tileops", TORCH_COMPILE_TAG)
        },
    )
