"""Benchmark for the per-group INT4 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import INT4QuantPerGroupFwdOp
from workloads.quantization.quantize import INT4QuantPerGroupWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("call", manifest_calls(INT4QuantPerGroupFwdOp))
def test_int4_quant_per_group_bench(call) -> None:
    workload = INT4QuantPerGroupWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = INT4QuantPerGroupFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    # No library kernel quantizes to GemmW4A16FwdOp's packing: vLLM's marlin, AWQ and
    # cutlass int4 entry points reorder weights that are already quantized.
    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )
