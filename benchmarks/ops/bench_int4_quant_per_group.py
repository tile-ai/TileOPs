"""Compare grouped INT4 quantization with DeepSpeed's fused CUDA kernel."""

import pytest

from benchmarks.baselines import DEEPSPEED_TAG, deepspeed_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import INT4QuantPerGroupFwdOp
from workloads.quantization.quantize import INT4QuantPerGroupWorkload


@pytest.mark.parametrize("call", manifest_calls(INT4QuantPerGroupFwdOp))
def test_int4_quant_per_group_bench(call) -> None:
    workload = INT4QuantPerGroupWorkload.from_call(call)
    inputs = workload.gen_inputs()
    op = INT4QuantPerGroupFwdOp(**call.arguments({}), tune=True)
    quantize = deepspeed_op("quantize")
    asymmetric = deepspeed_op("Asymmetric")

    def deepspeed(w):
        return quantize(w, w.numel() // workload.group_size, 4, asymmetric)

    ManifestBenchmark(op, workload).compare(
        {"tileops": op, DEEPSPEED_TAG: deepspeed},
        *inputs,
        noncomparable={
            DEEPSPEED_TAG: "vendor uses fast-math division for group scales; rounding can change packed INT4 codes"
        },
    )
