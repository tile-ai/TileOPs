"""Compare grouped INT4 quantization with DeepSpeed's fused CUDA kernel."""

import pytest
import torch

from benchmarks.baselines import DEEPSPEED_TAG, deepspeed_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Custom, zeroed_input
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

    def validate(got, expected):
        for output, target in zip(got, expected, strict=True):
            assert output.shape == target.shape and output.dtype == target.dtype
        torch.testing.assert_close(got[1], expected[1], rtol=1e-6, atol=0)

        def codes(packed):
            return torch.stack((packed >> 4, (packed << 4) >> 4), dim=-1).int()

        assert (codes(got[0]) - codes(expected[0])).abs().max() <= 1

    evidence = Custom(
        validate,
        "matching scale/offset; signed INT4 codes within one step at rounding boundaries",
        controls=(zeroed_input(0, "weight-zeroed"),),
    )
    ManifestBenchmark(op, workload).compare(
        {"tileops": op, DEEPSPEED_TAG: deepspeed},
        *inputs,
        evidence={"tileops": evidence, DEEPSPEED_TAG: evidence},
    )
