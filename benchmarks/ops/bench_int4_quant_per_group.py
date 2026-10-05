"""Benchmark for the per-group INT4 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Custom
from tileops.quantization import INT4QuantPerGroupFwdOp
from workloads.gemm import unrepack_w4a16_weight
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
    def validate(got, expected):
        for output, target in zip(got, expected, strict=True):
            assert output.shape == target.shape and output.dtype == target.dtype
        torch.testing.assert_close(got[1], expected[1], rtol=0, atol=0)

        def centered_codes(result):
            packed, scale, zero = result
            assert (zero <= 15).all()
            packed = unrepack_w4a16_weight(packed)
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).to(torch.int16)
            return (
                codes.reshape(*scale.shape, workload.group_size) - zero.to(torch.int16)[..., None]
            )

        assert (centered_codes(got) - centered_codes(expected)).abs().max() <= 1

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program, preserve_precision=True),
        },
        *inputs,
        evidence={
            TORCH_COMPILE_TAG: Custom(
                validate, "identical scales; reconstructed values within one INT4 quantization step"
            )
        },
    )
