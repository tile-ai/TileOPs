"""Benchmark for the FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops import FP8QuantOp
from workloads.fp8_quant import FP8QuantTest

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


class FP8QuantTestBaseline(FP8QuantTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input_tensor: (batch, seq_len_kv, kv_group, index_dim)
        amax_value = torch.abs(input_tensor).amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale_tensor = amax_value / 448.0
        output_tensor = torch.clamp(input_tensor / scale_tensor, min=-448.0, max=448.0)
        output_tensor = output_tensor.to(torch.float8_e4m3fn)
        return scale_tensor.squeeze(dim=-1), output_tensor


_FP8_QUANT_OP = "FP8QuantOp"
_FP8_QUANT_PARAMS = workload_field_params(
    load_workloads(_FP8_QUANT_OP),
    ("batch", "seq_len_kv", "kv_group", "index_dim", "in_dtype"),
)


@pytest.mark.parametrize("batch, seq_len_kv, kv_group, index_dim, in_dtype", _FP8_QUANT_PARAMS)
def test_fp8_quant_bench(batch: int, seq_len_kv: int, kv_group: int, index_dim: int,
                         in_dtype: torch.dtype) -> None:
    test = FP8QuantTestBaseline(batch, seq_len_kv, kv_group, index_dim, in_dtype)
    inputs = test.gen_inputs()

    op = FP8QuantOp(tune=_TUNE)
    bm = ManifestBenchmark(_FP8_QUANT_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
