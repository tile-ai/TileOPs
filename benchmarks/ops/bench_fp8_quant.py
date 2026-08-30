"""Benchmark for the FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    fields,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import FP8QuantFwdOp
from workloads.fp8_quant import FP8QuantWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


_FP8_QUANT_PARAMS = workload_params(
    load_workloads(FP8QuantFwdOp),
    fields("batch", "seq_len_kv", "kv_group", "index_dim", "in_dtype"),
    smoke_first=True,
)


@pytest.mark.parametrize("batch, seq_len_kv, kv_group, index_dim, in_dtype", _FP8_QUANT_PARAMS)
def test_fp8_quant_bench(
    batch: int, seq_len_kv: int, kv_group: int, index_dim: int, in_dtype: torch.dtype
) -> None:
    test = FP8QuantWorkload(batch, seq_len_kv, kv_group, index_dim, in_dtype)
    inputs = test.gen_inputs()

    op = FP8QuantFwdOp(tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
