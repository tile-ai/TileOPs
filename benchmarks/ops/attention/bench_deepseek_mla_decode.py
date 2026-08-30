import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from benchmarks.ops.attention.workload_args import mla_decode_args
from tileops.manifest import load_workloads
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import MlaDecodeWorkload

_OP_NAME = "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"


_MLA_DECODE_BENCH_PARAMS = workload_params(
    load_workloads(_OP_NAME), then_dtype(mla_decode_args, tune=True)
)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
    _MLA_DECODE_BENCH_PARAMS,
)
def test_mla_decode_bench(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_kv: int,
    dim: int,
    dim_pe: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MlaDecodeWorkload(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, tune=tune
    )
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        params=locals(),
    )
