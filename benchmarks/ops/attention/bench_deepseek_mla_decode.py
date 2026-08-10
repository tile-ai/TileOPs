import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params, mla_decode_args
from tileops.manifest import load_workloads
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import MlaDecodeWorkload

_OP_NAME = "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"


_MLA_DECODE_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), mla_decode_args)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
    _MLA_DECODE_BENCH_PARAMS,
)
def test_mla_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dim_pe: int, dtype: torch.dtype, tune: bool) -> None:
    test = MlaDecodeWorkload(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
