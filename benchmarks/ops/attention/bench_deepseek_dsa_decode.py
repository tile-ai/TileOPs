import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import dsa_decode_args, manifest_params
from tileops.manifest import load_workloads
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import DsaDecodeWorkload

_OP_NAME = "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"


_DSA_DECODE_BENCH_PARAMS = manifest_params(
    load_workloads(_OP_NAME),
    dsa_decode_args,
    tune=False,
)


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start_index_s, sm_scale, dtype, tune",
    _DSA_DECODE_BENCH_PARAMS,
)
def test_dsa_decode_bench(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                          dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                          q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                          tune: bool) -> None:
    test = DsaDecodeWorkload(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    inputs = test.gen_inputs()

    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
