import sys

import torch
import pytest

from benchmarks import DeepSeekSparseAttentionDecodeBenchmark
from top.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv, q_start_index_s, sm_scale, dtype, tune",
    [
        (1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False),
    ],
)
def test_sparse_mla_decode(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                           dim_tail: int, topk: int, stride_kv: int, group_kv: int,
                           q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                           tune: bool) -> None:
    op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        group_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        dtype=dtype,
        tune=tune)
    benchmark = DeepSeekSparseAttentionDecodeBenchmark(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        group_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        dtype=dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=3e-4, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
