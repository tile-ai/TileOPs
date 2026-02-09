import sys

import pytest
import torch

from benchmarks import DeepSeekSparseAttentionDecodeBenchmark
from top.functions import DeepSeekSparseAttentionDecodeWithKVCacheFunc
from top.layers import DeepSeekSparseAttentionDecodeLayer


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv, q_start_index_s, sm_scale, dtype, tune",
    [
        (1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False),
    ],
)
def test_sparse_mla_decode(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                           dim_tail: int, topk: int, stride_kv: int, group_kv: int,
                           q_start_index_s: int, sm_scale: float, dtype: torch.dtype, tune: bool):
    fn = DeepSeekSparseAttentionDecodeWithKVCacheFunc(
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
    layer = DeepSeekSparseAttentionDecodeLayer(
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

    try:
        print("Testing mla_fn...")
        benchmark.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)
        print("✅ mla_fn test passed")
    except Exception as e:
        print(f"❌ mla_fn test failed: {e}")
        raise

    try:
        print("Testing mla_layer...")
        benchmark.check_fn(layer, *inputs, grad=False, atol=3e-4, rtol=1e-5)
        print("✅ mla_layer test passed")
    except Exception as e:
        print(f"❌ mla_layer test failed: {e}")
        raise


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
