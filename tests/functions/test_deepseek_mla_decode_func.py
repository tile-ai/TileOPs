import pytest
import torch

from benchmarks import MultiHeadLatentAttentionDecodeBenchmark
from top.functions import MultiHeadLatentAttentionDecodeWithKVCacheFunc, mla_decode_with_kvcache
from top.layers import MultiHeadLatentAttentionDecodeLayer


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "batch, kv_head_num, seq_len_kv, heads, dim, pe_dim, dtype",
    [
        (32, 1, 8192, 128, 512, 64, torch.float16),
    ],
)
def test_mla_decode_fn(batch: int, kv_head_num: int, seq_len_kv: int, heads: int, dim: int,
                       pe_dim: int, dtype: torch.dtype):

    mla_layer = MultiHeadLatentAttentionDecodeLayer(batch, heads, kv_head_num, seq_len_kv, dim,
                                                    pe_dim, dtype)
    benchmark = MultiHeadLatentAttentionDecodeBenchmark(batch, heads, kv_head_num, seq_len_kv, dim,
                                                        pe_dim, dtype)

    inputs = benchmark.gen_inputs()

    try:
        print("Testing mla_fn interface...")
        benchmark.check_fn(mla_decode_with_kvcache, *inputs, grad=False, atol=3e-4, rtol=1e-5)
        print("✅ mla_fn test passed")
    except Exception as e:
        print(f"❌ mla_fn test failed: {e}")
        raise

    try:
        print("Testing mla_fn class...")
        fn = MultiHeadLatentAttentionDecodeWithKVCacheFunc(batch, heads, kv_head_num, seq_len_kv,
                                                           dim, pe_dim, dtype)
        benchmark.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)
        print("✅ mla_fn test passed")
    except Exception as e:
        print(f"❌ mla_fn test failed: {e}")
        raise

    try:
        print("Testing mla_layer...")
        benchmark.check_fn(mla_layer, *inputs, grad=False, atol=3e-4, rtol=1e-5)
        print("✅ mla_layer test passed")
    except Exception as e:
        print(f"❌ mla_layer test failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
