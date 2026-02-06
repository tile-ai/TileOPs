import argparse

from benchmarks import MultiHeadLatentAttentionDecodeBenchmark
from top.functions import MultiHeadLatentAttentionDecodeWithKVCacheFunc, mla_decode_with_kvcache
from top.layers import MultiHeadLatentAttentionDecodeLayer
from top.utils import str2dtype


def test_mla_decode_fn(batch, kv_head_num, seq_len_kv, heads, dim, pe_dim, dtype):

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--kv_head_num', type=int, default=1, help='number of key/value heads')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=128, help='num heads')
    parser.add_argument('--dim', type=int, default=512, help='head dim')
    parser.add_argument('--pe_dim', type=int, default=64, help='positional encoding dim')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_mla_decode_fn(args.batch, args.kv_head_num, args.seq_len_kv, args.heads, args.dim,
                       args.pe_dim, str2dtype[args.dtype])
