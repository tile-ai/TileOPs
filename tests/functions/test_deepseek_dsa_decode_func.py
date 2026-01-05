import argparse
from top.functions import DeepSeekSparseAttentionDecodeWithKVCacheFunc
from top.layers import DeepSeekSparseAttentionDecodeLayer
from top.utils import str2dtype
from benchmarks import DeepSeekSparseAttentionDecodeBenchmark


def test_sparse_mla_decode(batch,
                           heads,
                           seq_len_q,
                           seq_len_kv,
                           dim,
                           dim_tail,
                           topk,
                           stride_kv,
                           group_kv,
                           q_start_index_s,
                           sm_scale,
                           dtype,
                           tune=False):
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
        benchmark.check_fn(fn, *inputs, grad=False)
        print("✅ mla_fn test passed")
    except Exception as e:
        print(f"❌ mla_fn test failed: {e}")
        raise

    try:
        print("Testing mla_layer...")
        benchmark.check_fn(layer, *inputs, grad=False)
        print("✅ mla_layer test passed")
    except Exception as e:
        print(f"❌ mla_layer test failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--seq_len_kv', type=int, default=2048, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=128, help='num heads')
    parser.add_argument('--dim', type=int, default=512, help='head dim')
    parser.add_argument('--dim_tail', type=int, default=64, help='tail dim')
    parser.add_argument('--topk', type=int, default=2048, help='topk')
    parser.add_argument('--stride_kv', type=int, default=1, help='stride_kv')
    parser.add_argument('--group_kv', type=int, default=1, help='group_kv')
    parser.add_argument('--sm_scale', type=float, default=None, help='softmax scaling factor')
    parser.add_argument('--q_start_index_s', type=int, default=1024, help='query start index')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_sparse_mla_decode(args.batch, args.heads, args.seq_len, args.seq_len_kv, args.dim,
                           args.dim_tail, args.topk, args.stride_kv, args.group_kv,
                           args.q_start_index_s, args.sm_scale, str2dtype[args.dtype], args.tune)
