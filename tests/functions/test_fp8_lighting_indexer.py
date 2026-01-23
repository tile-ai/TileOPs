import argparse

from benchmarks.deepseek_mla import Fp8LightingIndexerBenchmark
from top.functions import Fp8LightingIndexerFunc
from top.layers import Fp8LightingIndexerDecodeLayer


def test_fp8_lighting_indexer(seq_len, heads, index_dim, seq_len_kv, clean_logits, config):
    fn = Fp8LightingIndexerFunc(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
    layer = Fp8LightingIndexerDecodeLayer(seq_len, heads, index_dim, seq_len_kv, clean_logits,
                                          config)
    benchmark = Fp8LightingIndexerBenchmark(seq_len, heads, index_dim, seq_len_kv, clean_logits,
                                            config)

    inputs = benchmark.gen_inputs()

    try:
        print("Testing indexer_fn...")
        benchmark.check_fn(fn, *inputs, grad=False)
        print("✅ indexer_fn test passed")
    except Exception as e:
        print(f"❌ indexer_fn test failed: {e}")
        raise

    try:
        print("Testing indexer_layer...")
        benchmark.check_fn(layer, *inputs, grad=False)
        print("✅ indexer_layer test passed")
    except Exception as e:
        print(f"❌ indexer_layer test failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='number of heads')
    parser.add_argument('--index_dim', type=int, default=64, help='index dim')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument(
        '--clean_logits',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='whether to clean logits outside the valid range')
    parser.add_argument('--config', type=str, default=None, help='positional encoding dim')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_fp8_lighting_indexer(args.seq_len, args.heads, args.index_dim, args.seq_len_kv,
                              args.clean_logits, args.config)
