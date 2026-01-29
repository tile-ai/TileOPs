import argparse

from benchmarks import Fp8QuantBenchmark
from top.functions import Fp8QuantFunc
from top.layers import Fp8QuantLayer
from top.utils import str2dtype

def test_fp8_quant(seq_len_kv,
                   index_dim,
                   in_dtype,
                   tune=False):
    fn = Fp8QuantFunc(
        seq_len_kv,
        index_dim,
        in_dtype,
        tune=tune)
    layer = Fp8QuantLayer(
        seq_len_kv,
        index_dim,
        in_dtype,
        tune=tune)
    benchmark = Fp8QuantBenchmark(
        seq_len_kv,
        index_dim,
        in_dtype,
        tune=tune)
    inputs = benchmark.gen_inputs()

    try:
        print("Testing fp8_quant_fn...")
        benchmark.check_fn(fn, *inputs, grad=False)
        print("✅ fp8_quant_fn test passed")
    except Exception as e:
        print(f"❌ fp8_quant_fn test failed: {e}")
        raise

    try:
        print("Testing fp8_quant_layer...")
        benchmark.check_fn(layer, *inputs, grad=False)
        print("✅ fp8_quant_layer test passed")
    except Exception as e:
        print(f"❌ fp8_quant_layer test failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len-kv", type=int, default=1024)
    parser.add_argument("--index-dim", type=int, default=64)
    parser.add_argument("--in-dtype", type=str, default="float16")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    test_fp8_quant(
        args.seq_len_kv,
        args.index_dim,
        str2dtype(args.in_dtype),
        tune=args.tune
    )