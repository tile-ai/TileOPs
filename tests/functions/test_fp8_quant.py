import pytest
import torch

from benchmarks import Fp8QuantBenchmark
from top.functions import Fp8QuantFunc
from top.layers import Fp8QuantLayer


@pytest.mark.parametrize(
    ("seq_len_kv, index_dim, in_dtype, tune"),
    [
        (8192, 64, torch.float16, False),
        (8192, 64, torch.bfloat16, False),
        (4096, 128, torch.float32, False),
        (16384, 32, torch.float32, False),
    ],
)
def test_fp8_quant(seq_len_kv, index_dim, in_dtype, tune=False):
    fn = Fp8QuantFunc(seq_len_kv, index_dim, in_dtype, tune=tune)
    layer = Fp8QuantLayer(seq_len_kv, index_dim, in_dtype, tune=tune)
    benchmark = Fp8QuantBenchmark(seq_len_kv, index_dim, in_dtype)
    inputs = benchmark.gen_inputs()

    try:
        print("Testing fp8_quant_fn...")
        benchmark.check_fn(fn, inputs, grad=False)
        print("✅ fp8_quant_fn test passed")
    except Exception as e:
        print(f"❌ fp8_quant_fn test failed: {e}")
        raise

    try:
        print("Testing fp8_quant_layer...")
        benchmark.check_fn(layer, inputs, grad=False)
        print("✅ fp8_quant_layer test passed")
    except Exception as e:
        print(f"❌ fp8_quant_layer test failed: {e}")
        raise


if __name__ == "__main__":
    test_fp8_quant(8192, 64, torch.float16, False)
    test_fp8_quant(8192, 64, torch.bfloat16, False)
    test_fp8_quant(4096, 128, torch.float32, False)
    test_fp8_quant(16384, 32, torch.float32, False)
