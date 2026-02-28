import pytest
import torch

from benchmarks import Fp8QuantBenchmark
from top.functions import Fp8QuantFunc
from top.layers import Fp8QuantLayer


@pytest.mark.parametrize(
    ("batch, seq_len_kv, kv_group, index_dim, in_dtype, tune"),
    [
        (1, 8192, 1, 64, torch.float16, False),
        (1, 8192, 1, 64, torch.bfloat16, False),
        (1, 4096, 1, 128, torch.float32, False),
        (1, 16384, 1, 32, torch.float32, False),
    ],
)
def test_fp8_quant(batch, seq_len_kv, kv_group, index_dim, in_dtype, tune=False):
    fn = Fp8QuantFunc(batch, seq_len_kv, kv_group, index_dim, in_dtype, tune=tune)
    layer = Fp8QuantLayer(batch, seq_len_kv, kv_group, index_dim, in_dtype, tune=tune)
    benchmark = Fp8QuantBenchmark(batch, seq_len_kv, kv_group, index_dim, in_dtype)
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
    test_fp8_quant(1, 8192, 1, 64, torch.float16, False)
    test_fp8_quant(1, 8192, 1, 64, torch.bfloat16, False)
    test_fp8_quant(1, 4096, 1, 128, torch.float32, False)
    test_fp8_quant(1, 16384, 1, 32, torch.float32, False)
