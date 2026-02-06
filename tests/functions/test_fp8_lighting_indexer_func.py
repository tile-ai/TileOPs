import pytest
import torch

from benchmarks.deepseek_mla import Fp8LightingIndexerBenchmark
from top.functions import Fp8LightingIndexerFunc
from top.layers import Fp8LightingIndexerDecodeLayer


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "seq_len, heads, index_dim, seq_len_kv, clean_logits, config",
    [
        (4096, 32, 64, 8192, True, None),
    ],
)
def test_fp8_lighting_indexer(seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                              clean_logits: bool, config):
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
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
