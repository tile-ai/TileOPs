import pytest
import torch

from benchmarks import TopkSelectorBenchmark
from top.functions import TopkSelectorFunc
from top.layers import TopkSelectorLayer


@pytest.mark.parametrize(
    "batch, seq_len, topk, in_dtype, out_dtype, tune",
    [
        (64, 32 * 1024, 2048, torch.float32, torch.int32, False),
    ],
)
def test_topk_selector(batch: int, seq_len: int, topk: int, in_dtype: torch.dtype,
                       out_dtype: torch.dtype, tune: bool) -> None:
    fn = TopkSelectorFunc(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    layer = TopkSelectorLayer(
        batch,
        seq_len,
        topk,
        in_dtype,
        out_dtype,
        tune=tune,
    )
    benchmark = TopkSelectorBenchmark(
        batch,
        seq_len,
        topk,
        in_dtype,
        out_dtype,
    )

    inputs = benchmark.gen_inputs()

    try:
        print("Testing topk_selector_fn...")
        benchmark.check_fn(fn, *inputs, grad=False)
        print("✅ topk_selector_fn test passed")
    except Exception as e:
        print(f"❌ topk_selector_fn test failed: {e}")
        raise

    try:
        print("Testing topk_selector_layer...")
        benchmark.check_fn(layer, *inputs, grad=False)
        print("✅ topk_selector_layer test passed")
    except Exception as e:
        print(f"❌ topk_selector_layer test failed: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
