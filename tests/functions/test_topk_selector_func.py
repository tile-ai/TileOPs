import argparse

import pytest
import torch

from benchmarks import TopkSelectorBenchmark
from top.functions import TopkSelectorFunc
from top.layers import TopkSelectorLayer
from top.utils import str2dtype


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--seq_len', type=int, default=32 * 1024, help='sequence length')
    parser.add_argument('--topk', type=int, default=2048, help='topk')
    parser.add_argument('--in_dtype', type=str, default="float32", help='input type')
    parser.add_argument('--out_dtype', type=str, default="int32", help='output type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_topk_selector(args.batch, args.seq_len, args.topk, str2dtype[args.in_dtype],
                       str2dtype[args.out_dtype], args.tune)
