import argparse

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import NsaTopkForwardBenchmark
from top.ops import NsaTopkForwardOp


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "M, N, topk, dtype, tune",
    [
        (320, 128, 6, "float32", False),
    ],
)
def test_nsa_topk_op(M, N, topk, dtype, tune):
    op = NsaTopkForwardOp(M, N, topk, dtype, tune=tune)
    benchmark = NsaTopkForwardBenchmark(M, N, topk, dtype, tune=tune)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)
    benchmark.baseline_profile(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=320, help='number of tokens')
    parser.add_argument('--N', type=int, default=128, help='number of experts')
    parser.add_argument('--topk', type=int, default=8, help='topk')
    parser.add_argument('--dtype', type=str, default='float32', help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()
    test_nsa_topk_op(args.M, args.N, args.topk, args.dtype, args.tune)
