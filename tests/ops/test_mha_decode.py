import argparse

import torch
import pytest

from benchmarks import MultiHeadAttentionDecodeBenchmark
from top.ops import MultiHeadAttentionDecodeWithKVCacheOp
from top.utils import str2dtype

# Set fixed seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@pytest.mark.parametrize(
    "b, h, s_q, s_kv, d, dtype, tune",
    [
        (1, 32, 128, 8192, 128, torch.bfloat16, False),
    ],
)
def test_mha_decode(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                    tune: bool) -> None:
    op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
    benchmark = MultiHeadAttentionDecodeBenchmark(b, h, s_q, s_kv, d, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=2e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seq_len_q', type=int, default=128, help='query sequence length')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument(
        '--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_mha_decode(args.batch, args.heads, args.seq_len_q, args.seq_len_kv, args.dim,
                    str2dtype["bfloat16"], args.tune)
    test_mha_decode(args.batch, args.heads, args.seq_len_q, args.seq_len_kv, args.dim,
                    str2dtype["float16"], args.tune)
    test_mha_decode(args.batch, args.heads, args.seq_len_q, 5, args.dim, str2dtype["float16"],
                    args.tune)
