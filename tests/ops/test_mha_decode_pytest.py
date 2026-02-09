"""Pytest version of test_mha_decode: parametrized correctness check for MultiHeadAttentionDecodeWithKVCacheOp."""

import pytest
import torch

from benchmarks import MultiHeadAttentionDecodeBenchmark
from top.ops import MultiHeadAttentionDecodeWithKVCacheOp


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(12345)


@pytest.mark.parametrize(
    ("batch", "heads", "seq_len_q", "seq_len_kv", "dim", "dtype", "tune"),
    [
        (1, 32, 128, 8192, 128, torch.float16, False),
        (1, 32, 128, 8192, 128, torch.bfloat16, False),
        (1, 32, 128, 5, 128, torch.float16, False),
    ],
)
def test_mha_decode_op(
    batch: int,
    heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    op = MultiHeadAttentionDecodeWithKVCacheOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dtype, tune=tune)
    benchmark = MultiHeadAttentionDecodeBenchmark(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
