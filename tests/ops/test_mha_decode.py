import torch
import pytest

from benchmarks import MultiHeadAttentionDecodeBenchmark
from top.ops import MultiHeadAttentionDecodeWithKVCacheOp


@pytest.mark.parametrize(
    ("b", "h", "s_q", "s_kv", "d", "dtype", "tune"),
    [
        (1, 32, 128, 8192, 128, torch.float16, False),
        (1, 32, 128, 8192, 128, torch.bfloat16, False),
        (1, 32, 128, 5, 128, torch.float16, False),
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
    pytest.main([__file__, "-vvs"])
