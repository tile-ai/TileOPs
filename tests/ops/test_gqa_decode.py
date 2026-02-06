import sys

import torch
import pytest

from benchmarks import GroupQueryAttentionDecodeBenchmark
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp


@pytest.mark.parametrize(
    "b, h, g, s_kv, d, dtype, tune",
    [
        (1, 32, 8, 8192, 128, torch.float16, False),
    ],
)
def test_gqa_decode(b: int, h: int, g: int, s_kv: int, d: int, dtype: torch.dtype,
                    tune: bool) -> None:
    op = GroupQueryAttentionDecodeWithKVCacheOp(b, h, g, s_kv, d, dtype, tune=tune)
    benchmark = GroupQueryAttentionDecodeBenchmark(b, h, g, s_kv, d, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=1e-2, rtol=1e-2)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
