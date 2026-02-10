# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import pytest
import torch

from benchmarks import MultiHeadAttentionFwdBenchmark
from top.ops import MultiHeadAttentionFwdOp


@pytest.mark.parametrize(
    "B, S, H, D, causal, dtype",
    [
        (8, 1024, 32, 128, False, torch.float16),
        (4, 512, 16, 64, True, torch.bfloat16),
        (2, 2048, 64, 128, False, torch.float16),
    ],
)
def test_mha_kernel_compile(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    op = MultiHeadAttentionFwdOp(B, H, S, D, causal, dtype)
    benchmark = MultiHeadAttentionFwdBenchmark(B, H, S, D, causal, dtype)

    compiled_op = torch.compile(op, fullgraph=True)
    inputs = benchmark.gen_inputs()
    benchmark.check(
        compiled_op, *inputs, atol=5e-3, rtol=1e-5)  # will throw an error if not compatible
    benchmark.profile(compiled_op, *inputs)

    print('Successfully validate the compatibility with torch.compile().✅')


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
