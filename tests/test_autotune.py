import sys

import pytest
import torch

from top.ops import MultiHeadAttentionFwdOp


@pytest.mark.parametrize(
    "B, S, H, D, causal, dtype",
    [
        (8, 1024, 32, 128, False, torch.float16),
    ],
)
def test_mha_kernel_autotune(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    # 1. test autotune at initialization
    op = MultiHeadAttentionFwdOp(B, H, S, D, causal, dtype, tune=True)

    # 2. test op.autotune()
    op.autotune()


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
