"""ALiBi position-encoding generative op."""

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import AlibiFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _apply_fp8_post_cast


class AlibiFwdOp(Op):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        dtype: Torch dtype.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "alibi"

    def __init__(
        self,
        seq_len: int,
        num_heads: int,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](seq_len, num_heads, dtype)

    @property
    def default_kernel_map(self):
        return {"alibi": AlibiFwdKernel}

    def forward(self) -> torch.Tensor:
        out = self.kernel()
        result = out.reshape(self.num_heads, self.seq_len, self.seq_len)
        return _apply_fp8_post_cast(result, self.kernel)
