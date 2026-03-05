import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.softmax import SoftmaxKernel

from .op import Op

__all__ = ["SoftmaxOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return math.ceil(n / alignment) * alignment


class SoftmaxOp(Op):

    def __init__(self,
                 m: int,
                 n: int,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.M = m
        self.N = n
        self.dtype = dtype
        self.padded_n = _align_up(n, ALIGNMENT)
        self.needs_padding = self.padded_n != n

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["softmax_kernel"](
            m, self.padded_n, n, dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"softmax_kernel": SoftmaxKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.contiguous()

        # Flatten arbitrary leading dims to 2D (M, N)
        x = x.reshape(-1, self.N)
        assert x.shape[0] == self.M, (
            f"Total number of rows ({x.shape[0]}) does not match M={self.M}")

        if self.needs_padding:
            pad_size = self.padded_n - self.N
            # Pad with -inf so exp(-inf) = 0, not affecting softmax sum
            x = F.pad(x, (0, pad_size), value=float('-inf'))

        y = self.kernel(x)

        if self.needs_padding:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
