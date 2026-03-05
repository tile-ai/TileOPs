import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.activation import GeluAndMulKernel

from .op import Op

__all__ = ["GeluAndMulOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return math.ceil(n / alignment) * alignment


class GeluAndMulOp(Op):
    """Fused gelu(x) * gate operator.

    Padding to 256-byte alignment is required by TileLang shared memory copy
    instructions. For non-aligned hidden sizes, padded elements are zero and
    trimmed from the output, so correctness is unaffected. The extra copy
    overhead is small relative to the fused kernel benefit.
    """

    def __init__(self,
                 m: int,
                 n: int,
                 dtype: torch.dtype = torch.float16,
                 approximate: str = 'tanh',
                 tune: bool = False,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.M = m
        self.N = n
        self.dtype = dtype
        self.approximate = approximate
        self.padded_n = _align_up(n, ALIGNMENT)
        self.needs_padding = self.padded_n != n

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gelu_and_mul_kernel"](
            m, self.padded_n, dtype, approximate=approximate, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gelu_and_mul_kernel": GeluAndMulKernel}

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape

        if gate.shape != x.shape:
            raise ValueError(
                f"gate shape {gate.shape} does not match x shape {x.shape}")

        x = x.contiguous()
        gate = gate.contiguous()

        # Flatten arbitrary leading dims to 2D (M, N)
        x = x.reshape(-1, self.N)
        gate = gate.reshape(-1, self.N)

        if x.shape[0] != self.M:
            raise ValueError(
                f"Total number of rows ({x.shape[0]}) does not match M={self.M}")

        if self.needs_padding:
            pad_size = self.padded_n - self.N
            x = F.pad(x, (0, pad_size))
            gate = F.pad(gate, (0, pad_size))

        y = self.kernel(x, gate)

        if self.needs_padding:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
