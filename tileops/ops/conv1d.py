from typing import Dict, Optional

import torch

from tileops.kernels.conv import Conv1dKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["Conv1dFwdOp", "Conv1dBiasFwdOp"]


class Conv1dFwdOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "conv1d_kernel" not in self.kernel_map:
            raise NotImplementedError("Conv1dFwdOp requires 'conv1d_kernel' in kernel_map")
        self.kernel = self.kernel_map["conv1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            kernel_l=kernel_size,
            stride_l=stride,
            pad_l=padding,
            dtype=dtype,
            has_bias=bias,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"conv1d_kernel": Conv1dKernel}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel(x, weight, bias)


class Conv1dBiasFwdOp(Conv1dFwdOp):
    """Conv1d forward with bias=True default.

    Identical to :class:`Conv1dFwdOp` but defaults ``bias=True`` so the
    manifest key ``Conv1dBiasFwdOp`` resolves to a distinct class name.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if not bias:
            raise ValueError(
                "Conv1dBiasFwdOp requires bias=True. "
                "Use Conv1dFwdOp for the no-bias variant."
            )
        super().__init__(
            n=n, c_in=c_in, l_in=l_in, c_out=c_out,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias, dtype=dtype, kernel_map=kernel_map, tune=tune,
        )
