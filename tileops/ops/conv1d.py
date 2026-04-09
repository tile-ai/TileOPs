from typing import Dict, Optional

import torch

from tileops.kernels.conv import Conv1dKernel
from tileops.kernels.conv.common import (
    resolve_conv1d_padding,
    validate_conv1d_params,
    validate_conv1d_tensors,
)
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["Conv1dFwdOp", "Conv1dBiasFwdOp"]


class _Conv1dBaseOp(Op):
    has_bias: bool = False

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        validate_conv1d_params(
            stride=stride,
            dilation=dilation,
            groups=groups,
            c_in=c_in,
            c_out=c_out,
        )

        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.pad_left, self.pad_right = resolve_conv1d_padding(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.dispatch_kernel(kernel_map)
        if "conv1d_kernel" not in self.kernel_map:
            raise NotImplementedError(f"{self.__class__.__name__} requires 'conv1d_kernel' in kernel_map")
        self.kernel = self.kernel_map["conv1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            kernel_l=kernel_size,
            stride_l=stride,
            pad_left=self.pad_left,
            pad_right=self.pad_right,
            dilation_l=dilation,
            groups=groups,
            dtype=dtype,
            has_bias=self.has_bias,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"conv1d_kernel": Conv1dKernel}

    def _run(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        validate_conv1d_tensors(
            op_name=type(self).__name__,
            input_shape=tuple(input.shape),
            expected_input_shape=(self.n, self.l_in, self.c_in),
            weight_shape=tuple(weight.shape),
            expected_weight_shape=(self.c_out, self.c_in // self.groups, self.kernel_size),
            bias_shape=None if bias is None else tuple(bias.shape),
            expected_bias_shape=(self.c_out,),
        )
        return self.kernel(input, weight, bias)


class Conv1dFwdOp(_Conv1dBaseOp):
    has_bias = False

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return self._run(input, weight)


class Conv1dBiasFwdOp(_Conv1dBaseOp):
    has_bias = True

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return self._run(input, weight, bias)
