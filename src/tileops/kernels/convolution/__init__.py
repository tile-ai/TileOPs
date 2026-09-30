"""Convolution kernels, one module per spatial rank."""

from tileops.kernels.convolution.call_spec import (
    Conv1dCall,
    Conv1dFwdInterface,
    Conv2dCall,
    Conv2dFwdInterface,
    Conv3dCall,
    Conv3dFwdInterface,
)
from tileops.kernels.convolution.conv1d import (
    Conv1dKernel,
    Conv1dPointwiseKernel,
    DepthwiseConv1dKernel,
    GroupConv1dKernel,
)
from tileops.kernels.convolution.conv2d import (
    Conv2d1x1Kernel,
    Conv2dKernel,
    Conv2dSymmetricKernel,
    DepthwiseConv2dKernel,
    GroupConv2dKernel,
)
from tileops.kernels.convolution.conv3d import Conv3dKernel, Conv3dNdhwcKernel, GroupConv3dKernel

__all__ = [
    "Conv1dCall",
    "Conv1dFwdInterface",
    "Conv1dKernel",
    "Conv1dPointwiseKernel",
    "Conv2d1x1Kernel",
    "Conv2dCall",
    "Conv2dFwdInterface",
    "Conv2dKernel",
    "Conv2dSymmetricKernel",
    "Conv3dCall",
    "Conv3dFwdInterface",
    "Conv3dKernel",
    "Conv3dNdhwcKernel",
    "DepthwiseConv1dKernel",
    "DepthwiseConv2dKernel",
    "GroupConv1dKernel",
    "GroupConv2dKernel",
    "GroupConv3dKernel",
]
