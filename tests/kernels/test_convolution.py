"""Convolution kernels validate the call they are built for."""

import pytest
import torch

from tileops.kernels.convolution import Conv2d1x1Kernel, Conv2dSymmetricKernel


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_conv2d_symmetric_kernel_refuses_a_tile_that_spans_two_images() -> None:
    # applies() keeps the dispatcher off this shape, so only a direct construction
    # reaches the kernel with a tile that would run off the end of an image. It has to
    # be told, rather than left to write the wrong rows.
    with pytest.raises(ValueError, match="spans two of this call's 5 images"):
        Conv2dSymmetricKernel(5, 96, 9, 9, 64, 3, 1, 0, 1, torch.float16)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_a_kernel_built_without_a_bias_refuses_one() -> None:
    """Bias presence is compiled in, so the two sides are different programs.

    Built directly: through the op the two always agree, so this guard has no op-level
    entry point.
    """
    kernel = Conv2d1x1Kernel(
        n=1,
        c_in=32,
        h=8,
        w=8,
        c_out=32,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        dtype=torch.float16,
    )
    x = torch.randn(1, 32, 8, 8, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(32, 32, 1, 1, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.randn(32, device="cuda", dtype=torch.float16).contiguous()

    with pytest.raises(ValueError, match="built without a bias"):
        kernel(x, weight, bias)
