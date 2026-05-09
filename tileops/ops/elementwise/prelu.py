"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import PreluFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _OP_REGISTRY, _apply_fp8_post_cast


class PreluFwdOp(Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs.

    Args:
        shape: Shape of the input tensor (must have a channel dimension).
        dtype: Torch dtype.
        num_channels: Number of channels (weight length).
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "prelu"
    _wrapped = None

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        num_channels: int,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.num_channels = num_channels
        N_total = prod(shape)
        self.N_total = N_total
        # PyTorch PReLU: channel dim is 1 for ndim>=2, else 0
        inner_size = (prod(shape[2:]) if len(shape) > 2 else 1) if len(shape) >= 2 else 1
        self.inner_size = inner_size
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](N_total, num_channels, inner_size, dtype)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"prelu": PreluFwdKernel}

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = input.shape
        result = self.kernel(
            input.contiguous().reshape(-1), weight.contiguous().reshape(-1),
        ).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if tuple(input.shape) != tuple(self.shape):
            raise ValueError(
                f"Expected input.shape {tuple(self.shape)}, got {tuple(input.shape)}"
            )
        # ``weight`` is part of the manifest contract; validate device,
        # dtype, and length so a malformed weight fails fast at the op
        # boundary instead of corrupting the kernel.
        if not weight.is_cuda:
            raise ValueError("Weight must be a CUDA tensor")
        if weight.dtype != self.dtype:
            raise ValueError(
                f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
            )
        if weight.numel() != self.num_channels:
            raise ValueError(
                f"Expected weight to have {self.num_channels} elements, "
                f"got {weight.numel()}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, weight, self._instance_key)
        return self._eager_forward(input, weight)
