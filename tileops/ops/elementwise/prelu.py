"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import PreluFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import KernelEntry, _PerDtypeKernels, resolve_output_dtype


class PreluFwdOp(_PerDtypeKernels, Op):
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
        num_channels: int,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.shape = shape
        self.num_channels = num_channels
        # Manifest input bindings for the synthesized eval_roofline
        # (docs/design/roofline.md §4.4.3): each signature.inputs entry
        # is exposed as self.<name>_shape so the codegen resolver can
        # reach it without family-specific aliases.
        self.input_shape = tuple(shape)
        self.weight_shape = (num_channels,)
        N_total = prod(shape)
        self.N_total = N_total
        # PyTorch PReLU: channel dim is 1 for ndim>=2, else 0
        inner_size = (prod(shape[2:]) if len(shape) > 2 else 1) if len(shape) >= 2 else 1
        self.inner_size = inner_size
        self.dispatch_kernel(kernel_map)
        self._init_entries()

    def _build_entry(self, dtype: torch.dtype, *shape: int) -> KernelEntry:
        kernel = self.kernel_map[self._op_name](
            self.N_total, self.num_channels, self.inner_size, dtype,
        )

        return KernelEntry(
            kernel=kernel,
            compute_dtype=dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    @property
    def default_kernel_map(self):
        return {"prelu": PreluFwdKernel}

    def _eager_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = input.shape
        return self._entry(input.dtype).kernel(
            input.contiguous().reshape(-1), weight.contiguous().reshape(-1),
        ).reshape(orig_shape)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        self._validate_dtypes(input, weight)
        if tuple(input.shape) != tuple(self.shape):
            raise ValueError(
                f"Expected input.shape {tuple(self.shape)}, got {tuple(input.shape)}"
            )
        # ``weight`` is part of the manifest contract; validate device,
        # dtype, and length so a malformed weight fails fast at the op
        # boundary instead of corrupting the kernel.
        if not weight.is_cuda:
            raise ValueError("Weight must be a CUDA tensor")
        if weight.dtype != input.dtype:
            raise ValueError(
                f"Expected weight.dtype {input.dtype}, got {weight.dtype}"
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
