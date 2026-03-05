from typing import Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel

__all__ = ["InstanceNormKernel"]


class InstanceNormKernel(Kernel):
    """Kernel wrapper for instance normalization forward."""

    # Marked as broadly supported because it dispatches to PyTorch CUDA kernel.
    supported_archs: list[int] = [70, 75, 80, 86, 89, 90]

    def __init__(self,
                 num_channels: int,
                 eps: float,
                 affine: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.dtype = dtype
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.device.type != "cuda":
            raise ValueError(f"InstanceNormKernel requires CUDA tensor, got {x.device}.")
        if x.ndim < 3:
            raise ValueError(f"instance_norm expects input ndim >= 3, got {x.ndim}.")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"input channel mismatch: expected {self.num_channels}, got {x.shape[1]}.")
        if x.dtype != self.dtype:
            raise ValueError(f"input dtype mismatch: expected {self.dtype}, got {x.dtype}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}.")

        if self.affine:
            if weight is None or bias is None:
                raise ValueError("weight and bias are required when affine=True.")
            if weight.shape != (self.num_channels,):
                raise ValueError(
                    f"weight shape mismatch: expected ({self.num_channels},), got {tuple(weight.shape)}."
                )
            if bias.shape != (self.num_channels,):
                raise ValueError(
                    f"bias shape mismatch: expected ({self.num_channels},), got {tuple(bias.shape)}.")
            if weight.dtype != x.dtype or bias.dtype != x.dtype:
                raise ValueError("weight and bias must match input dtype.")
            if weight.device != x.device or bias.device != x.device:
                raise ValueError("weight and bias must be on same device as input.")
        else:
            if weight is not None or bias is not None:
                raise ValueError("weight and bias must be None when affine=False.")

        return F.instance_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=bias,
            use_input_stats=True,
            momentum=0.1,
            eps=self.eps,
        )
