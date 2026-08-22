"""Input checks an op's ``forward`` runs before it reaches a kernel.

These take no op instance: an input's declared device and shape are properties of
the call, not of the op that received it.
"""

import torch

__all__ = ["check_tensor_shape"]


def check_tensor_shape(name: str, tensor: torch.Tensor, shape: "tuple[int, ...]") -> None:
    """Gate a declared input's device and shape. Dtypes are ``_validate_dtypes``' job."""
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(tensor.shape)}")
