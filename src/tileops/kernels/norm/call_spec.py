"""Call records for normalization kernels."""

from __future__ import annotations

import dataclasses

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = ["BatchNormCall"]


@dataclasses.dataclass(frozen=True)
class BatchNormCall(CallSpec):
    """The facts that select a batch or instance normalization implementation and build it.

    The input is ``(n, c, *spatial)``; ``spatial`` is the product of the trailing axes.
    ``eps`` and ``momentum`` are the op's construction parameters, which the programs
    compile in. ``input_dtype_params`` is set where the affine is in the input dtype and
    the running statistics are read rounded to it, as ``instance_norm`` reads them;
    ``has_weight`` and ``has_bias`` say which affine tensors are passed.
    """

    n: int = 0
    c: int = 0
    spatial: int = 0
    dtype: torch.dtype = torch.float16
    eps: float = 1e-5
    momentum: float = 0.1
    input_dtype_params: bool = False
    has_weight: bool = True
    has_bias: bool = True

    @property
    def passes_affine(self) -> bool:
        """Whether ``weight`` or ``bias`` is passed."""
        return self.has_weight or self.has_bias
