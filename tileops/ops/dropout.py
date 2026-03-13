"""Dropout op with PyTorch-compatible semantics.

Wraps DropoutKernel with shape handling and training/eval mode support.
Implements inverted dropout: output = x * mask / (1 - p) during training,
identity pass-through during eval (training=False).

Edge cases:
- p=0: identity (no dropout)
- p=1: all zeros
- training=False: identity pass-through
"""

from typing import Dict, Optional

import torch

from tileops.kernels.dropout import DropoutKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["DropoutOp"]


class DropoutOp(Op):
    """Dropout operation with Philox PRNG for deterministic replay.

    Compatible with PyTorch dropout semantics:
    - Training mode: output = x * mask / (1 - p), mask ~ Bernoulli(1 - p)
    - Eval mode (training=False): output = x (identity)
    - p=0: identity
    - p=1: all zeros

    Same seed produces identical masks for deterministic replay.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype (float16, bfloat16, float32).
        p: Drop probability in [0, 1].
        seed: Integer seed for Philox PRNG.
        training: If False, dropout is disabled (identity pass-through).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "dropout"
    kernel_cls = DropoutKernel

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        p: float = 0.5,
        seed: int = 0,
        training: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.N_total = N_total
        self.dtype = dtype
        self.p = p
        self.seed = seed
        self.training = training

        # Build kernel only when dropout is actually active
        self._skip = not training or p == 0.0
        if not self._skip:
            self.dispatch_kernel(kernel_map)
            self.kernel = self.kernel_map[self._op_name](
                N_total, dtype, p=p, seed=seed, tune=tune,
            )
        else:
            # Still need kernel_map for the interface but no kernel is built
            self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * self.dtype.itemsize * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(
                f"Expected {self.N_total} elements, got {x.numel()}"
            )

        # Edge cases: identity pass-through (return input directly to
        # preserve aliasing, matching torch.nn.functional.dropout semantics)
        if self._skip:
            return x

        # Edge case: p=1 means all zeros
        if self.p == 1.0:
            return torch.zeros_like(x)

        orig_shape = x.shape
        x_flat = x.contiguous().reshape(-1)
        y_flat = self.kernel(x_flat)
        return y_flat.reshape(orig_shape)
