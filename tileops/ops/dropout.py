"""Dropout op with PyTorch-compatible semantics.

Wraps DropoutKernel with shape handling and training/eval mode support.
Implements inverted dropout: output = x * mask / (1 - p) during training,
identity pass-through during eval (training=False).

Edge cases:
- p=0: identity (no dropout)
- p=1: all zeros
- training=False: identity pass-through
"""

import weakref
from typing import Dict, Optional

import torch

from tileops.kernels.dropout import DropoutKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["DropoutOp"]

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class DropoutOp(Op):
    """Dropout operation with deterministic replay via TileLang RNG.

    Compatible with PyTorch dropout semantics:
    - Training mode: output = x * mask / (1 - p), mask ~ Bernoulli(1 - p)
    - Eval mode (training=False): output = x (identity)
    - p=0: identity
    - p=1: all zeros

    Same seed produces identical masks for deterministic replay.
    Uses T.rng_init / T.rng_rand_float (backed by cuRAND Philox4_32_10
    by default) for per-thread random number generation.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype (float16, bfloat16, float32).
        p: Drop probability in [0, 1].
        seed: Integer seed for RNG.
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

        # Skip kernel build when dropout has no effect (identity or all-zeros)
        self._skip = not training or p == 0.0
        self._all_zero = training and p == 1.0

        # Always populate kernel_map for Op base class consistency
        self.dispatch_kernel(kernel_map)

        # Build kernel only when dropout is actually active
        if not self._skip and not self._all_zero:
            self.kernel = self.kernel_map[self._op_name](
                N_total, dtype, p=p, seed=seed, tune=tune,
            )
        else:
            self.kernel = None

        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * self.dtype.itemsize * 2

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._skip:
            return x.clone()
        if self._all_zero:
            return torch.zeros_like(x)
        orig_shape = x.shape
        x_flat = x.contiguous().reshape(-1)
        y_flat = self.kernel(x_flat)
        return y_flat.reshape(orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(
                f"Expected {self.N_total} elements, got {x.numel()}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)

    _wrapped = None


# ---------------------------------------------------------------------------
# torch.compile registration
# ---------------------------------------------------------------------------

@torch.library.custom_op("top::dropout", mutates_args=())
def _wrapped_dropout(x: torch.Tensor, instance_key: int) -> torch.Tensor:
    instance = _OP_REGISTRY[instance_key]
    return instance._eager_forward(x)


@_wrapped_dropout.register_fake
def _(x: torch.Tensor, instance_key: int) -> torch.Tensor:
    return torch.empty_like(x)


DropoutOp._wrapped = _wrapped_dropout
