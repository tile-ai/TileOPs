"""Dropout op with PyTorch-compatible semantics.

Implements inverted dropout: output = x * mask / (1 - p) during training,
identity pass-through during eval (training=False).

Edge cases:
- p=0: identity (no dropout)
- p=1: all zeros
- training=False: identity pass-through
"""

from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.dropout import DropoutKernel
from tileops.kernels.kernel_base import Entry, Kernel

from .op_base import Op

__all__ = ["DropoutFwdOp"]


class DropoutFwdOp(Op):
    """Dropout operation with deterministic replay via TileLang RNG.

    Compatible with PyTorch dropout semantics:
    - Training mode: output = x * mask / (1 - p), mask ~ Bernoulli(1 - p)
    - Eval mode (training=False): output = x (identity)
    - p=0: identity
    - p=1: all zeros

    Same seed produces identical masks for deterministic replay.
    Uses T.rng_init / T.rng_rand_float (backed by cuRAND Philox4_32_10
    by default) for per-thread random number generation.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"dropout": DropoutKernel}

    def __init__(
        self,
        p: float = 0.5,
        seed: int = 0,
        training: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            p: Drop probability in [0, 1].
            seed: Integer seed for RNG.
            training: If False, dropout is disabled (identity pass-through).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.p = p
        self.seed = seed
        self.training = training
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per element count, dtype and device."""
        count, dtype, _device_index = call
        return call, lambda: self.kernel_map["dropout"](
            count, dtype, p=self.p, seed=self.seed, tune=self.tune
        )

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input.clone()
        if self.p == 1.0:
            return torch.zeros_like(input)
        flat = input.contiguous().reshape(-1)
        # The kernel is handed the flat view; the signature declares `input` itself.
        kernel = self.kernel_for("dropout", (input,), (flat.numel(), flat.dtype, flat.device.index))
        return kernel(flat).reshape(input.shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)
