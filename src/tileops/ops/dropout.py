"""Dropout op with PyTorch-compatible semantics.

Wraps DropoutKernel with shape handling and training/eval mode support.
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

from ._compile_boundary_codegen import OperatorSpec
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

    _op_name = "dropout"
    kernel_cls = DropoutKernel

    def __init__(
        self,
        p: float = 0.5,
        seed: int = 0,
        training: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            p: Drop probability in [0, 1].
            seed: Integer seed for RNG.
            training: If False, dropout is disabled (identity pass-through).
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.target = target
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.N_total = None
        self.dtype = None
        self.p = p
        self.seed = seed
        self.training = training
        self.tune = tune

        # Skip kernel build when dropout has no effect (identity or all-zeros)
        self._skip = not training or p == 0.0
        self._all_zero = training and p == 1.0

        # Always populate kernel_map for Op base class consistency
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        if self.N_total is None or self.dtype is None:
            raise RuntimeError(
                "DropoutFwdOp.total_memory requires a prior forward() call to bind input shape and dtype"
            )
        return self.N_total * self.dtype.itemsize * 2

    def _get_kernel(self, x: torch.Tensor, rows: torch.Tensor) -> Kernel:
        """Fetch the kernel for *x*, handing over *x* itself rather than *rows*.

        *rows* is the flat view the kernel wants; *x* is what the signature declares.
        """
        return self.kernel_for(self._op_name, (x,), (rows.numel(), rows.dtype, rows.device.index))

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per element count, dtype and device."""
        count, dtype, _device_index = call
        return call, lambda: self.kernel_map[self._op_name](
            count, dtype, p=self.p, seed=self.seed, tune=self.tune
        )

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"input.dtype must be float16, bfloat16, or float32, got {x.dtype}")
        self.N_total = x.numel()
        self.dtype = x.dtype
        if self._skip:
            return x.clone()
        if self._all_zero:
            return torch.zeros_like(x)
        orig_shape = x.shape
        x_flat = x.contiguous().reshape(-1)
        self.kernel = self._get_kernel(x, x_flat)
        y_flat = self.kernel(x_flat)
        return y_flat.reshape(orig_shape)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: masking writes one element per input element."""
        return {"output": tuple(input_shape)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, as the manifest declares.
        """
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._serve(input)

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)
