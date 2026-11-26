"""Dropout op with PyTorch-compatible semantics.

Implements inverted dropout: output = x * mask / (1 - p) during training,
identity pass-through during eval (training=False).

Edge cases:
- p=0: identity (no dropout)
- p=1: all zeros
- training=False: identity pass-through
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.elementwise.dropout import DropoutCall, DropoutFwdInterface, DropoutKernel
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["DropoutFwdOp"]


class DropoutFwdOp(Op):
    """Dropout, following `torch.nn.functional.dropout`; in place when ``inplace``.

    - Training mode: output = x * mask / (1 - p), mask ~ Bernoulli(1 - p)
    - Eval mode (training=False): output = x (identity)
    - p=0: identity
    - p=1: all zeros

    Without ``seed``, each call draws a fresh mask from torch's default CUDA generator of
    the input's device, as torch's own dropout does: it reads the generator's seed and
    Philox offset and advances the offset past the draws it uses. ``torch.manual_seed``
    therefore makes the sequence of masks reproducible. Reading the offset is not allowed
    while a CUDA graph is captured, so a captured call needs ``seed``.

    With ``seed``, every call uses Philox stream ``seed`` at offset 0 and draws the same
    mask for the same input shape, without touching torch's generator.

    Each element is dropped by a Philox4x32-10 draw of its own, independently of every other
    element, with probability ``p`` rounded up to a multiple of ``2**-32``. The mask is
    TileOPs' own: it does not reproduce torch's for the same generator state.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"dropout": DropoutKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"dropout": DropoutFwdInterface}

    def __init__(
        self,
        p: float = 0.5,
        training: bool = True,
        *,
        inplace: bool = False,
        seed: int | None = None,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            p: Drop probability in [0, 1].
            training: If False, dropout is disabled (identity pass-through).
            inplace: When True, write the result into ``input`` and return ``input``.
            seed: ``None`` draws a fresh mask on each call from torch's default CUDA
                generator; a non-negative integer fixes the mask, the same on every call.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.p = p
        self.training = training
        self.inplace = inplace
        self.seed = seed
        super().__init__(target=target)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Drop each element of ``input`` with probability ``p`` and scale the rest.

        Args:
            input: The tensor to drop elements of.

        Returns:
            A tensor of ``input``'s shape and dtype; ``input`` itself when ``inplace``.
        """
        if not self.training or self.p == 0.0:
            return input if self.inplace else input.clone()
        if self.p == 1.0:
            return input.zero_() if self.inplace else torch.zeros_like(input)
        flat = input.contiguous().reshape(-1)
        # The kernel is handed the flat view; the signature declares `input` itself.
        kernel = self.kernel_for(
            "dropout",
            lambda: DropoutCall(count=flat.numel(), p=self.p, dtype=flat.dtype, device=flat.device),
        )
        result = kernel(flat, *self._philox_state(flat)).reshape(input.shape)
        if not self.inplace:
            return result
        input.copy_(result)
        return input

    def _philox_state(self, flat: torch.Tensor) -> tuple[int, int]:
        """The ``(seed, offset)`` this call draws from, advancing the generator past it."""
        if self.seed is not None:
            return self.seed, 0
        generator = torch.cuda.default_generators[flat.device.index]
        offset = generator.get_offset()
        # torch keeps offsets a multiple of 4.
        draws = max(flat.numel(), DropoutFwdInterface.MAX_DRAWS_PER_STREAM)
        generator.set_offset(offset + (draws + 3) // 4 * 4)
        # The seed is a 64-bit pattern; the kernel takes it as a signed 64-bit integer.
        seed = generator.initial_seed()
        return seed - (1 << 64) if seed >= 1 << 63 else seed, offset
