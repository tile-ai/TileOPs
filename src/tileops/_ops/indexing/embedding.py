"""Embedding lookup operator (host-side Op layer).

Provides:
  - EmbeddingFwdOp: output[*S, :] = weight[input[*S], :]
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.indexing import EmbeddingCall, EmbeddingFwdInterface, EmbeddingKernel
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["EmbeddingFwdOp"]


class EmbeddingFwdOp(Op):
    """Gather the ``weight`` rows ``input`` names; ``max_norm`` would rescale them in place.

    The lookup is ``torch.nn.functional.embedding``'s forward. ``padding_idx``,
    ``scale_grad_by_freq`` and ``sparse`` change only the gradient, so a call with them is the
    plain lookup. No kernel renormalizes by ``max_norm``: a call with it raises
    ``OpNotAvailableError`` and leaves ``weight`` unchanged.

    The caller guarantees every index lies in ``[0, V)``, or passes ``validate_inputs=True``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"embedding": EmbeddingKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"embedding": EmbeddingFwdInterface}

    def __init__(
        self,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op. The table shape and dtypes are taken from each call.

        Args:
            padding_idx: The row whose gradient is not accumulated; the forward reads it.
            max_norm: The norm each looked-up row is rescaled to at most, in place.
            norm_type: The p of the norm ``max_norm`` bounds.
            scale_grad_by_freq: Scale the gradient by index frequency; no effect forward.
            sparse: Return a sparse gradient; no effect forward.
            validate_inputs: Check on every call that each index lies in ``[0, V)``, at the
                cost of a device synchronization; not allowed during CUDA Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Gather a row of ``weight`` per index.

        Args:
            input: int32 or int64 indices of any shape ``[*S]``.
            weight: The table ``[V, D]``, float16, bfloat16, float32, int32 or int64.

        Returns:
            A new tensor ``[*S, D]`` in ``weight``'s dtype.
        """
        input, weight = input.contiguous(), weight.contiguous()
        return self.kernel_for(
            "embedding",
            lambda: EmbeddingCall(
                dim=weight.shape[1],
                dtype=weight.dtype,
                index_dtype=input.dtype,
                renorm=self.max_norm is not None,
                device=weight.device,
            ),
        )(input, weight)
