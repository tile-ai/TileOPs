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
    """Embedding lookup ``weight[input]``; ``max_norm`` rescales rows of ``weight`` in place.

    ``output[*S, :] = weight[input[*S], :]``, the forward of
    ``torch.nn.functional.embedding``: an exact copy of the rows ``input`` names, in the
    table's dtype. ``padding_idx``, ``scale_grad_by_freq`` and ``sparse`` change only the
    gradient, so the lookup ignores them. ``max_norm`` renormalizes the looked-up rows of
    ``weight`` in place; no in-tree kernel serves it, so on the in-tree kernels a call
    passing it is refused.

    The caller guarantees every id lies in ``[0, V)``, or passes ``validate_inputs=True``,
    which checks them on the CPU on every call. With the check off, the output row of an
    out-of-range id is unspecified.
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
        """Build the op. The ids, table shape and dtypes are taken from each call.

        Args:
            padding_idx: The row whose gradient stays zero; the lookup ignores it.
            max_norm: Renormalize each looked-up row of ``weight`` whose norm exceeds it.
            norm_type: The p of the p-norm ``max_norm`` compares against.
            scale_grad_by_freq: Scale gradients by id frequency; the lookup ignores it.
            sparse: Make ``weight``'s gradient sparse; the lookup ignores it.
            validate_inputs: Check on the CPU that every id lies in ``[0, V)``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Look up a row of ``weight`` for every id in ``input``.

        Args:
            input: Row ids of any shape $[*S]$, int32 or int64.
            weight: Table $[V \\times D]$, float16, bfloat16, float32, int32 or int64.

        Returns:
            A new $[*S \\times D]$ tensor in ``weight``'s dtype.
        """
        input = input.contiguous()
        return self.kernel_for(
            "embedding",
            lambda: EmbeddingCall(
                width=weight.shape[1],
                index_dtype=input.dtype,
                dtype=weight.dtype,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                device=weight.device,
            ),
        )(input, weight)
