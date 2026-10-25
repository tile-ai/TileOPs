from typing import ClassVar, Mapping

import torch

from tileops._kernels.gemm.grouped import (
    GroupedGEMMCall,
    GroupedGEMMFwdInterface,
    GroupedGEMMKernel,
    GroupedGEMMPersistentKernel,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["GroupedGEMMFwdOp"]


class GroupedGEMMFwdOp(Op):
    """Grouped GEMM with configurable transpose modes. Nothing is committed at construction:
    the extents and the dtype come from the ``forward`` inputs.

    The ``(trans_a, trans_b)`` pair selects one of four layouts:

    | Flags | Layout | Product |
    | --- | --- | --- |
    | ``(False, True)`` | NT, the default | $C = A \\mathbin{@} B^{\\top}$ |
    | ``(False, False)`` | NN | $C = A \\mathbin{@} B$ |
    | ``(True, False)`` | TN | $C = A^{\\top} \\mathbin{@} B$ |
    | ``(True, True)`` | TT | $C = A^{\\top} \\mathbin{@} B^{\\top}$ |

    The metadata values are the caller's obligation and are not checked, since checking
    them would synchronise: ``batch_sizes`` sums to the packed row count, and
    ``batch_offsets`` is its exclusive prefix sum.
    """

    # The SM90 template serves every layout whose extents TMA can address; the
    # general kernel takes what it refuses.
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "grouped_gemm": GroupedGEMMKernel,
        "grouped_gemm_persistent": GroupedGEMMPersistentKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "grouped_gemm": GroupedGEMMFwdInterface
    }

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            trans_a: Whether the groups split the contraction: ``a`` is
                $[\\mathit{batch\\_sum} \\times N]$ and the output keeps a group axis.
            trans_b: Whether the per-group operand is stored transposed. Default ``True`` (NT).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.trans_a = trans_a
        self.trans_b = trans_b
        super().__init__(target=target)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Run one GEMM per group, with the groups packed along a single axis.

        Args:
            a: Activations for every group, $[\\mathit{batch\\_sum} \\times K]$, or
                $[\\mathit{batch\\_sum} \\times N]$ when ``trans_a``.
            b: Per-group weights, $[\\mathit{batch\\_count} \\times N \\times K]$ under
                the default NT layout, $[\\mathit{batch\\_count} \\times K \\times N]$
                under NN; with ``trans_a``, $[K \\times \\mathit{batch\\_sum}]$ or
                $[\\mathit{batch\\_sum} \\times K]$.
            batch_sizes: Rows per group, 1D ``torch.int32``.
            batch_offsets: Start row of each group in ``a``, 1D ``torch.int32``.

        Returns:
            The per-group products in the dtype of the inputs:
            $[\\mathit{batch\\_sum} \\times N]$, or
            $[\\mathit{batch\\_count} \\times N \\times K]$ when ``trans_a``.

        Example:
            ```python linenums="1"
            op = GroupedGEMMFwdOp()               # NT by default
            d = op(a, b, batch_sizes, batch_offsets)
            ```
        """
        inputs = tuple(t.contiguous() for t in (a, b, batch_sizes, batch_offsets))
        batch_sum, width = a.shape
        if self.trans_a:
            n, k = width, b.shape[0 if self.trans_b else 1]
        else:
            n, k = b.shape[1 if self.trans_b else 2], width
        return self.kernel_for(
            "grouped_gemm",
            lambda: GroupedGEMMCall(
                numel=batch_sum,
                num_experts=batch_sizes.shape[0],
                n=n,
                k=k,
                dtype=a.dtype,
                transpose_a=self.trans_a,
                transpose_b=self.trans_b,
                device=a.device,
            ),
        )(*inputs)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.indices["T"])
