from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.grouped_gemm import (
    GroupedGemmCall,
    GroupedGemmKernel,
    GroupedGemmPersistentKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["GroupedGemmFwdOp"]


class GroupedGemmFwdOp(Op):
    """Grouped GEMM with configurable transpose modes. Nothing is committed at construction:
    the extents and the dtype come from the ``forward`` inputs.

    The ``(transpose_a, transpose_b)`` pair selects one of four layouts:

    | Flags | Layout | Product |
    | --- | --- | --- |
    | ``(False, True)`` | NT | $C = A \\mathbin{@} B^{\\top}$ |
    | ``(False, False)`` | NN | $C = A \\mathbin{@} B$ |
    | ``(True, False)`` | TN | $C = A^{\\top} \\mathbin{@} B$ |
    | ``(True, True)`` | TT | $C = A^{\\top} \\mathbin{@} B^{\\top}$ |

    The metadata values are the caller's obligation and are not checked, since checking
    them would synchronise: ``batch_sizes`` sums to the packed row count, and
    ``batch_offsets`` is its exclusive prefix sum.
    """

    compile_boundary: ClassVar[bool] = True

    # The SM90 template serves every layout whose extents TMA can address; the
    # general kernel takes what it refuses.
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "grouped_gemm_kernel": GroupedGemmKernel,
        "grouped_gemm_persistent": GroupedGemmPersistentKernel,
    }

    def __init__(
        self,
        transpose_a: bool = False,
        transpose_b: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            transpose_a: Whether the groups split the contraction: ``a`` is
                $[\\mathit{batch\\_sum} \\times N]$ and the output keeps a group axis.
            transpose_b: Whether the per-group operand is stored transposed. Default ``True`` (NT).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Run one GEMM per group, with the groups packed along a single axis.

        Args:
            a: Activations for every group, $[\\mathit{batch\\_sum} \\times K]$, or
                $[\\mathit{batch\\_sum} \\times N]$ when ``transpose_a``.
            b: Per-group weights, $[\\mathit{batch\\_count} \\times N \\times K]$ under
                the default NT layout, $[\\mathit{batch\\_count} \\times K \\times N]$
                under NN; with ``transpose_a``, $[K \\times \\mathit{batch\\_sum}]$ or
                $[\\mathit{batch\\_sum} \\times K]$.
            batch_sizes: Rows per group, 1D ``torch.int32``.
            batch_offsets: Start row of each group in ``a``, 1D ``torch.int32``.
            batch_padded_offsets: Start row of each group padded to 128 rows, 1D
                ``torch.int32``; no kernel reads it.

        Returns:
            The per-group products in the dtype of the inputs:
            $[\\mathit{batch\\_sum} \\times N]$, or
            $[\\mathit{batch\\_count} \\times N \\times K]$ when ``transpose_a``.

        Example:
            ```python linenums="1"
            op = GroupedGemmFwdOp()               # NT by default
            d = op(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
            ```
        """
        return self._call_boundary(a, b, batch_sizes, batch_offsets, batch_padded_offsets)

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = tuple(
            t.contiguous() for t in (a, b, batch_sizes, batch_offsets, batch_padded_offsets)
        )
        batch_sum, width = a.shape
        if self.transpose_a:
            n, k = width, b.shape[0 if self.transpose_b else 1]
        else:
            n, k = b.shape[1 if self.transpose_b else 2], width
        call = GroupedGemmCall(
            numel=batch_sum,
            num_experts=batch_sizes.shape[0],
            n=n,
            k=k,
            dtype=a.dtype,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
            tune=self.tune,
            device=a.device,
        )
        return self.kernel_for("grouped_gemm", inputs, call)(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])
