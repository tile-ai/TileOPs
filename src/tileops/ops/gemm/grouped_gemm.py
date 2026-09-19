from typing import ClassVar, Dict, Optional

import torch

from tileops.kernels.grouped_gemm import (
    GroupedGemmCall,
    GroupedGemmKernel,
    GroupedGemmPersistentKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof
from tileops.utils import get_sm_version

from .._compile_boundary_codegen import OperatorSpec
from ..op_base import Op

__all__ = ["GroupedGemmFwdOp"]


class GroupedGemmFwdOp(Op):
    """Grouped GEMM with configurable transpose modes.

    The ``(transpose_a, transpose_b)`` pair selects one of four layouts:

    | Flags | Layout | Product |
    | --- | --- | --- |
    | ``(False, True)`` | NT | $C = A \\mathbin{@} B^{\\top}$ |
    | ``(False, False)`` | NN | $C = A \\mathbin{@} B$ |
    | ``(True, False)`` | TN | $C = A^{\\top} \\mathbin{@} B$ |
    | ``(True, True)`` | TT | $C = A^{\\top} \\mathbin{@} B^{\\top}$ |

    Rows may reach the two M-grouped layouts (NT, NN) packed tight or padded, and
    ``batch_padded_offsets`` states which: passing it says every group starts on a
    row block, so no tile spans two groups, and the op then runs a kernel that
    stores whole tiles and keeps a deeper mainloop. The statement is a contract --
    a padded layout whose groups do not in fact start on a row block reads a
    neighbouring group's rows. Omit it for tight rows, which cost a row mask on
    each group's last tile. Groups split K under ``transpose_a``, which carries no
    row padding, so that layout takes no padded table.
    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)
    #: Rows a padded layout starts each group on, for a caller to pad against.
    row_block: ClassVar[int] = GroupedGemmPersistentKernel.row_block

    def __init__(
        self,
        transpose_a: bool = False,
        transpose_b: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            transpose_a: Manifest ``params.transpose_a``, ``bool``, default ``False``.
            transpose_b: Manifest ``params.transpose_b``, ``bool``, default ``True``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch_sum = None
        self.batch_count = None
        self.N = None
        self.K = None
        self.dtype = None
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    # The SM90 template serves every layout whose extents TMA can address; the
    # general kernel takes what it refuses.

    @property
    def default_kernel_map(self) -> Dict:
        return {
            "grouped_gemm_kernel": GroupedGemmKernel,
            "grouped_gemm_persistent": GroupedGemmPersistentKernel,
        }

    def _resolve_spec(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: Optional[torch.Tensor] = None,
    ) -> tuple[int, int, int, int, torch.dtype, int | None, bool]:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("a and b must be CUDA tensors")
        if a.dtype != b.dtype:
            raise ValueError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")
        if a.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"a.dtype must be float16 or bfloat16, got {a.dtype}")
        padded = batch_padded_offsets is not None
        tables = [batch_sizes, batch_offsets] + ([batch_padded_offsets] if padded else [])
        if any(t.ndim != 1 for t in tables):
            raise ValueError("batch metadata tensors must be 1D")
        batch_count = batch_sizes.shape[0]
        if any(t.shape[0] != batch_count for t in tables):
            raise ValueError("batch metadata tensors must have matching lengths")
        if any(t.dtype != torch.int32 for t in tables):
            raise ValueError("batch metadata tensors must use int32 dtype")
        if padded and self.transpose_a:
            raise ValueError(
                "GroupedGemmFwdOp takes no batch_padded_offsets when transpose_a=True: "
                "the groups split K there, and K carries no row padding"
            )
        if padded and a.shape[0] % self.row_block:
            raise ValueError(
                f"a padded layout runs every group to a {self.row_block}-row block, so a "
                f"holds a multiple of {self.row_block} rows; got {a.shape[0]}. Rows packed "
                f"tight are the call without batch_padded_offsets"
            )

        if not self.transpose_a:
            if a.ndim != 2 or b.ndim != 3:
                raise ValueError("GroupedGemmFwdOp expects 2D a and 3D b when transpose_a=False")
            batch_sum, k = a.shape
            if b.shape[0] != batch_count:
                raise ValueError(
                    f"b.shape[0] must match batch_count={batch_count}, got {b.shape[0]}"
                )
            if self.transpose_b:
                n, b_k = b.shape[1], b.shape[2]
            else:
                b_k, n = b.shape[1], b.shape[2]
            if b_k != k:
                raise ValueError(f"GroupedGemmFwdOp expected K={k}, got b K dimension {b_k}")
        else:
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError("GroupedGemmFwdOp expects 2D a and b when transpose_a=True")
            batch_sum, n = a.shape
            if self.transpose_b:
                k, b_batch_sum = b.shape
            else:
                b_batch_sum, k = b.shape
            if b_batch_sum != batch_sum:
                raise ValueError(
                    f"GroupedGemmFwdOp expected b batch_sum dimension {batch_sum}, got {b_batch_sum}"
                )
        return batch_sum, batch_count, n, k, a.dtype, a.device.index, padded

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch_sum: int,
        batch_count: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device_index: int | None,
        padded: bool,
    ) -> Kernel:
        call = GroupedGemmCall(
            arch=get_sm_version(device_index),
            numel=batch_sum,
            num_experts=batch_count,
            n=n,
            k=k,
            dtype=dtype,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
            padded=padded,
            tune=self.tune,
            device=None if device_index is None else torch.device("cuda", device_index),
        )
        return self.kernel_for("grouped_gemm", inputs, call)

    def layout_guard(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Asynchronous bool: the tables describe the row layout the call states.

        What only the device-resident values can answer -- a host check would
        synchronise, so ``forward`` never runs this. Tests and benchmarks consume
        it through ``torch._assert_async``. Groups run in order and stay inside
        ``a``; a padded layout additionally starts every group on a row block, the
        property that lets the kernel store whole tiles.

        Args:
            a: The activations the call passes.
            batch_sizes: Rows per group, 1D ``torch.int32``.
            batch_offsets: Start row of each group under a tight layout.
            batch_padded_offsets: Start row of each group under a padded layout,
                or ``None`` for a tight one.

        Returns:
            A 0-d ``torch.bool`` tensor on ``a``'s device.
        """
        starts = batch_offsets if batch_padded_offsets is None else batch_padded_offsets
        ends = starts + batch_sizes
        ok = (starts[1:] >= ends[:-1]).all() & (ends[-1] <= a.shape[0]) & (starts[0] == 0)
        if batch_padded_offsets is not None:
            ok = ok & (starts % self.row_block == 0).all()
        return ok

    def _infer_output_shapes(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
        batch_sizes_shape: tuple[int, ...],
        batch_offsets_shape: tuple[int, ...],
        batch_padded_offsets_shape: Optional[tuple[int, ...]] = None,
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``transpose_a`` decides whether the groups stay an axis."""
        if self.transpose_a:
            n = b_shape[0] if self.transpose_b else b_shape[1]
            return {"output": (batch_sizes_shape[0], a_shape[1], n)}
        n = b_shape[1] if self.transpose_b else b_shape[2]
        return {"output": (a_shape[0], n)}

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one GEMM per group, with the groups packed along a single axis.

        Args:
            a: Activations for every group, $[\\mathit{batch\\_sum} \\times K]$, or
                $[K \\times \\mathit{batch\\_sum}]$ when ``transpose_a``.
            b: Per-group weights, $[\\mathit{batch\\_count} \\times N \\times K]$ when
                ``transpose_a`` is false, or $[\\mathit{batch\\_sum} \\times N]$ when it is.
            batch_sizes: Rows per group, 1D ``torch.int32``.
            batch_offsets: Start row of each group in ``a``, 1D ``torch.int32``.
            batch_padded_offsets: Start row of each group in ``a`` under a padded
                layout, 1D ``torch.int32``. Passing it states that every group
                starts on a row block; omit it when the rows are packed tight.
                Takes ``transpose_a`` false.

        Returns:
            The per-group products, $[\\mathit{batch\\_sum} \\times N]$, in the dtype of
            the inputs.

        Raises:
            ValueError: ``a`` or ``b`` is not on CUDA, their dtypes differ or are
                neither float16 nor bfloat16, the metadata tensors are not 1D int32 of
                equal length, ``batch_padded_offsets`` arrives under ``transpose_a``,
                or the operand ranks and dims disagree with the layout flags.

        Example:
            ```python linenums="1"
            op = GroupedGemmFwdOp()                          # NT by default
            d = op(a, b, batch_sizes, batch_offsets)         # rows packed tight
            d = op(a, b, batch_sizes, batch_offsets, batch_padded_offsets)  # padded
            ```
        """
        return self._wrapped(
            a, b, batch_sizes, batch_offsets, batch_padded_offsets, self._instance_key
        )

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch_sum, batch_count, n, k, dtype, device_index, padded = self._resolve_spec(
            a,
            b,
            batch_sizes,
            batch_offsets,
            batch_padded_offsets,
        )
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.kernel = self._get_kernel(
            (a, b, batch_sizes, batch_offsets, batch_padded_offsets),
            batch_sum,
            batch_count,
            n,
            k,
            dtype,
            device_index,
            padded,
        )
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
