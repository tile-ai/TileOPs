from typing import ClassVar, Dict, Optional

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.topk_selector import TopkSelectorKernel

from ._compile_boundary_codegen import OperatorSpec
from .op_base import Op

__all__ = ["TopkSelectorFwdOp"]


class TopkSelectorFwdOp(Op):
    """The ``topk`` highest-scoring key positions of each query row's own window.

    Row ``(b, s, g)`` selects from ``index_score[b, s, starts[b, s]:ends[b, s], g]``.

    Two deviations from ``torch.topk``, which returns its indices sorted:

    - The indices come back in no particular order along the ``topk`` axis. Two calls on
      one input select the same positions and may place them in different slots.
    - A window holding fewer than ``topk`` positions fills the rest with ``seq_len_kv``,
      one past the last key, which selects nothing.
    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self, topk: int, kernel_map: Optional[Dict[str, Kernel]] = None, tune: bool = False
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            topk: Manifest ``params.topk``, ``int``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.seq_len = None
        self.seq_len_kv = None
        self.kv_group = None
        self.topk = topk
        self.in_dtype = None
        self.out_dtype = torch.int32
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"topk_selector_kernel": TopkSelectorKernel}

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        return self.kernel_for(
            "topk_selector_kernel",
            inputs,
            (batch, seq_len, seq_len_kv, kv_group, self.topk, in_dtype, device_index, self.tune),
        )

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device; ``out_dtype`` is the op's."""
        batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, _device_index, tune = call
        return call, lambda: self.kernel_map["topk_selector_kernel"](
            batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, self.out_dtype, tune=tune
        )

    def _infer_output_shapes(
        self,
        index_score_shape: tuple[int, ...],
        starts_shape: tuple[int, ...],
        ends_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: $[batch \\times seq\\_len \\times kv\\_group \\times topk]$."""
        batch, seq_len, _, kv_group = index_score_shape
        return {"indexes": (batch, seq_len, kv_group, self.topk)}

    def forward(self, index_score, starts, ends) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            index_score: Input tensor, dtype ``float32``.
            starts: Input tensor, dtype ``int32``.
            ends: Input tensor, dtype ``int32``.

        Returns:
            ``indexes``, as the manifest declares.
        """
        return self._wrapped(index_score, starts, ends, self._instance_key)

    def _eager_forward(self, index_score, starts, ends) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        if not index_score.is_cuda:
            raise ValueError("TopkSelectorFwdOp expects CUDA inputs")
        if index_score.ndim != 4:
            raise ValueError("TopkSelectorFwdOp expects index_score shape [B, S, S_kv, G]")
        if starts.ndim != 2 or ends.ndim != 2:
            raise ValueError("TopkSelectorFwdOp expects starts/ends shape [B, S]")
        if not starts.is_cuda or not ends.is_cuda:
            raise ValueError("starts and ends must be CUDA tensors")
        if starts.dtype != torch.int32 or ends.dtype != torch.int32:
            raise ValueError("TopkSelectorFwdOp expects int32 starts/ends tensors")

        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        if starts.shape != (batch, seq_len) or ends.shape != (batch, seq_len):
            raise ValueError("TopkSelectorFwdOp starts/ends must match index_score batch/seq_len")
        if not 0 < self.topk <= seq_len_kv:
            raise ValueError(f"topk must satisfy 0 < topk <= seq_len_kv={seq_len_kv}")

        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.in_dtype = index_score.dtype
        self.kernel = self._get_kernel(
            (index_score, starts, ends),
            batch,
            seq_len,
            seq_len_kv,
            kv_group,
            index_score.dtype,
            index_score.device.index,
        )

        return self.kernel(index_score, starts, ends)
