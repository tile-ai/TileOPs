from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.attention.topk_select import (
    TopKSelectCall,
    TopKSelectFwdInterface,
    TopKSelectKernel,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op

__all__ = ["TopKSelectFwdOp"]


class TopKSelectFwdOp(Op):
    """The ``topk`` highest-scoring key positions of each query row's own window.

    Row ``(b, s, g)`` selects from ``index_score[b, s, starts[b, s]:ends[b, s], g]``.

    Two deviations from ``torch.topk``, which returns its indices sorted:

    - The indices come back in no particular order along the ``topk`` axis. Two calls on
      one input select the same positions and may place them in different slots.
    - A window holding fewer than ``topk`` positions fills the rest with ``seq_len_kv``,
      one past the last key, which selects nothing.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"topk_select_kernel": TopKSelectKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "topk_select": TopKSelectFwdInterface
    }

    def roofline_data_terms(self) -> "dict[str, int]":
        """The scores this call's windows hold, which its flops and score reads follow."""
        from tileops.perf.formulas import topk_select_window_scores

        return {"window_scores": topk_select_window_scores(self.last_call)}

    def __init__(
        self,
        topk: int,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            topk: Manifest ``params.topk``, ``int``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.topk = topk
        self.out_dtype = torch.int32

        super().__init__(target=target)

    def forward(
        self, index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        """Select each query row's ``topk`` highest-scoring keys inside its window.

        Args:
            index_score: Scores [batch, seq_len, seq_len_kv, kv_group], ``float32``.
            starts: First key of each row's window [batch, seq_len], ``int32``.
            ends: One past the last key of each row's window [batch, seq_len], ``int32``.

        Returns:
            Selected key positions [batch, seq_len, kv_group, topk], ``int32``.
        """
        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        index_score = index_score.contiguous()
        starts, ends = starts.contiguous(), ends.contiguous()
        call = TopKSelectCall(
            batch=batch,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            topk=self.topk,
            dtype=index_score.dtype,
            out_dtype=self.out_dtype,
            device=index_score.device,
        )
        kernel = self.kernel_for("topk_select", call)
        return kernel(index_score, starts, ends)
