from typing import ClassVar, Dict, Mapping, Optional

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

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"topk_select_kernel": TopKSelectKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "topk_select": TopKSelectFwdInterface
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The scores this call's windows hold, which its flops and score reads follow."""
        from tileops.perf.formulas import topk_select_window_scores

        return {"window_scores": topk_select_window_scores(self.last_call)}

    def __init__(
        self,
        topk: int,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            topk: Manifest ``params.topk``, ``int``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.topk = topk
        self.out_dtype = torch.int32
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

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
        return self._call_boundary(index_score, starts, ends)

    def _eager_forward(
        self, index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
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
        self.kernel = self.kernel_for("topk_select", call)
        return self.kernel(index_score, starts, ends)
