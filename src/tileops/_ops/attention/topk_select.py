from typing import ClassVar, Mapping

import torch

from tileops._kernels.attention.topk_select import (
    TopKSelectCall,
    TopKSelectFwdInterface,
    TopKSelectKernel,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["TopKSelectFwdOp"]


class TopKSelectFwdOp(Op):
    """The ``top_k`` highest-scoring key positions of each query row's own window.

    It returns attention key positions only, where ``FusedTopKFwdOp`` returns routing
    weights and expert ids from router logits and ``TopKMaskFwdOp`` returns the logits
    with every entry outside the top k set to ``-inf``.

    Row ``(b, s, g)`` selects from ``logits[b, s, starts[b, s]:ends[b, s], g]``.

    Two deviations from ``torch.topk``, which returns its indices sorted:

    - The indices come back in no particular order along the ``top_k`` axis. Two calls on
      one input select the same positions and may place them in different slots.
    - A window holding fewer than ``top_k`` positions fills the rest with ``seq_len_kv``,
      one past the last key, which selects nothing.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"topk_select_kernel": TopKSelectKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "topk_select": TopKSelectFwdInterface
    }

    def roofline_data_terms(self) -> "dict[str, int]":
        """The scores this call's windows hold, which its flops and score reads follow."""
        from tileops._perf.formulas import topk_select_window_scores

        return {"window_scores": topk_select_window_scores(self.last_call)}

    def __init__(
        self,
        top_k: int,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: Manifest ``params.top_k``, ``int``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.top_k = top_k
        self.out_dtype = torch.int32

        super().__init__(target=target)

    def forward(
        self, logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        """Select each query row's ``top_k`` highest-scoring keys inside its window.

        Args:
            logits: Scores [batch, seq_len, seq_len_kv, kv_group], ``float32``.
            starts: First key of each row's window [batch, seq_len], ``int32``.
            ends: One past the last key of each row's window [batch, seq_len], ``int32``.

        Returns:
            Selected key positions [batch, seq_len, kv_group, top_k], ``int32``.
        """
        batch, seq_len, seq_len_kv, kv_group = logits.shape
        logits = logits.contiguous()
        starts, ends = starts.contiguous(), ends.contiguous()
        kernel = self.kernel_for(
            "topk_select",
            lambda: TopKSelectCall(
                batch=batch,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                kv_group=kv_group,
                topk=self.top_k,
                dtype=logits.dtype,
                out_dtype=self.out_dtype,
                device=logits.device,
            ),
        )
        return kernel(logits, starts, ends)
