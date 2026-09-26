from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.topk_selector import TopkSelectorKernel

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

    The in-tree kernel keeps at most 4096 candidates that share the score bucket the
    ``topk``-th score falls in; a window with more such ties may select a lower score.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "topk_selector_kernel": TopkSelectorKernel
    }

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

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device; ``out_dtype`` is the op's."""
        batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, _device_index = call
        return call, lambda: self.kernel_map["topk_selector_kernel"](
            batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, self.out_dtype, tune=self.tune
        )

    def forward(self, index_score, starts, ends) -> torch.Tensor:
        """Select each query row's ``topk`` highest-scoring keys inside its window.

        Args:
            index_score: Scores [batch, seq_len, seq_len_kv, kv_group], ``float32``.
            starts: First key of each row's window [batch, seq_len], ``int32``.
            ends: One past the last key of each row's window [batch, seq_len], ``int32``.

        Returns:
            Selected key positions [batch, seq_len, kv_group, topk], ``int32``.
        """
        return self._call_boundary(index_score, starts, ends)

    def _eager_forward(self, index_score, starts, ends) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        self.kernel = self.kernel_for(
            "topk_selector_kernel",
            (index_score, starts, ends),
            (
                batch,
                seq_len,
                seq_len_kv,
                kv_group,
                self.topk,
                index_score.dtype,
                index_score.device.index,
            ),
        )
        return self.kernel(index_score, starts, ends)
