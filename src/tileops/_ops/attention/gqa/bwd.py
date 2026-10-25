from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.attention import (
    GQABwdMMAKernel,
    GQABwdPreprocessKernel,
    GQABwdWGMMAPipelinedKernel,
    GQABwdWSKernel,
)
from tileops._kernels.attention.call_spec import (
    AttentionCall,
    GQABwdInterface,
    GQAPreprocessBwdInterface,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.attention.gqa.parameters import score_softcap
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["GQADenseBwdOp"]


class GQADenseBwdOp(Op):
    r"""Grouped-Query Attention (GQA) backward over dense tensors. Layout: BSHD.

    The backward of :class:`GQADenseWithLSEFwdOp` over a square call ($S_q = S_{kv} = S$):
    ``o`` and ``lse`` are the pair that forward returns, and the parameters mean what the
    forward's do. ``lse`` is the float32 natural-log log-sum-exp of each query row's scaled
    scores, ``[B, H, S]``; a row with ``lse`` ``-inf`` saw no key and gets zero gradients.

    The in-tree kernels serve ``window_size_left = window_size_right = -1`` and no
    ``softcap``; a call with either is refused when a kernel is selected, naming the
    limit. Pass them anyway, equal to the forward's, so a refused call says why.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_bwd_preprocess_kernel": GQABwdPreprocessKernel,
        "gqa_bwd_kernel": GQABwdWGMMAPipelinedKernel,
        "gqa_bwd_ws_kernel": GQABwdWSKernel,
        "gqa_bwd_mma_kernel": GQABwdMMAKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gqa_bwd_preprocess": GQAPreprocessBwdInterface,
        "gqa_bwd": GQABwdInterface,
    }

    def __init__(
        self,
        is_causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            is_causal: Apply the causal mask. Default ``False``, as in ``flash_attn_func``.
            window_size_left: Keys admitted left of a query row; ``-1`` means unlimited.
                The in-tree kernels serve only ``-1``.
            window_size_right: Keys admitted right of a query row; ``-1`` means unlimited.
                The in-tree kernels serve only ``-1``.
            sm_scale: Score scale, or ``None`` for ``D ** -0.5``, ``D`` the head dimension
                of ``q`` and ``k``.
            softcap: Score cap of the forward; ``None`` or ``0`` disables it. The in-tree
                kernels serve only a disabled cap.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.

        Raises:
            ValueError: ``softcap`` is negative.
        """
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.sm_scale = sm_scale
        self.softcap = score_softcap(softcap)
        super().__init__(target=target)

    def _attention_call(self, q: torch.Tensor, k: torch.Tensor) -> AttentionCall:
        """State what one backward call is, for selection to filter against."""
        batch, seq_len, heads, dim = q.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=k.shape[2],
            dim=dim,
            max_seqlen_q=seq_len,
            seqlen_kv=seq_len,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            device=q.device,
        )

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the gradients of ``q``, ``k`` and ``v``.

        Args:
            do: Gradient of the loss with respect to ``o``, ``[B, S, H, D]``, same dtype
                as ``q``.
            q: Queries, ``[B, S, H, D]``, ``float16`` or ``bfloat16``.
            k: Keys, ``[B, S, H_kv, D]``, same dtype as ``q``.
            v: Values, ``[B, S, H_kv, D]``, same dtype as ``q``.
            o: The forward output, ``[B, S, H, D]``, same dtype as ``q``.
            lse: The forward's natural-log log-sum-exp, ``[B, H, S]``, ``float32``.

        Returns:
            ``(dq, dk, dv)``, shaped like ``q``, ``k`` and ``v``, in their dtype.
        """
        do = do.contiguous()

        def call() -> AttentionCall:
            return self._attention_call(q, k)

        # Reject unsupported backward calls before compiling or launching preprocess.
        backward = self.kernel_for("gqa_bwd", call)
        delta, lse_log2, dq_accum = self.kernel_for("gqa_bwd_preprocess", call)(
            o,
            do,
            lse.contiguous(),
            with_dq_accum=not backward.owns_dq_accum,
            launch_dependents=backward.waits_for_preprocess,
        )
        return backward(q, k, v, do, lse_log2, delta, dq_accum)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
