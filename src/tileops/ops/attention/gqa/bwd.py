from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    GQABwdMMAKernel,
    GQABwdPreprocessKernel,
    GQABwdWGMMAPipelinedKernel,
    MHABwdWSKernel,
)
from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQABwdInterface,
    GQAPreprocessBwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GQABwdOp"]


class GQABwdOp(Op):
    """Layout: BSHD"""

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_bwd_preprocess_kernel": GQABwdPreprocessKernel,
        "gqa_bwd_kernel": GQABwdWGMMAPipelinedKernel,
        "gqa_bwd_ws_kernel": MHABwdWSKernel,
        "gqa_bwd_mma_kernel": GQABwdMMAKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gqa_bwd_preprocess": GQAPreprocessBwdInterface,
        "gqa_bwd": GQABwdInterface,
    }

    def __init__(
        self,
        is_causal: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.is_causal = is_causal

        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, same dtype as ``q``.
            v: Input tensor, same dtype as ``q``.
            o: Input tensor, same dtype as ``q``.
            do: Input tensor, same dtype as ``q``.
            lse: Input tensor, dtype ``float32``.

        Returns:
            ``dq``, ``dk``, ``dv``, as the manifest declares. Shape rules: ``dq.shape == (B, S, H, D)``; ``dk.shape == (B, S, H_kv, D)``; ``dv.shape == (B, S, H_kv, D)``.
        """
        return self._call_boundary(q, k, v, o, do, lse)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        do = do.contiguous()
        call = self._attention_call(q, k)
        # Reject unsupported backward calls before compiling or launching preprocess.
        backward = self.kernel_for("gqa_bwd", call)
        delta, dq_accum = self.kernel_for("gqa_bwd_preprocess", call)(o, do)
        inputs = (q, k, v, do, lse, delta, dq_accum)
        return backward(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
