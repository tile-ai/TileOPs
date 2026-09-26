from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.gla_recurrence import GLADecodeFP32Kernel, GLADecodeKernel

from ..op_base import Op

__all__ = ["GLADecodeFwdOp"]


class GLADecodeFwdOp(Op):
    """GLA (Gated Linear Attention) decode (single-step recurrence).

    Computes one step of the gated linear attention recurrence:
        S_new = diag(exp(gk)) @ S + outer(k, v)
        o     = scale * q^T @ S_new

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "GLADecodeKernel": GLADecodeKernel,
        "GLADecodeFP32Kernel": GLADecodeFP32Kernel,
    }

    def __init__(
        self,
        scale: float = -1.0,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            scale: Query scale; a non-positive value means ``DK ** -0.5``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.scale = scale
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """The dtype picks the implementation, so it is in the identity."""
        batch, heads, dim_k, dim_v, dtype, _device = call
        name = "GLADecodeFP32Kernel" if dtype == torch.float32 else "GLADecodeKernel"
        return call, lambda: self.kernel_map[name](
            batch,
            heads,
            dim_k,
            dim_v,
            scale=self.scale,
            dtype=Kernel.dtype_to_str(dtype),
            tune=self.tune,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step.

        Args:
            q: Query [B, H, DK].
            k: Key [B, H, DK].
            v: Value [B, H, DV].
            gk: Log-space key gate [B, H, DK].
            state: Recurrent state [B, H, DK, DV].

        Returns:
            ``o`` [B, H, DV] and ``new_state`` [B, H, DK, DV].
        """
        return self._call_boundary(q, k, v, gk, state)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, heads, dim_k = q.shape
        kernel = self.kernel_for(
            "gla_decode",
            (q, k, v, gk, state),
            (batch, heads, dim_k, v.shape[2], q.dtype, q.device.index),
        )
        return kernel(q, k, v, gk, state)
