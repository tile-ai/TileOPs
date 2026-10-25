from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.linear_attention import (
    GLADecodeCall,
    GLADecodeFP32Kernel,
    GLADecodeFwdInterface,
    GLADecodeKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["GLARecurrentFwdOp"]


class GLARecurrentFwdOp(Op):
    """Gated Linear Attention (GLA) decode (single-step recurrence).

    Computes one step of the gated linear attention recurrence:
        S_new = diag(exp(g)) @ S + outer(k, v)
        o     = scale * q^T @ S_new

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gla_decode": GLADecodeKernel,
        "gla_decode_fp32": GLADecodeFP32Kernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gla_decode": GLADecodeFwdInterface
    }

    def __init__(
        self,
        scale: Optional[float] = None,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            scale: Positive query scale, or ``None`` for ``DK ** -0.5`` (key head dim).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.scale = scale
        super().__init__(target=target)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step.

        Args:
            q: Query [B, H, DK].
            k: Key [B, H, DK].
            v: Value [B, H, DV].
            g: Log-space key gate [B, H, DK].
            state: Recurrent state [B, H, DK, DV].

        Returns:
            ``o`` [B, H, DV] and ``new_state`` [B, H, DK, DV].
        """
        batch, heads, dim_k = q.shape
        return self.kernel_for(
            "gla_decode",
            lambda: GLADecodeCall(
                batch=batch,
                heads=heads,
                dim_k=dim_k,
                dim_v=v.shape[2],
                scale=self.scale if self.scale is not None else dim_k**-0.5,
                dtype=q.dtype,
                device=q.device,
            ),
        )(q, k, v, g, state)
