from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.gla_recurrence import GLADecodeFP32Kernel, GLADecodeKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["GLADecodeOp"]


class GLADecodeOp(Op):
    """GLA (Gated Linear Attention) decode (single-step recurrence).

    Computes one step of the gated linear attention recurrence:
        S_new = diag(exp(gk)) @ S + outer(k, v)
        o     = scale * q^T @ S_new

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Dispatch: fp32 → FP32 kernel (no TF32), fp16/bf16 → TC kernel
        if dtype == torch.float32:
            kernel_cls = self.kernel_map["GLADecodeFP32Kernel"]
        else:
            kernel_cls = self.kernel_map["GLADecodeKernel"]
        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = kernel_cls(
            batch, heads, dim_k, dim_v,
            scale=scale,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLADecodeKernel": GLADecodeKernel,
            "GLADecodeFP32Kernel": GLADecodeFP32Kernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(q, k, v, gk, state)
