from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
)
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["DeltaNetDecodeOp"]


class DeltaNetDecodeOp(Op):
    """DeltaNet decode (single-step recurrence, ungated).

    Computes one step of the delta rule (no gate):
        v_new = beta * (v - S @ k)
        o     = S @ q + (q . k) * v_new
        S_new = S + outer(k, v_new)

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
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Dispatch: fp32 -> FP32 kernel (no TF32), fp16/bf16 -> TC kernel
        if dtype == torch.float32:
            kernel_cls = self.kernel_map["DeltaNetDecodeFP32Kernel"]
        else:
            kernel_cls = self.kernel_map["DeltaNetDecodeKernel"]
        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = kernel_cls(
            batch, heads, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetDecodeKernel": DeltaNetDecodeKernel,
            "DeltaNetDecodeFP32Kernel": DeltaNetDecodeFP32Kernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(q, k, v, beta, state)
