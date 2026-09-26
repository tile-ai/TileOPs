from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.deltanet_call import DeltaNetDecodeCall
from tileops.kernels.linear_attention.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)

from ..op_base import Op

__all__ = ["DeltaNetDecodeFwdOp"]


class DeltaNetDecodeFwdOp(Op):
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

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "DeltaNetDecodeKernel": DeltaNetDecodeKernel,
        "DeltaNetDecodeFP32Kernel": DeltaNetDecodeFP32Kernel,
        "DeltaNetDecodeRawCudaFlaStyleKernel": DeltaNetDecodeRawCudaFlaStyleKernel,
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step.

        Args:
            q: Query [B, H, DK].
            k: Key [B, H, DK].
            v: Value [B, H, DV].
            beta: Delta-rule step size [B, H].
            state: Recurrent state [B, H, DK, DV].

        Returns:
            ``o`` [B, H, DV] and ``new_state`` [B, H, DK, DV].
        """
        return self._call_boundary(q, k, v, beta, state)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, heads, dim_k = q.shape
        call = DeltaNetDecodeCall(
            batch=batch,
            heads=heads,
            dim_k=dim_k,
            dim_v=v.shape[2],
            dtype=q.dtype,
            tune=self.tune,
            device=q.device,
        )
        kernel = self.kernel_for("deltanet_decode", (q, k, v, beta, state), call)
        return kernel(q, k, v, beta, state)
