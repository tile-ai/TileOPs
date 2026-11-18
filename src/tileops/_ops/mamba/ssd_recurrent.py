from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.mamba import SSDDecodeCall, SSDDecodeFwdInterface, SSDDecodeKernel
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["SSDRecurrentFwdOp"]


class SSDRecurrentFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) recurrent decode step; updates ``state`` in place.

    Returns the output y for the current token:

      g                = h // (n_heads // n_groups)
      dA[b, h, p, n]   = exp(dt[b, h, p] * A[h, p, n])
      state[b,h,p,n]  <- dA[b,h,p,n] * state[b,h,p,n]
                         + dt[b,h,p] * B[b,g,n] * x[b,h,p]
      y[b, h, p]       = sum_n  state[b, h, p, n] * C[b, g, n]

    The caller applies the skip connection (D * x) and the output gate (z * silu).
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"ssd_decode": SSDDecodeKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ssd_decode": SSDDecodeFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self,
        A: torch.Tensor,
        dt: torch.Tensor,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single Mamba-2 decode step.

        Args:
            A:     (n_heads, d_head, d_state) float32  -- SSM decay parameter (A <= 0)
            dt:    (batch, n_heads, d_head) float32  -- discretization step (post-softplus)
            x:     (batch, n_heads, d_head) dtype  -- input features per head
            B:     (batch, n_groups, d_state) dtype  -- SSM B matrix (per group)
            C:     (batch, n_groups, d_state) dtype  -- SSM C matrix (per group)
            state: (batch, n_heads, d_head, d_state) float32  -- recurrent state (mutated in-place)

        Returns:
            y: (batch, n_heads, d_head) float32
        """
        batch, n_heads, d_head = x.shape
        d_state = state.shape[3]
        n_groups = B.shape[1]
        kernel = self.kernel_for(
            "ssd_decode",
            lambda: SSDDecodeCall(
                batch=batch,
                n_heads=n_heads,
                d_head=d_head,
                d_state=d_state,
                n_groups=n_groups,
                dtype=x.dtype,
                device=x.device,
            ),
        )
        return kernel(
            A.contiguous(),
            dt.contiguous(),
            x.contiguous(),
            B.contiguous(),
            C.contiguous(),
            state,
        )
