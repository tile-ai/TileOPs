from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba import SSDDecodeKernel

from .op import Op

__all__ = ["SSDDecodeOp"]


class SSDDecodeOp(Op):
    """Mamba-2 State-Space Dual (SSD) recurrent decode (step) operator.

    Performs a single decode step of the Mamba-2 State Space Model (SSM) core: updates the
    recurrent state in-place and returns the output y for the current token:

      g                = h // (n_heads // n_groups)
      dA[b, h, p, n]   = exp(dt[b, h, p] * A[h, p, n])
      state[b,h,p,n]  <- dA[b,h,p,n] * state[b,h,p,n]
                         + dt[b,h,p] * B_in[b,g,n] * x[b,h,p]
      y_out[b, h, p]   = sum_n  state[b, h, p, n] * C_in[b, g, n]

    The skip connection (D * x) and output gate (z * silu) are not fused
    here and must be applied by the caller if needed.

    Args:
        batch:    Batch size.
        n_heads:  Number of attention heads.
        d_head:   Head dimension (headdim in Mamba-2 notation).
        d_state:  SSM state dimension.
        n_groups: Number of B/C groups (n_heads must be divisible by n_groups).
        dtype:    Data type for x, B_in, C_in (float16 or bfloat16).
        tune:     Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        dtype: torch.dtype = torch.float16,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        if n_heads % n_groups != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})"
            )
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ssd_decode"](
            batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_decode": SSDDecodeKernel}

    def forward(
        self,
        A: torch.Tensor,
        dt: torch.Tensor,
        x: torch.Tensor,
        B_in: torch.Tensor,
        C_in: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single Mamba-2 decode step.

        Args:
            A:     (n_heads, d_head, d_state) float32  -- SSM decay parameter (A <= 0)
            dt:    (batch, n_heads, d_head) float32  -- discretization step (post-softplus)
            x:     (batch, n_heads, d_head) dtype  -- input features per head
            B_in:  (batch, n_groups, d_state) dtype  -- SSM B matrix (per group)
            C_in:  (batch, n_groups, d_state) dtype  -- SSM C matrix (per group)
            state: (batch, n_heads, d_head, d_state) float32  -- recurrent state (mutated in-place)

        Returns:
            y_out: (batch, n_heads, d_head) float32
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x dtype {self.dtype}, got {x.dtype}")
        if dt.dtype != torch.float32:
            raise ValueError(f"Expected float32 dt, got {dt.dtype}")
        if state.dtype != torch.float32:
            raise ValueError(f"Expected float32 state, got {state.dtype}")
        if not state.is_contiguous():
            raise ValueError("state must be contiguous for in-place update")

        A = A.contiguous()
        dt = dt.contiguous()
        x = x.contiguous()
        B_in = B_in.contiguous()
        C_in = C_in.contiguous()

        return self.kernel(A, dt, x, B_in, C_in, state)
