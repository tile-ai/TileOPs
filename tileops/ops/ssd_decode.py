from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import SSDDecodeKernel

from .op_base import Op

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
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = None
        self.n_heads = None
        self.d_head = None
        self.d_state = None
        self.n_groups = None
        self.dtype = None
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_decode": SSDDecodeKernel}

    def _get_kernel(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, n_heads, d_head, d_state, n_groups, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["ssd_decode"](
                batch, n_heads, d_head, d_state, n_groups, dtype, tune=self.tune)
        return self._kernel_cache[key]

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
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, n_heads, d_head]")
        batch, n_heads, d_head = x.shape
        if state.ndim != 4:
            raise ValueError("state must have shape [batch, n_heads, d_head, d_state]")
        if state.shape[:3] != (batch, n_heads, d_head):
            raise ValueError("state must match x batch, n_heads, and d_head")
        d_state = state.shape[3]
        if B_in.ndim != 3 or B_in.shape[0] != batch or B_in.shape[2] != d_state:
            raise ValueError("B_in must have shape [batch, n_groups, d_state]")
        n_groups = B_in.shape[1]
        if n_heads % n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")
        if C_in.shape != (batch, n_groups, d_state):
            raise ValueError("C_in must have shape [batch, n_groups, d_state]")
        if A.shape != (n_heads, d_head, d_state):
            raise ValueError("A must have shape [n_heads, d_head, d_state]")
        if dt.shape != (batch, n_heads, d_head):
            raise ValueError("dt must have shape [batch, n_heads, d_head]")
        if dt.dtype != torch.float32:
            raise ValueError(f"Expected float32 dt, got {dt.dtype}")
        if state.dtype != torch.float32:
            raise ValueError(f"Expected float32 state, got {state.dtype}")
        if not state.is_contiguous():
            raise ValueError("state must be contiguous for in-place update")

        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = x.dtype
        self.kernel = self._get_kernel(
            batch, n_heads, d_head, d_state, n_groups, x.dtype, x.device.index)

        A = A.contiguous()
        dt = dt.contiguous()
        x = x.contiguous()
        B_in = B_in.contiguous()
        C_in = C_in.contiguous()

        return self.kernel(A, dt, x, B_in, C_in, state)
