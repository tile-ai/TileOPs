from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba import SSDDecodeKernel

from ..op_base import Op

__all__ = ["SSDDecodeFwdOp"]


class SSDDecodeFwdOp(Op):
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

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"ssd_decode": SSDDecodeKernel}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune the tile config when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        batch, n_heads, d_head, d_state, n_groups, dtype, _device = call
        return call, lambda: self.kernel_map["ssd_decode"](
            batch, n_heads, d_head, d_state, n_groups, dtype, tune=self.tune
        )

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
        return self._call_boundary(A, dt, x, B_in, C_in, state)

    def _eager_forward(
        self,
        A: torch.Tensor,
        dt: torch.Tensor,
        x: torch.Tensor,
        B_in: torch.Tensor,
        C_in: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, n_heads, d_head = x.shape
        d_state = state.shape[3]
        n_groups = B_in.shape[1]
        kernel = self.kernel_for(
            "ssd_decode",
            (A, dt, x, B_in, C_in, state),
            (batch, n_heads, d_head, d_state, n_groups, x.dtype, x.device.index),
        )
        return kernel(
            A.contiguous(),
            dt.contiguous(),
            x.contiguous(),
            B_in.contiguous(),
            C_in.contiguous(),
            state,
        )
