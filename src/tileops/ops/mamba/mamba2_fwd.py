"""Mamba-2 end-to-end SSD forward operator.

Chains the five sub-ops in order:
  1. DaCumsumFwdOp        — dt preprocessing + dA cumulative sum
  2. CBProducerFwdOp      — causal C@B coupling matrix per chunk and group
  3. SSDChunkStateFwdOp   — per-chunk SSM state computation
  4. SSDStatePassingFwdOp — inter-chunk recurrent state scan
  5. SSDChunkScanFwdOp    — final output scan

* SSDChunkStateFwdOp output is float32 with shape (B, C, H, P, N). It is
  reshaped to (B, C, H, P*N) so SSDStatePassingFwdOp scans the flattened state.

* cb holds only the C@B term; SSDChunkScanFwdKernel applies the decay
  exp(dA[l] - dA[s]) * dt[s] itself.

* All intermediate tensors remain on-device; no host syncs between sub-ops.
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op
from .cb_producer import CBProducerFwdOp
from .da_cumsum import DaCumsumFwdOp
from .ssd_chunk_scan import SSDChunkScanFwdOp
from .ssd_chunk_state import SSDChunkStateFwdOp
from .ssd_state_passing import SSDStatePassingFwdOp

__all__ = ["Mamba2FwdOp"]


class Mamba2FwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) full forward pass operator.

    Combines DaCumsum → CBProducer → SSDChunkState → SSDStatePassing → SSDChunkScan
    into a single callable whose interface matches mamba_chunk_scan_combined from
    the official mamba_ssm library, except that ``final_states`` is always returned.

    """

    def __init__(
        self,
        chunk_size: int = 256,
        dt_softplus: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op and its sub-ops. Shapes and dtype are taken from each call.

        Args:
            chunk_size:         Tokens per chunk (default 256).
            dt_softplus:        Apply softplus to (dt + dt_bias) before use.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override, passed to every sub-op.
            tune:               Whether to autotune the sub-ops' kernels when first built.
        """
        self.chunk_size = chunk_size
        self.dt_softplus = dt_softplus
        self.target = target
        self.tune = tune
        # This composite owns no kernel; the override reaches the sub-ops that do.
        self.dispatch_kernel(kernel_map)
        shared = {"target": target, "kernel_map": kernel_map, "tune": tune}
        # dt_out is stored in x's dtype, a construction parameter of DaCumsumFwdOp.
        self._da_cumsum_ops = {
            dtype: DaCumsumFwdOp(chunk_size, out_dtype=dtype, dt_softplus=dt_softplus, **shared)
            for dtype in (torch.float16, torch.bfloat16)
        }
        self._cb_producer_op = CBProducerFwdOp(chunk_size, **shared)
        self._chunk_state_op = SSDChunkStateFwdOp(**shared)
        self._state_passing_op = SSDStatePassingFwdOp(**shared)
        self._chunk_scan_op = SSDChunkScanFwdOp(**shared)

    def kernel_delegates(self) -> tuple[Op, ...]:
        return (
            *self._da_cumsum_ops.values(),
            self._cb_producer_op,
            self._chunk_state_op,
            self._state_passing_op,
            self._chunk_scan_op,
        )

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full Mamba-2 SSD forward pass.

        Args:
            x:               (batch, seqlen, n_heads, d_head)          dtype
            dt:              (batch, seqlen, n_heads)                   float32
            A:               (n_heads,)                                 float32  (log-space, ≤ 0)
            B:               (batch, seqlen, n_groups, d_state)         dtype
            C:               (batch, seqlen, n_groups, d_state)         dtype
            dt_bias:         (n_heads,) float32, optional
            initial_states:  (batch, n_heads, d_head, d_state) float32, optional

        Returns:
            y:            (batch, seqlen, n_heads, d_head)   float32
            final_states: (batch, n_heads, d_head, d_state)  float32
        """
        batch, seqlen, n_heads, d_head = x.shape
        d_state = B.shape[3]
        chunk_size = self.chunk_size
        num_chunks = seqlen // chunk_size

        # dt_out: (B, H, C, Q) in x's dtype; dA_cumsum: (B, H, C, Q) float32.
        dt_out, dA_cumsum = self._da_cumsum_ops[x.dtype](dt, A, dt_bias)

        # cb[b,c,g,l,s] = C[b,c*Q+l,g,:] @ B[b,c*Q+s,g,:]^T for s <= l, else 0.
        cb = self._cb_producer_op(C, B)

        # No seq_idx: this composite does not segment a chunk, so the kernel is
        # built without that branch.
        chunk_states = self._chunk_state_op(x, B, dt_out, dA_cumsum)  # (B, C, H, P, N) float32

        # The state scan runs over the flattened P * N state.
        chunk_states_flat = chunk_states.reshape(batch, num_chunks, n_heads, d_head * d_state)
        # Last dA value per chunk. The slice of a 4D tensor is non-contiguous, so
        # contiguous() always copies.
        dA_chunk_cumsum = dA_cumsum[..., chunk_size - 1].contiguous()  # (B, H, C)
        init_flat = (
            None
            if initial_states is None
            else initial_states.reshape(batch, n_heads, d_head * d_state)
        )
        prev_states_flat, final_states_flat = self._state_passing_op(
            chunk_states_flat, dA_chunk_cumsum, init_flat
        )
        prev_states = prev_states_flat.reshape(batch, num_chunks, n_heads, d_head, d_state)

        y = self._chunk_scan_op(x, cb, dA_cumsum, C, prev_states, dt_out)  # (B, S, H, P) float32
        return y, final_states_flat.reshape(batch, n_heads, d_head, d_state)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["x"][1])
