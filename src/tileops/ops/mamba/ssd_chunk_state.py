from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.mamba import (
    SSDChunkStateCall,
    SSDChunkStateFwdInterface,
    SSDChunkStateFwdKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["SSDChunkStateFwdOp"]


class SSDChunkStateFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) chunk state forward operator.

    Computes the chunk-end State Space Model (SSM) state for each chunk:

      out[b, c, h, p, n] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]
              * (1 if seq_idx is None else (seq_idx[b,c*Q+Q-1] >= 0 and seq_idx[b,c*Q+l] == seq_idx[b,c*Q+Q-1]))

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "ssd_chunk_state_fwd": SSDChunkStateFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ssd_chunk_state_fwd": SSDChunkStateFwdInterface
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
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the SSD chunk state forward pass.

        Args:
            x:          (batch, seq_len, n_heads, d_head)
            Bmat:       (batch, seq_len, n_groups, d_state)
            dt:         (batch, n_heads, num_chunks, chunk_len), x's dtype or float32
            dA_cumsum:  (batch, n_heads, num_chunks, chunk_len) float32
            seq_idx:    (batch, seq_len) int32, optional

        Returns:
            states: (batch, num_chunks, n_heads, d_head, d_state) float32
        """
        batch, seq_len, n_heads, d_head = x.shape
        num_chunks, chunk_len = dt.shape[2], dt.shape[3]
        n_groups, d_state = Bmat.shape[2], Bmat.shape[3]
        call = SSDChunkStateCall(
            batch=batch,
            num_chunks=num_chunks,
            chunk_len=chunk_len,
            n_heads=n_heads,
            d_head=d_head,
            d_state=d_state,
            n_groups=n_groups,
            dtype=x.dtype,
            dt_dtype=dt.dtype,
            has_seq_idx=seq_idx is not None,
            device=x.device,
        )
        kernel = self.kernel_for("ssd_chunk_state_fwd", call)

        x = x.contiguous()
        Bmat = Bmat.contiguous()
        dt = dt.contiguous()
        dA_cumsum = dA_cumsum.contiguous()

        if seq_idx is None:
            # The kernel built for this call has no seq_idx branch, so this
            # buffer only fills the argument slot and is never read.
            seq_idx = x.new_empty(batch, seq_len, dtype=torch.int32)
        else:
            seq_idx = seq_idx.contiguous()

        return kernel(x, Bmat, dt, dA_cumsum, seq_idx)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["x"][1])
