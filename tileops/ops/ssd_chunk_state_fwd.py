from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba import SsdChunkStateFwdKernel

from .op import Op

__all__ = ["SsdChunkStateFwdOp"]


class SsdChunkStateFwdOp(Op):
    """Mamba-2 SSD chunk state forward operator.

    Computes the chunk-end SSM state for each chunk:

      out[b, c, h, n, p] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]

    Args:
        batch:      Batch size.
        num_chunks: Number of chunks (seq_len / chunk_len).
        chunk_len:  Tokens per chunk.
        n_heads:    Number of attention heads.
        d_head:     Head dimension.
        d_state:    SSM state dimension.
        n_groups:   Number of B-matrix groups (n_heads must be divisible by n_groups).
        dtype:      Data type for inputs (float16 or bfloat16).
        tune:       Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ssd_chunk_state_fwd"](
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_state_fwd": SsdChunkStateFwdKernel}

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
    ) -> torch.Tensor:
        """Run the SSD chunk state forward pass.

        Args:
            x:          (batch, seq_len, n_heads, d_head)
            Bmat:       (batch, seq_len, n_groups, d_state)
            dt:         (batch, n_heads, num_chunks, chunk_len) float32
            dA_cumsum:  (batch, n_heads, num_chunks, chunk_len) float32

        Returns:
            out: (batch, num_chunks, n_heads, d_state, d_head) float32
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {x.dtype}")

        x = x.contiguous()
        Bmat = Bmat.contiguous()
        dt = dt.contiguous()
        dA_cumsum = dA_cumsum.contiguous()

        return self.kernel(x, Bmat, dt, dA_cumsum)
