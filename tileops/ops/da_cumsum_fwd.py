from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba import DaCumsumFwdKernel

from .op import Op

__all__ = ["DaCumsumFwdOp"]


class DaCumsumFwdOp(Op):
    """Mamba-2 dA_cumsum forward operator.

    Computes the chunk-local inclusive prefix sum of dA = dt * A:

      dA_cumsum[b, h, c, l] = sum_{i=0}^{l} dt[b, c*Q+i, h] * A[h]

    Args:
        batch:      Batch size.
        num_chunks: Number of chunks (seq_len / chunk_len).
        chunk_len:  Tokens per chunk.
        n_heads:    Number of attention heads.
        seq_len:    Total sequence length (= num_chunks * chunk_len).
        tune:       Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dtype = torch.float32
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["da_cumsum_fwd"](
            batch, num_chunks, chunk_len, n_heads, seq_len, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"da_cumsum_fwd": DaCumsumFwdKernel}

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Run the dA_cumsum forward pass.

        Args:
            dt: (batch, seq_len, n_heads) float32
            A:  (n_heads,) float32

        Returns:
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32
        """
        if not dt.is_cuda:
            raise ValueError("dt must be a CUDA tensor")
        if dt.dtype != torch.float32:
            raise ValueError(f"Expected float32 dt, got {dt.dtype}")

        dt = dt.contiguous()
        A = A.contiguous()

        return self.kernel(dt, A)
