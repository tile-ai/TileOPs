from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import SSDChunkScanFwdKernel

from .op_base import Op

__all__ = ["SSDChunkScanFwdOp"]


class SSDChunkScanFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) fused chunk output operator.

    Fuses the history (prev_states) contribution and intra-chunk causal decay
    into a single pass, computing:

      out[l, p] = exp(dA_cumsum[l]) * (C[l] @ prev_states)
                + sum_{s <= l} cb[l, s] * exp(dA_cumsum[l] - dA_cumsum[s]) * dt[s] * x[s, p]

    Args:
        batch:      Batch size.
        num_chunks: Number of chunks (T / chunk_len).
        chunk_len:  Tokens per chunk.
        n_heads:    Number of heads.
        d_head:     Head dimension.
        d_state:    State Space Model (SSM) state dimension.
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
        self.kernel = self.kernel_map["ssd_chunk_scan_fwd"](
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_scan_fwd": SSDChunkScanFwdKernel}

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused SSD chunk scan.

        Args:
            x:           (batch, seqlen, n_heads, d_head)                    dtype
            cb:          (batch, num_chunks, n_groups, chunk_len, chunk_len)  dtype
            dA_cumsum:   (batch, n_heads, num_chunks, chunk_len)              float32
            C:           (batch, seqlen, n_groups, d_state)                   dtype
            prev_states: (batch, num_chunks, n_heads, d_head, d_state)        dtype
            dt:          (batch, n_heads, num_chunks, chunk_len)              dtype

        Returns:
            out: (batch, seqlen, n_heads, d_head)  float32
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {x.dtype}")

        x = x.contiguous()
        cb = cb.contiguous()
        dA_cumsum = dA_cumsum.contiguous()
        C = C.contiguous()
        prev_states = prev_states.contiguous()
        dt = dt.contiguous()

        return self.kernel(x, cb, dA_cumsum, C, prev_states, dt)
