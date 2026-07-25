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
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = None
        self.num_chunks = None
        self.chunk_len = None
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
        return {"ssd_chunk_scan_fwd": SSDChunkScanFwdKernel}

    def _get_kernel(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["ssd_chunk_scan_fwd"](
                batch,
                num_chunks,
                chunk_len,
                n_heads,
                d_head,
                d_state,
                n_groups,
                dtype,
                tune=self.tune,
            )
        return self._kernel_cache[key]

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
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, seq_len, n_heads, d_head]")
        batch, seq_len, n_heads, d_head = x.shape
        if dA_cumsum.ndim != 4:
            raise ValueError("dA_cumsum must have shape [batch, n_heads, num_chunks, chunk_len]")
        if dA_cumsum.shape[0] != batch or dA_cumsum.shape[1] != n_heads:
            raise ValueError("dA_cumsum must match x batch and n_heads")
        num_chunks, chunk_len = dA_cumsum.shape[2], dA_cumsum.shape[3]
        if seq_len != num_chunks * chunk_len:
            raise ValueError("x seq_len must equal num_chunks * chunk_len")
        if C.ndim != 4 or C.shape[0] != batch or C.shape[1] != seq_len:
            raise ValueError("C must have shape [batch, seq_len, n_groups, d_state]")
        n_groups, d_state = C.shape[2], C.shape[3]
        if n_heads % n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")
        if cb.shape != (batch, num_chunks, n_groups, chunk_len, chunk_len):
            raise ValueError("cb must have shape [batch, num_chunks, n_groups, chunk_len, chunk_len]")
        if prev_states.shape != (batch, num_chunks, n_heads, d_head, d_state):
            raise ValueError("prev_states must have shape [batch, num_chunks, n_heads, d_head, d_state]")
        if dt.shape != (batch, n_heads, num_chunks, chunk_len):
            raise ValueError("dt must have shape [batch, n_heads, num_chunks, chunk_len]")

        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = x.dtype
        self.kernel = self._get_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, x.dtype,
            x.device.index)

        x = x.contiguous()
        cb = cb.contiguous()
        dA_cumsum = dA_cumsum.contiguous()
        C = C.contiguous()
        prev_states = prev_states.contiguous()
        dt = dt.contiguous()

        return self.kernel(x, cb, dA_cumsum, C, prev_states, dt)
