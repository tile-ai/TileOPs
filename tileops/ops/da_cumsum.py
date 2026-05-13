from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import DaCumsumFwdKernel

from .op_base import Op

__all__ = ["DaCumsumFwdOp"]


class DaCumsumFwdOp(Op):
    """Mamba-2 dA_cumsum forward operator.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    Args:
        batch:        Batch size.
        num_chunks:   Number of chunks (seq_len / chunk_len).
        chunk_len:    Tokens per chunk.
        n_heads:      Number of attention heads.
        seq_len:      Total sequence length (= num_chunks * chunk_len).
        dt_softplus:  Whether to apply softplus (with bypass for dt > 20) to dt.
        has_dt_bias:  Whether a per-head dt_bias is added before softplus/clamp.
        dt_min:       Lower clamp bound applied after bias and softplus.
        dt_max:       Upper clamp bound applied after bias and softplus.
        tune:         Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        dt_softplus: bool = False,
        has_dt_bias: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dt_softplus = dt_softplus
        self.has_dt_bias = has_dt_bias
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dtype = torch.float32
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["da_cumsum_fwd"](
            batch, num_chunks, chunk_len, n_heads, seq_len,
            dt_softplus=dt_softplus,
            has_dt_bias=has_dt_bias,
            dt_min=dt_min,
            dt_max=dt_max,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"da_cumsum_fwd": DaCumsumFwdKernel}

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the dA_cumsum forward pass.

        Args:
            dt: (batch, seq_len, n_heads) float32 — raw dt values.
            A:  (n_heads,) float32 — SSM decay parameters.
            dt_bias: (n_heads,) float32, optional — per-head dt bias.
                Required when the op was constructed with has_dt_bias=True.

        Returns:
            dt_out: (batch, n_heads, num_chunks, chunk_len) float32 — processed dt.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
        """
        if not dt.is_cuda:
            raise ValueError("dt must be a CUDA tensor")
        if dt.dtype != torch.float32:
            raise ValueError(f"Expected float32 dt, got {dt.dtype}")

        dt = dt.contiguous()
        A = A.contiguous()

        return self.kernel(dt, A, dt_bias)
