from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.mamba import (
    SSDChunkCumsumCall,
    SSDChunkCumsumFwdInterface,
    SSDChunkCumsumFwdKernel,
)
from tileops.ops.op_base import Op

__all__ = ["SSDChunkCumsumFwdOp"]


class SSDChunkCumsumFwdOp(Op):
    """Mamba-2 dA_cumsum forward operator.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    Note: dt_out is cast to the target dtype for storage efficiency, but dA_cumsum
    is computed from the fp32 dt values before casting, ensuring numerical precision.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "ssd_chunk_cumsum_fwd": SSDChunkCumsumFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ssd_chunk_cumsum_fwd": SSDChunkCumsumFwdInterface
    }

    def __init__(
        self,
        chunk_len: int,
        out_dtype: torch.dtype = torch.float32,
        dt_softplus: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        *,
        target: Target = None,
    ):
        """Build the op. Shapes are taken from each call.

        Args:
            chunk_len: Tokens per chunk.
            out_dtype: Storage dtype of ``dt_out``.
            dt_softplus: Whether to apply softplus (with bypass for dt > 20) to dt.
            dt_min: Lower clamp bound applied after bias and softplus.
            dt_max: Upper clamp bound applied after bias and softplus.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.chunk_len = chunk_len
        self.out_dtype = out_dtype
        self.dt_softplus = dt_softplus
        self.dt_min = dt_min
        self.dt_max = dt_max
        super().__init__(target=target)

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

        Returns:
            dt_out: (batch, n_heads, num_chunks, chunk_len) ``out_dtype`` — processed dt.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum
                of dA = dt_val * A, computed from fp32 dt_val before casting dt_out.
        """
        batch, seq_len, n_heads = dt.shape
        dt = dt.contiguous()
        A = A.contiguous()
        call = SSDChunkCumsumCall(
            batch=batch,
            seq_len=seq_len,
            n_heads=n_heads,
            chunk_len=self.chunk_len,
            has_dt_bias=dt_bias is not None,
            dt_softplus=self.dt_softplus,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            out_dtype=self.out_dtype,
            device=dt.device,
        )
        kernel = self.kernel_for("ssd_chunk_cumsum_fwd", call)
        return kernel(dt, A, dt_bias)
