"""
CB Producer Op - High-level interface for CB matrix computation.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba.cb_producer import CBProducerKernel
from tileops.ops.op_base import Op

__all__ = ["CBProducerOp"]


class CBProducerOp(Op):
    """CB (C@B) matrix producer operator.

    Computes cb[b,c,g,l,s] = sum_n C[b,c,g,l,n] * B[b,c,g,s,n]
    with causal masking (cb[l,s] = 0 if s > l).

    Args:
        batch: Batch size
        num_chunks: Number of chunks
        n_groups: Number of groups
        chunk_len: Chunk length (Q)
        d_state: State dimension (N)
        dtype: Data type
        tune: Whether to autotune
        kernel_map: Optional pre-initialized kernels
    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        dtype: torch.dtype = torch.float16,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.dtype = dtype

        if kernel_map is None:
            self.kernel = CBProducerKernel(
                batch, num_chunks, n_groups, chunk_len, d_state, dtype, tune=tune
            )
        else:
            self.kernel = kernel_map["cb_producer"]

        print(
            f"CBProducerOp initialized with config: {self.kernel.config}"
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """Default kernel map."""
        return {
            "cb_producer": CBProducerKernel(
                self.batch, self.num_chunks, self.n_groups,
                self.chunk_len, self.d_state, self.dtype, tune=False
            )
        }

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, C, G, Q, N]  dtype
            B_mat: [B, C, G, Q, N]  dtype

        Returns:
            cb: [B, C, G, Q, Q]  float32
        """
        return self.kernel(C_mat, B_mat)
