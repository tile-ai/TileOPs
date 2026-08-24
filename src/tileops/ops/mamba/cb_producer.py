"""
CB Producer Op - High-level interface for CB matrix computation.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba.cb_producer import CBProducerKernel

from .._validation import check_tensor_shape
from ..op_base import Op

__all__ = ["CBProducerFwdOp"]


class CBProducerFwdOp(Op):
    """CB (C@B) matrix producer operator.

    Computes cb[b,c,g,l,s] = sum_n C[b,c,g,l,n] * B[b,c,g,s,n]
    with causal masking (cb[l,s] = 0 if s > l).

    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            batch: Batch size
            num_chunks: Number of chunks
            n_groups: Number of groups
            chunk_len: Chunk length (Q)
            d_state: State dimension (N)
            tune: Whether to autotune
            kernel_map: Optional pre-initialized kernels
        """
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.tune = tune

        # Use standard Op dispatch pattern
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "cb_producer",
            inputs,
            key=dtype,
            build=lambda: self.kernel_map["cb_producer"](
                self.batch,
                self.num_chunks,
                self.n_groups,
                self.chunk_len,
                self.d_state,
                dtype,
                tune=self.tune,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """Default kernel map - returns kernel class, not instance."""
        return {"cb_producer": CBProducerKernel}

    def _infer_output_shapes(
        self,
        C_mat_shape: tuple[int, ...],
        B_mat_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: one causal ``(Q, Q)`` block per batch, chunk and group."""
        batch, _, groups, _ = C_mat_shape
        return {"cb": (batch, self.num_chunks, groups, self.chunk_len, self.chunk_len)}

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, S, G, N]  dtype (contiguous)
            B_mat: [B, S, G, N]  dtype (contiguous)

        Returns:
            cb: [B, C, G, Q, Q]  dtype
        """
        self._validate_dtypes(C_mat, B_mat)
        S = self.num_chunks * self.chunk_len
        expected_shape = (self.batch, S, self.n_groups, self.d_state)
        self.dtype = C_mat.dtype
        check_tensor_shape("C_mat", C_mat, expected_shape)
        check_tensor_shape("B_mat", B_mat, expected_shape)
        C_mat = C_mat.contiguous()
        B_mat = B_mat.contiguous()
        return self._get_kernel((C_mat, B_mat), C_mat.dtype)(C_mat, B_mat)
