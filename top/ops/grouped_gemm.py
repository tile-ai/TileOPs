from typing import Dict, Optional

import torch

from top.kernels.grouped_gemm import (
    grouped_gemm_nn_kernel,
    grouped_gemm_nt_kernel,
    grouped_gemm_tn_kernel,
    grouped_gemm_tt_kernel,
)
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = ['GroupedGemmNTOp', 'GroupedGemmNNOp', 'GroupedGemmTNOp', 'GroupedGemmTTOp']


class GroupedGemmNTOp(Op):
    """
    Grouped GEMM forward with no transpose on A and transpose on B:

     A @ B^T -> C
    """

    def __init__(self,
                 batch_sum: int,
                 batch_count: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_nt_kernel"](
            batch_sum, batch_count, n, k, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict:
        return {"grouped_gemm_nt_kernel": grouped_gemm_nt_kernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)


class GroupedGemmNNOp(Op):
    """
    Grouped GEMM backward that calculates dA with no transpose on both A and B:

      A @ B -> C
    """

    def __init__(self,
                 batch_sum: int,
                 batch_count: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_nn_kernel"](
            batch_sum, batch_count, n, k, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict:
        return {"grouped_gemm_nn_kernel": grouped_gemm_nn_kernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)


class GroupedGemmTNOp(Op):
    """
    Grouped GEMM backward that calculates dB with transpose on A and no transpose on B:

      A^T @ B -> C
    """

    def __init__(self,
                 batch_sum: int,
                 batch_count: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_tn_kernel"](
            batch_sum, batch_count, n, k, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict:
        return {"grouped_gemm_tn_kernel": grouped_gemm_tn_kernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)


class GroupedGemmTTOp(Op):
    """
    Grouped GEMM backward that calculates dB with transpose on A and transpose on B^T:

      A^T @ B^T -> C
    """

    def __init__(self,
                 batch_sum: int,
                 batch_count: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_tt_kernel"](
            batch_sum, batch_count, n, k, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict:
        return {"grouped_gemm_tt_kernel": grouped_gemm_tt_kernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
