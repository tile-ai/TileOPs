import torch
from .op import Op
from top.kernels import Kernel
from top.kernels.grouped_gemm.grouped_gemm import grouped_gemm_nt_kernel, grouped_gemm_nn_kernel, grouped_gemm_tn_kernel, grouped_gemm_tt_kernel
from typing import Optional, Dict

__all__ = ['grouped_gemm_nt', 'grouped_gemm_nn', 'grouped_gemm_tn', 'grouped_gemm_tt']

class grouped_gemm_nt(Op):
    """
    Grouped GEMM forward with no transpose on A and transpose on B: A @ B^T -> C
    """
    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_nt_kernel"](batch_sum, batch_count, N, K, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"grouped_gemm_nt_kernel": grouped_gemm_nt_kernel}
    
    def forward(self, 
                A: torch.Tensor, 
                B: torch.Tensor, 
                batch_sizes: torch.Tensor, 
                batch_offsets: torch.Tensor,
                batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)

class grouped_gemm_nn(Op):
    """
    Grouped GEMM backward that calculates dA with no transpose on both A and B: A @ B -> C
    """
    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_nn_kernel"](batch_sum, batch_count, N, K, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"grouped_gemm_nn_kernel": grouped_gemm_nn_kernel}
    
    def forward(self, 
                A: torch.Tensor, 
                B: torch.Tensor, 
                batch_sizes: torch.Tensor, 
                batch_offsets: torch.Tensor,
                batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)

class grouped_gemm_tn(Op):
    """
    Grouped GEMM backward that calculates dB with transpose on A and no transpose on B: A^T @ B -> C
    """
    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_tn_kernel"](batch_sum, batch_count, N, K, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"grouped_gemm_tn_kernel": grouped_gemm_tn_kernel}
    
    def forward(self,
                A: torch.Tensor, 
                B: torch.Tensor, 
                batch_sizes: torch.Tensor, 
                batch_offsets: torch.Tensor,
                batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)

class grouped_gemm_tt(Op):
    """
    Grouped GEMM backward that calculates dB with transpose on A and transpose on B^T: A^T @ (B^T)^T -> C
    """
    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_tt_kernel"](batch_sum, batch_count, N, K, dtype=dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"grouped_gemm_tt_kernel": grouped_gemm_tt_kernel}
    
    def forward(self,
                A: torch.Tensor, 
                B: torch.Tensor, 
                batch_sizes: torch.Tensor, 
                batch_offsets: torch.Tensor,
                batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
