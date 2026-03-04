from typing import Dict, Optional

import torch

from tileops.kernels.grouped_gemm import grouped_gemm_kernel
from tileops.kernels.kernel import Kernel
from tileops.ops.op import Op

__all__ = ['GroupedGemmOp']


class GroupedGemmOp(Op):
    """
    Grouped GEMM with configurable transpose modes.

    Supports four layouts via (transpose_a, transpose_b):
      - (False, True)  NT: A @ B^T -> C
      - (False, False)  NN: A @ B   -> C
      - (True,  False) TN: A^T @ B  -> C
      - (True,  True)  TT: A^T @ B^T -> C
    """

    def __init__(self,
                 batch_sum: int,
                 batch_count: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.float16,
                 transpose_a: bool = False,
                 transpose_b: bool = True,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["grouped_gemm_kernel"](
            batch_sum,
            batch_count,
            n,
            k,
            dtype=dtype,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            tune=tune)

    @property
    def default_kernel_map(self) -> Dict:
        return {"grouped_gemm_kernel": grouped_gemm_kernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
