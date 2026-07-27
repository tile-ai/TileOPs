from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.fp8_quant import FP8QuantKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["FP8QuantOp"]


class FP8QuantOp(Op):

    def __init__(self,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.batch = None
        self.seq_len_kv = None
        self.kv_group = None
        self.index_dim = None
        self.in_dtype = None
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_quant_kernel": FP8QuantKernel}

    def _get_kernel(
        self,
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, seq_len_kv, kv_group, index_dim, in_dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fp8_quant_kernel"](
                batch, seq_len_kv, kv_group, index_dim, in_dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not input_tensor.is_cuda:
            raise ValueError("FP8QuantOp expects a CUDA input tensor")
        if input_tensor.ndim != 4:
            raise ValueError("FP8QuantOp expects input_tensor shape [B, S, G, D]")
        if input_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"FP8QuantOp does not support dtype {input_tensor.dtype}")

        batch, seq_len_kv, kv_group, index_dim = input_tensor.shape
        if min(batch, seq_len_kv, kv_group, index_dim) <= 0:
            raise ValueError("FP8QuantOp input dimensions must be positive")

        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = input_tensor.dtype
        self.kernel = self._get_kernel(
            batch, seq_len_kv, kv_group, index_dim, input_tensor.dtype, input_tensor.device.index)
        return self.kernel(input_tensor)
