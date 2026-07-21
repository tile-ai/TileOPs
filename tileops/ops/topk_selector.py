from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.topk_selector import TopkSelectorKernel

from .op_base import Op

__all__ = ["TopkSelectorOp"]


class TopkSelectorOp(Op):

    def __init__(self,
                 topk: int,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = None
        self.seq_len = None
        self.seq_len_kv = None
        self.kv_group = None
        self.topk = topk
        self.in_dtype = None
        self.out_dtype = torch.int32
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"topk_selector_kernel": TopkSelectorKernel}

    def _get_kernel(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, seq_len, seq_len_kv, kv_group, self.topk, in_dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["topk_selector_kernel"](
                batch,
                seq_len,
                seq_len_kv,
                kv_group,
                self.topk,
                in_dtype,
                self.out_dtype,
                tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, index_score, starts, ends) -> torch.Tensor:
        if not index_score.is_cuda:
            raise ValueError("TopkSelectorOp expects CUDA inputs")
        if index_score.ndim != 4:
            raise ValueError("TopkSelectorOp expects index_score shape [B, S, S_kv, G]")
        if starts.ndim != 2 or ends.ndim != 2:
            raise ValueError("TopkSelectorOp expects starts/ends shape [B, S]")
        if not starts.is_cuda or not ends.is_cuda:
            raise ValueError("starts and ends must be CUDA tensors")
        if starts.dtype != torch.int32 or ends.dtype != torch.int32:
            raise ValueError("TopkSelectorOp expects int32 starts/ends tensors")

        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        if starts.shape != (batch, seq_len) or ends.shape != (batch, seq_len):
            raise ValueError("TopkSelectorOp starts/ends must match index_score batch/seq_len")
        if not 0 < self.topk <= seq_len_kv:
            raise ValueError(f"topk must satisfy 0 < topk <= seq_len_kv={seq_len_kv}")

        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.in_dtype = index_score.dtype
        self.kernel = self._get_kernel(
            batch, seq_len, seq_len_kv, kv_group, index_score.dtype, index_score.device.index)

        return self.kernel(index_score, starts, ends)
