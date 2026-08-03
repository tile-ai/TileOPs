from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.topk_selector import TopkSelectorKernel
from kernels.topk_selector_torch import TopkSelectorTorchKernel
from perf.formulas import topk_selector_roofline

from .op_base import Op

__all__ = ["TopkSelectorOp"]


class TopkSelectorOp(Op):
    """Top-K selector op — selects top-k indices along seq_len_kv dimension.

    Input:  index_score [batch, seq_len, seq_len_kv, kv_group]
    Output: indexes     [batch, seq_len, kv_group, topk]

    By default uses ``TopkSelectorTorchKernel`` (PyTorch-based, NPU-ready).
    Pass ``kernel_map={"topk_selector_kernel": TopkSelectorKernel}`` to use
    the TileLang radix kernel (requires a backend that supports shared
    memory + thread sync, e.g. CUDA or a future TileLang NPU backend).
    """

    def __init__(self, topk: int,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.topk = topk
        self.tune = tune
        self.out_dtype = torch.int32

        self.batch = None
        self.seq_len = None
        self.seq_len_kv = None
        self.kv_group = None
        self.in_dtype = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"topk_selector_kernel": TopkSelectorTorchKernel}

    def _get_kernel(self, batch, seq_len, seq_len_kv, kv_group, in_dtype,
                    device_index) -> Kernel:
        key = (batch, seq_len, seq_len_kv, kv_group, self.topk,
               in_dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["topk_selector_kernel"](
                batch, seq_len, seq_len_kv, kv_group, self.topk,
                in_dtype, self.out_dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, index_score, starts, ends) -> torch.Tensor:
        from utils import device_str, is_available

        if not is_available():
            raise ValueError("TopkSelectorOp expects an accelerator device")
        if index_score.ndim != 4:
            raise ValueError("index_score shape must be [B, S, S_kv, G]")
        if starts.ndim != 2 or ends.ndim != 2:
            raise ValueError("starts/ends shape must be [B, S]")
        if starts.dtype != torch.int32 or ends.dtype != torch.int32:
            raise ValueError("starts/ends must be int32")

        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        if starts.shape != (batch, seq_len) or ends.shape != (batch, seq_len):
            raise ValueError("starts/ends must match index_score [B, S]")
        if not 0 < self.topk <= seq_len_kv:
            raise ValueError(f"topk must satisfy 0 < topk <= {seq_len_kv}")

        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.in_dtype = index_score.dtype

        self.kernel = self._get_kernel(
            batch, seq_len, seq_len_kv, kv_group,
            index_score.dtype, index_score.device.index)
        return self.kernel(index_score, starts, ends)

    def eval_roofline(self) -> tuple[int, int]:
        return topk_selector_roofline(self)
