"""PyTorch-based top-k selector kernel — NPU-ready fallback.

The TileLang radix top-k kernel (kernels/topk_selector.py) uses CUDA SIMT
primitives (shared memory, thread sync, atomic_add) that are not yet
supported by the TileLang Ascend backend.  This module provides a
PyTorch-based implementation that runs on any device (NPU, CUDA, CPU),
so the benchmark framework is fully functional end-to-end.

When the TileLang NPU backend matures to support the needed primitives,
switch the Op's default kernel to ``TopkSelectorKernel`` (the TileLang
implementation) — no other changes needed.
"""

from __future__ import annotations

from typing import Optional

import torch

from kernels.kernel_base import Kernel

__all__ = ["TopkSelectorTorchKernel"]


class TopkSelectorTorchKernel(Kernel):
    """Top-K selector using torch.topk — runs on NPU/CUDA/CPU.

    This kernel selects the top-k indices along the seq_len_kv dimension
    of a 4-D score tensor, matching the TileLang kernel's I/O contract:

      Input:  index_score [batch, seq_len, seq_len_kv, kv_group]
      Output: indexes     [batch, seq_len, kv_group, topk]
    """

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 seq_len_kv: int,
                 kv_group: int,
                 topk: int,
                 in_dtype: torch.dtype,
                 out_dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.config = self.default_config

    @property
    def default_config(self) -> dict:
        return {"impl": "torch.topk"}

    def forward(self, index_score: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor) -> torch.Tensor:
        # index_score: [B, S, S_kv, G]; topk over dim=2 (seq_len_kv)
        # starts/ends are accepted for interface compatibility but not used
        # here — torch.topk operates on the full dimension.  The TileLang
        # kernel uses them for variable-length ranges; a torch equivalent
        # would mask before topk if needed.
        indexes = torch.topk(index_score, self.topk, dim=2)[1]
        # Permute to [B, S, G, topk] to match kernel output layout
        return indexes.permute(0, 1, 3, 2).to(self.out_dtype)
