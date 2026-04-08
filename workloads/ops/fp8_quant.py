from typing import Tuple

import torch

from workloads.base import WorkloadBase


class FP8QuantTest(WorkloadBase):

    def __init__(self, batch: int, seq_len_kv: int, kv_group: int, index_dim: int,
                 in_dtype: torch.dtype):
        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = in_dtype

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        input_tensor = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            dtype=self.in_dtype,
            device="cuda")
        return (input_tensor,)
