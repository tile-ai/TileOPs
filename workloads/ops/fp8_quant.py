from typing import Tuple

import torch

from workloads.base import WorkloadBase


class Fp8QuantTest(WorkloadBase):

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

    def ref_program(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_tensor: (batch, seq_len_kv, kv_group, index_dim)
        amax_value = torch.abs(input_tensor).amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale_tensor = amax_value / 448.0
        output_tensor = torch.clamp(input_tensor / scale_tensor, min=-448.0, max=448.0)
        output_tensor = output_tensor.to(torch.float8_e4m3fn)
        return scale_tensor.squeeze(dim=-1), output_tensor
