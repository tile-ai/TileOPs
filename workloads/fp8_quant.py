from typing import Any

import torch

from workloads.workload_base import WorkloadBase


class FP8QuantWorkload(WorkloadBase):
    def __init__(
        self, batch: int, seq_len_kv: int, kv_group: int, index_dim: int, in_dtype: torch.dtype
    ):
        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = in_dtype

    @classmethod
    def from_call(cls, call: Any) -> "FP8QuantWorkload":
        """The workload of one manifest call of ``FP8QuantFwdOp``."""
        ix = call.ix
        return cls(
            ix["batch"], ix["seq_len_kv"], ix["kv_group"], ix["index_dim"], getattr(torch, ix["T"])
        )

    def gen_inputs(self) -> tuple[torch.Tensor]:
        input_tensor = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            dtype=self.in_dtype,
            device="cuda",
        )
        return (input_tensor,)

    def ref_program(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input_tensor: (batch, seq_len_kv, kv_group, index_dim)
        x = input_tensor.float()
        amax_value = torch.abs(x).amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale_tensor = amax_value / 448.0
        output_tensor = torch.clamp(x / scale_tensor, min=-448.0, max=448.0)
        output_tensor = output_tensor.to(torch.float8_e4m3fn)
        return scale_tensor.squeeze(dim=-1), output_tensor
