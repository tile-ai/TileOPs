from typing import Any

import torch

from workloads.device import run_device
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
        """The workload of one manifest call of ``FP8QuantPerTokenGroupFwdOp``."""
        indices = call.indices
        return cls(
            indices["batch"],
            indices["seq_len_kv"],
            indices["kv_group"],
            indices["index_dim"],
            getattr(torch, indices["T"]),
        )

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            dtype=self.in_dtype,
            device=run_device(),
        )
        return (x,)

    def ref_program(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len_kv, kv_group, index_dim)
        x = x.float()
        amax_value = torch.abs(x).amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale = amax_value / 448.0
        q = torch.clamp(x / scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
        return q, scale.squeeze(dim=-1)

    def verification(self, *inputs):
        from workloads.numerics import Custom, assert_quantized

        return Custom(assert_quantized, "FP8 codes and scales")
