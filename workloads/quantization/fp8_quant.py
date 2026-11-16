from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase


class FP8QuantWorkload(WorkloadBase):
    """Group-wise FP8 quantization of a ``shape`` tensor, groups of ``group_size`` columns."""

    def __init__(
        self,
        shape: tuple[int, ...],
        group_size: int,
        in_dtype: torch.dtype,
        eps: float = 1e-10,
        out_q: bool = False,
    ):
        self.shape = tuple(shape)
        self.group_size = group_size
        self.in_dtype = in_dtype
        self.eps = eps
        self.out_q = out_q

    @classmethod
    def from_call(cls, call: Any) -> "FP8QuantWorkload":
        """The workload of one manifest call of ``FP8QuantPerTokenGroupFwdOp``."""
        shape, dtype = call.tensors["x"]
        params = call.params
        return cls(
            shape,
            params["group_size"],
            getattr(torch, dtype),
            params["eps"],
            call.specs["out_q"] is not None,
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape, dtype=self.in_dtype, device=run_device())
        if not self.out_q:
            return (x,)
        return x, torch.empty(self.shape, dtype=torch.float8_e4m3fn, device=run_device())

    def ref_program(
        self, x: torch.Tensor, out_q: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = x.float().unflatten(-1, (-1, self.group_size))
        amax_value = torch.abs(groups).amax(dim=-1, keepdim=True).clamp(min=self.eps)
        scale = amax_value / 448.0
        q = torch.clamp(groups / scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
        return q.flatten(-2), scale.squeeze(dim=-1)

    def verification(self, *inputs):
        from workloads.numerics import Custom, assert_quantized

        return Custom(assert_quantized, "FP8 codes and scales")
