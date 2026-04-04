import torch

from workloads.base import WorkloadBase


class FusedAddRmsNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, residual, weight

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_result = (x.float() + residual.float()).to(x.dtype)
        add_f32 = add_result.float()
        rms = torch.sqrt(add_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = ((add_f32 / rms) * weight.float()).to(x.dtype)
        return y, add_result
