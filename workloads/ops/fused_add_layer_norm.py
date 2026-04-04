import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase


class FusedAddLayerNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, residual, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_result = (x.float() + residual.float()).to(x.dtype)
        y = F.layer_norm(
            add_result.float(),
            (self.n,),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)
        return y, add_result
