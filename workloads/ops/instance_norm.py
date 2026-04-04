import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase


class InstanceNormTest(WorkloadBase):

    def __init__(self, n: int, c: int, spatial: tuple,
                 dtype: torch.dtype, eps: float = 1e-5):
        self.n = n
        self.c = c
        self.spatial = spatial
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.n, self.c, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.c, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.c, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(
            x.float(),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)
