import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase


class AdaLayerNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return x, scale, shift

    def ref_program(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # AdaLN: y = scale * LayerNorm(x) + shift
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = scale.float() * normed + shift.float()
        return y.to(x.dtype)
