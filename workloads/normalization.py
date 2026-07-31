"""Workload definitions for the normalization op family."""

import torch

from workloads.workload_base import WorkloadBase


class RMSNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, weight


class LayerNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, weight, bias


class FusedAddRMSNormTest(WorkloadBase):

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


class AdaLayerNormZeroTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        gate = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return x, scale, shift, gate


class GroupNormTest(WorkloadBase):

    def __init__(self, n: int, c: int, spatial: tuple, g: int,
                 dtype: torch.dtype, eps: float = 1e-5):
        self.n = n
        self.c = c
        self.spatial = spatial
        self.g = g
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.n, self.c, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.c, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.c, dtype=self.dtype, device="cuda")
        return x, weight, bias


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


def _make_tensors(N, C, spatial, dtype, device="cuda"):
    shape = (N, C, *spatial)
    x = torch.randn(*shape, device=device, dtype=dtype)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)
    running_mean = torch.zeros(C, device=device, dtype=torch.float32)
    running_var = torch.ones(C, device=device, dtype=torch.float32)
    return x, weight, bias, running_mean, running_var

class BatchNormBwdTest(WorkloadBase):

    def __init__(self, N, C, spatial, dtype):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, weight, bias, running_mean, running_var = _make_tensors(
            self.N, self.C, self.spatial, self.dtype)
        grad_out = torch.randn_like(x)
        # Need mean/rstd from a forward pass.
        x32 = x.float()
        # Compute mean and rstd via native batch norm internals.
        C = self.C
        L = x32.numel() // C
        x_cl = x32.permute(1, 0, *range(2, x32.ndim)).reshape(C, L).contiguous()
        mean = x_cl.mean(dim=1)
        var = x_cl.var(dim=1, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + 1e-5)
        return grad_out, x, weight, mean, rstd

class BatchNormFwdTest(WorkloadBase):

    def __init__(self, N, C, spatial, dtype, training):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.training = training

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return _make_tensors(self.N, self.C, self.spatial, self.dtype)
