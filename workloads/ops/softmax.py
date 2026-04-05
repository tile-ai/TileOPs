import torch

from workloads.base import WorkloadBase


class SoftmaxTest(WorkloadBase):
    """Workload definition for SoftmaxOp (2D, dim=-1)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)


class LogSoftmaxTest(WorkloadBase):
    """Workload definition for LogSoftmaxOp (2D, dim=-1)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)


class LogSumExpTest(WorkloadBase):
    """Workload definition for LogSumExpOp (2D, dim=-1)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)
