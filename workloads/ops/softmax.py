import torch

from workloads.base import WorkloadBase


class SoftmaxTest(WorkloadBase):
    """Workload definition for SoftmaxOp (spec interface: shape + dtype)."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)


class LogSoftmaxTest(WorkloadBase):
    """Workload definition for LogSoftmaxOp (spec interface: shape + dtype)."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)


class LogSumExpTest(WorkloadBase):
    """Workload definition for LogSumExpOp (spec interface: shape + dtype)."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)
