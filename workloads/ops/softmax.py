import torch

from workloads.base import WorkloadBase


class LogSoftmaxTest(WorkloadBase):
    """Workload definition for spec-interface LogSoftmaxOp."""

    def __init__(self, shape: tuple[int, ...], dim: int, dtype: torch.dtype):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

class LogSumExpTest(WorkloadBase):
    """Workload definition for spec-interface LogSumExpOp."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        dtype: torch.dtype,
        keepdim: bool = False,
    ):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype
        self.keepdim = keepdim

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

class SoftmaxTest(WorkloadBase):
    """Workload definition for spec-interface SoftmaxOp."""

    def __init__(self, shape: tuple[int, ...], dim: int, dtype: torch.dtype):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)
