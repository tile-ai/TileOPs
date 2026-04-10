import torch

from workloads.base import WorkloadBase


class ArgmaxTest(WorkloadBase):
    """Workload definition for ArgmaxFwdOp (spec interface: shape + dtype)."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)


class ArgminTest(WorkloadBase):
    """Workload definition for ArgminFwdOp (spec interface: shape + dtype)."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)
