import torch

from workloads.base import WorkloadBase


class _RandnTest(WorkloadBase):
    """Base for workloads that generate inputs via torch.randn."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)


class ArgmaxTest(_RandnTest):
    """Workload definition for ArgmaxFwdOp."""


class ArgminTest(_RandnTest):
    """Workload definition for ArgminFwdOp."""
