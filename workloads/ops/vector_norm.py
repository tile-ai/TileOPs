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


class L1NormTest(_RandnTest):
    """Workload definition for L1NormFwdOp."""


class L2NormTest(_RandnTest):
    """Workload definition for L2NormFwdOp."""


class InfNormTest(_RandnTest):
    """Workload definition for InfNormFwdOp."""
