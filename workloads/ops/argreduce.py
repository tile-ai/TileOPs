import torch

from workloads.base import WorkloadBase


class _RandnWorkload(WorkloadBase):
    """Base for workloads that generate inputs via torch.randn."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)


class ArgmaxWorkload(_RandnWorkload):
    """Workload definition for ArgmaxFwdOp."""


class ArgminWorkload(_RandnWorkload):
    """Workload definition for ArgminFwdOp."""


# Backward-compatible aliases for existing consumers
ArgmaxTest = ArgmaxWorkload
ArgminTest = ArgminWorkload
