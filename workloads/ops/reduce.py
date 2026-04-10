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


class SumWorkload(_RandnWorkload):
    """Workload definition for SumFwdOp."""


class MeanWorkload(_RandnWorkload):
    """Workload definition for MeanFwdOp."""


class AmaxWorkload(_RandnWorkload):
    """Workload definition for AmaxFwdOp."""


class AminWorkload(_RandnWorkload):
    """Workload definition for AminFwdOp."""


class ProdWorkload(WorkloadBase):
    """Workload definition for ProdFwdOp.

    Uses small-range values (0.99..1.0) to avoid overflow in product reduction.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.rand(*self.shape, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        return (x,)


class StdWorkload(_RandnWorkload):
    """Workload definition for StdFwdOp."""


class VarWorkload(_RandnWorkload):
    """Workload definition for VarFwdOp."""


class VarMeanWorkload(_RandnWorkload):
    """Workload definition for VarMeanFwdOp."""
