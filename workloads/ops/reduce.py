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


class SumTest(_RandnTest):
    """Workload definition for SumFwdOp."""


class MeanTest(_RandnTest):
    """Workload definition for MeanFwdOp."""


class AmaxTest(_RandnTest):
    """Workload definition for AmaxFwdOp."""


class AminTest(_RandnTest):
    """Workload definition for AminFwdOp."""


class ProdTest(WorkloadBase):
    """Workload definition for ProdFwdOp.

    Uses small-range values (0.99..1.0) to avoid overflow in product reduction.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.rand(*self.shape, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        return (x,)


class StdTest(_RandnTest):
    """Workload definition for StdFwdOp."""


class VarTest(_RandnTest):
    """Workload definition for VarFwdOp."""


class VarMeanTest(_RandnTest):
    """Workload definition for VarMeanFwdOp."""
