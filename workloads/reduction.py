"""Workload definitions for the reduction op family."""

import torch

from workloads.workload_base import RandnTest, WorkloadBase


class SumTest(RandnTest):
    """Workload definition for SumFwdOp."""


class MeanTest(RandnTest):
    """Workload definition for MeanFwdOp."""


class AmaxTest(RandnTest):
    """Workload definition for AmaxFwdOp."""


class AminTest(RandnTest):
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


class StdTest(RandnTest):
    """Workload definition for StdFwdOp."""


class VarTest(RandnTest):
    """Workload definition for VarFwdOp."""


class VarMeanTest(RandnTest):
    """Workload definition for VarMeanFwdOp."""


class ArgmaxTest(RandnTest):
    """Workload definition for ArgmaxFwdOp."""


class ArgminTest(RandnTest):
    """Workload definition for ArgminFwdOp."""


class SoftmaxTest(RandnTest):
    """Workload definition for SoftmaxFwdOp (spec interface: shape + dtype)."""


class LogSoftmaxTest(RandnTest):
    """Workload definition for LogSoftmaxFwdOp (spec interface: shape + dtype)."""


class LogSumExpTest(RandnTest):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""


class L1NormTest(RandnTest):
    """Workload definition for L1NormFwdOp."""


class L2NormTest(RandnTest):
    """Workload definition for L2NormFwdOp."""


class InfNormTest(RandnTest):
    """Workload definition for InfNormFwdOp."""


class _LogicalTest(WorkloadBase):
    """Shared workload base for logical reduce ops (any, all, count_nonzero).

    Generates inputs with a mix of zeros and non-zeros for meaningful
    logical reduction testing. Boolean, integer, float, and complex
    dtypes are supported.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, self.dtype),)


class AnyTest(_LogicalTest):
    """Workload definition for AnyFwdOp."""


class AllTest(_LogicalTest):
    """Workload definition for AllFwdOp."""


class CountNonzeroTest(_LogicalTest):
    """Workload definition for CountNonzeroFwdOp."""


# ---------------------------------------------------------------------------
# Shared input-generation helper
# ---------------------------------------------------------------------------


def _make_logical_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor with a mix of zeros and non-zeros.

    When the first dimension is large enough (>4), the first row is forced
    to all-zero (meaningful for ``any``) and the second row to all-nonzero
    (meaningful for ``all``).
    """
    m = shape[0] if len(shape) >= 1 else 1

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
        if m > 4:
            x[0] = False
            x[1] = True
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(*shape, dtype=torch.float32, device="cuda")
        imag = torch.randn(*shape, dtype=torch.float32, device="cuda")
        x = torch.complex(real, imag).to(dtype)
        if m > 4:
            x[0] = 0 + 0j
            x[1] = 1 + 1j
    elif dtype in (torch.int32, torch.int64):
        x = torch.randint(-5, 6, shape, dtype=dtype, device="cuda")
        if m > 4:
            x[0] = 0
            x[1] = 1
    else:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        if m > 4:
            x[0] = 0.0
            x[1] = 1.0

    return x
