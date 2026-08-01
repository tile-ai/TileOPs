"""Workload definitions for the reduction op family."""

import torch

from workloads.workload_base import RandnWorkload, WorkloadBase


class SumWorkload(RandnWorkload):
    """Workload definition for SumFwdOp."""


class MeanWorkload(RandnWorkload):
    """Workload definition for MeanFwdOp."""


class AmaxWorkload(RandnWorkload):
    """Workload definition for AmaxFwdOp."""


class AminWorkload(RandnWorkload):
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


class StdWorkload(RandnWorkload):
    """Workload definition for StdFwdOp."""


class VarWorkload(RandnWorkload):
    """Workload definition for VarFwdOp."""


class VarMeanWorkload(RandnWorkload):
    """Workload definition for VarMeanFwdOp."""


class ArgmaxWorkload(RandnWorkload):
    """Workload definition for ArgmaxFwdOp."""


class ArgminWorkload(RandnWorkload):
    """Workload definition for ArgminFwdOp."""


class SoftmaxWorkload(RandnWorkload):
    """Workload definition for SoftmaxFwdOp (spec interface: shape + dtype)."""


class LogSoftmaxWorkload(RandnWorkload):
    """Workload definition for LogSoftmaxFwdOp (spec interface: shape + dtype)."""


class LogSumExpWorkload(RandnWorkload):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""


class L1NormWorkload(RandnWorkload):
    """Workload definition for L1NormFwdOp."""


class L2NormWorkload(RandnWorkload):
    """Workload definition for L2NormFwdOp."""


class InfNormWorkload(RandnWorkload):
    """Workload definition for InfNormFwdOp."""


class _LogicalWorkload(WorkloadBase):
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


class AnyWorkload(_LogicalWorkload):
    """Workload definition for AnyFwdOp."""


class AllWorkload(_LogicalWorkload):
    """Workload definition for AllFwdOp."""


class CountNonzeroWorkload(_LogicalWorkload):
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


class CumulativeWorkload(WorkloadBase):
    """Inputs for cumsum / cumprod over the last dimension.

    ``cumprod`` defaults to a narrow band around 1.0 so a long scan stays in
    range; pass ``use_small_range`` explicitly to override.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        op_kind: str,
        use_small_range: bool | None = None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.op_kind = op_kind
        self.use_small_range = (
            op_kind == "cumprod" if use_small_range is None else use_small_range
        )

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.use_small_range:
            x = torch.rand(*self.shape, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)
