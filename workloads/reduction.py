"""Workload definitions for the reduction op family."""

import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, RandnWorkload, WorkloadBase


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
        x = torch.rand(*self.shape, dtype=self.dtype, device=run_device()) * 0.01 + 0.99
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().prod(dim=-1).to(x.dtype)


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


class ReductionCall(CallWorkload):
    """One manifest call of a reduction op, with the input shape and dtype a report shows."""

    def __init__(self, call, device: "torch.device | str | None" = None):
        super().__init__(call, device)
        spec = call.specs["x"]
        self.shape, self.dtype = spec.shape, spec.dtype


class ProdCall(ReductionCall):
    """A product over thousands of elements stays finite only near 1: values in [0.99, 1)."""

    def gen_inputs(self) -> tuple[torch.Tensor]:
        dtype = getattr(torch, self.dtype)
        return (torch.rand(*self.shape, dtype=dtype, device=self.device) * 0.01 + 0.99,)


class LogicalCall(ReductionCall):
    """A logical reduction's input mixes zeros and nonzeros, with an all-zero and an
    all-nonzero leading row."""

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, getattr(torch, self.dtype), self.device),)


# ---------------------------------------------------------------------------
# Shared input-generation helper
# ---------------------------------------------------------------------------


def _make_logical_input(shape: tuple, dtype: torch.dtype, device=None) -> torch.Tensor:
    """Create a tensor with a mix of zeros and non-zeros.

    When the first dimension is large enough (>4), the first row is forced
    to all-zero (meaningful for ``any``) and the second row to all-nonzero
    (meaningful for ``all``).
    """
    device = device or run_device()
    m = shape[0] if len(shape) >= 1 else 1

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
        if m > 4:
            x[0] = False
            x[1] = True
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(*shape, dtype=torch.float32, device=device)
        imag = torch.randn(*shape, dtype=torch.float32, device=device)
        x = torch.complex(real, imag).to(dtype)
        if m > 4:
            x[0] = 0 + 0j
            x[1] = 1 + 1j
    elif dtype in (torch.int32, torch.int64):
        x = torch.randint(-5, 6, shape, dtype=dtype, device=device)
        if m > 4:
            x[0] = 0
            x[1] = 1
    else:
        x = torch.randn(*shape, dtype=dtype, device=device)
        if m > 4:
            x[0] = 0.0
            x[1] = 1.0

    return x


class CumulativeWorkload(WorkloadBase):
    """Inputs for cumsum / cumprod along ``dim``, and the fp32 scan they are checked against.

    ``cumprod`` defaults to a narrow band around 1.0 so a long scan stays in
    range; pass ``use_small_range`` explicitly to override.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        op_kind: str,
        use_small_range: bool | None = None,
        dim: int = -1,
    ):
        if op_kind not in ("cumsum", "cumprod"):
            raise ValueError(f"Unknown op_kind: {op_kind}")
        self.shape = tuple(shape)
        self.dtype = dtype
        self.op_kind = op_kind
        self.dim = dim
        self.use_small_range = op_kind == "cumprod" if use_small_range is None else use_small_range

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.use_small_range:
            x = torch.rand(*self.shape, dtype=self.dtype, device=run_device()) * 0.01 + 0.99
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device=run_device())
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        scan = torch.cumsum if self.op_kind == "cumsum" else torch.cumprod
        return scan(x.float(), dim=self.dim).to(x.dtype)


class CumulativeCall(CallWorkload, CumulativeWorkload):
    """A manifest call of CumsumFwdOp or CumprodFwdOp, with the workload's inputs."""

    def __init__(self, call, op_kind: str) -> None:
        CallWorkload.__init__(self, call)
        shape, dtype = call.tensors["x"]
        CumulativeWorkload.__init__(
            self, shape, getattr(torch, dtype), op_kind, dim=call.params["dim"]
        )

    gen_inputs = CumulativeWorkload.gen_inputs
