"""Workload definitions for the reduction op family."""

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import CallWorkload, RandnWorkload, WorkloadBase


class SumWorkload(RandnWorkload):
    """Workload definition for SumFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


class MeanWorkload(RandnWorkload):
    """Workload definition for MeanFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


class AmaxWorkload(RandnWorkload):
    """Workload definition for AmaxFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


class AminWorkload(RandnWorkload):
    """Workload definition for AminFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


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

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, product=True)


class StdWorkload(RandnWorkload):
    """Workload definition for StdFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


class VarWorkload(RandnWorkload):
    """Workload definition for VarFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


class VarMeanWorkload(RandnWorkload):
    """Workload definition for VarMeanFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype)


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


class VectorNormWorkload(RandnWorkload):
    """Workload definition for VectorNormFwdOp."""

    def verification(self, *inputs):
        return vector_norm_verification(inputs[0].dtype)


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

    def ref_program(self, *inputs):
        p = self.call.params
        name = self.call.signature.name
        x = inputs[0]
        dim, keep = p.get("dim"), p.get("keepdim", False)
        dtype = getattr(torch, p["dtype"]) if p.get("dtype") else x.dtype
        if name in ("ArgmaxFwdOp", "ArgminFwdOp"):
            return (torch.argmax if name == "ArgmaxFwdOp" else torch.argmin)(
                x, dim=dim, keepdim=keep
            )
        if name in ("AnyFwdOp", "AllFwdOp"):
            return (torch.any if name == "AnyFwdOp" else torch.all)(x.bool(), dim=dim, keepdim=keep)
        if name == "CountNonzeroFwdOp":
            return torch.count_nonzero(x, dim=dim).to(torch.int64)
        if name in ("SoftmaxFwdOp", "LogSoftmaxFwdOp"):
            fn = torch.softmax if name == "SoftmaxFwdOp" else torch.log_softmax
            return fn(x.float(), dim=dim).to(dtype)
        if name == "LogSumExpFwdOp":
            return torch.logsumexp(x.float(), dim=dim, keepdim=keep).to(dtype)
        if name == "VectorNormFwdOp":
            return torch.linalg.vector_norm(x.float(), ord=p["ord"], dim=dim, keepdim=keep).to(
                dtype
            )
        if name in ("StdFwdOp", "VarFwdOp", "VarMeanFwdOp"):
            fn = {"StdFwdOp": torch.std, "VarFwdOp": torch.var, "VarMeanFwdOp": torch.var_mean}[
                name
            ]
            out = fn(
                x.float(),
                dim=dim,
                keepdim=keep,
                correction=1 if p["correction"] is None else p["correction"],
            )
            return tuple(v.to(dtype) for v in out) if isinstance(out, tuple) else out.to(dtype)
        fn = {
            "SumFwdOp": torch.sum,
            "MeanFwdOp": torch.mean,
            "AmaxFwdOp": torch.amax,
            "AminFwdOp": torch.amin,
            "ProdFwdOp": torch.prod,
        }[name]
        return fn(x.float(), dim=dim, keepdim=keep).to(dtype)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        name = self.call.signature.name
        if name in (
            "SumFwdOp",
            "MeanFwdOp",
            "AmaxFwdOp",
            "AminFwdOp",
            "StdFwdOp",
            "VarFwdOp",
            "VarMeanFwdOp",
            "ProdFwdOp",
        ):
            return reduction_verification(inputs[0].dtype, product=name == "ProdFwdOp")
        if name == "VectorNormFwdOp":
            return VectorNormWorkload.verification(self, *inputs)
        return Exact()


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

    def verification(self, *inputs):
        from workloads.numerics import Exact

        if self.op_kind == "cumprod":
            tol = 1e-3 if inputs[0].dtype == torch.float32 else 5e-2
        else:
            tol = (
                1e-5
                if inputs[0].dtype == torch.float32
                else 1e-2
                if inputs[0].dtype == torch.float16
                else 1.6e-2
            )
        return Exact(atol=tol, rtol=tol)


class CumulativeCall(CallWorkload, CumulativeWorkload):
    """A manifest call of CumsumFwdOp or CumprodFwdOp, with the workload's inputs."""

    def __init__(self, call, op_kind: str) -> None:
        CallWorkload.__init__(self, call)
        shape, dtype = call.tensors["x"]
        CumulativeWorkload.__init__(
            self, shape, getattr(torch, dtype), op_kind, dim=call.params["dim"]
        )

    gen_inputs = CumulativeWorkload.gen_inputs


def reduction_verification(dtype, *, product=False):
    """Long reductions use the existing reduction bound, shared by both consumers."""
    from workloads.numerics import Exact

    tol = (
        (1e-3 if dtype == torch.float32 else 5e-2)
        if product
        else (1e-4 if dtype == torch.float32 else 1e-2)
    )
    return Exact(atol=tol, rtol=tol)


class ArgreduceCase(ArgmaxWorkload):
    """Parameterized test helper for argreduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        if self.op_kind == "argmax":
            return x.argmax(dim=-1)
        elif self.op_kind == "argmin":
            return x.argmin(dim=-1)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class SoftmaxCase(SoftmaxWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


class LogSoftmaxCase(LogSoftmaxWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


class LogSumExpCase(LogSumExpWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim).to(x.dtype)

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


class WelfordNonAlignedCase(RandnWorkload):
    def __init__(self, shape: tuple, dtype, op_kind: str, correction: int = 1):
        super().__init__(shape, dtype)
        self.op_kind = op_kind
        self.correction = correction

    """Test helper for Welford ops with non-aligned N values."""

    def ref_program(self, x: torch.Tensor) -> object:
        x_f32 = x.float()
        if self.op_kind == "var":
            return x_f32.var(dim=-1, correction=self.correction).to(x.dtype)
        elif self.op_kind == "std":
            return x_f32.std(dim=-1, correction=self.correction).to(x.dtype)
        elif self.op_kind == "var_mean":
            v = x_f32.var(dim=-1, correction=self.correction).to(x.dtype)
            m = x_f32.mean(dim=-1).to(x.dtype)
            return (v, m)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class LogicalReduceCase(AnyWorkload):
    """Parameterized test helper for logical reduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "any":
            return x.bool().any(dim=-1)
        elif self.op_kind == "all":
            return x.bool().all(dim=-1)
        elif self.op_kind == "count_nonzero":
            return torch.count_nonzero(x, dim=-1).to(torch.int64)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class ReduceCase(SumWorkload):
    """Parameterized test helper for simple reduce ops (sum/mean/amax/amin)."""

    def __init__(
        self,
        m: int,
        n: int,
        dtype: torch.dtype,
        op_kind: str,
    ):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "sum":
            return x_f32.sum(dim=-1).to(x.dtype)
        elif self.op_kind == "mean":
            return x_f32.mean(dim=-1).to(x.dtype)
        elif self.op_kind == "amax":
            return x_f32.amax(dim=-1).to(x.dtype)
        elif self.op_kind == "amin":
            return x_f32.amin(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class WelfordCase(StdWorkload):
    """Test helper for Welford-based ops (std, var, var_mean)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str, correction: int = 1):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind
        self.correction = correction

    def ref_program(self, x: torch.Tensor) -> object:
        x_f32 = x.float()
        if self.op_kind == "var":
            return x_f32.var(dim=-1, correction=self.correction).to(x.dtype)
        elif self.op_kind == "std":
            return x_f32.std(dim=-1, correction=self.correction).to(x.dtype)
        elif self.op_kind == "var_mean":
            v = x_f32.var(dim=-1, correction=self.correction).to(x.dtype)
            m = x_f32.mean(dim=-1).to(x.dtype)
            return (v, m)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class VectorNormCase(VectorNormWorkload):
    """Parameterized test helper for vector norm ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for reference, then cast back to input dtype
        ord_val = {"l1": 1, "l2": 2, "inf": float("inf")}[self.op_kind]
        ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=-1)
        return ref.to(self.dtype)


def reduction_tolerance(dtype: torch.dtype) -> dict[str, float]:
    """The reduction policy, exposed for algebraic property assertions."""
    return reduction_verification(dtype).tolerance({})


def vector_norm_verification(dtype):
    from workloads.numerics import Exact

    tol = 1e-5 if dtype == torch.float32 else 1e-2
    return Exact(atol=tol, rtol=tol)
