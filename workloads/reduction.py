"""Workload definitions for the reduction op family."""

import math

import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, RandnWorkload, WorkloadBase


class SumWorkload(RandnWorkload):
    """Workload definition for SumFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


class MeanWorkload(RandnWorkload):
    """Workload definition for MeanFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


class AmaxWorkload(RandnWorkload):
    """Workload definition for AmaxFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


class AminWorkload(RandnWorkload):
    """Workload definition for AminFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


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
        return _reduction_reference("ProdFwdOp", x, dict(dim=-1))

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, product=True, scalar=inputs[0].ndim == 0)


class StdWorkload(RandnWorkload):
    """Workload definition for StdFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


class VarWorkload(RandnWorkload):
    """Workload definition for VarFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


class VarMeanWorkload(RandnWorkload):
    """Workload definition for VarMeanFwdOp."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


class ArgmaxWorkload(RandnWorkload):
    """Workload definition for ArgmaxFwdOp."""


class ArgminWorkload(RandnWorkload):
    """Workload definition for ArgminFwdOp."""


class SoftmaxWorkload(RandnWorkload):
    """Workload definition for SoftmaxFwdOp (spec interface: shape + dtype)."""

    def verification(self, *inputs):
        return softmax_verification(inputs[0].dtype, logarithmic=False)


class LogSoftmaxWorkload(RandnWorkload):
    """Workload definition for LogSoftmaxFwdOp (spec interface: shape + dtype)."""

    def verification(self, *inputs):
        return softmax_verification(inputs[0].dtype, logarithmic=True)


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
        spec = call.specs[next(iter(call.signature.inputs))]
        self.shape, self.dtype = spec.shape, spec.dtype

    def ref_program(self, *inputs):
        return _reduction_reference(self.call.signature.name, inputs[0], self.call.params)

    def torch_program(self, *inputs):
        return _reduction_reference(
            self.call.signature.name, inputs[0], self.call.params, native=True
        )

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
            return reduction_verification(
                inputs[0].dtype, product=name == "ProdFwdOp", scalar=inputs[0].ndim == 0
            )
        if name in ("SoftmaxFwdOp", "LogSoftmaxFwdOp"):
            dtype = self.call.params.get("dtype")
            dtype = getattr(torch, dtype) if dtype else inputs[0].dtype
            return softmax_verification(
                dtype, input_dtype=inputs[0].dtype, logarithmic=name == "LogSoftmaxFwdOp"
            )
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
        out_dtype: torch.dtype | None = None,
    ):
        if op_kind not in ("cumsum", "cumprod"):
            raise ValueError(f"Unknown op_kind: {op_kind}")
        self.shape = tuple(shape)
        self.dtype = dtype
        self.op_kind = op_kind
        self.dim = dim
        self.out_dtype = out_dtype
        self.use_small_range = op_kind == "cumprod" if use_small_range is None else use_small_range

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.use_small_range:
            x = torch.rand(*self.shape, dtype=self.dtype, device=run_device()) * 0.01 + 0.99
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device=run_device())
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        scan = torch.cumsum if self.op_kind == "cumsum" else torch.cumprod
        return scan(x.float(), dim=self.dim).to(self.out_dtype or x.dtype)

    def torch_program(self, x: torch.Tensor) -> torch.Tensor:
        scan = torch.cumsum if self.op_kind == "cumsum" else torch.cumprod
        return scan(x, dim=self.dim, dtype=self.out_dtype)

    def verification(self, *inputs):
        x = inputs[0]
        if self.op_kind == "cumprod":
            return reduction_verification(x.dtype, product=True)
        scale = (
            x.shape[self.dim] ** 0.5 * x.float().nan_to_num(0, 0, 0).square().mean().sqrt().item()
        )
        return reduction_verification(x.dtype, scale=scale)


class CumprodExceptionalWorkload(CumulativeWorkload):
    """Cumprod rows whose running product meets every exceptional value along the row.

    Row ``r % 5``: 0 a random walk of the log of the product; 1 that walk with a zero, then
    a NaN; 2 the walk with an inf, then a zero; 3 factors of +-1 and +-2 that overflow in
    the first half of the row; 4 factors of +-1 and +-0.5 that underflow there. Signs are
    random. Powers of two multiply exactly, and with every factor of a row on one side of 1
    a partial product over- or underflows only where the running product does, so the
    expected output does not depend on how a scan associates the products.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(shape, dtype, "cumprod", out_dtype=out_dtype)

    def gen_inputs(self) -> tuple[torch.Tensor]:
        *lead, n = self.shape
        m = math.prod(lead)
        device = run_device()
        out_dtype = self.out_dtype or self.dtype
        # The walk ends with its log spread a few units inside the output's range; the
        # powers of two take the product's exponent twice past the output's largest, and
        # past its smallest subnormal.
        spread = 3.0 if out_dtype == torch.float16 else 10.0
        steps = 2 * math.ceil(math.log2(torch.finfo(out_dtype).max))
        sign = torch.where(torch.rand(m, n, device=device) < 0.5, -1.0, 1.0)
        x = sign * torch.exp(spread / math.sqrt(n) * torch.randn(m, n, device=device))
        step = torch.zeros(n, dtype=torch.bool, device=device)
        step[torch.arange(steps, device=device) * (n // 2) // steps] = True
        kind = torch.arange(m, device=device) % 5
        x[kind == 1, n // 3] = 0.0
        x[kind == 1, 2 * n // 3] = float("nan")
        x[kind == 2, n // 3] = -float("inf")
        x[kind == 2, 2 * n // 3] = 0.0
        x[kind == 3] = (sign * torch.where(step, 2.0, 1.0))[kind == 3]
        x[kind == 4] = (sign * torch.where(step, 0.5, 1.0))[kind == 4]
        return (x.to(self.dtype).view(self.shape),)

    def verification(self, *inputs):
        """The cumprod tolerance, and the zeros, signs and non-finite values exactly."""
        from workloads.numerics import Custom, compare_outputs

        close = super().verification(*inputs)

        def validator(got: torch.Tensor, expected: torch.Tensor) -> None:
            compare_outputs(got, expected, close)
            wrong_zero = (got == 0) != (expected == 0)
            assert not wrong_zero.any(), f"{int(wrong_zero.sum())} zeros differ"
            wrong_sign = (got.signbit() != expected.signbit()) & ~expected.isnan()
            assert not wrong_sign.any(), f"{int(wrong_sign.sum())} signs differ"

        return Custom(validator, "the cumprod tolerance, with exact zeros and signs")


class CumulativeCall(CallWorkload, CumulativeWorkload):
    """A manifest call of CumsumFwdOp or CumprodFwdOp, with the workload's inputs."""

    def __init__(self, call, op_kind: str) -> None:
        CallWorkload.__init__(self, call)
        shape, dtype = call.tensors["input"]
        out_dtype = call.params.get("dtype")
        CumulativeWorkload.__init__(
            self,
            shape,
            getattr(torch, dtype),
            op_kind,
            dim=call.params["dim"],
            out_dtype=getattr(torch, out_dtype) if out_dtype else None,
        )

    gen_inputs = CumulativeWorkload.gen_inputs


def reduction_verification(dtype, *, product=False, scalar=False, scale=1.0):
    """Long reductions use the existing reduction bound, shared by both consumers.

    ``scale`` is the magnitude of the running sum: a partial sum near zero
    cancels terms of that size and keeps their rounding error, so the absolute
    tolerance is the relative one at that scale, and a normalized-error bound
    rejects an output lost to that cancellation.
    """
    from workloads.numerics import Exact

    # Scalar closed forms and integer/boolean outputs have no rounding budget.
    if scalar or not (dtype.is_floating_point or dtype.is_complex):
        return Exact(atol=0, rtol=0)
    tol = (
        (1e-3 if dtype == torch.float32 else 5e-2)
        if product
        else (1e-4 if dtype == torch.float32 else 1e-2)
    )
    if scale > 1.0:
        return Exact(atol=tol * scale, rtol=tol, normalized=1e-3)
    return Exact(atol=tol, rtol=tol)


class ArgreduceCase(ArgmaxWorkload):
    """Parameterized test helper for argreduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        return _reduction_reference(
            {"argmax": "ArgmaxFwdOp", "argmin": "ArgminFwdOp"}[self.op_kind],
            inputs[0],
            dict(dim=-1),
        )


class SoftmaxCase(SoftmaxWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _reduction_reference("SoftmaxFwdOp", x, dict(dim=self.dim))

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


class LogSoftmaxCase(LogSoftmaxWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _reduction_reference("LogSoftmaxFwdOp", x, dict(dim=self.dim))

    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype)
        self.dim = dim


class LogSumExpCase(LogSumExpWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _reduction_reference("LogSumExpFwdOp", x, dict(dim=self.dim))

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
        return _reduction_reference(
            {"var": "VarFwdOp", "std": "StdFwdOp", "var_mean": "VarMeanFwdOp"}[self.op_kind],
            x,
            dict(dim=-1, correction=self.correction),
        )


class VarianceConstantCase(StdWorkload):
    """Finite constant inputs whose variance must stay zero despite a large origin."""

    def __init__(self, shape: tuple, axes: tuple, op_kind: str, value: float):
        super().__init__(shape, torch.float32)
        self.axes = axes
        self.op_kind = op_kind
        self.value = value

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.full(self.shape, self.value, dtype=self.dtype, device=run_device()),)

    def ref_program(self, x: torch.Tensor) -> object:
        return _reduction_reference(
            {"var": "VarFwdOp", "std": "StdFwdOp", "var_mean": "VarMeanFwdOp"}[self.op_kind],
            x,
            dict(dim=self.axes, correction=0),
        )


class LogicalReduceCase(AnyWorkload):
    """Parameterized test helper for logical reduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _reduction_reference(
            {"any": "AnyFwdOp", "all": "AllFwdOp", "count_nonzero": "CountNonzeroFwdOp"}[
                self.op_kind
            ],
            x,
            dict(dim=-1),
        )


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
        return _reduction_reference(
            {"sum": "SumFwdOp", "mean": "MeanFwdOp", "amax": "AmaxFwdOp", "amin": "AminFwdOp"}[
                self.op_kind
            ],
            x,
            dict(dim=-1),
        )


class ExtremaNanCase(SumWorkload):
    """Extrema inputs with both NaN signs at opposite ends of reduced dimensions."""

    def __init__(self, shape: tuple, dtype: torch.dtype, axes: tuple, kind: str):
        super().__init__(shape, dtype)
        self.axes = axes
        self.kind = kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        (x,) = super().gen_inputs()
        kept = next(axis for axis in range(x.ndim) if axis not in self.axes)
        first = [0] * x.ndim
        last = [size - 1 for size in self.shape]
        last[kept] = 1
        x[tuple(first)] = float("nan")
        x[tuple(last)] = -float("nan")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _reduction_reference(
            {"amax": "AmaxFwdOp", "amin": "AminFwdOp"}[self.kind], x, dict(dim=self.axes)
        )


class WelfordCase(StdWorkload):
    """Test helper for Welford-based ops (std, var, var_mean)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str, correction: int = 1):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind
        self.correction = correction

    def ref_program(self, x: torch.Tensor) -> object:
        return _reduction_reference(
            {"var": "VarFwdOp", "std": "StdFwdOp", "var_mean": "VarMeanFwdOp"}[self.op_kind],
            x,
            dict(dim=-1, correction=self.correction),
        )


class VectorNormCase(VectorNormWorkload):
    """Parameterized test helper for vector norm ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for reference, then cast back to input dtype
        return _reduction_reference(
            "VectorNormFwdOp",
            x,
            dict(dim=-1, ord={"l1": 1, "l2": 2, "inf": float("inf")}[self.op_kind]),
        )


def reduction_tolerance(dtype: torch.dtype) -> dict[str, float]:
    """The reduction policy, exposed for algebraic property assertions."""
    return reduction_verification(dtype).tolerance({})


def vector_norm_verification(dtype):
    from workloads.numerics import Exact

    tol = 1e-5 if dtype == torch.float32 else 1e-2
    return Exact(atol=tol, rtol=tol)


def softmax_verification(dtype, *, input_dtype=None, logarithmic=False):
    """FP32 probabilities need a relative bound: an absolute floor can accept all zeros.

    Widening the result must also exclude an intermediate rounding to the input dtype.
    """
    from workloads.numerics import Exact, reference_tolerance

    if dtype == torch.float32 and (not logarithmic or input_dtype not in (None, dtype)):
        return Exact(atol=0, rtol=1e-4)
    return Exact(**reference_tolerance(dtype))


def _reduction_reference(name, x, p, native=False):
    """One reduction oracle; adapters supply parameters, never another formula.

    ``native`` runs the same torch call on ``x`` at its own dtype, passing a ``dtype``
    parameter through: the op as torch runs it, the benchmark baseline.
    """
    dim, keep = p.get("dim"), p.get("keepdim", False)
    dtype = getattr(torch, p["dtype"]) if p.get("dtype") else x.dtype
    kw = {"dtype": dtype} if native and p.get("dtype") else {}
    if name in ("ArgmaxFwdOp", "ArgminFwdOp"):
        return (torch.argmax if name == "ArgmaxFwdOp" else torch.argmin)(x, dim=dim, keepdim=keep)
    if name in ("AnyFwdOp", "AllFwdOp"):
        return (torch.any if name == "AnyFwdOp" else torch.all)(x.bool(), dim=dim, keepdim=keep)
    if name == "CountNonzeroFwdOp":
        return torch.count_nonzero(x, dim=dim).to(torch.int64)
    wide = x if native else x.float()
    if name in ("SoftmaxFwdOp", "LogSoftmaxFwdOp"):
        fn = torch.softmax if name == "SoftmaxFwdOp" else torch.log_softmax
        return fn(wide, dim=dim, **kw).to(dtype)
    if name == "LogSumExpFwdOp":
        return torch.logsumexp(wide, dim=dim, keepdim=keep).to(dtype)
    if name == "VectorNormFwdOp":
        return torch.linalg.vector_norm(wide, ord=p["ord"], dim=dim, keepdim=keep, **kw).to(dtype)
    if name in ("StdFwdOp", "VarFwdOp", "VarMeanFwdOp"):
        fn = {"StdFwdOp": torch.std, "VarFwdOp": torch.var, "VarMeanFwdOp": torch.var_mean}[name]
        out = fn(
            wide,
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
    return fn(wide, dim=dim, keepdim=keep, **kw).to(dtype)
