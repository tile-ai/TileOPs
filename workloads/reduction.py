"""Workload definitions for the reduction op family."""

import math
from dataclasses import dataclass

import torch

from workloads.device import run_device
from workloads.numerics import Custom, Exact, compare_outputs, reference_tolerance
from workloads.workload_base import CallWorkload, WorkloadBase


class _ReductionCase(WorkloadBase):
    """Standard-normal ``x`` of *shape* through op *name* with the parameters *params*."""

    def __init__(self, shape: tuple, dtype: torch.dtype, name: str, **params):
        self.shape = shape
        self.dtype = dtype
        self._name = name
        self._params = params

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(*self.shape, dtype=self.dtype, device=run_device()),)

    def ref_program(self, x: torch.Tensor):
        return _reduction_reference(self._name, x, self._params)


class _ReductionWorkload(_ReductionCase):
    """A `_ReductionCase` checked against the long-reduction bound."""

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, scalar=inputs[0].ndim == 0)


_WELFORD = {"var": "VarFwdOp", "std": "StdFwdOp", "var_mean": "VarMeanFwdOp"}


def _near_one(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """Values in [0.99, 1): a product or product scan over thousands of them stays finite."""
    return torch.rand(*shape, dtype=dtype, device=device) * 0.01 + 0.99


@dataclass(eq=False)
class ProdWorkload(WorkloadBase):
    """Workload definition for ProdFwdOp, on inputs near 1."""

    shape: tuple
    dtype: torch.dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_near_one(self.shape, self.dtype, run_device()),)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return _reduction_reference("ProdFwdOp", x, dict(dim=-1))

    def verification(self, *inputs):
        return reduction_verification(inputs[0].dtype, product=True, scalar=inputs[0].ndim == 0)


class ReductionCall(CallWorkload):
    """One manifest call of a reduction op, with its input's shape and dtype name."""

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
            return vector_norm_verification(inputs[0].dtype)
        return Exact()


class ProdCall(ReductionCall):
    """A manifest call of ProdFwdOp, on inputs near 1."""

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_near_one(self.shape, getattr(torch, self.dtype), self.device),)


class LogicalCall(ReductionCall):
    """A logical reduction's input mixes zeros and nonzeros, with an all-zero and an
    all-nonzero leading row."""

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, getattr(torch, self.dtype), self.device),)


def _make_logical_input(shape: tuple, dtype: torch.dtype, device=None) -> torch.Tensor:
    """Zeros and nonzeros of *dtype*; with more than four rows, the first is all zero and the
    second all nonzero."""
    device = device or run_device()
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(*shape, dtype=torch.float32, device=device)
        imag = torch.randn(*shape, dtype=torch.float32, device=device)
        x = torch.complex(real, imag).to(dtype)
    elif dtype in (torch.int32, torch.int64):
        x = torch.randint(-5, 6, shape, dtype=dtype, device=device)
    else:
        x = torch.randn(*shape, dtype=dtype, device=device)
    if shape and shape[0] > 4:
        x[0] = 0
        x[1] = 1 + 1j if dtype.is_complex else 1
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
            return (_near_one(self.shape, self.dtype, run_device()),)
        return (torch.randn(*self.shape, dtype=self.dtype, device=run_device()),)

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


class ArgreduceCase(_ReductionCase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__(
            (m, n), dtype, {"argmax": "ArgmaxFwdOp", "argmin": "ArgminFwdOp"}[op_kind], dim=-1
        )


class SoftmaxCase(_ReductionCase):
    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype, "SoftmaxFwdOp", dim=dim)

    def verification(self, *inputs):
        return softmax_verification(inputs[0].dtype, logarithmic=False)


class LogSoftmaxCase(_ReductionCase):
    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype, "LogSoftmaxFwdOp", dim=dim)

    def verification(self, *inputs):
        return softmax_verification(inputs[0].dtype, logarithmic=True)


class LogSumExpCase(_ReductionCase):
    def __init__(self, shape: tuple, dtype: torch.dtype, dim: int = -1):
        super().__init__(shape, dtype, "LogSumExpFwdOp", dim=dim)


class WelfordNonAlignedCase(_ReductionCase):
    def __init__(self, shape: tuple, dtype, op_kind: str, correction: int = 1):
        super().__init__(shape, dtype, _WELFORD[op_kind], dim=-1, correction=correction)


class VarianceConstantCase(_ReductionWorkload):
    """Finite constant inputs whose variance must stay zero despite a large origin."""

    def __init__(self, shape: tuple, axes: tuple, op_kind: str, value: float):
        super().__init__(shape, torch.float32, _WELFORD[op_kind], dim=axes, correction=0)
        self.value = value

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.full(self.shape, self.value, dtype=self.dtype, device=run_device()),)


class LogicalReduceCase(_ReductionCase):
    """`_make_logical_input` rows through a logical reduction."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        name = {"any": "AnyFwdOp", "all": "AllFwdOp", "count_nonzero": "CountNonzeroFwdOp"}
        super().__init__((m, n), dtype, name[op_kind], dim=-1)

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, self.dtype),)


class ReduceCase(_ReductionWorkload):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        name = {"sum": "SumFwdOp", "mean": "MeanFwdOp", "amax": "AmaxFwdOp", "amin": "AminFwdOp"}
        super().__init__((m, n), dtype, name[op_kind], dim=-1)


class ExtremaNanCase(_ReductionWorkload):
    """Extrema inputs with both NaN signs at opposite ends of reduced dimensions."""

    def __init__(self, shape: tuple, dtype: torch.dtype, axes: tuple, kind: str):
        super().__init__(shape, dtype, {"amax": "AmaxFwdOp", "amin": "AminFwdOp"}[kind], dim=axes)
        self.axes = axes

    def gen_inputs(self) -> tuple[torch.Tensor]:
        (x,) = super().gen_inputs()
        kept = next(axis for axis in range(x.ndim) if axis not in self.axes)
        first = [0] * x.ndim
        last = [size - 1 for size in self.shape]
        last[kept] = 1
        x[tuple(first)] = float("nan")
        x[tuple(last)] = -float("nan")
        return (x,)


class WelfordCase(_ReductionWorkload):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str, correction: int = 1):
        super().__init__((m, n), dtype, _WELFORD[op_kind], dim=-1, correction=correction)


class VectorNormCase(_ReductionCase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        order = {"l1": 1, "l2": 2, "inf": float("inf")}[op_kind]
        super().__init__((m, n), dtype, "VectorNormFwdOp", dim=-1, ord=order)

    def verification(self, *inputs):
        return vector_norm_verification(inputs[0].dtype)


def reduction_tolerance(dtype: torch.dtype) -> dict[str, float]:
    """The reduction policy, exposed for algebraic property assertions."""
    return reduction_verification(dtype).tolerance({})


def vector_norm_verification(dtype):
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    return Exact(atol=tol, rtol=tol)


def softmax_verification(dtype, *, input_dtype=None, logarithmic=False):
    """FP32 probabilities need a relative bound: an absolute floor can accept all zeros.

    Widening the result must also exclude an intermediate rounding to the input dtype.
    """
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
