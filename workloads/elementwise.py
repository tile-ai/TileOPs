"""Workload definitions for elementwise op workloads with custom generators."""

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


class ReluWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n_total, dtype=self.dtype, device=run_device())
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.float()).to(x.dtype)


class FusedGatedBenchCase:
    """Minimal workload for fused gated ops."""

    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = M
        self.N = N
        self.n_total = M * N
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.M, 2 * self.N, device=run_device(), dtype=self.dtype),)

    def verification(self, *inputs):
        return fused_gated_verification(inputs[0].dtype)


class AddBroadcastWorkload(WorkloadBase):
    def __init__(self, a_shape: tuple, b_shape: tuple, dtype: torch.dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.a_shape, dtype=self.dtype, device=run_device())
        b = torch.randn(self.b_shape, dtype=self.dtype, device=run_device())
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


class PowPositiveWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device=run_device()) + 0.5
        b = torch.rand(self.n_total, dtype=self.dtype, device=run_device()) * 2.0
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.pow(a.float(), b.float()).to(a.dtype)


class BitwiseNotWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device=run_device()) > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, (self.n_total,), device=run_device(), dtype=self.dtype)
        else:
            x = torch.randint(-128, 128, (self.n_total,), device=run_device(), dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


class AddCompileWorkload(WorkloadBase):
    def __init__(self, a_shape, b_shape, dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self):
        a = torch.randn(self.a_shape, dtype=self.dtype, device=run_device())
        b = torch.randn(self.b_shape, dtype=self.dtype, device=run_device())
        return a, b

    def ref_program(self, a, b):
        return (a.float() + b.float()).to(a.dtype)


class EqCompileWorkload(WorkloadBase):
    def __init__(self, a_shape, b_shape, dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self):
        a = torch.randn(self.a_shape, dtype=self.dtype, device=run_device())
        b = a.clone()
        mask = torch.rand_like(a, dtype=torch.float32) > 0.5
        b[mask] = torch.randn_like(b[mask])
        return a, b

    def ref_program(self, a, b):
        return a == b


class SiluAndMulCompileWorkload(WorkloadBase):
    def __init__(self, M, N, dtype):
        self.M = M
        self.N = N
        self.dtype = dtype

    def gen_inputs(self):
        x = torch.randn(self.M, 2 * self.N, dtype=self.dtype, device=run_device())
        return (x,)

    def ref_program(self, x):
        gate = x[:, : self.N].float()
        value = x[:, self.N :].float()
        return (torch.nn.functional.silu(gate) * value).to(x.dtype)

    def verification(self, *inputs):
        return fused_gated_verification(inputs[0].dtype)


class LogicalNotWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device=run_device()) > 0.5
            return (x,)

        if self.dtype == torch.uint8:
            x = torch.randint(0, 8, (self.n_total,), device=run_device(), dtype=self.dtype)
        elif not (self.dtype.is_floating_point or self.dtype.is_complex) and self.dtype.is_signed:
            x = torch.randint(-4, 4, (self.n_total,), device=run_device(), dtype=self.dtype)
        else:
            x = torch.randn(self.n_total, device=run_device(), dtype=self.dtype)

        mask = torch.rand(self.n_total, device=run_device()) > 0.5
        x[mask] = 0
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x)


class BitwiseWorkload(WorkloadBase):
    def __init__(self, n_total: int):
        self.n_total = n_total
        self.dtype = torch.int32

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device=run_device())
        b = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device=run_device())
        return a, b


class LogicalWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device=run_device()) > 0
        b = torch.randn(self.n_total, dtype=self.dtype, device=run_device()) > 0
        a = a.to(self.dtype)
        b = b.to(self.dtype)
        return a, b


class SpecialWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device=run_device(), dtype=self.dtype)
        quarter = self.n_total // 4
        x[:quarter] = float("nan")
        x[quarter : 2 * quarter] = float("inf")
        x[2 * quarter : 3 * quarter] = float("-inf")
        return (x,)


class RandnFlatWorkload(WorkloadBase):
    """One ``randn`` vector of ``n_total`` elements.

    ``gen_fn`` lets a caller substitute a domain-restricted draw (positive-only,
    NaN-seeded, ...) without another class.
    """

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        return (torch.randn(self.n_total, device=run_device(), dtype=self.dtype),)


class RandnPairWorkload(WorkloadBase):
    """Two same-shape ``randn`` vectors — the default binary-op input."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device=run_device())
        b = torch.randn(self.n_total, dtype=self.dtype, device=run_device())
        return a, b


class PositivePairWorkload(WorkloadBase):
    """Two same-shape vectors in ``[0.1, 1.1)`` — for ops undefined at or below 0."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device=run_device()) + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device=run_device()) + 0.1
        return a, b


class GatedRandnWorkload(WorkloadBase):
    """One ``(m, 2 * n)`` tensor — gate and value halves for a fused gated op."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.m, 2 * self.n, dtype=self.dtype, device=run_device()),)


# One manifest call of an elementwise op (docs/design/manifest.md § Workloads).


def _draw_normal(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).bool()
    if not dtype.is_floating_point:
        info = torch.iinfo(dtype)
        return torch.randint(
            max(info.min, -1000), min(info.max, 1000) + 1, shape, device=device, dtype=dtype
        )
    return torch.randn(shape, device=device, dtype=dtype)


def _draw_positive(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    # For ops undefined at or below 0: log, sqrt, reciprocal, division.
    if not dtype.is_floating_point:
        return torch.randint(1, 100, shape, device=device, dtype=dtype)
    return torch.rand(shape, device=device, dtype=dtype) + 0.5


def _draw_sparse(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    # Half zeros, so a logical op sees both truth values.
    x = _draw_normal(shape, dtype, device)
    return x.masked_fill(torch.rand(shape, device=device) < 0.5, 0)


def _draw_special(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    # A quarter each of NaN, +Inf, -Inf and finite values, for the isnan family.
    x = _draw_normal(shape, dtype, device)
    if dtype.is_floating_point:
        flat = x.view(-1)
        quarter = flat.numel() // 4
        flat[:quarter] = float("nan")
        flat[quarter : 2 * quarter] = float("inf")
        flat[2 * quarter : 3 * quarter] = float("-inf")
    return x


_DOMAINS = {
    **dict.fromkeys(
        (
            "LogFwdOp",
            "SqrtFwdOp",
            "RsqrtFwdOp",
            "Log1pFwdOp",
            "ReciprocalFwdOp",
            "DivFwdOp",
            "RemainderFwdOp",
            "PowFwdOp",
            "FloorDivideFwdOp",
        ),
        _draw_positive,
    ),
    **dict.fromkeys(("LogicalNotFwdOp", "LogicalAndFwdOp", "LogicalOrFwdOp"), _draw_sparse),
    **dict.fromkeys(("IsnanFwdOp", "IsinfFwdOp", "IsfiniteFwdOp"), _draw_special),
}


def alibi_reference(seq_len: int, num_heads: int, dtype: torch.dtype, device=None) -> torch.Tensor:
    """Full ALiBi bias: (num_heads, seq_len, seq_len), bias[h,i,j] = -slope_h * |i-j|."""
    device = device or run_device()
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device=device, dtype=torch.float32) / num_heads,
    )
    return (-slopes[:, None, None] * dist[None, :, :]).to(dtype)


def sinusoidal_reference(
    seq_len: int, d_model: int, dtype: torch.dtype, device=None
) -> torch.Tensor:
    """Sinusoidal encoding: (seq_len, d_model), sin on even columns, cos on odd ones."""
    device = device or run_device()
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    angles = pos / torch.pow(10000.0, dim / d_model)
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe.to(dtype)


def _gated(activation):
    def reference(a: dict, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return activation(x[..., :half]) * x[..., half:]

    return reference


# Each op's reference: its constructor arguments, then its inputs in signature order.
_REFERENCES = {
    "ExpFwdOp": lambda a, x: torch.exp(x),
    "LogFwdOp": lambda a, x: torch.log(x),
    "SqrtFwdOp": lambda a, x: torch.sqrt(x),
    "RsqrtFwdOp": lambda a, x: torch.rsqrt(x),
    "AbsFwdOp": lambda a, x: torch.abs(x),
    "NegFwdOp": lambda a, x: torch.neg(x),
    "ReciprocalFwdOp": lambda a, x: torch.reciprocal(x),
    "SignFwdOp": lambda a, x: torch.sign(x),
    "SinFwdOp": lambda a, x: torch.sin(x),
    "CosFwdOp": lambda a, x: torch.cos(x),
    "FloorFwdOp": lambda a, x: torch.floor(x),
    "CeilFwdOp": lambda a, x: torch.ceil(x),
    "RoundFwdOp": lambda a, x: torch.round(x, decimals=a.get("decimals", 0)),
    "TruncFwdOp": lambda a, x: torch.trunc(x),
    "ErfFwdOp": lambda a, x: torch.erf(x),
    "Log1pFwdOp": lambda a, x: torch.log1p(x),
    "Expm1FwdOp": lambda a, x: torch.expm1(x),
    "SigmoidFwdOp": lambda a, x: torch.sigmoid(x),
    "TanhFwdOp": lambda a, x: torch.tanh(x),
    "LogicalNotFwdOp": lambda a, x: torch.logical_not(x),
    "BitwiseNotFwdOp": lambda a, x: torch.bitwise_not(x),
    "IsnanFwdOp": lambda a, x: torch.isnan(x),
    "IsinfFwdOp": lambda a, x: torch.isinf(x),
    "IsfiniteFwdOp": lambda a, x: torch.isfinite(x),
    "DropoutFwdOp": lambda a, x: F.dropout(x, p=a.get("p", 0.5), training=a.get("training", True)),
    "ReluFwdOp": lambda a, x: F.relu(x, a.get("inplace", False)),
    "GeluFwdOp": lambda a, x: F.gelu(x, approximate=a.get("approximate", "none")),
    "SiluFwdOp": lambda a, x: F.silu(x, a.get("inplace", False)),
    "HardswishFwdOp": lambda a, x: F.hardswish(x, a.get("inplace", False)),
    "HardsigmoidFwdOp": lambda a, x: F.hardsigmoid(x, a.get("inplace", False)),
    "MishFwdOp": lambda a, x: F.mish(x, a.get("inplace", False)),
    "SeluFwdOp": lambda a, x: F.selu(x, a.get("inplace", False)),
    "LeakyReluFwdOp": lambda a, x: F.leaky_relu(
        x, a.get("negative_slope", 0.01), a.get("inplace", False)
    ),
    "EluFwdOp": lambda a, x: F.elu(x, a.get("alpha", 1.0), a.get("inplace", False)),
    "HardtanhFwdOp": lambda a, x: F.hardtanh(
        x, a.get("min_val", -1.0), a.get("max_val", 1.0), a.get("inplace", False)
    ),
    "SoftplusFwdOp": lambda a, x: F.softplus(x, a.get("beta", 1.0), a.get("threshold", 20.0)),
    "ClampTensorFwdOp": lambda a, x, lo=None, hi=None: torch.clamp(x, lo, hi),
    "ClampScalarFwdOp": lambda a, x: torch.clamp(x, a.get("min", None), a.get("max", None)),
    "NanToNumFwdOp": lambda a, x: torch.nan_to_num(
        x, a.get("nan", 0.0), a.get("posinf", None), a.get("neginf", None)
    ),
    "PreluFwdOp": lambda a, x, w: F.prelu(x, w),
    "MaskedFillTensorFwdOp": lambda a, x, m, v: x.masked_fill(m, v),
    "MaskedFillScalarFwdOp": lambda a, x, m: x.masked_fill(m, a["value"]),
    "AddFwdOp": lambda a, x, y: torch.add(x, y, alpha=a.get("alpha", 1.0)),
    "SubFwdOp": lambda a, x, y: torch.sub(x, y, alpha=a.get("alpha", 1.0)),
    "MulFwdOp": lambda a, x, y: torch.mul(x, y),
    "DivFwdOp": lambda a, x, y: torch.div(x, y, rounding_mode=a.get("rounding_mode", None)),
    "RemainderFwdOp": lambda a, x, y: torch.remainder(x, y),
    "PowFwdOp": lambda a, x, y: torch.pow(x, y),
    "FloorDivideFwdOp": lambda a, x, y: torch.floor_divide(x, y),
    "LerpScalarFwdOp": lambda a, x, y: torch.lerp(x, y, a.get("weight", 0.5)),
    "MaximumFwdOp": lambda a, x, y: torch.maximum(x, y),
    "MinimumFwdOp": lambda a, x, y: torch.minimum(x, y),
    "EqFwdOp": lambda a, x, y: torch.eq(x, y),
    "NeFwdOp": lambda a, x, y: torch.ne(x, y),
    "GtFwdOp": lambda a, x, y: torch.gt(x, y),
    "LtFwdOp": lambda a, x, y: torch.lt(x, y),
    "GeFwdOp": lambda a, x, y: torch.ge(x, y),
    "LeFwdOp": lambda a, x, y: torch.le(x, y),
    "LogicalAndFwdOp": lambda a, x, y: torch.logical_and(x, y),
    "LogicalOrFwdOp": lambda a, x, y: torch.logical_or(x, y),
    "BitwiseAndFwdOp": lambda a, x, y: torch.bitwise_and(x, y),
    "BitwiseOrFwdOp": lambda a, x, y: torch.bitwise_or(x, y),
    "BitwiseXorFwdOp": lambda a, x, y: torch.bitwise_xor(x, y),
    "WhereFwdOp": lambda a, c, x, y: torch.where(c, x, y),
    "LerpTensorFwdOp": lambda a, x, e, w: torch.lerp(x, e, w),
    "SiluAndMulFwdOp": _gated(F.silu),
    "GeluAndMulFwdOp": _gated(F.gelu),
    "GeluTanhAndMulFwdOp": _gated(lambda g: F.gelu(g, approximate="tanh")),
    "AlibiFwdOp": lambda a: alibi_reference(
        a["seq_len"], a["num_heads"], a["out_dtype"], a["device"]
    ),
    "SinusoidalFwdOp": lambda a: sinusoidal_reference(
        a["seq_len"], a["d_model"], a["out_dtype"], a["device"]
    ),
}


class ElementwiseWorkload(WorkloadBase):
    """One elementwise reference and contract for supplied inputs or a manifest call.

    Parameters describe the operation; consumers cannot supply numerical policy.
    """

    def __init__(self, name: str, inputs: tuple, **parameters):
        self.name = name
        self.inputs = inputs
        self.parameters = parameters

    def gen_inputs(self):
        return self.inputs

    def arguments(self):
        return self.parameters

    def ref_program(self, *inputs):
        return _REFERENCES[self.name](self.arguments(), *inputs)

    def verification(self, *inputs):
        from workloads.numerics import Custom, Exact

        name = self.name
        if name in ("SiluAndMulFwdOp", "GeluAndMulFwdOp", "GeluTanhAndMulFwdOp"):
            return fused_gated_verification(inputs[0].dtype)
        if name in {
            "FloorDivideFwdOp",
            "RemainderFwdOp",
            "WhereFwdOp",
            "ClampScalarFwdOp",
            "ClampTensorFwdOp",
            "MaskedFillScalarFwdOp",
            "MaskedFillTensorFwdOp",
            "MaximumFwdOp",
            "MinimumFwdOp",
            "NanToNumFwdOp",
        } or (name == "DivFwdOp" and self.arguments().get("rounding_mode") is not None):
            return Exact(atol=0, rtol=0)
        if name == "LerpScalarFwdOp":
            return lerp_verification(inputs[0].dtype)
        if name != "DropoutFwdOp":
            return Exact()
        x = inputs[0]
        a = self.arguments()
        p = a.get("p", 0.5) if a.get("training", True) else 0.0

        def validate(got, _expected):
            if p in (0.0, 1.0):
                torch.testing.assert_close(
                    got, x if p == 0 else torch.zeros_like(x), atol=0, rtol=0
                )
                return
            # Independent generators need not choose identical masks.
            torch.testing.assert_close(got, torch.where(got != 0, x / (1 - p), 0))
            eligible = x != 0
            n = eligible.sum()
            dropped = ((got == 0) & eligible).sum()
            assert (dropped - n * p).abs() <= 6 * (n * p * (1 - p)).sqrt() + 1

        return Custom(validate, "dropout scaling and six-sigma mask rate")


class ElementwiseCall(CallWorkload, ElementwiseWorkload):
    """One manifest call of an elementwise op: the row's tensors, drawn from the op's value
    domain, and the op's reference.

    ``shape`` (the output's) and ``dtype`` (the call's element type) name the case in the
    benchmark report.
    """

    def __init__(self, call, device=None):
        CallWorkload.__init__(self, call, device)
        self.name = call.signature.name
        self.shape = call.tensors["output"][0]
        self.dtype = getattr(torch, call.ix.get("T") or call.tensors["output"][1])

    def gen_inputs(self) -> tuple:
        draw = _DOMAINS.get(self.call.signature.name, _draw_normal)
        specs = (self.call.specs[t] for t in self.call.signature.inputs)
        return tuple(
            None if s is None else draw(s.shape, getattr(torch, s.dtype), self.device)
            for s in specs
        )

    def arguments(self) -> dict:
        # No elementwise op takes a construction-time tensor. A tensorless op's unset
        # `device` parameter follows the workload's device.
        arguments = self.call.arguments({})
        if "device" in arguments and arguments["device"] is None:
            arguments["device"] = self.device
        return arguments


def fused_gated_verification(dtype):
    from workloads.numerics import Exact, reference_tolerance

    return (
        Exact(atol=1e-2, rtol=1e-2)
        if dtype == torch.float16
        else Exact(**reference_tolerance(dtype))
    )


class ReluCompileCase(RandnFlatWorkload):
    def ref_program(self, x):
        return torch.relu(x.float()).to(x.dtype)


class AbsCompileCase(RandnFlatWorkload):
    def ref_program(self, x):
        return torch.abs(x.float()).to(x.dtype)


class SignCompileCase(RandnFlatWorkload):
    def ref_program(self, x):
        return torch.sign(x.float()).to(x.dtype)


class SiluAndMulCase(GatedRandnWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate = x_f32[:, : self.n]
        value = x_f32[:, self.n :]
        return (F.silu(gate) * value).to(x.dtype)

    def verification(self, *inputs):
        return fused_gated_verification(inputs[0].dtype)


class GeluAndMulCase(GatedRandnWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate = x_f32[:, : self.n]
        value = x_f32[:, self.n :]
        return (F.gelu(gate) * value).to(x.dtype)

    def verification(self, *inputs):
        return fused_gated_verification(inputs[0].dtype)


class GeluTanhAndMulCase(GatedRandnWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate = x_f32[:, : self.n]
        value = x_f32[:, self.n :]
        return (F.gelu(gate, approximate="tanh") * value).to(x.dtype)

    def verification(self, *inputs):
        return fused_gated_verification(inputs[0].dtype)


class AddSameShapeCase(RandnPairWorkload):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


class BinarySameShapeCase(RandnPairWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, **parameters):
        RandnPairWorkload.__init__(self, n_total, dtype)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class BinaryPositiveCase(PositivePairWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, **parameters):
        PositivePairWorkload.__init__(self, n_total, dtype)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class RemainderCase(PositivePairWorkload):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a, b)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=0, rtol=0)


class FloorDivideCase(PositivePairWorkload):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(a, b)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=0, rtol=0)


class LerpCase(RandnPairWorkload):
    def __init__(self, n_total: int, dtype, weight: float = 0.5):
        super().__init__(n_total, dtype)
        self.weight = weight

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.lerp(a.float(), b.float(), self.weight).to(a.dtype)

    def verification(self, *inputs):
        return lerp_verification(inputs[0].dtype)


class SpecialCase(SpecialWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, gen_fn=None, **parameters):
        SpecialWorkload.__init__(self, n_total, dtype, gen_fn=gen_fn)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class BitwiseCase(BitwiseWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, op_name, **parameters):
        BitwiseWorkload.__init__(self, n_total)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class UnaryActivationCase(RandnFlatWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, gen_fn=None, **parameters):
        RandnFlatWorkload.__init__(self, n_total, dtype, gen_fn=gen_fn)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class UnaryMathCase(RandnFlatWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, gen_fn=None, **parameters):
        RandnFlatWorkload.__init__(self, n_total, dtype, gen_fn=gen_fn)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class ComparisonCase(RandnPairWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, **parameters):
        RandnPairWorkload.__init__(self, n_total, dtype)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


class LogicalCase(LogicalWorkload, ElementwiseWorkload):
    """Input fixture using the shared operator reference and verification."""

    def __init__(self, n_total, dtype, op_name, **parameters):
        LogicalWorkload.__init__(self, n_total, dtype)
        ElementwiseWorkload.__init__(self, op_name, (), **parameters)


def lerp_verification(dtype):
    """Native-dtype multiply/add round separately before the final lerp result."""
    from workloads.numerics import Exact

    return Exact(atol=5e-3, rtol=5e-3) if dtype == torch.float16 else Exact()


class GeluTailWorkload(ElementwiseWorkload):
    """The saturated tails have exact closed forms in the storage dtype."""

    def __init__(self, dtype):
        largest = torch.finfo(dtype).max
        x = torch.tensor(
            [
                -largest,
                -1000.0,
                -8.0,
                8.0,
                1000.0,
                largest,
                -float("inf"),
                float("inf"),
                float("nan"),
            ],
            device=run_device(),
            dtype=dtype,
        )
        super().__init__("GeluFwdOp", (x,))

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=0, rtol=0)


class ErfRoundingWorkload(ElementwiseWorkload):
    """Exhaust the 16-bit domain; the polynomial must stay within one storage step."""

    def __init__(self, dtype):
        codes = torch.arange(1 << 16, dtype=torch.int32, device=run_device()).to(torch.int16)
        super().__init__("ErfFwdOp", (codes.view(dtype),))

    def ref_program(self, x):
        return torch.erf(x.float()).to(x.dtype)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        x = inputs[0]

        def ordered(t):
            code = t.view(torch.int16).to(torch.int32)
            return torch.where(code < 0, -32768 - code, code)

        def validate(got, expected):
            assert torch.equal(got.isnan(), expected.isnan())
            finite = torch.isfinite(x)
            assert ((ordered(got[finite]) - ordered(expected[finite])).abs() <= 1).all()
            # Infinity must saturate exactly; one-ULP slack here loses the closed form.
            torch.testing.assert_close(
                got[~finite], expected[~finite], atol=0, rtol=0, equal_nan=True
            )

        return Custom(validate, "one storage step over the 16-bit domain; exact special values")


class LerpCancellationWorkload(ElementwiseWorkload):
    """A representable nonzero midpoint is lost if end-start is narrowed early."""

    def __init__(self, dtype):
        a = torch.full((256,), -4 - 4 * torch.finfo(dtype).eps, dtype=dtype, device=run_device())
        b, w = torch.full_like(a, 4), torch.full_like(a, 0.5)
        super().__init__("LerpTensorFwdOp", (a, b, w))

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=0, rtol=0)
