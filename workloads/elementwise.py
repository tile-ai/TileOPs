"""Workload definitions for elementwise op workloads with custom generators."""

from math import prod

import torch
import torch.nn.functional as F

from workloads.workload_base import CallWorkload, WorkloadBase


class ReluWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.float()).to(x.dtype)


class BinaryBenchCase:
    """Two same-shape tensors drawn from a named value domain."""

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        domain: str = "normal",
    ):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype
        self.output_dtype = output_dtype
        self.domain = domain

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return PAIR_DOMAINS[self.domain](self.shape, self.dtype)


class FusedGatedBenchCase:
    """Minimal workload for fused gated ops."""

    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = M
        self.N = N
        self.n_total = M * N
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.M, 2 * self.N, device="cuda", dtype=self.dtype),)


class BroadcastBenchCase:
    """Workload for broadcast binary ops with asymmetric shapes."""

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        domain: str = "normal",
    ):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.n_total = prod(a_shape)  # output size = broadcast result
        self.dtype = dtype
        self.output_dtype = output_dtype
        self.domain = domain

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return BROADCAST_DOMAINS[self.domain](self.a_shape, self.b_shape, self.dtype)


class AddBroadcastWorkload(WorkloadBase):
    def __init__(self, a_shape: tuple, b_shape: tuple, dtype: torch.dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = torch.randn(self.b_shape, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


class PowPositiveWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.5
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") * 2.0
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.pow(a.float(), b.float()).to(a.dtype)


class BitwiseNotWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randint(-128, 128, (self.n_total,), device="cuda", dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


class AddCompileWorkload(WorkloadBase):
    def __init__(self, a_shape, b_shape, dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self):
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = torch.randn(self.b_shape, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a, b):
        return (a.float() + b.float()).to(a.dtype)


class EqCompileWorkload(WorkloadBase):
    def __init__(self, a_shape, b_shape, dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self):
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
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
        x = torch.randn(self.M, 2 * self.N, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x):
        gate = x[:, : self.N].float()
        value = x[:, self.N :].float()
        return (torch.nn.functional.silu(gate) * value).to(x.dtype)


class LogicalNotWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
            return (x,)

        if self.dtype == torch.uint8:
            x = torch.randint(0, 8, (self.n_total,), device="cuda", dtype=self.dtype)
        elif self.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            x = torch.randint(-4, 4, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)

        mask = torch.rand(self.n_total, device="cuda") > 0.5
        x[mask] = 0
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x)


class BitwiseWorkload(WorkloadBase):
    def __init__(self, n_total: int):
        self.n_total = n_total
        self.dtype = torch.int32

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device="cuda")
        b = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device="cuda")
        return a, b


class LogicalWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
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
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
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
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)


class RandnPairWorkload(WorkloadBase):
    """Two same-shape ``randn`` vectors — the default binary-op input."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b


class PositivePairWorkload(WorkloadBase):
    """Two same-shape vectors in ``[0.1, 1.1)`` — for ops undefined at or below 0."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b


class GatedRandnWorkload(WorkloadBase):
    """One ``(m, 2 * n)`` tensor — gate and value halves for a fused gated op."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.m, 2 * self.n, dtype=self.dtype, device="cuda"),)


# Value domains of the benchmark studies. A study names the domain its op requires;
# the draw itself belongs to this layer.


def draw_normal_pair(shape: tuple, dtype: torch.dtype):
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    return a, b


def draw_positive_pair(shape: tuple, dtype: torch.dtype):
    a = torch.rand(*shape, device="cuda", dtype=dtype) + 0.1
    b = torch.rand(*shape, device="cuda", dtype=dtype) + 0.1
    return a, b


def draw_int_pair(shape: tuple, dtype: torch.dtype):
    a = torch.randint(-1000, 1000, shape, device="cuda", dtype=torch.int32)
    b = torch.randint(-1000, 1000, shape, device="cuda", dtype=torch.int32)
    return a, b


def draw_bool_pair(shape: tuple, dtype: torch.dtype):
    a = (torch.randn(*shape, device="cuda", dtype=dtype) > 0).to(dtype)
    b = (torch.randn(*shape, device="cuda", dtype=dtype) > 0).to(dtype)
    return a, b


def draw_normal_broadcast_pair(a_shape, b_shape, dtype):
    a = torch.randn(*a_shape, device="cuda", dtype=dtype)
    b = torch.randn(*b_shape, device="cuda", dtype=dtype)
    return a, b


def draw_positive_broadcast_pair(a_shape, b_shape, dtype):
    a = torch.rand(*a_shape, device="cuda", dtype=dtype) + 0.1
    b = torch.rand(*b_shape, device="cuda", dtype=dtype) + 0.1
    return a, b


# Domain name -> draw function. The mapping lives here, so a caller names a
# domain rather than passing a draw, and the set is statically visible.
PAIR_DOMAINS = {
    "normal": draw_normal_pair,
    "positive": draw_positive_pair,
    "int": draw_int_pair,
    "bool": draw_bool_pair,
}

BROADCAST_DOMAINS = {
    "normal": draw_normal_broadcast_pair,
    "positive": draw_positive_broadcast_pair,
}


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


def alibi_reference(
    seq_len: int, num_heads: int, dtype: torch.dtype, device="cuda"
) -> torch.Tensor:
    """Full ALiBi bias: (num_heads, seq_len, seq_len), bias[h,i,j] = -slope_h * |i-j|."""
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device=device, dtype=torch.float32) / num_heads,
    )
    return (-slopes[:, None, None] * dist[None, :, :]).to(dtype)


def sinusoidal_reference(
    seq_len: int, d_model: int, dtype: torch.dtype, device="cuda"
) -> torch.Tensor:
    """Sinusoidal encoding: (seq_len, d_model), sin on even columns, cos on odd ones."""
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
    "RoundFwdOp": lambda a, x: torch.round(x, decimals=a["decimals"]),
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
    "DropoutFwdOp": lambda a, x: F.dropout(x, p=a["p"], training=a["training"]),
    "ReluFwdOp": lambda a, x: F.relu(x, a["inplace"]),
    "GeluFwdOp": lambda a, x: F.gelu(x, approximate=a["approximate"]),
    "SiluFwdOp": lambda a, x: F.silu(x, a["inplace"]),
    "HardswishFwdOp": lambda a, x: F.hardswish(x, a["inplace"]),
    "HardsigmoidFwdOp": lambda a, x: F.hardsigmoid(x, a["inplace"]),
    "MishFwdOp": lambda a, x: F.mish(x, a["inplace"]),
    "SeluFwdOp": lambda a, x: F.selu(x, a["inplace"]),
    "LeakyReluFwdOp": lambda a, x: F.leaky_relu(x, a["negative_slope"], a["inplace"]),
    "EluFwdOp": lambda a, x: F.elu(x, a["alpha"], a["inplace"]),
    "HardtanhFwdOp": lambda a, x: F.hardtanh(x, a["min_val"], a["max_val"], a["inplace"]),
    "SoftplusFwdOp": lambda a, x: F.softplus(x, a["beta"], a["threshold"]),
    "ClampFwdOp": lambda a, x, lo, hi: torch.clamp(x, lo, hi),
    "ClampScalarFwdOp": lambda a, x: torch.clamp(x, a["min"], a["max"]),
    "NanToNumFwdOp": lambda a, x: torch.nan_to_num(x, a["nan"], a["posinf"], a["neginf"]),
    "PreluFwdOp": lambda a, x, w: F.prelu(x, w),
    "MaskedFillFwdOp": lambda a, x, m, v: x.masked_fill(m, v),
    "MaskedFillScalarFwdOp": lambda a, x, m: x.masked_fill(m, a["value"]),
    "AddFwdOp": lambda a, x, y: torch.add(x, y, alpha=a["alpha"]),
    "SubFwdOp": lambda a, x, y: torch.sub(x, y, alpha=a["alpha"]),
    "MulFwdOp": lambda a, x, y: torch.mul(x, y),
    "DivFwdOp": lambda a, x, y: torch.div(x, y, rounding_mode=a["rounding_mode"]),
    "RemainderFwdOp": lambda a, x, y: torch.remainder(x, y),
    "PowFwdOp": lambda a, x, y: torch.pow(x, y),
    "FloorDivideFwdOp": lambda a, x, y: torch.floor_divide(x, y),
    "LerpFwdOp": lambda a, x, y: torch.lerp(x, y, a["weight"]),
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
        a["seq_len"], a["num_heads"], a["out_dtype"], a["device"] or "cuda"
    ),
    "SinusoidalFwdOp": lambda a: sinusoidal_reference(
        a["seq_len"], a["d_model"], a["out_dtype"], a["device"] or "cuda"
    ),
}


class ElementwiseCall(CallWorkload):
    """One manifest call of an elementwise op: the row's tensors, drawn from the op's value
    domain, and the op's reference.

    ``shape`` (the output's) and ``dtype`` (the call's element type) name the case in the
    benchmark report.
    """

    def __init__(self, call, device="cuda"):
        super().__init__(call, device)
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
        # No elementwise op takes a construction-time tensor.
        return self.call.arguments({})

    def ref_program(self, *inputs):
        return _REFERENCES[self.call.signature.name](self.arguments(), *inputs)
