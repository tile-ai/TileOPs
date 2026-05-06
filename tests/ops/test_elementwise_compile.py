"""Tests for torch.compile compatibility of elementwise ops.

Section 1: Detailed compile tests for 6 representative ops (relu, add, eq,
silu_and_mul, abs, sign) with full fixture/test structure.

Section 2: Parametrized compile-smoke tests covering every remaining registered
op to ensure the registration table is fully exercised.

Validates that torch.compile(op, fullgraph=True) produces correct output.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import (
    AbsFwdOp,
    AddFwdOp,
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    CeilFwdOp,
    ClampFwdOp,
    ClampMaxFwdOp,
    ClampMinFwdOp,
    ClampScalarFwdOp,
    CosFwdOp,
    DivFwdOp,
    EqFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorDivideFwdOp,
    FloorFwdOp,
    GeFwdOp,
    GeluAndMulFwdOp,
    GeluFwdOp,
    GeluTanhAndMulFwdOp,
    GtFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
    LeFwdOp,
    LerpFwdOp,
    LerpTensorFwdOp,
    Log1pFwdOp,
    LogFwdOp,
    LogicalAndFwdOp,
    LogicalNotFwdOp,
    LogicalOrFwdOp,
    LtFwdOp,
    MaskedFillFwdOp,
    MaskedFillScalarFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MishFwdOp,
    MulFwdOp,
    NeFwdOp,
    NegFwdOp,
    PowFwdOp,
    ReciprocalFwdOp,
    ReluFwdOp,
    RemainderFwdOp,
    RoundFwdOp,
    RsqrtFwdOp,
    SeluFwdOp,
    SigmoidFwdOp,
    SignFwdOp,
    SiluAndMulFwdOp,
    SiluFwdOp,
    SinFwdOp,
    SqrtFwdOp,
    SubFwdOp,
    TanhFwdOp,
    TruncFwdOp,
    WhereFwdOp,
)


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Reset torch._dynamo before each test to avoid recompile limit."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Unary compile test: relu
# ---------------------------------------------------------------------------


class ReluCompileFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


class ReluCompileTest(TestBase):
    def __init__(self, n_total, dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self):
        return (torch.randn(self.n_total, dtype=self.dtype, device="cuda"),)

    def ref_program(self, x):
        return torch.relu(x.float()).to(x.dtype)


@ReluCompileFixture
def test_relu_compile(n_total, dtype):
    test = ReluCompileTest(n_total, dtype)
    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Binary compile test: add
# ---------------------------------------------------------------------------


class AddCompileFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype", [
            pytest.param((1024, 1024), (1024, 1024), torch.float16, marks=pytest.mark.full),
            pytest.param((1024, 1024), (1, 1024), torch.float16, marks=pytest.mark.full),
        ]),
    ]


class AddCompileTest(TestBase):
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


@AddCompileFixture
def test_add_compile(a_shape, b_shape, dtype):
    test = AddCompileTest(a_shape, b_shape, dtype)
    op = AddFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Comparison compile test: eq (bool output)
# ---------------------------------------------------------------------------


class EqCompileFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype", [
            pytest.param((1024, 1024), (1024, 1024), torch.float16, marks=pytest.mark.full),
        ]),
    ]


class EqCompileTest(TestBase):
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


@EqCompileFixture
def test_eq_compile(a_shape, b_shape, dtype):
    test = EqCompileTest(a_shape, b_shape, dtype)
    op = EqFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, compare=exact_compare)


# ---------------------------------------------------------------------------
# FusedGated compile test: silu_and_mul
# ---------------------------------------------------------------------------


class SiluAndMulCompileFixture(FixtureBase):
    PARAMS = [
        ("M, N, dtype", [
            pytest.param(512, 1024, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class SiluAndMulCompileTest(TestBase):
    def __init__(self, M, N, dtype):
        self.M = M
        self.N = N
        self.dtype = dtype

    def gen_inputs(self):
        x = torch.randn(self.M, 2 * self.N, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x):
        gate = x[:, :self.N].float()
        value = x[:, self.N:].float()
        return (torch.nn.functional.silu(gate) * value).to(x.dtype)


@SiluAndMulCompileFixture
def test_silu_and_mul_compile(M, N, dtype):
    test = SiluAndMulCompileTest(M, N, dtype)
    op = SiluAndMulFwdOp(M=M, N=N, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Additional unary compile tests: abs, sign
# ---------------------------------------------------------------------------


class AbsCompileFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class AbsCompileTest(TestBase):
    def __init__(self, n_total, dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self):
        return (torch.randn(self.n_total, dtype=self.dtype, device="cuda"),)

    def ref_program(self, x):
        return torch.abs(x.float()).to(x.dtype)


@AbsCompileFixture
def test_abs_compile(n_total, dtype):
    test = AbsCompileTest(n_total, dtype)
    op = AbsFwdOp(N_total=n_total, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


class SignCompileFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class SignCompileTest(TestBase):
    def __init__(self, n_total, dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self):
        return (torch.randn(self.n_total, dtype=self.dtype, device="cuda"),)

    def ref_program(self, x):
        return torch.sign(x.float()).to(x.dtype)


@SignCompileFixture
def test_sign_compile(n_total, dtype):
    test = SignCompileTest(n_total, dtype)
    op = SignFwdOp(N_total=n_total, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# register_fake shape/dtype correctness
# ---------------------------------------------------------------------------


class FakeUnaryFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1024, torch.float16, marks=pytest.mark.full),
        ]),
    ]


@FakeUnaryFixture
def test_register_fake_unary_shape_dtype(n_total, dtype):
    """Verify register_fake returns correct shape and dtype for unary ops."""
    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == x.dtype, f"Dtype mismatch: {out.dtype} vs {x.dtype}"


class FakeComparisonFixture(FixtureBase):
    PARAMS = [
        ("shape, dtype", [
            pytest.param((256, 256), torch.float16, marks=pytest.mark.full),
        ]),
    ]


@FakeComparisonFixture
def test_register_fake_comparison_bool_dtype(shape, dtype):
    """Verify register_fake returns torch.bool for comparison ops."""
    op = EqFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    assert out.dtype == torch.bool, f"Expected bool, got {out.dtype}"


class FakeFusedGatedFixture(FixtureBase):
    PARAMS = [
        ("M, N, dtype", [
            pytest.param(64, 128, torch.float16, marks=pytest.mark.full),
        ]),
    ]


@FakeFusedGatedFixture
def test_register_fake_fused_gated_shape(M, N, dtype):
    """Verify register_fake returns correct shape for fused gated ops."""
    op = SiluAndMulFwdOp(M=M, N=N, dtype=dtype)
    x = torch.randn(M, 2 * N, dtype=dtype, device="cuda")
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x)
    assert out.shape == (M, N), f"Shape mismatch: {out.shape} vs {(M, N)}"
    assert out.dtype == dtype


# ---------------------------------------------------------------------------
# Exhaustive compile-smoke: every registered op
# ---------------------------------------------------------------------------
# These tests instantiate and torch.compile each op to verify that the
# custom_op registration, register_fake, and CUDA codegen all succeed.
# They are marked "smoke" so CI catches registration regressions early.

_N = 1024 * 1024
_SHAPE = (1024, 1024)
_SMALL = (256, 256)
_DTYPE = torch.float16


# --- Remaining unary ops (not covered by detailed tests above) ---

def _positive_input(n, dtype):
    """Generate strictly positive inputs for log/sqrt/rsqrt/log1p domains."""
    return torch.rand(n, dtype=dtype, device="cuda").clamp(min=0.01) * 10.0


_UNARY_FLOAT_OPS = [
    pytest.param(ExpFwdOp, torch.exp, None, "exp", marks=pytest.mark.full),
    pytest.param(LogFwdOp, lambda x: torch.log(x.float()).to(x.dtype), _positive_input, "log", marks=pytest.mark.full),
    pytest.param(SqrtFwdOp, lambda x: torch.sqrt(x.float()).to(x.dtype), _positive_input, "sqrt", marks=pytest.mark.full),
    pytest.param(RsqrtFwdOp, lambda x: torch.rsqrt(x.float()).to(x.dtype), _positive_input, "rsqrt", marks=pytest.mark.full),
    pytest.param(NegFwdOp, torch.neg, None, "neg", marks=pytest.mark.full),
    pytest.param(ReciprocalFwdOp, lambda x: torch.reciprocal(x.float()).to(x.dtype), None, "reciprocal", marks=pytest.mark.full),
    pytest.param(SinFwdOp, lambda x: torch.sin(x.float()).to(x.dtype), None, "sin", marks=pytest.mark.full),
    pytest.param(CosFwdOp, lambda x: torch.cos(x.float()).to(x.dtype), None, "cos", marks=pytest.mark.full),
    pytest.param(FloorFwdOp, lambda x: torch.floor(x.float()).to(x.dtype), None, "floor", marks=pytest.mark.full),
    pytest.param(CeilFwdOp, lambda x: torch.ceil(x.float()).to(x.dtype), None, "ceil", marks=pytest.mark.full),
    pytest.param(RoundFwdOp, lambda x: torch.round(x.float()).to(x.dtype), None, "round", marks=pytest.mark.full),
    pytest.param(TruncFwdOp, lambda x: torch.trunc(x.float()).to(x.dtype), None, "trunc", marks=pytest.mark.full),
    pytest.param(ErfFwdOp, lambda x: torch.erf(x.float()).to(x.dtype), None, "erf", marks=pytest.mark.full),
    pytest.param(Log1pFwdOp, lambda x: torch.log1p(x.float()).to(x.dtype), _positive_input, "log1p", marks=pytest.mark.full),
    pytest.param(Expm1FwdOp, lambda x: torch.expm1(x.float()).to(x.dtype), None, "expm1", marks=pytest.mark.full),
    pytest.param(GeluFwdOp, lambda x: torch.nn.functional.gelu(x.float()).to(x.dtype), None, "gelu", marks=pytest.mark.full),
    pytest.param(SiluFwdOp, lambda x: torch.nn.functional.silu(x.float()).to(x.dtype), None, "silu", marks=pytest.mark.full),
    pytest.param(SigmoidFwdOp, lambda x: torch.sigmoid(x.float()).to(x.dtype), None, "sigmoid", marks=pytest.mark.full),
    pytest.param(TanhFwdOp, lambda x: torch.tanh(x.float()).to(x.dtype), None, "tanh", marks=pytest.mark.full),
    pytest.param(HardswishFwdOp, lambda x: torch.nn.functional.hardswish(x.float()).to(x.dtype), None, "hardswish", marks=pytest.mark.full),
    pytest.param(HardsigmoidFwdOp, lambda x: torch.nn.functional.hardsigmoid(x.float()).to(x.dtype), None, "hardsigmoid", marks=pytest.mark.full),
    pytest.param(MishFwdOp, lambda x: torch.nn.functional.mish(x.float()).to(x.dtype), None, "mish", marks=pytest.mark.full),
    pytest.param(SeluFwdOp, lambda x: torch.nn.functional.selu(x.float()).to(x.dtype), None, "selu", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, ref_fn, input_fn, name", _UNARY_FLOAT_OPS)
def test_unary_float_compile(op_cls, ref_fn, input_fn, name):
    """Compile-smoke for remaining float unary ops."""
    n = _N
    op = op_cls(N_total=n, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    x = input_fn(n, _DTYPE) if input_fn is not None else torch.randn(n, dtype=_DTYPE, device="cuda")
    out = compiled_op(x)
    ref = ref_fn(x)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- Unary bool-output ops ---

_UNARY_BOOL_OPS = [
    pytest.param(LogicalNotFwdOp, lambda x: ~(x != 0), torch.float16, "logical_not", marks=pytest.mark.full),
    pytest.param(IsnanFwdOp, torch.isnan, torch.float16, "isnan", marks=pytest.mark.full),
    pytest.param(IsinfFwdOp, torch.isinf, torch.float16, "isinf", marks=pytest.mark.full),
    pytest.param(IsfiniteFwdOp, torch.isfinite, torch.float16, "isfinite", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, ref_fn, dtype, name", _UNARY_BOOL_OPS)
def test_unary_bool_compile(op_cls, ref_fn, dtype, name):
    """Compile-smoke for unary ops with bool output."""
    n = _N
    op = op_cls(N_total=n, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    x = torch.randn(n, dtype=dtype, device="cuda")
    out = compiled_op(x)
    ref = ref_fn(x)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref)


# --- Unary bitwise op ---

@pytest.mark.full
def test_bitwise_not_compile():
    """Compile-smoke for BitwiseNotFwdOp."""
    n = _N
    x_int = torch.randint(0, 256, (n,), dtype=torch.uint8, device="cuda")
    op = BitwiseNotFwdOp(N_total=n, dtype=torch.uint8)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x_int)
    ref = ~x_int
    assert torch.equal(out, ref)


# --- Remaining binary same-dtype ops ---

_BINARY_ARITH_OPS = [
    pytest.param(SubFwdOp, lambda a, b: (a.float() - b.float()).half(), "sub", marks=pytest.mark.full),
    pytest.param(MulFwdOp, lambda a, b: (a.float() * b.float()).half(), "mul", marks=pytest.mark.full),
    pytest.param(DivFwdOp, lambda a, b: (a.float() / b.float()).half(), "div", marks=pytest.mark.full),
    pytest.param(RemainderFwdOp, lambda a, b: a - torch.floor(a.float() / b.float()).half() * b, "remainder", marks=pytest.mark.full),
    pytest.param(FloorDivideFwdOp, lambda a, b: torch.floor(a.float() / b.float()).half(), "floor_divide", marks=pytest.mark.full),
    pytest.param(MaximumFwdOp, lambda a, b: torch.maximum(a.float(), b.float()).half(), "maximum", marks=pytest.mark.full),
    pytest.param(MinimumFwdOp, lambda a, b: torch.minimum(a.float(), b.float()).half(), "minimum", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, ref_fn, name", _BINARY_ARITH_OPS)
def test_binary_arith_compile(op_cls, ref_fn, name):
    """Compile-smoke for remaining binary arithmetic ops."""
    shape = _SMALL
    a = torch.randn(shape, dtype=_DTYPE, device="cuda")
    b = torch.randn(shape, dtype=_DTYPE, device="cuda").abs().clamp(min=0.1)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = ref_fn(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.full
def test_pow_compile():
    """Compile-smoke for PowFwdOp with positive inputs to avoid NaN domain issues."""
    shape = _SMALL
    # Use positive base and small positive exponent to stay in valid domain
    a = torch.rand(shape, dtype=_DTYPE, device="cuda").clamp(min=0.1) * 5.0
    b = torch.rand(shape, dtype=_DTYPE, device="cuda") * 2.0
    op = PowFwdOp(a_shape=shape, b_shape=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = torch.pow(a.float(), b.float()).half()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- Lerp (special binary with weight) ---

@pytest.mark.full
def test_lerp_compile():
    """Compile-smoke for LerpFwdOp."""
    shape = _SMALL
    a = torch.randn(shape, dtype=_DTYPE, device="cuda")
    b = torch.randn(shape, dtype=_DTYPE, device="cuda")
    op = LerpFwdOp(a_shape=shape, b_shape=shape, dtype=_DTYPE, weight=0.3)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = torch.lerp(a.float(), b.float(), 0.3).half()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.full
def test_lerp_tensor_compile():
    """Compile-smoke for LerpTensorFwdOp (Tensor-weight overload)."""
    shape = _SMALL
    a = torch.randn(shape, dtype=_DTYPE, device="cuda")
    b = torch.randn(shape, dtype=_DTYPE, device="cuda")
    w = torch.rand(shape, dtype=_DTYPE, device="cuda")
    op = LerpTensorFwdOp(input=shape, end=shape, weight=shape, dtype=_DTYPE)
    assert type(op)._wrapped is not None, (
        "LerpTensorFwdOp._wrapped must be populated by registration"
    )
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b, w)
    ref = torch.lerp(a, b, w)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- Remaining comparison ops ---

_COMPARISON_OPS = [
    pytest.param(NeFwdOp, lambda a, b: a != b, "ne", marks=pytest.mark.full),
    pytest.param(GtFwdOp, lambda a, b: a > b, "gt", marks=pytest.mark.full),
    pytest.param(LtFwdOp, lambda a, b: a < b, "lt", marks=pytest.mark.full),
    pytest.param(GeFwdOp, lambda a, b: a >= b, "ge", marks=pytest.mark.full),
    pytest.param(LeFwdOp, lambda a, b: a <= b, "le", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, ref_fn, name", _COMPARISON_OPS)
def test_comparison_compile(op_cls, ref_fn, name):
    """Compile-smoke for remaining comparison ops (bool output)."""
    shape = _SMALL
    a = torch.randn(shape, dtype=_DTYPE, device="cuda")
    b = torch.randn(shape, dtype=_DTYPE, device="cuda")
    op = op_cls(a_shape=shape, b_shape=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = ref_fn(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref)


# --- Logical binary ops ---

_LOGICAL_OPS = [
    pytest.param(LogicalAndFwdOp, lambda a, b: (a != 0) & (b != 0), "logical_and", marks=pytest.mark.full),
    pytest.param(LogicalOrFwdOp, lambda a, b: (a != 0) | (b != 0), "logical_or", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, ref_fn, name", _LOGICAL_OPS)
def test_logical_binary_compile(op_cls, ref_fn, name):
    """Compile-smoke for logical binary ops (bool output)."""
    shape = _SMALL
    a = torch.randn(shape, dtype=_DTYPE, device="cuda")
    b = torch.randn(shape, dtype=_DTYPE, device="cuda")
    op = op_cls(a_shape=shape, b_shape=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = ref_fn(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref)


# --- Bitwise binary ops ---

_BITWISE_BINARY_OPS = [
    pytest.param(BitwiseAndFwdOp, lambda a, b: a & b, "bitwise_and", marks=pytest.mark.full),
    pytest.param(BitwiseOrFwdOp, lambda a, b: a | b, "bitwise_or", marks=pytest.mark.full),
    pytest.param(BitwiseXorFwdOp, lambda a, b: a ^ b, "bitwise_xor", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, ref_fn, name", _BITWISE_BINARY_OPS)
def test_bitwise_binary_compile(op_cls, ref_fn, name):
    """Compile-smoke for bitwise binary ops."""
    shape = _SMALL
    a = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    op = op_cls(a_shape=shape, b_shape=shape, dtype=torch.uint8)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = ref_fn(a, b)
    assert torch.equal(out, ref)


# --- Remaining fused gated ops ---

_FUSED_GATED_OPS = [
    pytest.param(GeluAndMulFwdOp, "gelu_and_mul", marks=pytest.mark.full),
    pytest.param(GeluTanhAndMulFwdOp, "gelu_tanh_and_mul", marks=pytest.mark.full),
]


@pytest.mark.parametrize("op_cls, name", _FUSED_GATED_OPS)
def test_fused_gated_compile(op_cls, name):
    """Compile-smoke for remaining fused gated ops."""
    M, N = 64, 128
    x = torch.randn(M, 2 * N, dtype=_DTYPE, device="cuda")
    op = op_cls(M=M, N=N, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x)
    assert out.shape == (M, N)
    assert out.dtype == _DTYPE


# --- Where op (cond, x, y -> out): same-shape and broadcasting ---

@pytest.mark.full
def test_where_compile_same_shape():
    """Compile-smoke for WhereFwdOp with all three inputs same-shape.

    Regression: ensures WhereFwdOp registers a custom_op so
    torch.compile(fullgraph=True) does not fail with
    "torch.* op returned non-Tensor".
    """
    shape = (16,)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    y = torch.randn(shape, dtype=_DTYPE, device="cuda")
    op = WhereFwdOp(condition=shape, input=shape, other=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(cond, x, y)
    ref = torch.where(cond, x, y)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_where_compile_broadcast():
    """Compile-smoke for WhereFwdOp with broadcasting inputs."""
    cond_shape = (4, 1)
    x_shape = (1, 8)
    y_shape = (1,)
    cond = torch.randint(0, 2, cond_shape, dtype=torch.bool, device="cuda")
    x = torch.randn(x_shape, dtype=_DTYPE, device="cuda")
    y = torch.randn(y_shape, dtype=_DTYPE, device="cuda")
    op = WhereFwdOp(condition=cond_shape, input=x_shape, other=y_shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(cond, x, y)
    ref = torch.where(cond, x, y)
    assert out.shape == ref.shape == (4, 8)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# --- ClampScalarFwdOp (input -> out, scalar min/max baked) ---

@pytest.mark.full
def test_clamp_scalar_compile():
    """Compile-smoke for ClampScalarFwdOp (Number min/max baked into __init__)."""
    shape = (1024, 1024)
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    op = ClampScalarFwdOp(input=shape, min=-0.5, max=0.5, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x)
    ref = torch.clamp(x.float(), min=-0.5, max=0.5).to(_DTYPE)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# --- Tensor-bound ClampFwdOp (input, min?, max? -> out) ---

@pytest.mark.full
def test_clamp_tensor_compile_same_shape():
    """Compile-smoke for ClampFwdOp with both Tensor bounds at same shape.

    Regression: ensures ClampFwdOp registers a custom_op so
    torch.compile(fullgraph=True) does not fail with
    "torch.* op returned non-Tensor".
    """
    shape = (16, 16)
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    lo = torch.full(shape, -0.5, dtype=_DTYPE, device="cuda")
    hi = torch.full(shape, 0.5, dtype=_DTYPE, device="cuda")
    op = ClampFwdOp(input=shape, min=shape, max=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, lo, hi)
    ref = torch.clamp(x.float(), lo.float(), hi.float()).to(_DTYPE)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_clamp_tensor_compile_broadcast():
    """Compile-smoke for ClampFwdOp with broadcasting Tensor bounds."""
    input_shape = (4, 8)
    min_shape = (1, 8)
    max_shape = (4, 1)
    x = torch.randn(input_shape, dtype=_DTYPE, device="cuda")
    lo = torch.full(min_shape, -0.5, dtype=_DTYPE, device="cuda")
    hi = torch.full(max_shape, 0.5, dtype=_DTYPE, device="cuda")
    op = ClampFwdOp(input=input_shape, min=min_shape, max=max_shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, lo, hi)
    ref = torch.clamp(x.float(), lo.float(), hi.float()).to(_DTYPE)
    assert out.shape == ref.shape == input_shape
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# --- Single-bound Tensor clamp variants ---

@pytest.mark.full
def test_clamp_min_compile_same_shape():
    """Compile-smoke for ClampMinFwdOp at same shape."""
    shape = (16, 16)
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    lo = torch.full(shape, -0.5, dtype=_DTYPE, device="cuda")
    op = ClampMinFwdOp(input=shape, min=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, lo)
    ref = torch.clamp(x.float(), min=lo.float()).to(_DTYPE)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_clamp_min_compile_broadcast():
    """Compile-smoke for ClampMinFwdOp with broadcasting min."""
    input_shape = (4, 8)
    min_shape = (1, 8)
    x = torch.randn(input_shape, dtype=_DTYPE, device="cuda")
    lo = torch.full(min_shape, -0.5, dtype=_DTYPE, device="cuda")
    op = ClampMinFwdOp(input=input_shape, min=min_shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, lo)
    ref = torch.clamp(x.float(), min=lo.float()).to(_DTYPE)
    assert out.shape == ref.shape == input_shape
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_clamp_max_compile_same_shape():
    """Compile-smoke for ClampMaxFwdOp at same shape."""
    shape = (16, 16)
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    hi = torch.full(shape, 0.5, dtype=_DTYPE, device="cuda")
    op = ClampMaxFwdOp(input=shape, max=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, hi)
    ref = torch.clamp(x.float(), max=hi.float()).to(_DTYPE)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_clamp_max_compile_broadcast():
    """Compile-smoke for ClampMaxFwdOp with broadcasting max."""
    input_shape = (4, 8)
    max_shape = (4, 1)
    x = torch.randn(input_shape, dtype=_DTYPE, device="cuda")
    hi = torch.full(max_shape, 0.5, dtype=_DTYPE, device="cuda")
    op = ClampMaxFwdOp(input=input_shape, max=max_shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, hi)
    ref = torch.clamp(x.float(), max=hi.float()).to(_DTYPE)
    assert out.shape == ref.shape == input_shape
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# --- MaskedFillFwdOp (Tensor value) ---

@pytest.mark.full
def test_masked_fill_tensor_compile_same_shape():
    """Compile-smoke for MaskedFillFwdOp (0-dim Tensor value) at same shape.

    Regression: ensures MaskedFillFwdOp registers a custom_op so
    torch.compile(fullgraph=True) does not fail with
    "torch.* op returned non-Tensor".
    """
    shape = (16, 16)
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    value = torch.tensor(-1.0, dtype=_DTYPE, device="cuda")
    op = MaskedFillFwdOp(input=shape, mask=shape, value=(), dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, mask, value)
    ref = torch.where(mask, value.expand(shape), x)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_masked_fill_tensor_compile_broadcast():
    """Compile-smoke for MaskedFillFwdOp with broadcasting input/mask."""
    input_shape = (4, 8)
    mask_shape = (1, 8)
    x = torch.randn(input_shape, dtype=_DTYPE, device="cuda")
    mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device="cuda")
    value = torch.tensor(-1.0, dtype=_DTYPE, device="cuda")
    op = MaskedFillFwdOp(input=input_shape, mask=mask_shape, value=(), dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, mask, value)
    ref = torch.where(
        mask.expand(input_shape), value.expand(input_shape), x,
    )
    assert out.shape == ref.shape == input_shape
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# --- MaskedFillScalarFwdOp (broadcast path now uses custom_op) ---

@pytest.mark.full
def test_masked_fill_scalar_compile_same_shape():
    """Compile-smoke for MaskedFillScalarFwdOp at same shape."""
    shape = (16, 16)
    x = torch.randn(shape, dtype=_DTYPE, device="cuda")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    op = MaskedFillScalarFwdOp(input=shape, mask=shape, value=-1.0, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, mask)
    ref = x.masked_fill(mask, -1.0)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_masked_fill_scalar_compile_broadcast():
    """Compile-smoke for MaskedFillScalarFwdOp with broadcasting input/mask.

    Regression for the removed ``not self._needs_broadcast`` guard:
    register_fake is now broadcast-aware so the custom_op path works.
    """
    input_shape = (4, 8)
    mask_shape = (1, 8)
    x = torch.randn(input_shape, dtype=_DTYPE, device="cuda")
    mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device="cuda")
    op = MaskedFillScalarFwdOp(
        input=input_shape, mask=mask_shape, value=-1.0, dtype=_DTYPE,
    )
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x, mask)
    ref = x.masked_fill(mask.expand(input_shape), -1.0)
    assert out.shape == ref.shape == input_shape
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
