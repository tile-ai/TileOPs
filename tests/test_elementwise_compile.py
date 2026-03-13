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
    AbsOp,
    AddOp,
    BitwiseAndOp,
    BitwiseNotOp,
    BitwiseOrOp,
    BitwiseXorOp,
    CeilOp,
    CosOp,
    DivOp,
    EqOp,
    ErfOp,
    Expm1Op,
    ExpOp,
    FloorDivideOp,
    FloorOp,
    GeluAndMulOp,
    GeluOp,
    GeluTanhAndMulOp,
    GeOp,
    GtOp,
    HardsigmoidOp,
    HardswishOp,
    IsfiniteOp,
    IsinfOp,
    IsnanOp,
    LeOp,
    LerpOp,
    Log1pOp,
    LogicalAndOp,
    LogicalNotOp,
    LogicalOrOp,
    LogOp,
    LtOp,
    MaximumOp,
    MinimumOp,
    MishOp,
    MulOp,
    NegOp,
    NeOp,
    PowOp,
    ReciprocalOp,
    ReluOp,
    RemainderOp,
    RoundOp,
    RsqrtOp,
    SeluOp,
    SigmoidOp,
    SignOp,
    SiluAndMulOp,
    SiluOp,
    SinOp,
    SqrtOp,
    SubOp,
    TanhOp,
    TruncOp,
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
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
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
    op = ReluOp(N_total=n_total, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Binary compile test: add
# ---------------------------------------------------------------------------


class AddCompileFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype", [
            pytest.param((1024, 1024), (1024, 1024), torch.float16, marks=pytest.mark.smoke),
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
    op = AddOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Comparison compile test: eq (bool output)
# ---------------------------------------------------------------------------


class EqCompileFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype", [
            pytest.param((1024, 1024), (1024, 1024), torch.float16, marks=pytest.mark.smoke),
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
    op = EqOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, compare=exact_compare)


# ---------------------------------------------------------------------------
# FusedGated compile test: silu_and_mul
# ---------------------------------------------------------------------------


class SiluAndMulCompileFixture(FixtureBase):
    PARAMS = [
        ("M, N, dtype", [
            pytest.param(512, 1024, torch.float16, marks=pytest.mark.smoke),
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
    op = SiluAndMulOp(M=M, N=N, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Additional unary compile tests: abs, sign
# ---------------------------------------------------------------------------


class AbsCompileFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
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
    op = AbsOp(N_total=n_total, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


class SignCompileFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
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
    op = SignOp(N_total=n_total, dtype=dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# register_fake shape/dtype correctness
# ---------------------------------------------------------------------------


class FakeUnaryFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1024, torch.float16, marks=pytest.mark.smoke),
        ]),
    ]


@FakeUnaryFixture
def test_register_fake_unary_shape_dtype(n_total, dtype):
    """Verify register_fake returns correct shape and dtype for unary ops."""
    op = ReluOp(N_total=n_total, dtype=dtype)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == x.dtype, f"Dtype mismatch: {out.dtype} vs {x.dtype}"


class FakeComparisonFixture(FixtureBase):
    PARAMS = [
        ("shape, dtype", [
            pytest.param((256, 256), torch.float16, marks=pytest.mark.smoke),
        ]),
    ]


@FakeComparisonFixture
def test_register_fake_comparison_bool_dtype(shape, dtype):
    """Verify register_fake returns torch.bool for comparison ops."""
    op = EqOp(a_shape=shape, b_shape=shape, dtype=dtype)
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    assert out.dtype == torch.bool, f"Expected bool, got {out.dtype}"


class FakeFusedGatedFixture(FixtureBase):
    PARAMS = [
        ("M, N, dtype", [
            pytest.param(64, 128, torch.float16, marks=pytest.mark.smoke),
        ]),
    ]


@FakeFusedGatedFixture
def test_register_fake_fused_gated_shape(M, N, dtype):
    """Verify register_fake returns correct shape for fused gated ops."""
    op = SiluAndMulOp(M=M, N=N, dtype=dtype)
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
    pytest.param(ExpOp, torch.exp, None, "exp", marks=pytest.mark.smoke),
    pytest.param(LogOp, lambda x: torch.log(x.float()).to(x.dtype), _positive_input, "log", marks=pytest.mark.smoke),
    pytest.param(SqrtOp, lambda x: torch.sqrt(x.float()).to(x.dtype), _positive_input, "sqrt", marks=pytest.mark.smoke),
    pytest.param(RsqrtOp, lambda x: torch.rsqrt(x.float()).to(x.dtype), _positive_input, "rsqrt", marks=pytest.mark.smoke),
    pytest.param(NegOp, torch.neg, None, "neg", marks=pytest.mark.smoke),
    pytest.param(ReciprocalOp, lambda x: torch.reciprocal(x.float()).to(x.dtype), None, "reciprocal", marks=pytest.mark.smoke),
    pytest.param(SinOp, lambda x: torch.sin(x.float()).to(x.dtype), None, "sin", marks=pytest.mark.smoke),
    pytest.param(CosOp, lambda x: torch.cos(x.float()).to(x.dtype), None, "cos", marks=pytest.mark.smoke),
    pytest.param(FloorOp, lambda x: torch.floor(x.float()).to(x.dtype), None, "floor", marks=pytest.mark.smoke),
    pytest.param(CeilOp, lambda x: torch.ceil(x.float()).to(x.dtype), None, "ceil", marks=pytest.mark.smoke),
    pytest.param(RoundOp, lambda x: torch.round(x.float()).to(x.dtype), None, "round", marks=pytest.mark.smoke),
    pytest.param(TruncOp, lambda x: torch.trunc(x.float()).to(x.dtype), None, "trunc", marks=pytest.mark.smoke),
    pytest.param(ErfOp, lambda x: torch.erf(x.float()).to(x.dtype), None, "erf", marks=pytest.mark.smoke),
    pytest.param(Log1pOp, lambda x: torch.log1p(x.float()).to(x.dtype), _positive_input, "log1p", marks=pytest.mark.smoke),
    pytest.param(Expm1Op, lambda x: torch.expm1(x.float()).to(x.dtype), None, "expm1", marks=pytest.mark.smoke),
    pytest.param(GeluOp, lambda x: torch.nn.functional.gelu(x.float()).to(x.dtype), None, "gelu", marks=pytest.mark.smoke),
    pytest.param(SiluOp, lambda x: torch.nn.functional.silu(x.float()).to(x.dtype), None, "silu", marks=pytest.mark.smoke),
    pytest.param(SigmoidOp, lambda x: torch.sigmoid(x.float()).to(x.dtype), None, "sigmoid", marks=pytest.mark.smoke),
    pytest.param(TanhOp, lambda x: torch.tanh(x.float()).to(x.dtype), None, "tanh", marks=pytest.mark.smoke),
    pytest.param(HardswishOp, lambda x: torch.nn.functional.hardswish(x.float()).to(x.dtype), None, "hardswish", marks=pytest.mark.smoke),
    pytest.param(HardsigmoidOp, lambda x: torch.nn.functional.hardsigmoid(x.float()).to(x.dtype), None, "hardsigmoid", marks=pytest.mark.smoke),
    pytest.param(MishOp, lambda x: torch.nn.functional.mish(x.float()).to(x.dtype), None, "mish", marks=pytest.mark.smoke),
    pytest.param(SeluOp, lambda x: torch.nn.functional.selu(x.float()).to(x.dtype), None, "selu", marks=pytest.mark.smoke),
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
    pytest.param(LogicalNotOp, lambda x: ~(x != 0), torch.float16, "logical_not", marks=pytest.mark.smoke),
    pytest.param(IsnanOp, torch.isnan, torch.float16, "isnan", marks=pytest.mark.smoke),
    pytest.param(IsinfOp, torch.isinf, torch.float16, "isinf", marks=pytest.mark.smoke),
    pytest.param(IsfiniteOp, torch.isfinite, torch.float16, "isfinite", marks=pytest.mark.smoke),
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

@pytest.mark.smoke
def test_bitwise_not_compile():
    """Compile-smoke for BitwiseNotOp."""
    n = _N
    x_int = torch.randint(0, 256, (n,), dtype=torch.uint8, device="cuda")
    op = BitwiseNotOp(N_total=n, dtype=torch.uint8)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x_int)
    ref = ~x_int
    assert torch.equal(out, ref)


# --- Remaining binary same-dtype ops ---

_BINARY_ARITH_OPS = [
    pytest.param(SubOp, lambda a, b: (a.float() - b.float()).half(), "sub", marks=pytest.mark.smoke),
    pytest.param(MulOp, lambda a, b: (a.float() * b.float()).half(), "mul", marks=pytest.mark.smoke),
    pytest.param(DivOp, lambda a, b: (a.float() / b.float()).half(), "div", marks=pytest.mark.smoke),
    pytest.param(RemainderOp, lambda a, b: a - torch.floor(a.float() / b.float()).half() * b, "remainder", marks=pytest.mark.smoke),
    pytest.param(FloorDivideOp, lambda a, b: torch.floor(a.float() / b.float()).half(), "floor_divide", marks=pytest.mark.smoke),
    pytest.param(MaximumOp, lambda a, b: torch.maximum(a.float(), b.float()).half(), "maximum", marks=pytest.mark.smoke),
    pytest.param(MinimumOp, lambda a, b: torch.minimum(a.float(), b.float()).half(), "minimum", marks=pytest.mark.smoke),
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


@pytest.mark.smoke
def test_pow_compile():
    """Compile-smoke for PowOp with positive inputs to avoid NaN domain issues."""
    shape = _SMALL
    # Use positive base and small positive exponent to stay in valid domain
    a = torch.rand(shape, dtype=_DTYPE, device="cuda").clamp(min=0.1) * 5.0
    b = torch.rand(shape, dtype=_DTYPE, device="cuda") * 2.0
    op = PowOp(a_shape=shape, b_shape=shape, dtype=_DTYPE)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = torch.pow(a.float(), b.float()).half()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- Lerp (special binary with weight) ---

@pytest.mark.smoke
def test_lerp_compile():
    """Compile-smoke for LerpOp."""
    shape = _SMALL
    a = torch.randn(shape, dtype=_DTYPE, device="cuda")
    b = torch.randn(shape, dtype=_DTYPE, device="cuda")
    op = LerpOp(a_shape=shape, b_shape=shape, dtype=_DTYPE, weight=0.3)
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(a, b)
    ref = torch.lerp(a.float(), b.float(), 0.3).half()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- Remaining comparison ops ---

_COMPARISON_OPS = [
    pytest.param(NeOp, lambda a, b: a != b, "ne", marks=pytest.mark.smoke),
    pytest.param(GtOp, lambda a, b: a > b, "gt", marks=pytest.mark.smoke),
    pytest.param(LtOp, lambda a, b: a < b, "lt", marks=pytest.mark.smoke),
    pytest.param(GeOp, lambda a, b: a >= b, "ge", marks=pytest.mark.smoke),
    pytest.param(LeOp, lambda a, b: a <= b, "le", marks=pytest.mark.smoke),
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
    pytest.param(LogicalAndOp, lambda a, b: (a != 0) & (b != 0), "logical_and", marks=pytest.mark.smoke),
    pytest.param(LogicalOrOp, lambda a, b: (a != 0) | (b != 0), "logical_or", marks=pytest.mark.smoke),
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
    pytest.param(BitwiseAndOp, lambda a, b: a & b, "bitwise_and", marks=pytest.mark.smoke),
    pytest.param(BitwiseOrOp, lambda a, b: a | b, "bitwise_or", marks=pytest.mark.smoke),
    pytest.param(BitwiseXorOp, lambda a, b: a ^ b, "bitwise_xor", marks=pytest.mark.smoke),
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
    pytest.param(GeluAndMulOp, "gelu_and_mul", marks=pytest.mark.smoke),
    pytest.param(GeluTanhAndMulOp, "gelu_tanh_and_mul", marks=pytest.mark.smoke),
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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
