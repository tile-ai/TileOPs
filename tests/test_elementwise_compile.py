"""Tests for torch.compile compatibility of elementwise ops.

Covers 6 representative cases: relu (unary), add (binary), eq (comparison/bool output),
silu_and_mul (fused gated), abs (clamp-like), sign (where-like).
Validates that torch.compile(op, fullgraph=True) produces correct output.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare


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
    from tileops.ops.elementwise import ReluOp

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
    from tileops.ops.elementwise import AddOp

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
    from tileops.ops.elementwise import EqOp

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
    from tileops.ops.elementwise import SiluAndMulOp

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
    from tileops.ops.elementwise import AbsOp

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
    from tileops.ops.elementwise import SignOp

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
    from tileops.ops.elementwise import ReluOp

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
    from tileops.ops.elementwise import EqOp

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
    from tileops.ops.elementwise import SiluAndMulOp

    op = SiluAndMulOp(M=M, N=N, dtype=dtype)
    x = torch.randn(M, 2 * N, dtype=dtype, device="cuda")
    compiled_op = torch.compile(op, fullgraph=True)
    out = compiled_op(x)
    assert out.shape == (M, N), f"Shape mismatch: {out.shape} vs {(M, N)}"
    assert out.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
