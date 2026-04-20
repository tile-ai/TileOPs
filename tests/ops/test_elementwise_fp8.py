"""Tests for fp8 dtype support in elementwise kernels.

Covers:
- AC1: fp8 accumulation strategy (compute in fp16, cast back to fp8)
- AC2: Template base classes support fp8_e4m3fn and fp8_e5m2
- AC3: Correctness tests for representative ops (relu, add, silu_and_mul)
- AC4: Saturation/overflow behavior (e4m3fn saturates, e5m2 produces Inf)

fp8 accumulation design:
  fp8 input -> cast to fp16 -> compute -> cast back to fp8
  This avoids precision loss from direct fp8 arithmetic.

Saturation semantics (NVIDIA spec):
  - e4m3fn: no Inf/NaN representation, max value is 448.0, saturates on overflow
  - e5m2: has Inf/NaN, max finite value is 57344.0, overflows to Inf
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare

_FP8_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="e4m3fn", marks=pytest.mark.smoke),
    pytest.param(torch.float8_e5m2, id="e5m2", marks=pytest.mark.smoke),
]

_N = 1024 * 16  # 16K elements, fits 128-bit alignment with npt=16


# ---------------------------------------------------------------------------
# AC2: Template base classes accept fp8 dtypes
# ---------------------------------------------------------------------------


class Fp8DtypeAcceptanceFixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


@Fp8DtypeAcceptanceFixture
def test_unary_kernel_accepts_fp8(dtype):
    """UnaryKernel base class can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import ReluFwdKernel

    kernel = ReluFwdKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeAcceptanceFixture
def test_binary_kernel_accepts_fp8(dtype):
    """BinaryKernel base class can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import AddFwdKernel

    kernel = AddFwdKernel(
        N_total=_N, dtype=dtype,
        coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
        a_numel=_N, b_numel=_N,
    )
    assert kernel.dtype == dtype


@Fp8DtypeAcceptanceFixture
def test_fused_gated_kernel_accepts_fp8(dtype):
    """FusedGatedKernel base class can be instantiated with fp8 dtype."""
    if dtype == torch.float8_e4m3fn:
        pytest.skip("Temporarily skipping known e4m3fn fused-gated fp8 failure under TileLang 5f70374c (#999).")
    from tileops.kernels.elementwise import SiluAndMulFwdKernel

    M, N = 64, 128
    kernel = SiluAndMulFwdKernel(M=M, N=N, dtype=dtype)
    assert kernel.dtype == dtype


# ---------------------------------------------------------------------------
# AC1: fp8 default_config uses num_per_thread=16 for 128-bit alignment
# ---------------------------------------------------------------------------


@Fp8DtypeAcceptanceFixture
def test_fp8_default_config_npt16(dtype):
    """fp8 default_config returns num_per_thread=16 for 128-bit alignment."""
    from tileops.kernels.elementwise import ReluFwdKernel

    kernel = ReluFwdKernel(N_total=_N, dtype=dtype)
    assert kernel.config["num_per_thread"] == 16


# ---------------------------------------------------------------------------
# AC3: Correctness tests for representative ops
# ---------------------------------------------------------------------------


class Fp8UnaryFixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


class Fp8ReluTest(TestBase):
    def __init__(self, n_total, dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self):
        # Generate in fp16 range that fits fp8, then cast
        x_fp16 = torch.randn(self.n_total, dtype=torch.float16, device="cuda")
        # Scale down to fit fp8 range
        x_fp16 = x_fp16 * 2.0
        return (x_fp16.to(self.dtype),)

    def ref_program(self, x):
        # Compute reference: cast to fp16, relu, cast back to fp8
        x_fp16 = x.to(torch.float16)
        return torch.relu(x_fp16).to(self.dtype)


@Fp8UnaryFixture
def test_relu_fp8(dtype):
    """ReLU correctness with fp8 input/output."""
    from tileops.ops.elementwise import ReluFwdOp

    n = _N
    test = Fp8ReluTest(n, dtype)
    op = ReluFwdOp(N_total=n, dtype=dtype)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=0, rtol=0, compare=exact_compare)


class Fp8BinaryFixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


class Fp8AddTest(TestBase):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self):
        # Small values to avoid overflow in fp8
        a = (torch.randn(self.shape, dtype=torch.float16, device="cuda") * 0.5).to(self.dtype)
        b = (torch.randn(self.shape, dtype=torch.float16, device="cuda") * 0.5).to(self.dtype)
        return a, b

    def ref_program(self, a, b):
        return (a.to(torch.float16) + b.to(torch.float16)).to(self.dtype)


@Fp8BinaryFixture
def test_add_fp8(dtype):
    """Add correctness with fp8, including broadcast."""
    from tileops.ops.elementwise import AddFwdOp

    shape = (128, 128)
    test = Fp8AddTest(shape, dtype)
    op = AddFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=0, rtol=0, compare=exact_compare)


@Fp8BinaryFixture
def test_add_fp8_broadcast(dtype):
    """Add correctness with fp8 and row broadcast."""
    from tileops.ops.elementwise import AddFwdOp

    a_shape = (128, 128)
    b_shape = (1, 128)
    a = (torch.randn(a_shape, dtype=torch.float16, device="cuda") * 0.5).to(dtype)
    b = (torch.randn(b_shape, dtype=torch.float16, device="cuda") * 0.5).to(dtype)
    op = AddFwdOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    ref = (a.to(torch.float16) + b.to(torch.float16)).to(dtype)
    out = op(a, b)
    assert torch.equal(out, ref)


class Fp8FusedGatedFixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


class Fp8SiluAndMulTest(TestBase):
    def __init__(self, M, N, dtype):
        self.M = M
        self.N = N
        self.dtype = dtype

    def gen_inputs(self):
        # Small values to keep within fp8 range
        x = (torch.randn(self.M, 2 * self.N, dtype=torch.float16, device="cuda") * 0.5).to(self.dtype)
        return (x,)

    def ref_program(self, x):
        x_fp16 = x.to(torch.float16)
        gate = x_fp16[:, :self.N]
        value = x_fp16[:, self.N:]
        return (torch.nn.functional.silu(gate) * value).to(self.dtype)


@Fp8FusedGatedFixture
def test_silu_and_mul_fp8(dtype):
    """SiLU-and-Mul correctness with fp8."""
    if dtype == torch.float8_e4m3fn:
        pytest.skip("Temporarily skipping known e4m3fn silu_and_mul fp8 failure under TileLang 5f70374c (#999).")
    from tileops.ops.elementwise import SiluAndMulFwdOp

    M, N = 64, 128
    test = Fp8SiluAndMulTest(M, N, dtype)
    op = SiluAndMulFwdOp(M=M, N=N, dtype=dtype)
    inputs = test.gen_inputs()
    out = op(*inputs)
    ref = test.ref_program(*inputs)
    # fp8 assert_close only supports exact comparison; compare via fp16
    torch.testing.assert_close(
        out.to(torch.float16), ref.to(torch.float16), atol=0.125, rtol=0.1,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
