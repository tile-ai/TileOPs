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
    from tileops.kernels.elementwise import ReluKernel

    kernel = ReluKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeAcceptanceFixture
def test_binary_kernel_accepts_fp8(dtype):
    """BinaryKernel base class can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import AddKernel

    kernel = AddKernel(
        N_total=_N, dtype=dtype,
        coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
        a_numel=_N, b_numel=_N,
    )
    assert kernel.dtype == dtype


@Fp8DtypeAcceptanceFixture
def test_fused_gated_kernel_accepts_fp8(dtype):
    """FusedGatedKernel base class can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import SiluAndMulKernel

    M, N = 64, 128
    kernel = SiluAndMulKernel(M=M, N=N, dtype=dtype)
    assert kernel.dtype == dtype


# ---------------------------------------------------------------------------
# AC1: fp8 default_config uses num_per_thread=16 for 128-bit alignment
# ---------------------------------------------------------------------------


@Fp8DtypeAcceptanceFixture
def test_fp8_default_config_npt16(dtype):
    """fp8 default_config returns num_per_thread=16 for 128-bit alignment."""
    from tileops.kernels.elementwise import ReluKernel

    kernel = ReluKernel(N_total=_N, dtype=dtype)
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
    from tileops.ops.elementwise import ReluOp

    n = _N
    test = Fp8ReluTest(n, dtype)
    op = ReluOp(N_total=n, dtype=dtype)
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
    from tileops.ops.elementwise import AddOp

    shape = (128, 128)
    test = Fp8AddTest(shape, dtype)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=0, rtol=0, compare=exact_compare)


@Fp8BinaryFixture
def test_add_fp8_broadcast(dtype):
    """Add correctness with fp8 and row broadcast."""
    from tileops.ops.elementwise import AddOp

    a_shape = (128, 128)
    b_shape = (1, 128)
    a = (torch.randn(a_shape, dtype=torch.float16, device="cuda") * 0.5).to(dtype)
    b = (torch.randn(b_shape, dtype=torch.float16, device="cuda") * 0.5).to(dtype)
    op = AddOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
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
    from tileops.ops.elementwise import SiluAndMulOp

    M, N = 64, 128
    test = Fp8SiluAndMulTest(M, N, dtype)
    op = SiluAndMulOp(M=M, N=N, dtype=dtype)
    inputs = test.gen_inputs()
    out = op(*inputs)
    ref = test.ref_program(*inputs)
    # fp8 assert_close only supports exact comparison; compare via fp16
    torch.testing.assert_close(
        out.to(torch.float16), ref.to(torch.float16), atol=0.125, rtol=0.1,
    )


# ---------------------------------------------------------------------------
# AC4: Saturation/overflow behavior
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_e4m3fn_saturates_on_overflow():
    """e4m3fn saturates to max value (448.0) instead of producing Inf.

    Per NVIDIA spec, e4m3fn has no Inf representation. Values exceeding
    the max representable magnitude clamp to +/-448.0.
    """
    from tileops.ops.elementwise import AddOp

    # Create values near e4m3fn max (448.0)
    n = 1024
    a_shape = (n,)
    # Use values that would overflow when added
    a_fp16 = torch.full((n,), 400.0, dtype=torch.float16, device="cuda")
    b_fp16 = torch.full((n,), 400.0, dtype=torch.float16, device="cuda")
    dtype = torch.float8_e4m3fn
    a = a_fp16.to(dtype)
    b = b_fp16.to(dtype)
    op = AddOp(a_shape=a_shape, b_shape=a_shape, dtype=dtype)
    out = op(a, b)
    out_fp32 = out.to(torch.float32)
    # Result should be 448.0 (saturated max), not Inf
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    assert torch.all(out_fp32 <= e4m3_max), (
        f"e4m3fn output should saturate to <= {e4m3_max}, got max={out_fp32.max().item()}"
    )
    assert not torch.any(torch.isinf(out_fp32)), "e4m3fn should not produce Inf"
    assert not torch.any(torch.isnan(out_fp32)), "e4m3fn should not produce NaN"


@pytest.mark.smoke
def test_e5m2_overflow_produces_inf():
    """e5m2 produces Inf on overflow (has Inf/NaN representation).

    Per NVIDIA spec, e5m2 follows IEEE-like overflow semantics.
    The kernel produces fp16 output to preserve Inf, then the Op layer
    casts to e5m2 via PyTorch's non-saturating conversion.
    """
    from tileops.ops.elementwise import AddOp

    n = 1024
    a_shape = (n,)
    dtype = torch.float8_e5m2
    # Use values near e5m2 max (57344.0) that will overflow when added
    a_fp16 = torch.full((n,), 40000.0, dtype=torch.float16, device="cuda")
    b_fp16 = torch.full((n,), 40000.0, dtype=torch.float16, device="cuda")
    a = a_fp16.to(dtype)
    b = b_fp16.to(dtype)
    op = AddOp(a_shape=a_shape, b_shape=a_shape, dtype=dtype)
    out = op(a, b)
    out_fp32 = out.to(torch.float32)
    # e5m2 supports Inf, so overflowed values should be Inf
    assert torch.any(torch.isinf(out_fp32)), (
        f"e5m2 should produce Inf on overflow, got max={out_fp32.max().item()}"
    )


@pytest.mark.smoke
def test_e5m2_exp_overflow_produces_inf():
    """e5m2 exp(large) should produce Inf, matching PyTorch reference."""
    from tileops.ops.elementwise import ExpOp

    n = 1024
    dtype = torch.float8_e5m2
    # exp(16) in fp16 overflows to Inf
    x_fp16 = torch.full((n,), 16.0, dtype=torch.float16, device="cuda")
    x = x_fp16.to(dtype)
    op = ExpOp(N_total=n, dtype=dtype)
    out = op(x)
    ref = torch.exp(x.to(torch.float16)).to(dtype)
    assert torch.equal(out, ref), (
        f"e5m2 exp overflow should match reference. "
        f"Got {out.to(torch.float32)[:3]}, expected {ref.to(torch.float32)[:3]}"
    )


@pytest.mark.smoke
def test_e5m2_div_by_zero_produces_inf():
    """e5m2 1/0 should produce Inf, matching PyTorch reference."""
    from tileops.ops.elementwise import DivOp

    n = 1024
    dtype = torch.float8_e5m2
    a_shape = (n,)
    a = torch.ones(n, dtype=torch.float16, device="cuda").to(dtype)
    b = torch.zeros(n, dtype=torch.float16, device="cuda").to(dtype)
    op = DivOp(a_shape=a_shape, b_shape=a_shape, dtype=dtype)
    out = op(a, b)
    out_fp32 = out.to(torch.float32)
    assert torch.all(torch.isinf(out_fp32)), (
        f"e5m2 1/0 should produce Inf, got {out_fp32[:3]}"
    )


@pytest.mark.smoke
def test_e5m2_log_zero_produces_neg_inf():
    """e5m2 log(0) should produce -Inf, matching PyTorch reference."""
    from tileops.ops.elementwise import LogOp

    n = 1024
    dtype = torch.float8_e5m2
    x = torch.zeros(n, dtype=torch.float16, device="cuda").to(dtype)
    op = LogOp(N_total=n, dtype=dtype)
    out = op(x)
    ref = torch.log(x.to(torch.float16)).to(dtype)
    assert torch.equal(out, ref), (
        f"e5m2 log(0) should produce -Inf. "
        f"Got {out.to(torch.float32)[:3]}, expected {ref.to(torch.float32)[:3]}"
    )


@pytest.mark.smoke
def test_e4m3fn_exp_overflow_saturates():
    """e4m3fn exp(large) should saturate to 448.0, not produce Inf."""
    from tileops.ops.elementwise import ExpOp

    n = 1024
    dtype = torch.float8_e4m3fn
    x_fp16 = torch.full((n,), 10.0, dtype=torch.float16, device="cuda")
    x = x_fp16.to(dtype)
    op = ExpOp(N_total=n, dtype=dtype)
    out = op(x)
    out_fp32 = out.to(torch.float32)
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    assert torch.all(out_fp32 <= e4m3_max), (
        f"e4m3fn exp should saturate to <= {e4m3_max}, got max={out_fp32.max().item()}"
    )
    assert not torch.any(torch.isinf(out_fp32)), "e4m3fn should not produce Inf"


# ---------------------------------------------------------------------------
# AC1: fp8 accumulation strategy -- verify computation happens in fp16
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_fp8_accumulation_in_higher_precision():
    """Verify fp8 computation uses fp16 accumulation for non-trivial ops.

    SiLU involves sigmoid which requires higher precision. The result
    should match fp16 computation cast back to fp8, not direct fp8 arithmetic.
    """
    from tileops.ops.elementwise import SiluOp

    n = _N
    dtype = torch.float8_e4m3fn
    # Values in a range where fp8 precision matters
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = SiluOp(N_total=n, dtype=dtype)
    out = op(x)
    # Reference: compute in fp16, cast back to fp8
    ref = torch.nn.functional.silu(x.to(torch.float16)).to(dtype)
    assert torch.equal(out, ref), (
        "fp8 SiLU should match fp16-accumulated reference"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
