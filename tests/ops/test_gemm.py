import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GemmFp8Op, GemmOp, GemmW4A16Op
from workloads.gemm import GemmFp8Workload, GemmW4A16Workload, GemmWorkload, quantize_weight_int4


class GemmTest(GemmWorkload, TestBase):
    pass


class GemmFp8Test(GemmFp8Workload, TestBase):
    pass


class GemmW4A16Test(GemmW4A16Workload, TestBase):
    pass


class GemmFixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, trans_a, trans_b, tune", [
            pytest.param(
                1024, 1024, 1024, torch.float16, False, False, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fp16-square",
            ),
            pytest.param(
                1024, 1024, 1024, torch.bfloat16, False, False, False,
                marks=pytest.mark.smoke,
                id="smoke-bf16-square",
            ),
            pytest.param(
                1, 1024, 1024, torch.float16, False, True, False,
                marks=pytest.mark.full,
                id="full-fp16-trans-b-small-m",
            ),
            pytest.param(
                128, 2112, 4096, torch.float16, False, True, False,
                marks=pytest.mark.full,
                id="full-fp16-nt-dense-ws",
            ),
            pytest.param(
                1, 7168, 16384, torch.float16, False, True, True,
                marks=pytest.mark.full,
                id="full-fp16-tuned-wide",
            ),
            pytest.param(
                1, 18432, 7168, torch.float16, False, True, False,
                marks=pytest.mark.full,
                id="full-fp16-tuned-wide-alt",
            ),
            pytest.param(
                1024, 1, 1024, torch.float16, False, False, False,
                marks=pytest.mark.full,
                id="full-fp16-thin-n",
            ),
            pytest.param(
                7168, 1, 16384, torch.float16, False, False, False,
                marks=pytest.mark.full,
                id="full-fp16-tuned-thin-n",
            ),
            pytest.param(
                18432, 1, 7168, torch.float16, False, False, False,
                marks=pytest.mark.full,
                id="full-fp16-tuned-thin-n-alt",
            ),
            pytest.param(
                1, 1024, 1024, torch.bfloat16, False, True, False,
                marks=pytest.mark.full,
                id="full-bf16-trans-b-small-m",
            ),
            pytest.param(
                1, 7168, 16384, torch.bfloat16, False, True, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-wide",
            ),
            pytest.param(
                1, 18432, 7168, torch.bfloat16, False, True, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-wide-alt",
            ),
            pytest.param(
                1024, 1, 1024, torch.bfloat16, False, False, False,
                marks=pytest.mark.full,
                id="full-bf16-thin-n",
            ),
            pytest.param(
                7168, 1, 16384, torch.bfloat16, False, False, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-thin-n",
            ),
            pytest.param(
                18432, 1, 7168, torch.bfloat16, False, False, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-thin-n-alt",
            ),
        ]),
    ]


class GemvBoundaryFixture(FixtureBase):
    """GEMV cases with non-aligned n/k to exercise partial-tile paths."""
    PARAMS = [
        ("n, k, dtype, tune", [
            # lhs_row: m=1, trans_b=True — non-aligned n
            pytest.param(3000, 1024, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(3000, 1024, torch.bfloat16, False, marks=pytest.mark.smoke),
            # lhs_row: non-aligned k
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            # rhs_col: n=1 — non-aligned m (mapped to gemv n param)
            pytest.param(3001, 1024, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


class GemmFp8Fixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, scale_mode, out_dtype, bias", [
            pytest.param(
                128, 128, 128, torch.float8_e4m3fn, "per_tensor", torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp8-e4m3-per-tensor",
            ),
            pytest.param(
                128, 256, 256, torch.float8_e4m3fn, "block128", torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp8-e4m3-block128",
            ),
            pytest.param(
                128, 128, 128, torch.float8_e5m2, "per_tensor", torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp8-e5m2-per-tensor",
            ),
            pytest.param(
                4096, 256, 256, torch.float8_e4m3fn, "block128", torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-fp8-e4m3-block128-large-m",
            ),
            pytest.param(
                8, 256, 128, torch.float8_e4m3fn, "per_tensor", torch.float16, True,
                marks=pytest.mark.full,
                id="full-fp8-e4m3-per-tensor-small-m-bias",
            ),
            pytest.param(
                1, 256, 128, torch.float8_e4m3fn, "per_tensor", torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-fp8-e4m3-per-tensor-gemv",
            ),
        ]),
    ]


class GemmW4A16Fixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype", [
            pytest.param(
                64, 64, 128, torch.float16,
                marks=pytest.mark.smoke,
                id="smoke-w4a16-square",
            ),
            pytest.param(
                128, 256, 256, torch.float16,
                marks=pytest.mark.smoke,
                id="smoke-w4a16-rect",
            ),
            pytest.param(
                1, 512, 512, torch.float16,
                marks=pytest.mark.full,
                id="full-w4a16-m1",
            ),
            pytest.param(
                16, 1024, 1024, torch.float16,
                marks=pytest.mark.full,
                id="full-w4a16-m16",
            ),
        ]),
    ]


@GemmFixture
def test_gemm(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
              tune: bool) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmOp(trans_a=trans_a, trans_b=trans_b, tune=tune)
    if dtype == torch.float16:
        # Only GEMV sums in a different order than cuBLAS; there cancellation
        # leaves atol alone to carry the reduction error, 3.3e-3 at K=16384.
        gemv = not trans_a and ((m == 1 and trans_b) or (n == 1 and not trans_b))
        atol = 1e-3 * max(1.0, k / 2048) if gemv else 1e-3
        tolerances = {"atol": atol, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)

@GemmFp8Fixture
def test_gemm_fp8(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    scale_mode: str,
    out_dtype: torch.dtype,
    bias: bool,
) -> None:
    test = GemmFp8Test(m, n, k, dtype, scale_mode, out_dtype=out_dtype, bias=bias)
    op = GemmFp8Op(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    if dtype != torch.float8_e4m3fn:
        with pytest.raises(ValueError, match="only supports torch.float8_e4m3fn"):
            op(*inputs)
        return
    test.check(op, *inputs, atol=2e-2, rtol=2e-2)


@GemmW4A16Fixture
def test_gemm_w4a16(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = GemmW4A16Test(m, n, k, dtype)
    op = GemmW4A16Op()
    test.check(op, *test.gen_inputs(), atol=7e-2, rtol=5e-2)


@pytest.mark.smoke
def test_quantize_weight_int4_keeps_one_sided_groups_in_range() -> None:
    weight = torch.tensor(
        [
            [0.25, 0.50, 0.75, 1.00],
            [-1.00, -0.75, -0.50, -0.25],
        ],
        dtype=torch.float32,
    )

    _, scale, zero, dequantized = quantize_weight_int4(weight, group_size=4)

    assert torch.equal(zero, torch.tensor([[0], [15]], dtype=torch.uint8))
    assert torch.all(scale > 0)
    torch.testing.assert_close(dequantized[0].max(), weight[0].max())
    torch.testing.assert_close(dequantized[1].min(), weight[1].min())


@pytest.mark.smoke
def test_quantize_weight_int4_preserves_nibble_order_across_group_boundary() -> None:
    weight = torch.tensor(
        [[0.0, 0.2, 0.4, 0.6, 0.0, -0.2, -0.4, -0.6]],
        dtype=torch.float32,
    )

    packed, scale, zero, dequantized = quantize_weight_int4(weight, group_size=4)

    assert torch.equal(packed, torch.tensor([[0x50, 0xFA, 0xAF, 0x05]], dtype=torch.uint8))
    assert torch.equal(zero, torch.tensor([[0, 15]], dtype=torch.uint8))
    torch.testing.assert_close(scale, torch.full((1, 2), 0.04))
    torch.testing.assert_close(dequantized, weight)


@pytest.mark.smoke
def test_gemm_fp8_block128_single_k_block_uses_block_kernel() -> None:
    test = GemmFp8Test(128, 256, 128, torch.float8_e4m3fn, "block128")
    op = GemmFp8Op()
    test.check(op, *test.gen_inputs(), atol=2e-2, rtol=2e-2)
    assert op.kernel.__class__.__name__ == "GemmFp8BlockScaledKernel"


@pytest.mark.smoke
def test_gemm_fp8_rejects_unsupported_scale_grids() -> None:
    m, n, k = 128, 256, 256
    test = GemmFp8Test(m, n, k, torch.float8_e4m3fn, "per_tensor")
    a, b, _, _ = test.gen_inputs()
    op = GemmFp8Op()

    with pytest.raises(ValueError, match="supports scale shapes"):
        op(
            a,
            b,
            torch.ones((1, k // 128), device="cuda", dtype=torch.float32),
            torch.ones((1, k // 128), device="cuda", dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="supports scale shapes"):
        op(
            a,
            b,
            torch.ones((m, 1), device="cuda", dtype=torch.float32),
            torch.ones((n, 1), device="cuda", dtype=torch.float32),
        )


@pytest.mark.smoke
def test_gemm_fp8_revalidates_cached_signature_dtypes() -> None:
    test = GemmFp8Test(
        128,
        128,
        128,
        torch.float8_e4m3fn,
        "per_tensor",
        out_dtype=torch.bfloat16,
        bias=True,
    )
    a, b, scale_a, scale_b, bias = test.gen_inputs()
    op = GemmFp8Op(out_dtype=torch.bfloat16)
    op(a, b, scale_a, scale_b, bias)

    with pytest.raises(ValueError, match="expects b dtype"):
        op(a, b.to(torch.float8_e5m2), scale_a, scale_b, bias)

    with pytest.raises(ValueError, match="scale_a and scale_b"):
        op(a, b, scale_a.to(torch.float16), scale_b, bias)

    with pytest.raises(ValueError, match="expects bias dtype"):
        op(a, b, scale_a, scale_b, bias.to(torch.float16))


@pytest.mark.smoke
def test_gemm_w4a16_rejects_invalid_metadata_shapes() -> None:
    test = GemmW4A16Test(64, 64, 128, torch.float16)
    activation, packed_weight, weight_scale, weight_zero = test.gen_inputs()
    op = GemmW4A16Op()

    with pytest.raises(ValueError, match="weight_scale must have shape"):
        op(activation, packed_weight, weight_scale[:, :0], weight_zero)

    with pytest.raises(ValueError, match="packed_weight shape mismatch"):
        op(activation, packed_weight[:, :-1], weight_scale, weight_zero)


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmOp(trans_a=False, trans_b=True, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n  # reuse fixture's n as the non-aligned m dimension
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmOp(trans_a=False, trans_b=False, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
