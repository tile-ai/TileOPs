import pytest
import torch

from tests.test_base import FixtureBase, TestBase, served_in_tree
from tileops.backend import BUILTIN
from tileops.kernels.gemm import (
    GemmCpAsyncKernel,
    GemmTmaKernel,
    GemmW4A16Kernel,
    GemvKernel,
    W4A16RepackKernel,
)
from tileops.kernels.gemm.dense import (
    GemmFp8BlockScaleKernel,
    _b_eviction,
    _bandwidth_autotune_grid,
    _gemm_pingpong_kernel,
)
from tileops.kernels.gemm.heuristics import (
    best_config,
    gemv_config,
    small_batch_config,
    small_m_splitk_config,
)
from tileops.kernels.gemm.w4a16 import _stage_meta_per_tile
from tileops.ops import GemmFp8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from workloads.gemm import (
    GemmFp8Workload,
    GemmW4A16Workload,
    GemmWorkload,
    quantize_weight_int4,
    repack_w4a16_weight,
)


class GemmTest(GemmWorkload, TestBase):
    pass


class GemmFp8Test(GemmFp8Workload, TestBase):
    pass


class GemmW4A16Test(GemmW4A16Workload, TestBase):
    pass


class GemmFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, k, dtype, trans_a, trans_b, tune",
            [
                pytest.param(
                    1024,
                    1024,
                    1024,
                    torch.float16,
                    False,
                    False,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-fp16-square",
                ),
                pytest.param(
                    1024,
                    1024,
                    1024,
                    torch.bfloat16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bf16-square",
                ),
                pytest.param(
                    1,
                    1024,
                    1024,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-trans-b-small-m",
                ),
                pytest.param(
                    128,
                    2112,
                    4096,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-nt-dense-ws",
                ),
                pytest.param(
                    256,
                    512,
                    128,
                    torch.float16,
                    True,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-tn-trans-a",
                ),
                pytest.param(
                    1,
                    7168,
                    16384,
                    torch.float16,
                    False,
                    True,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned-wide",
                ),
                pytest.param(
                    1,
                    18432,
                    7168,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned-wide-alt",
                ),
                pytest.param(
                    1024,
                    1,
                    1024,
                    torch.float16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-thin-n",
                ),
                pytest.param(
                    7168,
                    1,
                    16384,
                    torch.float16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned-thin-n",
                ),
                pytest.param(
                    18432,
                    1,
                    7168,
                    torch.float16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned-thin-n-alt",
                ),
                pytest.param(
                    1,
                    1024,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-trans-b-small-m",
                ),
                pytest.param(
                    1,
                    7168,
                    16384,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned-wide",
                ),
                pytest.param(
                    1,
                    18432,
                    7168,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned-wide-alt",
                ),
                pytest.param(
                    1024,
                    1,
                    1024,
                    torch.bfloat16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-thin-n",
                ),
                pytest.param(
                    7168,
                    1,
                    16384,
                    torch.bfloat16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned-thin-n",
                ),
                pytest.param(
                    18432,
                    1,
                    7168,
                    torch.bfloat16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned-thin-n-alt",
                ),
                pytest.param(
                    2,
                    2112,
                    7168,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-small-batch-m2",
                ),
                pytest.param(
                    4,
                    7168,
                    2048,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-small-m4-swap-ab",
                ),
                pytest.param(
                    4,
                    3000,
                    2048,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-small-m4-basic-ntail",
                ),
                pytest.param(
                    8,
                    2112,
                    7168,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-small-m8-splitk",
                ),
                pytest.param(
                    4,
                    5000,
                    2048,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-small-m4-swap-ab-ntail",
                ),
                pytest.param(
                    1536,
                    2112,
                    256,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-persistent",
                ),
                pytest.param(
                    1440,
                    2080,
                    256,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-mn-tail",
                ),
                pytest.param(
                    4096,
                    2112,
                    256,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-persistent",
                ),
                pytest.param(
                    4000,
                    2080,
                    256,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-mn-tail",
                ),
                pytest.param(
                    64,
                    7168,
                    2048,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-simple-plain",
                ),
                pytest.param(
                    128,
                    7168,
                    2048,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-simple-cluster",
                ),
            ],
        ),
    ]


class GemvBoundaryFixture(FixtureBase):
    """GEMV cases with non-aligned n/k to exercise partial-tile paths."""

    PARAMS = [
        (
            "n, k, dtype, tune",
            [
                # lhs_row: m=1, trans_b=True — non-aligned n
                pytest.param(3000, 1024, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(3000, 1024, torch.bfloat16, False, marks=pytest.mark.smoke),
                # lhs_row: non-aligned k
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                # rhs_col: n=1 — non-aligned m (mapped to gemv n param)
                pytest.param(3001, 1024, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


class GemmFp8Fixture(FixtureBase):
    PARAMS = [
        (
            "m, n, k, dtype, scale_mode, out_dtype, bias",
            [
                pytest.param(
                    128,
                    128,
                    128,
                    torch.float8_e4m3fn,
                    "per_tensor",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fp8-e4m3-per-tensor",
                ),
                pytest.param(
                    128,
                    256,
                    256,
                    torch.float8_e4m3fn,
                    "block128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fp8-e4m3-block128",
                ),
                pytest.param(
                    128,
                    128,
                    128,
                    torch.float8_e5m2,
                    "per_tensor",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fp8-e5m2-per-tensor",
                ),
                pytest.param(
                    4096,
                    256,
                    256,
                    torch.float8_e4m3fn,
                    "block128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128-large-m",
                ),
                pytest.param(
                    8,
                    256,
                    128,
                    torch.float8_e4m3fn,
                    "per_tensor",
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-per-tensor-small-m-bias",
                ),
                pytest.param(
                    1,
                    256,
                    128,
                    torch.float8_e4m3fn,
                    "per_tensor",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-per-tensor-gemv",
                ),
                pytest.param(
                    128,
                    256,
                    6144,
                    torch.float8_e4m3fn,
                    "block128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128-split-k",
                ),
                pytest.param(
                    128,
                    256,
                    6144,
                    torch.float8_e4m3fn,
                    "per_tensor",
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-per-tensor-split-k-bias",
                ),
            ],
        ),
    ]


class GemmW4A16Fixture(FixtureBase):
    PARAMS = [
        (
            "m, n, k, dtype",
            [
                pytest.param(
                    64,
                    64,
                    128,
                    torch.float16,
                    marks=pytest.mark.smoke,
                    id="smoke-w4a16-square",
                ),
                pytest.param(
                    128,
                    256,
                    256,
                    torch.float16,
                    marks=pytest.mark.smoke,
                    id="smoke-w4a16-rect",
                ),
                pytest.param(
                    1,
                    512,
                    512,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-m1",
                ),
                pytest.param(
                    16,
                    1024,
                    1024,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-m16",
                ),
                pytest.param(
                    1,
                    35,
                    384,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-decode-short-k-n-tail",
                ),
                pytest.param(
                    1,
                    65,
                    1024,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-decode-n-tail",
                ),
                pytest.param(
                    1,
                    96,
                    8192,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-decode-staged-k",
                ),
                pytest.param(
                    17,
                    65,
                    256,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-small-mn-tail",
                ),
            ],
        ),
    ]


@GemmFixture
def test_gemm(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool, tune: bool
) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b, tune=tune)
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
    op = GemmFp8FwdOp(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    if dtype != torch.float8_e4m3fn:
        with pytest.raises(ValueError, match="only supports torch.float8_e4m3fn"):
            op(*inputs)
        return
    test.check(op, *inputs, atol=2e-2, rtol=2e-2)


@GemmW4A16Fixture
def test_gemm_w4a16(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = GemmW4A16Test(m, n, k, dtype)
    op = GemmW4A16FwdOp()
    test.check(op, *test.gen_inputs(), atol=7e-2, rtol=5e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("k_index", [0, 1, 127, 128, 200, 383])
def test_gemm_w4a16_is_exact_on_a_basis_vector(k_index: int) -> None:
    """Check nibble, group, and zero-point indexing."""
    n, k = 35, 384
    torch.manual_seed(0)
    rows = torch.arange(n)[:, None]
    quantized = torch.randint(0, 16, (n, k))
    zero = ((3 * rows + torch.arange(k // 128)[None, :]) % 16).to(torch.uint8)
    scale = (
        0.03137 + torch.arange(n * (k // 128), dtype=torch.float32).reshape(n, -1) * 0.001147
    ).to(torch.float16)
    packed = (quantized[:, 0::2] | (quantized[:, 1::2] << 4)).to(torch.uint8)
    group = k_index // 128
    centered = quantized[:, k_index].float() - zero[:, group].float()
    expected = (centered * scale[:, group].float()).half()[None, :]
    activation = torch.zeros((1, k), device="cuda", dtype=torch.float16)
    activation[0, k_index] = 1
    prepacked = repack_w4a16_weight(packed)
    actual = GemmW4A16FwdOp()(activation, prepacked.cuda(), scale.cuda(), zero.cuda())
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


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
    fp16_scale_ulp = 2.0**-11
    torch.testing.assert_close(dequantized[0].max(), weight[0].max(), rtol=fp16_scale_ulp, atol=0)
    torch.testing.assert_close(dequantized[1].min(), weight[1].min(), rtol=fp16_scale_ulp, atol=0)


@pytest.mark.smoke
def test_gemm_fp8_block128_single_k_block_uses_block_kernel() -> None:
    test = GemmFp8Test(128, 256, 128, torch.float8_e4m3fn, "block128")
    op = GemmFp8FwdOp()
    test.check(op, *test.gen_inputs(), atol=2e-2, rtol=2e-2)
    if served_in_tree(op):
        assert op.kernel.__class__.__name__ == "GemmFp8BlockScaleKernel"


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            (4096, 2112, 7168),
            (128, 4),
            marks=pytest.mark.smoke,
            id="prefill-gate-up",
        ),
        pytest.param(
            (128, 7168, 2048),
            (64, 5),
            marks=pytest.mark.full,
            id="decode-grid-underfills",
        ),
        pytest.param(
            (8, 7168, 2048),
            (128, 6),
            marks=pytest.mark.full,
            id="tiny-m-widens-the-tile",
        ),
        pytest.param(
            (4096, 7168, 16384),
            (128, 3),
            marks=pytest.mark.full,
            id="long-k-shallow-ring",
        ),
    ],
)
def test_gemm_fp8_block128_default_config(
    shape: tuple[int, int, int], expected: tuple[int, int]
) -> None:
    kernel = GemmFp8BlockScaleKernel(
        *shape,
        dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
    )

    assert (kernel.config["block_n"], kernel.config["num_stages"]) == expected


@pytest.mark.smoke
def test_gemm_fp8_rejects_unsupported_scale_grids() -> None:
    m, n, k = 128, 256, 256
    test = GemmFp8Test(m, n, k, torch.float8_e4m3fn, "per_tensor")
    a, b, _, _ = test.gen_inputs()
    op = GemmFp8FwdOp()

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
    op = GemmFp8FwdOp(out_dtype=torch.bfloat16)
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
    op = GemmW4A16FwdOp()

    with pytest.raises(ValueError, match="weight_scale must have shape"):
        op(activation, packed_weight, weight_scale[:, :0], weight_zero)

    with pytest.raises(ValueError, match="packed_weight shape mismatch"):
        op(activation, packed_weight[:, :-1], weight_scale, weight_zero)


@pytest.mark.smoke
def test_gemm_w4a16_rejects_a_scale_outside_the_activation_dtype() -> None:
    test = GemmW4A16Test(64, 64, 128, torch.float16)
    activation, packed_weight, weight_scale, weight_zero = test.gen_inputs()
    op = GemmW4A16FwdOp()

    with pytest.raises(ValueError, match="weight_scale in the activation dtype"):
        op(activation, packed_weight, weight_scale.float(), weight_zero)


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmFwdOp(trans_a=False, trans_b=True, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n  # reuse fixture's n as the non-aligned m dimension
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmFwdOp(trans_a=False, trans_b=False, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@pytest.mark.smoke
def test_lhs_rows_band_dispatch() -> None:
    """``GemvKernel`` takes its ``lhs_rows`` band only at m == 2, on the n band swap_ab leaves it.

    One case per clause of ``GemvKernel.band_for``: m == 1 lands on the same class in
    its ``lhs_row`` band, m >= 3 and non-NT stay on the generic kernel (whose small-m
    band picks swap_ab / split-K / simple configs analytically), and so does any n wide
    enough for the operand-swapped grid. Selection only — no kernel is built, so this
    stays smoke-fast.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() not in (GemvKernel.supported_archs or []):
        pytest.skip("the bandwidth-bound band is SM90-only")

    nt = GemmFwdOp(trans_a=False, trans_b=True)
    fp = torch.float16
    two_rows = nt._call_spec(2, 2112, 7168, fp)
    assert nt.select_kernel(two_rows) is GemvKernel
    assert GemvKernel.band_for(two_rows) == "lhs_rows"
    assert nt.select_kernel(nt._call_spec(2, 7168, 2048, fp)) is GemmTmaKernel
    assert nt.select_kernel(nt._call_spec(3, 2112, 7168, fp)) is GemmTmaKernel
    one_row = nt._call_spec(1, 2112, 7168, fp)
    assert nt.select_kernel(one_row) is GemvKernel
    assert GemvKernel.band_for(one_row) == "lhs_row"
    nn = GemmFwdOp(trans_a=False, trans_b=False)
    assert nn.select_kernel(nn._call_spec(2, 2112, 7168, fp)) is GemmTmaKernel


@pytest.mark.smoke
def test_gemv_bands_build_their_own_body_and_config() -> None:
    """Each band states its own body shape and config band; the band is in the identity.

    The three bands share ``_gemm_small_batch_kernel``, so what separates them is what
    this asserts: how many rows the body contracts, which config rule picks its
    parameters, and that two bands never share a cache entry.
    """
    from tileops.utils import get_sm_count, get_sm_version

    if get_sm_version() not in (GemvKernel.supported_archs or []):
        pytest.skip("the bandwidth-bound band is SM90-only")

    fp = torch.float16
    nt = GemmFwdOp(trans_a=False, trans_b=True)
    nn = GemmFwdOp(trans_a=False, trans_b=False)

    rows_identity, _ = GemvKernel.entry_for(nt._call_spec(2, 2112, 7168, fp))
    row_identity, _ = GemvKernel.entry_for(nt._call_spec(1, 2112, 7168, fp))
    col_identity, _ = GemvKernel.entry_for(nn._call_spec(2112, 1, 7168, fp))
    assert rows_identity[0] == "lhs_rows"
    assert row_identity[0] == "lhs_row"
    assert col_identity[0] == "rhs_col"
    assert len({rows_identity, row_identity, col_identity}) == 3

    rows = GemvKernel("lhs_rows", 2, 2112, 7168, fp)
    row = GemvKernel("lhs_row", 1, 2112, 7168, fp)
    col = GemvKernel("rhs_col", 2112, 1, 7168, fp)
    assert (rows.out_len, row.out_len, col.out_len) == (2112, 2112, 2112)
    assert rows.default_config == small_batch_config(2112, 7168, get_sm_count())
    assert row.default_config == gemv_config(7168) == col.default_config
    assert rows.autotune_configs == _bandwidth_autotune_grid((32, 64, 128), (1, 2, 4), (2, 3, 4, 5))
    assert row.autotune_configs == _bandwidth_autotune_grid(
        (32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6)
    )
    assert col.autotune_configs == row.autotune_configs

    with pytest.raises(ValueError, match="serves bands"):
        GemvKernel("m2", 2, 2112, 7168, fp)


@pytest.mark.smoke
def test_explicit_structure_config_is_taken_verbatim() -> None:
    """A structure-flagged ``config=`` survives instead of being merged away.

    ``GemmTmaKernel`` has one config schema per structure, so the base's
    merge-over-``default_config`` would drop the caller's flag and keep their tile
    values — asking for ``coop2s`` on a shape the selector serves with ``coop2``
    yielded ``coop2`` at ``coop2s``' ``block_n``, which no measurement covers.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("the GEMM structures are SM90-only")

    assert GemmTmaKernel(1536, 2112, 256, torch.bfloat16, trans_b=True).config["block_n"] == 192

    requested = {"coop2s": True, "block_n": 64, "block_k": 128, "num_stages": 4}
    kernel = GemmTmaKernel(1536, 2112, 256, torch.bfloat16, trans_b=True, config=dict(requested))
    assert kernel.config == requested

    merged = GemmTmaKernel(512, 512, 512, torch.float16, config={"block_k": 32}).config
    assert merged["block_k"] == 32
    assert "block_m" in merged and "panel_size" in merged


@pytest.mark.smoke
def test_gemm_routes_tma_misaligned_shapes_to_the_pipelined_mainloop() -> None:
    """An unaligned innermost dimension leaves ``GemmTmaKernel``, which names the dim.

    Every ``GemmTmaKernel`` structure loads through TMA, which addresses the
    innermost dimension in 16-byte units; which logical dim that is follows the
    layout, so the same extent is served in one layout and refused in another.
    ``GemmCpAsyncKernel`` loads through ``cp.async`` and takes what is refused.
    Undeclared, these calls died inside TileLang's descriptor check instead
    ("Check failed: (result.supported) is false"), naming nothing to change.

    Routing only — the aligned shapes already run end to end in ``GemmFixture``.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("the TMA alignment region is SM90-specific")

    nt, nn = GemmFwdOp(trans_a=False, trans_b=True), GemmFwdOp(trans_a=False, trans_b=False)
    fp = torch.bfloat16

    misaligned_k = nt._call_spec(256, 512, 1001, fp)
    assert nt.select_kernel(misaligned_k) is GemmCpAsyncKernel
    assert "multiple of 8 elements" in GemmTmaKernel.refusal(misaligned_k)
    assert "k=1001" in GemmTmaKernel.refusal(misaligned_k)

    misaligned_n = nn._call_spec(256, 511, 1024, fp)
    assert nn.select_kernel(misaligned_n) is GemmCpAsyncKernel
    assert "n=511" in GemmTmaKernel.refusal(misaligned_n)

    assert nt.select_kernel(nt._call_spec(256, 511, 1024, fp)) is GemmTmaKernel
    assert nt.select_kernel(nt._call_spec(1, 512, 1001, fp)) is GemvKernel
    assert nt._call_spec(1, 512, 1001, fp).gemv_mode == "lhs_row"

    with pytest.raises(ValueError, match=r"cannot serve 256x512x1001"):
        GemmTmaKernel(256, 512, 1001, fp, trans_a=False, trans_b=True)


@pytest.mark.smoke
def test_gemm_revalidates_cached_signature_dtypes() -> None:
    """A changed ``b`` dtype reaches the op's gate, not TileLang's.

    ``forward`` skips validation when the input signature matches the previous
    call, so that signature has to carry every dtype the gate reads. It carried
    only ``a``'s: behind an fp16 warm-up a bf16 ``b`` went unvalidated into the
    fp16 kernel and failed inside TileLang. ``_validate_dtypes`` is the only
    dtype gate an op has, and it runs per call.
    """
    a = torch.randn(256, 128, dtype=torch.float16, device="cuda")
    b = torch.randn(512, 128, dtype=torch.float16, device="cuda")
    op = GemmFwdOp()
    op(a, b)

    with pytest.raises(ValueError, match=r"input 'b' has dtype torch.bfloat16"):
        op(a, b.to(torch.bfloat16))


@pytest.mark.smoke
def test_gemm_refuses_non_matrix_operands_before_building_anything() -> None:
    """A rank-3 operand is refused at the op boundary, not inside TileLang.

    ``GemmFwdOp``'s manifest inputs declare no ``shape``, so rank is stated
    nowhere but here (both sibling ops check it themselves). Without the check
    the trailing axis was dropped: ``(4, 16, 64)`` NT inferred ``m=4, n=4,
    k=16``, bound those on the op, compiled a kernel for them, and only then
    failed TileLang's argument check.
    """
    op = GemmFwdOp()
    a = torch.empty(4, 16, 64, dtype=torch.float16, device="cuda")

    with pytest.raises(ValueError, match=r"contracts two matrices.*a\.ndim=3"):
        op(a, a)
    assert not any(hasattr(op, dim) for dim in ("m", "n", "k"))
    assert not op.built_kernels("gemm")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "block_n, num_stages, stage_buf", [(256, 4, 1), (256, 4, 2), (256, 3, 4), (176, 4, 1)]
)
def test_coop2_epilogue_staging_matches_reference(
    block_n: int, num_stages: int, stage_buf: int
) -> None:
    """Shape coverage: the coop2 epilogue over one, two and four staging tiles.

    A ``block_n``-wide output tile always leaves as ``block_n / stage_n`` slices; what
    ``stage_buf`` sets is how many of them are in flight, from one (each TMA store waited
    on before the next slice is written) to all four (none waited on within a tile). Four
    tiles only fit alongside a three-deep mainloop ring, which is why the depth moves with
    the count. 176 is not a whole number of swizzle atoms, so it stages the whole tile
    through one buffer, and 3072 is not a multiple of it, so its last column of tiles is
    ragged.
    """
    m, n, k = 1536, 3072, 256
    test = GemmTest(m, n, k, torch.bfloat16, False, True)
    a, b = test.gen_inputs()
    kernel = GemmTmaKernel(
        m,
        n,
        k,
        torch.bfloat16,
        trans_a=False,
        trans_b=True,
        config={
            "coop2": True,
            "block_n": block_n,
            "block_k": 64,
            "num_stages": num_stages,
            "group_size_m": 16,
            "stage_n": 0,
            "stage_buf": stage_buf,
        },
    )
    torch.testing.assert_close(kernel.forward(a, b), torch.matmul(a, b.T), atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.smoke
def test_b_tile_eviction_hint_follows_the_m_tile_count() -> None:
    """Dispatch branch: the streaming-B hint is on at one or two M-tiles, off above."""
    assert _b_eviction(64, 64) == "evict_first"
    assert _b_eviction(128, 64) == "evict_first"
    assert _b_eviction(129, 64) is None
    assert _b_eviction(4096, 128) is None


@pytest.mark.smoke
@pytest.mark.parametrize("block_n, num_stages, stage_n", [(176, 5, 16), (176, 4, 88), (128, 6, 64)])
def test_pingpong_staging_matches_reference(block_n: int, num_stages: int, stage_n: int) -> None:
    """Shape coverage: the ping-pong epilogue at its shipped slice and two others.

    ``stage_n`` only has to be a TMA-legal divisor of ``block_n``; the slice hides
    under the other consumer's mainloop, so the plan takes whatever leaves room for
    the deepest ring. 2080 is not a multiple of 176 and 4000 not of 128, so the
    last row and column of tiles are ragged and store through TMA's clipping.
    """
    m, n, k = 4000, 2080, 256
    test = GemmTest(m, n, k, torch.bfloat16, False, True)
    a, b = test.gen_inputs()
    kernel = GemmTmaKernel(
        m,
        n,
        k,
        torch.bfloat16,
        trans_a=False,
        trans_b=True,
        config={
            "pingpong": True,
            "block_n": block_n,
            "block_k": 64,
            "num_stages": num_stages,
            "group_size_m": 16,
            "stage_n": stage_n,
        },
    )
    torch.testing.assert_close(kernel.forward(a, b), torch.matmul(a, b.T), atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.smoke
def test_pingpong_refuses_a_grid_its_second_consumer_cannot_share() -> None:
    """A grid of at most ``sm_count`` tiles gives the odd consumer nothing to do.

    TileLang then proves its TMA store dead and rejects the build with a message
    naming no shape; the builder refuses first, and the selector never offers
    ping-pong below two tiles per CTA.
    """
    build = _gemm_pingpong_kernel(1024, 2112, 256, False, True, "bfloat16", sm_count=132)
    with pytest.raises(ValueError, match="more than 132 tiles"):
        build(176, 64, 5, 16, 16)
    assert not best_config(1024, 2112, 256, False, True, 132, "NVIDIA H200").get("pingpong")


@pytest.mark.smoke
def test_structure_routing_matches_test_ids() -> None:
    """Each ``GemmFixture`` case reaches the structure its id names.

    The correctness cases above are the only coverage several structures have,
    and two of them (coop2) get there through ``heuristics.best_config``
    rather than a pin — so a change to the selector's scoring could silently
    route them elsewhere and leave ``coop2`` untested while every test still
    passes. This pins the mapping: when it fails, the correctness case named in
    the assertion needs a new shape, not a new expectation.

    Routing only — construction builds no JIT, so this stays smoke-fast.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("structure routing is SM90-specific")

    expected = [
        ("smoke-fp16-square", 1024, 1024, 1024, torch.float16, False, "coop2s"),
        ("smoke-bf16-square", 1024, 1024, 1024, torch.bfloat16, False, "coop2s"),
        ("full-bf16-coop2-persistent", 1536, 2112, 256, torch.bfloat16, True, "coop2"),
        ("full-bf16-coop2-mn-tail", 1440, 2080, 256, torch.bfloat16, True, "coop2"),
        ("full-bf16-pingpong-persistent", 4096, 2112, 256, torch.bfloat16, True, "pingpong"),
        ("full-bf16-pingpong-mn-tail", 4000, 2080, 256, torch.bfloat16, True, "pingpong"),
        ("full-bf16-simple-plain", 64, 7168, 2048, torch.bfloat16, True, "simple"),
        ("full-bf16-simple-cluster", 128, 7168, 2048, torch.bfloat16, True, "simple"),
        ("full-fp16-nt-dense-ws", 128, 2112, 4096, torch.float16, True, "coop2_splitk"),
        ("full-fp16-small-m4-swap-ab", 4, 7168, 2048, torch.float16, True, "swap_ab"),
        ("full-bf16-small-m4-swap-ab-ntail", 4, 5000, 2048, torch.bfloat16, True, "swap_ab"),
        ("full-fp16-small-m8-splitk", 8, 2112, 7168, torch.float16, True, "splitK4"),
        ("full-fp16-small-m4-basic-ntail", 4, 3000, 2048, torch.float16, True, "basic"),
    ]
    flags = GemmTmaKernel._STRUCTURE_FLAGS

    for test_id, m, n, k, dtype, trans_b, want in expected:
        op = GemmFwdOp(trans_a=False, trans_b=trans_b)
        call = op._call_spec(m, n, k, dtype)
        cls = op.select_kernel(call)
        assert cls is GemmTmaKernel, f"{test_id}: expected the generic kernel, got {cls.__name__}"
        _identity, build = cls.entry_for(call)
        config = build().config
        got = next((f for f in flags if config.get(f)), None)
        if got is None:
            split_k = config.get("split_k", 1)
            got = f"splitK{split_k}" if split_k > 1 else "basic"
        assert got == want, (
            f"{test_id} ({m}x{n}x{k}) now routes to {got}, not {want} — "
            f"that structure has lost its correctness coverage"
        )


@pytest.mark.smoke
def test_gemm_tma_kernel_tune_falls_back_to_default() -> None:
    """``GemmTmaKernel`` defines no ``autotune_configs``: ``tune=True`` must warn
    and fall back to ``default_config``.

    The in-tree tuner sweeps only the basic mainloop builder, so a silent
    basic-grid sweep would downgrade shapes whose default is a structure-
    flagged config (coop2 / split-K). Construction only — no JIT compile.
    """
    with pytest.warns(UserWarning, match="does not define autotune_configs"):
        kernel = GemmTmaKernel(
            4096, 4096, 7168, torch.float16, tune=True, trans_a=False, trans_b=True
        )
    assert kernel.config == kernel.default_config


@pytest.mark.smoke
def test_config_selector_declines_a_board_it_was_not_measured_on() -> None:
    """The ranking constants are achieved rates for one board.

    A board whose profile carries no ``gemm_selector`` section is not ranked:
    ``best_config`` returns ``None`` so the kernel takes its modal default,
    rather than a ranking measured somewhere else.
    """
    assert best_config(1024, 1024, 1024, False, False, 132, "NVIDIA H200") is not None
    assert best_config(1024, 1024, 1024, False, False, 132, "NVIDIA H20-3e") is None
    assert best_config(1024, 1024, 1024, False, False, 132, "no such board") is None


@pytest.mark.smoke
def test_small_m_splitk_config_selects_a_shape_band() -> None:
    assert small_m_splitk_config(32, 7168, 18432, 132, "NVIDIA H200") == {
        "block_m": 32,
        "block_n": 112,
        "block_k": 128,
        "num_stages": 2,
        "threads": 128,
        "split_k": 4,
    }
    assert small_m_splitk_config(64, 7168, 18432, 132, "NVIDIA H200") is None
    assert small_m_splitk_config(32, 7168, 2048, 132, "NVIDIA H200") is None
    assert small_m_splitk_config(32, 7168, 18432, 132, "NVIDIA H100") is None
    # The band is fitted on the board, not on the exact name CUDA reports for it,
    # and the name arrives both as CUDA spells it and through a call record.
    assert small_m_splitk_config(32, 7168, 18432, 132, "NVIDIA H200 NVL") is not None
    assert small_m_splitk_config(32, 7168, 18432, 132, "nvidia h200") is not None


@pytest.mark.smoke
def test_dense_splitk_interfaces_match_reference() -> None:
    m, n, k = 32, 112, 512
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    basic_config = {
        "block_m": m,
        "block_n": n,
        "block_k": 128,
        "num_stages": 2,
        "threads": 128,
        "split_k": 4,
    }
    actual = GemmCpAsyncKernel(m, n, k, torch.bfloat16, basic_config, trans_b=True)(a, b)
    torch.testing.assert_close(actual.float(), a.float() @ b.float().T, rtol=2e-2, atol=1e-1)

    gated_b = torch.randn(2 * n, k, dtype=torch.bfloat16, device="cuda")
    gated_config = {
        "block_m": 64,
        "block_n": n,
        "block_k": 128,
        "num_stages": 4,
        "panel_size": 8,
        "split_k": 4,
    }
    actual = GemmTmaKernel(
        m,
        2 * n,
        k,
        torch.bfloat16,
        gated_config,
        trans_b=True,
        activation="silu_and_mul",
    )(a, gated_b)
    gate, up = (a.float() @ gated_b.float().T).chunk(2, dim=1)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=1e-1)


@pytest.mark.smoke
def test_gemm_cp_async_kernel_k_tail_padding() -> None:
    """Non-16-aligned k rides on the backend zero-padding the K tail.

    Verified on real sm80 and sm89 hardware: any k with
    k * itemsize >= 4 compiles and matches the reference with
    block_k = 16 (the mma.sync floor); the K tail is zero-padded.
    """
    for k in (2, 8, 24):
        kern = GemmCpAsyncKernel(m=32, n=64, k=k, dtype=torch.bfloat16, trans_b=True)
        a = torch.randn(32, k, dtype=torch.bfloat16, device="cuda") * 0.05
        b = torch.randn(64, k, dtype=torch.bfloat16, device="cuda") * 0.05
        out = kern(a, b)
        ref = a.float() @ b.float().t()
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("m", [100, 257])
def test_gemm_w4a16_kernel_predicates_a_ragged_token_count(m: int) -> None:
    test = GemmW4A16Test(m, 1024, 512, torch.float16)
    activation, prepacked, scale, zero = test.gen_inputs()
    kernel = GemmW4A16Kernel(m, 1024, 512, torch.float16)
    torch.testing.assert_close(
        kernel(activation, prepacked, scale, zero),
        test.ref_program(activation, prepacked, scale, zero),
        atol=7e-2,
        rtol=5e-2,
    )


@pytest.mark.smoke
def test_gemm_w4a16_long_k_stages_metadata_per_tile() -> None:
    groups_at_crossover = 256  # 64 rows * 256 groups * 3 bytes = 48 KiB.
    assert not _stage_meta_per_tile(128, 512, 64, groups_at_crossover)
    assert _stage_meta_per_tile(128, 512, 64, groups_at_crossover + 1)
    assert not _stage_meta_per_tile(256, 512, 64, groups_at_crossover + 1)

    test = GemmW4A16Test(1, 64, 32896, torch.float16)
    activation, prepacked, scale, zero = test.gen_inputs()
    kernel = GemmW4A16Kernel(1, 64, 32896, torch.float16)
    assert _stage_meta_per_tile(
        kernel.config["threads"], kernel.config["block_k"], 64, 32896 // 128
    )
    torch.testing.assert_close(
        kernel(activation, prepacked, scale, zero),
        test.ref_program(activation, prepacked, scale, zero),
        atol=7e-2,
        rtol=5e-2,
    )


@pytest.mark.smoke
def test_gemm_w4a16_slices_k_only_where_the_grid_underfills() -> None:
    assert GemmW4A16Kernel(1, 1024, 8192, torch.float16).config["split_k"] > 1
    assert GemmW4A16Kernel(1, 8192, 8192, torch.float16).config["split_k"] == 1


@pytest.mark.smoke
@pytest.mark.parametrize("split_k", [2, 8])
def test_gemm_w4a16_sliced_k_matches_the_reference(split_k: int) -> None:
    """The fp32 partials reduce to what the whole K loop computes."""
    test = GemmW4A16Test(1, 1024, 8192, torch.float16)
    activation, prepacked, scale, zero = test.gen_inputs()
    base = GemmW4A16Kernel(1, 1024, 8192, torch.float16).config
    kernel = GemmW4A16Kernel(
        1,
        1024,
        8192,
        torch.float16,
        config={**base, "block_k": 256, "num_stages": 4, "split_k": split_k},
    )
    torch.testing.assert_close(
        kernel(activation, prepacked, scale, zero),
        test.ref_program(activation, prepacked, scale, zero),
        atol=7e-2,
        rtol=5e-2,
    )


@pytest.mark.smoke
def test_gemm_w4a16_autotune_keeps_composite_runtime_state() -> None:
    test = GemmW4A16Test(1, 1024, 8192, torch.float16)
    inputs = test.gen_inputs()
    op = GemmW4A16FwdOp(target=BUILTIN)
    expected = op(*inputs)
    kernel = op.kernel
    state = (dict(kernel.config), kernel.m_pad, kernel.kernel, kernel._reduce)

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        op.autotune()

    assert op.tune is False
    assert (kernel.config, kernel.m_pad, kernel.kernel, kernel._reduce) == state
    torch.testing.assert_close(op(*inputs), expected, atol=0, rtol=0)

    next_test = GemmW4A16Test(2, 1024, 8192, torch.float16)
    next_inputs = next_test.gen_inputs()
    torch.testing.assert_close(
        op(*next_inputs),
        next_test.ref_program(*next_inputs),
        atol=7e-2,
        rtol=5e-2,
    )

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        new_op = GemmW4A16FwdOp(tune=True)
    assert new_op.tune is False


@pytest.mark.smoke
def test_repack_w4a16_weight_permutes_nibbles_inside_a_step() -> None:
    packed = torch.randint(0, 256, (7, 256), dtype=torch.uint8)

    repacked = repack_w4a16_weight(packed)

    assert repacked.shape == packed.shape
    assert repacked.is_contiguous()

    def nibbles(tile: torch.Tensor) -> torch.Tensor:
        return torch.cat([tile & 0xF, tile >> 4], dim=1).sort(dim=1).values

    step = 64
    for start in range(0, packed.shape[1], step):
        torch.testing.assert_close(
            nibbles(repacked[:, start : start + step]),
            nibbles(packed[:, start : start + step]),
        )


@pytest.mark.smoke
@pytest.mark.parametrize(("n", "k"), [(64, 256), (1024, 512)])
def test_w4a16_repack_kernel_matches_the_reference(n: int, k: int) -> None:
    """The kernel and the tensor-expression repack agree bit for bit."""
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")

    actual = W4A16RepackKernel(n, k // 2)(packed)

    assert actual.dtype == torch.uint8
    assert actual.shape == packed.shape
    assert torch.equal(actual, repack_w4a16_weight(packed))


@pytest.mark.smoke
def test_gemm_w4a16_repack_feeds_forward() -> None:
    test = GemmW4A16Test(64, 1024, 512, torch.float16)
    activation, _, scale, zero = test.gen_inputs()
    packed = test.row_major_weight

    prepacked = GemmW4A16FwdOp.repack(packed)
    actual = GemmW4A16FwdOp()(activation, prepacked, scale, zero)

    torch.testing.assert_close(
        actual, test.ref_program(activation, packed, scale, zero), atol=7e-2, rtol=5e-2
    )


@pytest.mark.smoke
def test_gemm_w4a16_repack_refuses_a_partial_k_step() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        GemmW4A16FwdOp.repack(torch.zeros((8, 96), dtype=torch.uint8, device="cuda"))
