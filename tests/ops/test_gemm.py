import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.gemm import GemmKernel, SmallBatchGemmKernel
from tileops.kernels.gemm.dense import GemmFp8BlockScaledKernel
from tileops.ops import GemmFp8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from workloads.gemm import GemmFp8Workload, GemmW4A16Workload, GemmWorkload, quantize_weight_int4


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
                # small-batch NT kernel-mode: tier-1 (m<=4) on both decode regimes +
                # both dtypes + a non-power-of-two n tail; tier-2 (5<=m<=8) on a
                # severely-underfilled grid.
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
                # swap_ab band (n wide enough for the operand-swapped grid): bf16,
                # a non-block_nn-divisible n exercising the predicated transpose
                # epilogue, and the deep-ring stage count of the mid-CTA bucket.
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
                # coop2 (persistent 2-consumer): the structure serving every dense
                # prefill row. Both cases select block_n=192 / num_stages=5 /
                # stage_n=96 — field-for-field the shipped prefill-gate-up pin — on
                # a K short enough to keep the test cheap.
                #   no tails: every tile takes the chunked TMA-store epilogue.
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
                #   m tail 32 + n tail 160: the last M tile leaves consumer 0 with
                #   32 rows (predicated scalar store) and consumer 1 with none at
                #   all (its ``arows > 0`` guard), while interior tiles still take
                #   the TMA path.
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
                # simple (non-warp-specialized pipelined): reachable only from a
                # ``_TUNED_CONFIGS`` pin, so these two shapes are the only way to
                # cover it. m=64 takes the plain grid, m=128 the cluster_m=2
                # ClusterKernel path.
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
def test_gemm_fp8_block128_single_k_block_uses_block_kernel() -> None:
    test = GemmFp8Test(128, 256, 128, torch.float8_e4m3fn, "block128")
    op = GemmFp8FwdOp()
    test.check(op, *test.gen_inputs(), atol=2e-2, rtol=2e-2)
    assert op.kernel.__class__.__name__ == "GemmFp8BlockScaledKernel"


@pytest.mark.parametrize(
    ("shape", "expected_tile"),
    [
        pytest.param(
            (4096, 2112, 7168),
            (128, 64),
            marks=pytest.mark.smoke,
            id="prefill-gate-up",
        ),
        pytest.param(
            (4096, 4096, 7168),
            (64, 128),
            marks=pytest.mark.full,
            id="prefill-attn-proj",
        ),
        pytest.param(
            (4096, 7168, 2048),
            (128, 128),
            marks=pytest.mark.full,
            id="prefill-down-default",
        ),
    ],
)
def test_gemm_fp8_block128_default_config(
    shape: tuple[int, int, int], expected_tile: tuple[int, int]
) -> None:
    kernel = GemmFp8BlockScaledKernel(
        *shape,
        dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
    )

    assert (kernel.config["block_m"], kernel.config["block_n"]) == expected_tile


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
def test_small_batch_dispatch() -> None:
    """small_batch dispatches only at m == 2 on grids no generic path fills.

    m == 1 stays on gemv, m >= 3 / full grids / non-NT stay on the generic
    kernel (whose small-m band picks swap_ab / split-K / simple configs
    analytically). Dispatch only — ``_get_kernel`` constructs kernel objects
    without triggering a JIT compile (that happens on first forward), so this
    stays smoke-fast.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() not in (SmallBatchGemmKernel.supported_archs or []):
        pytest.skip("small_batch kernel-mode is SM90-only")

    op = GemmFwdOp(trans_a=False, trans_b=True)  # NT
    # m == 2, and n too narrow for the operand-swapped grid (2112 / 64 = 33
    # CTAs): the bandwidth kernel's remaining band.
    assert op._get_kernel(2, 2112, 7168, torch.float16)[0] == "small_batch"
    # m == 2 but n wide enough for swap_ab (7168 / 64 = 112 CTAs, 4096 -> 64):
    # the generic kernel streams the same weights on a wider grid and wins.
    assert op._get_kernel(2, 7168, 2048, torch.float16)[0] == "gemm"
    assert op._get_kernel(2, 4096, 7168, torch.float16)[0] == "gemm"
    # m >= 3: the split-K / simple generic configs overtake the bandwidth
    # kernel (its per-element CUDA-core cost scales with m).
    assert op._get_kernel(3, 2112, 7168, torch.float16)[0] == "gemm"
    assert op._get_kernel(4, 7168, 2048, torch.float16)[0] == "gemm"
    assert op._get_kernel(8, 2112, 7168, torch.float16)[0] == "gemm"
    # boundaries
    assert op._get_kernel(1, 2112, 7168, torch.float16)[0] == "lhs_row"  # gemv
    from tileops.kernels.gemm.heuristics import TINY_M_BLOCK_N
    from tileops.utils import get_sm_count

    wide_n = get_sm_count() * TINY_M_BLOCK_N  # generic fills the GPU
    assert op._get_kernel(2, wide_n, 2048, torch.float16)[0] == "gemm"
    op_nn = GemmFwdOp(trans_a=False, trans_b=False)  # non-NT
    assert op_nn._get_kernel(2, 2112, 7168, torch.float16)[0] == "gemm"


@pytest.mark.smoke
def test_explicit_structure_config_is_taken_verbatim() -> None:
    """A structure-flagged ``config=`` survives instead of being merged away.

    ``GemmKernel`` has one config schema per structure, so the base's
    merge-over-``default_config`` would drop the caller's flag and keep their tile
    values — asking for ``coop2s`` on a shape the selector serves with ``coop2``
    yielded ``coop2`` at ``coop2s``' ``block_n``, which no measurement covers.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("the GEMM structures are SM90-only")

    # 1536x2112x256 NT is served by coop2 (block_n=192) analytically.
    assert GemmKernel(1536, 2112, 256, torch.bfloat16, trans_b=True).config["block_n"] == 192

    requested = {"coop2s": True, "block_n": 64, "block_k": 128, "num_stages": 4}
    kernel = GemmKernel(1536, 2112, 256, torch.bfloat16, trans_b=True, config=dict(requested))
    assert kernel.config == requested

    # A config without a structure flag still merges over the default.
    merged = GemmKernel(512, 512, 512, torch.float16, config={"block_k": 32}).config
    assert merged["block_k"] == 32
    assert "block_m" in merged and "panel_size" in merged


@pytest.mark.smoke
def test_registered_wrapped_ops_keep_their_contracts() -> None:
    """The two ``top::`` ops stay callable at the ranks they advertise.

    Nothing in-tree calls either — ``forward`` builds the JIT directly and these
    exist for ``torch.compile`` — so their bodies rot unwatched. Both were
    changed here: the GEMM op gained ``panel_size`` / ``split_k`` and a split-K
    branch, and the GEMV op now delegates to the shared ``[m, k] -> [m, n]``
    small-batch body while still advertising ``a[k] -> c[n]``. That rank
    adaptation was in fact missing and silent until this test existed.
    """
    from torch.library import opcheck

    from tileops.kernels.gemm import _gemm_wrapped_kernel, _gemv_wrapped_kernel
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("both bodies are SM90-only")

    m, n, k = 128, 512, 1024
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ b.float().T

    for split_k in (1, 4):  # split_k > 1 takes the branch added here
        args = (m, n, k, False, True, "bfloat16", 64, 128, 128, 4, 16, split_k, a, b)
        out = _gemm_wrapped_kernel(*args)
        assert out.shape == (m, n)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
        opcheck(
            torch.ops.top.gemm_wrapped_kernel, args, test_utils=("test_schema", "test_faketensor")
        )

    vec_args = (n, k, "bfloat16", 1, 128, 4, a[0].contiguous(), b)
    out = _gemv_wrapped_kernel(*vec_args)
    assert out.shape == (n,)
    torch.testing.assert_close(out.float(), ref[0], atol=2e-2, rtol=2e-2)
    opcheck(
        torch.ops.top.gemv_wrapped_kernel, vec_args, test_utils=("test_schema", "test_faketensor")
    )


@pytest.mark.smoke
def test_gemm_refuses_tma_misaligned_shapes_by_naming_the_dim() -> None:
    """An unaligned innermost dimension is refused, with the dimension named.

    Every ``GemmKernel`` structure loads through TMA, which addresses the
    innermost dimension in 16-byte units; which logical dim that is follows the
    layout, so the same extent is served in one layout and refused in another.
    Undeclared, these calls died inside TileLang's descriptor check instead
    ("Check failed: (result.supported) is false"), naming nothing to change.

    Routing only — the aligned shapes already run end to end in ``GemmFixture``.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("the TMA alignment region is SM90-specific")

    nt, nn = GemmFwdOp(trans_a=False, trans_b=True), GemmFwdOp(trans_a=False, trans_b=False)
    fp = torch.bfloat16

    # k is innermost for both operands under NT, so 1001 is refused there...
    with pytest.raises(ValueError, match=r"multiple of 8 elements.*k=1001"):
        nt._get_kernel(256, 512, 1001, fp)
    # ...while n = 511 is innermost only where b is not transposed.
    with pytest.raises(ValueError, match=r"multiple of 8 elements.*n=511"):
        nn._get_kernel(256, 511, 1024, fp)
    assert nt._get_kernel(256, 511, 1024, fp)[0] == "gemm"

    # m == 1 stays served at any extent: the bandwidth body uses cp.async.
    assert nt._get_kernel(1, 512, 1001, fp)[0] == "lhs_row"

    # Constructing the kernel directly bypasses selection, so it refuses too.
    with pytest.raises(ValueError, match=r"cannot serve 256x512x1001"):
        GemmKernel(256, 512, 1001, fp, trans_a=False, trans_b=True)


@pytest.mark.smoke
def test_structure_routing_matches_test_ids() -> None:
    """Each ``GemmFixture`` case reaches the structure its id names.

    The correctness cases above are the only coverage several structures have,
    and two of them (coop2) get there through ``gemm_heuristics.best_config``
    rather than a pin — so a change to the selector's scoring could silently
    route them elsewhere and leave ``coop2`` untested while every test still
    passes. This pins the mapping: when it fails, the correctness case named in
    the assertion needs a new shape, not a new expectation.

    Routing only — ``_get_kernel`` builds no JIT, so this stays smoke-fast.
    """
    from tileops.utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("structure routing is SM90-specific")

    # (test id, m, n, k, dtype, trans_b, expected structure)
    expected = [
        ("smoke-fp16-square", 1024, 1024, 1024, torch.float16, False, "coop2s"),
        ("smoke-bf16-square", 1024, 1024, 1024, torch.bfloat16, False, "coop2s"),
        ("full-bf16-coop2-persistent", 1536, 2112, 256, torch.bfloat16, True, "coop2"),
        ("full-bf16-coop2-mn-tail", 1440, 2080, 256, torch.bfloat16, True, "coop2"),
        ("full-bf16-simple-plain", 64, 7168, 2048, torch.bfloat16, True, "simple"),
        ("full-bf16-simple-cluster", 128, 7168, 2048, torch.bfloat16, True, "simple"),
        ("full-fp16-nt-dense-ws", 128, 2112, 4096, torch.float16, True, "coop2_splitk"),
        ("full-fp16-small-m4-swap-ab", 4, 7168, 2048, torch.float16, True, "swap_ab"),
        ("full-bf16-small-m4-swap-ab-ntail", 4, 5000, 2048, torch.bfloat16, True, "swap_ab"),
        ("full-fp16-small-m8-splitk", 8, 2112, 7168, torch.float16, True, "splitK4"),
        ("full-fp16-small-m4-basic-ntail", 4, 3000, 2048, torch.float16, True, "basic"),
    ]
    flags = GemmKernel._STRUCTURE_FLAGS

    for test_id, m, n, k, dtype, trans_b, want in expected:
        op = GemmFwdOp(trans_a=False, trans_b=trans_b)
        mode, kernel = op._get_kernel(m, n, k, dtype)
        assert mode == "gemm", f"{test_id}: expected the generic kernel, got {mode}"
        config = kernel.config
        got = next((f for f in flags if config.get(f)), None)
        if got is None:
            split_k = config.get("split_k", 1)
            got = f"splitK{split_k}" if split_k > 1 else "basic"
        assert got == want, (
            f"{test_id} ({m}x{n}x{k}) now routes to {got}, not {want} — "
            f"that structure has lost its correctness coverage"
        )


@pytest.mark.smoke
def test_gemm_kernel_tune_falls_back_to_default() -> None:
    """``GemmKernel`` defines no ``autotune_configs``: ``tune=True`` must warn
    and fall back to ``default_config``.

    The in-tree tuner sweeps only the basic mainloop builder, so a silent
    basic-grid sweep would downgrade shapes whose default is a structure-
    flagged config (coop2 / split-K). Construction only — no JIT compile.
    """
    with pytest.warns(UserWarning, match="does not define autotune_configs"):
        kernel = GemmKernel(4096, 4096, 7168, torch.float16, tune=True, trans_a=False, trans_b=True)
    assert kernel.config == kernel.default_config

