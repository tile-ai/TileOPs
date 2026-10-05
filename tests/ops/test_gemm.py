import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase, served_in_tree
from tileops.backend import BUILTIN
from tileops.kernels.gemm import (
    GemmCpAsyncKernel,
    GemmTMAKernel,
    GemmW4A16Kernel,
    GemvKernel,
    W4A16RepackKernel,
)
from tileops.kernels.gemm.call_spec import GemmCall, GemmFP8Call
from tileops.kernels.gemm.dense import (
    GemmFP8TensorScaleKernel,
    _b_eviction,
)
from tileops.kernels.gemm.fp8_1d2d import GemmFP81D2DFwdKernel
from tileops.kernels.gemm.heuristics import (
    best_config,
    small_m_splitk_config,
)
from tileops.kernels.gemm.w4a16 import GROUP_SIZE, _select_config, _stage_meta_per_tile
from tileops.ops import GemmFP8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from tileops.utils import get_sm_version
from workloads.device import run_device
from workloads.gemm import (
    GemmFP8Workload,
    GemmW4A16BasisWorkload,
    GemmW4A16Workload,
    GemmWorkload,
    quantize_weight_int4,
    repack_w4a16_weight,
    w4a16_partition_verification,
)
from workloads.numerics import compare_outputs


def _gemm_call(m: int, n: int, k: int, *, dtype=torch.float16, trans_b: bool = True) -> GemmCall:
    """One dense GEMM call on the SM90 board the regions were fitted on."""
    return GemmCall(arch=90, sm_count=132, m=m, n=n, k=k, dtype=dtype, trans_b=trans_b)


def _selects(op: GemmFwdOp, call: GemmCall) -> type:
    return op.kernel_map[op.select_implementation("gemm", call)]


class GemmTest(GemmWorkload, TestBase):
    pass


class GemmFP8Test(GemmFP8Workload, TestBase):
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
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="gemm")],
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


class GemmFP8Fixture(FixtureBase):
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
                    128,
                    256,
                    512,
                    torch.float8_e4m3fn,
                    "block128x128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fp8-e4m3-block128x128",
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
                pytest.param(
                    200,
                    300,
                    1536,
                    torch.float8_e4m3fn,
                    "block128x128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128x128-mn-tail",
                ),
                pytest.param(
                    512,
                    8576,
                    512,
                    torch.float8_e4m3fn,
                    "block128x128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128x128-multi-wave",
                ),
                pytest.param(
                    8,
                    300,
                    256,
                    torch.float8_e4m3fn,
                    "block128x128",
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128x128-general-kernel",
                ),
                pytest.param(
                    64,
                    256,
                    6144,
                    torch.float8_e4m3fn,
                    "block128x128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128x128-general-split-k",
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
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    trans_a: bool,
    trans_b: bool,
    tune: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b, tune=tune)
    test.check(op, *test.gen_inputs())


@GemmFP8Fixture
def test_gemm_fp8(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    scale_mode: str,
    out_dtype: torch.dtype,
    bias: bool,
) -> None:
    test = GemmFP8Test(m, n, k, dtype, scale_mode, out_dtype=out_dtype, bias=bias)
    op = GemmFP8FwdOp(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    if dtype != torch.float8_e4m3fn:
        with pytest.raises(ValueError, match=r"outside \['float8_e4m3fn'\]"):
            op(*inputs)
        return
    test.check(op, *inputs)


@GemmW4A16Fixture
def test_gemm_w4a16(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = GemmW4A16Test(m, n, k, dtype)
    op = GemmW4A16FwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("k_index", [0, 1, 127, 128, 200, 383])
def test_gemm_w4a16_is_exact_on_a_basis_vector(k_index: int) -> None:
    """Check nibble, group, and zero-point indexing."""
    workload = GemmW4A16BasisWorkload(k_index)
    TestBase.check(workload, GemmW4A16FwdOp(), *workload.gen_inputs())


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
    # Both groups span one unit including zero: their 15-step scale is known exactly.
    expected_scale = torch.full_like(scale, 1 / 15)
    assert torch.equal(scale, expected_scale)
    assert dequantized[0].max() == 15 * expected_scale[0, 0].float()
    assert dequantized[1].min() == -15 * expected_scale[1, 0].float()


@pytest.mark.smoke
def test_gemm_fp8_block128_single_k_block_uses_block_kernel() -> None:
    test = GemmFP8Test(128, 256, 128, torch.float8_e4m3fn, "block128")
    op = GemmFP8FwdOp()
    test.check(op, *test.gen_inputs())
    if served_in_tree(op):
        assert op.kernel.__class__.__name__ == "GemmFP8BlockScaleKernel"


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("m", "scale_b_rows", "bias", "out_dtype", "expected"),
    [
        pytest.param(128, 128, False, torch.bfloat16, "GemmFP81D2DFwdKernel", id="1d2d"),
        pytest.param(64, 128, False, torch.bfloat16, "GemmFP8BlockScaleKernel", id="1d2d-small-m"),
        pytest.param(128, 128, True, torch.bfloat16, "GemmFP8BlockScaleKernel", id="1d2d-bias"),
        pytest.param(128, 128, False, torch.float16, "GemmFP8BlockScaleKernel", id="1d2d-fp16"),
        pytest.param(128, 1, False, torch.bfloat16, "GemmFP8BlockScaleKernel", id="1d1d"),
    ],
)
def test_gemm_fp8_block_scale_selection(
    m: int, scale_b_rows: int, bias: bool, out_dtype: torch.dtype, expected: str
) -> None:
    """The 1D2D kernel serves its region; the general block kernel serves the rest."""
    n, k = 256, 512
    call = GemmFP8Call(
        m=m,
        n=n,
        k=k,
        dtype=torch.float8_e4m3fn,
        scale_a_shape=(m, k // 128),
        scale_b_shape=(-(-n // scale_b_rows), k // 128),
        out_dtype=out_dtype,
        has_bias=bias,
    )
    op = GemmFP8FwdOp(out_dtype=out_dtype)
    assert op.kernel_map[op.select_implementation("gemm_fp8", call)].__name__ == expected


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gemm_fp8_serves_sm89_by_scale_grid() -> None:
    m, n, k = 128, 256, 512
    for scale_a_shape, scale_b_shape, expected in (
        ((1, 1), (1, 1), "gemm_fp8_tensor_scale"),
        ((m, k // 128), (n, k // 128), "gemm_fp8_block_scale"),
        ((m, k // 128), (n // 128, k // 128), "gemm_fp8_block_scale"),
    ):
        call = GemmFP8Call(
            arch=89,
            sm_count=1,
            m=m,
            n=n,
            k=k,
            dtype=torch.float8_e4m3fn,
            scale_a_shape=scale_a_shape,
            scale_b_shape=scale_b_shape,
            out_dtype=torch.bfloat16,
        )
        assert GemmFP8FwdOp().select_implementation("gemm_fp8", call) == expected


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("arch", [89, 90])
def test_gemm_fp8_builds_from_the_calls_device_facts(arch: int) -> None:
    from tileops.utils import get_sm_version

    device = torch.device(run_device())
    # Construction still checks the call's device, whatever arch the call states.
    if get_sm_version(device.index) not in GemmFP8TensorScaleKernel.supported_archs:
        pytest.skip("the FP8 kernel is built only on SM89 and SM90")
    call = GemmFP8Call(
        arch=arch,
        sm_count=1,
        m=128,
        n=256,
        k=512,
        dtype=torch.float8_e4m3fn,
        scale_a_shape=(1, 1),
        scale_b_shape=(1, 1),
        out_dtype=torch.bfloat16,
        device=device,
    )
    _identity, build = GemmFP8TensorScaleKernel.entry_for(call)
    kernel = build()
    assert (kernel.ws_refusal is None) == (arch == 90)
    assert kernel.sm_count == 1


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_1d2d_shared_epilogue_matches_reference() -> None:
    """The shared-memory epilogue publishes the whole tile."""

    class SharedEpilogue(GemmFP81D2DFwdKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "shared_epilogue": True})

    test = GemmFP8Test(128, 256, 512, torch.float8_e4m3fn, "block128x128")
    op = GemmFP8FwdOp(kernel_map={"gemm_fp8_1d2d": SharedEpilogue}, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("gemm_fp8").values()
    assert type(kernel) is SharedEpilogue


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmFwdOp(trans_a=False, trans_b=True, tune=tune)
    test.check(op, *test.gen_inputs())


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmFwdOp(trans_a=False, trans_b=False, tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_lhs_rows_band_dispatch() -> None:
    """``GemvKernel`` takes its ``lhs_rows`` band only at m == 2, on the n band swap_ab leaves it.

    One case per clause of ``GemvKernel.band_for``: m == 1 lands on the same class in
    its ``lhs_row`` band, m >= 3 and non-NT stay on the generic kernel (whose small-m
    band picks swap_ab / split-K / simple configs analytically), and so does any n wide
    enough for the operand-swapped grid. Selection only — no kernel is built, so this
    stays smoke-fast.
    """
    nt, nn = GemmFwdOp(trans_a=False, trans_b=True), GemmFwdOp(trans_a=False, trans_b=False)
    two_rows = _gemm_call(2, 2112, 7168)
    assert _selects(nt, two_rows) is GemvKernel
    assert GemvKernel.band_for(two_rows) == "lhs_rows"
    assert _selects(nt, _gemm_call(2, 7168, 2048)) is GemmTMAKernel
    assert _selects(nt, _gemm_call(3, 2112, 7168)) is GemmTMAKernel
    one_row = _gemm_call(1, 2112, 7168)
    assert _selects(nt, one_row) is GemvKernel
    assert GemvKernel.band_for(one_row) == "lhs_row"
    assert _selects(nn, _gemm_call(2, 2112, 7168, trans_b=False)) is GemmTMAKernel


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_routes_tma_misaligned_shapes_to_the_pipelined_mainloop() -> None:
    """An unaligned innermost dimension leaves ``GemmTMAKernel``, which names the dim.

    Every ``GemmTMAKernel`` structure loads through TMA, which addresses the
    innermost dimension in 16-byte units; which logical dim that is follows the
    layout, so the same extent is served in one layout and refused in another.
    ``GemmCpAsyncKernel`` loads through ``cp.async`` and takes what is refused.
    Undeclared, these calls died inside TileLang's descriptor check instead
    ("Check failed: (result.supported) is false"), naming nothing to change.

    Routing only — the aligned shapes already run end to end in ``GemmFixture``.
    """
    nt, nn = GemmFwdOp(trans_a=False, trans_b=True), GemmFwdOp(trans_a=False, trans_b=False)
    bf = torch.bfloat16

    misaligned_k = _gemm_call(256, 512, 1001, dtype=bf)
    assert _selects(nt, misaligned_k) is GemmCpAsyncKernel
    assert "multiple of 8 elements" in GemmTMAKernel.refusal(misaligned_k)
    assert "k=1001" in GemmTMAKernel.refusal(misaligned_k)

    misaligned_n = _gemm_call(256, 511, 1024, dtype=bf, trans_b=False)
    assert _selects(nn, misaligned_n) is GemmCpAsyncKernel
    assert "n=511" in GemmTMAKernel.refusal(misaligned_n)

    assert _selects(nt, _gemm_call(256, 511, 1024, dtype=bf)) is GemmTMAKernel
    assert _selects(nt, _gemm_call(1, 512, 1001, dtype=bf)) is GemvKernel


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("block_k, stage_n", [(32, 0), (64, 128)])
def test_coop2_epilogue_chunking_matches_reference(block_k: int, stage_n: int) -> None:
    """Shape coverage: the coop2 epilogue staged in one SMEM chunk and in two.

    ``stage_n`` cuts a ``block_n``-wide output tile into ``block_n / stage_n`` chunks. Each
    pin is the coop2 candidate the selector emits at that ``block_k``.
    """
    m, n, k = 1536, 3072, 256
    test = GemmTest(m, n, k, torch.bfloat16, False, True)
    config = {
        "coop2": True,
        "block_n": 256,
        "block_k": block_k,
        "num_stages": 4,
        "group_size_m": 16,
        "stage_n": stage_n,
    }

    class Coop2(GemmTMAKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "config": config})

    op = GemmFwdOp(trans_b=True, kernel_map={"gemm_tma": Coop2}, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("gemm").values()
    assert type(kernel) is Coop2


@pytest.mark.smoke
def test_b_tile_eviction_hint_follows_the_m_tile_count() -> None:
    """Dispatch branch: the streaming-B hint is on at one or two M-tiles, off above."""
    assert _b_eviction(64, 64) == "evict_first"
    assert _b_eviction(128, 64) == "evict_first"
    assert _b_eviction(129, 64) is None
    assert _b_eviction(4096, 128) is None


@pytest.mark.sm90
@pytest.mark.cuda_only
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
    flags = GemmTMAKernel._STRUCTURE_FLAGS

    for test_id, m, n, k, dtype, trans_b, want in expected:
        op = GemmFwdOp(trans_a=False, trans_b=trans_b)
        call = _gemm_call(m, n, k, dtype=dtype, trans_b=trans_b)
        cls = _selects(op, call)
        assert cls is GemmTMAKernel, f"{test_id}: expected the generic kernel, got {cls.__name__}"
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


class _PipelinedUnderTMA(GemmCpAsyncKernel):
    """The pipelined mainloop installed under the TMA key, as an sm80 or sm89 device selects it."""

    # Available where the key it replaces is, so other devices keep their own dispatch.
    supported_archs = GemmTMAKernel.supported_archs


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("k", [2, 8, 24])
def test_gemm_cp_async_kernel_k_tail_padding(k: int) -> None:
    """A K shorter than, or not a multiple of, ``block_k = 16`` rides on the zero-padded K tail.

    ``k=2`` is TMA-misaligned, so dispatch takes the pipelined mainloop. ``k=8`` and
    ``k=24`` take the TMA key on SM90, which then runs the pipelined mainloop.
    """
    test = GemmTest(32, 64, k, torch.bfloat16, False, True)
    op = GemmFwdOp(trans_b=True, kernel_map={"gemm_tma": _PipelinedUnderTMA}, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("gemm").values()
    pinned = k != 2 and get_sm_version() == 90
    assert type(kernel) is (_PipelinedUnderTMA if pinned else GemmCpAsyncKernel)
    assert kernel.config["block_k"] == 16


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("m", [100, 257])
def test_gemm_w4a16_kernel_predicates_a_ragged_token_count(m: int) -> None:
    test = GemmW4A16Test(m, 1024, 512, torch.float16)
    op = GemmW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("gemm_w4a16").values()
    assert kernel.m_pad != m


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_long_k_stages_metadata_per_tile() -> None:
    groups_at_crossover = 256  # 64 rows * 256 groups * 3 bytes = 48 KiB.
    assert not _stage_meta_per_tile(128, 512, 64, groups_at_crossover)
    assert _stage_meta_per_tile(128, 512, 64, groups_at_crossover + 1)
    assert not _stage_meta_per_tile(256, 512, 64, groups_at_crossover + 1)

    test = GemmW4A16Test(1, 64, 32896, torch.float16)
    op = GemmW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("gemm_w4a16").values()
    assert _stage_meta_per_tile(
        kernel.config["threads"], kernel.config["block_k"], 64, 32896 // 128
    )


def _pinned_w4a16(config: dict) -> type:
    """``GemmW4A16Kernel`` fixed to *config*, for ``kernel_map=``."""

    class Pinned(GemmW4A16Kernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "config": config})

    return Pinned


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(("sms", "split_k"), [(60, 2), (132, 8)])
def test_gemm_w4a16_sliced_k_matches_the_reference(sms: int, split_k: int) -> None:
    """The fp32 partials reduce to what the whole K loop computes.

    The config is the one the selector picks for a device with ``sms`` SMs.
    """
    m, n, k = 1, 1024, 8192
    config = _select_config(m, n, k, GROUP_SIZE, sms=sms)
    assert config["split_k"] == split_k
    pinned = _pinned_w4a16(config)
    test = GemmW4A16Test(m, n, k, torch.float16)
    op = GemmW4A16FwdOp(kernel_map={"gemm_w4a16": pinned}, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("gemm_w4a16").values()
    assert type(kernel) is pinned
    assert kernel.config == config


@pytest.mark.sm90
@pytest.mark.cuda_only
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
    assert torch.equal(op(*inputs), expected)

    next_test = GemmW4A16Test(2, 1024, 8192, torch.float16)
    next_inputs = next_test.gen_inputs()
    next_test.check(op, *next_inputs)

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        new_op = GemmW4A16FwdOp(tune=True)
    assert new_op.tune is False


@pytest.mark.smoke
def test_gemm_w4a16_select_config_streams_only_the_underfilled_grid() -> None:
    """Stream-K takes the long-K decode row, whose 128 N tiles leave SMs idle, and no other."""
    assert _select_config(1, 8192, 81920, GROUP_SIZE, sms=132)["stream_ctas"] == 132
    assert _select_config(1, 8192, 8192, GROUP_SIZE, sms=132)["stream_ctas"] == 0
    assert _select_config(128, 4096, 14336, GROUP_SIZE, sms=132)["stream_ctas"] == 0


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_stream_k_matches_the_unstreamed_tile() -> None:
    """Crossing and exact two-way Stream-K partitions handle a partial K tile.

    Each config is the one the selector picks for a device with that many SMs. The
    34 N tiles over 67 CTAs exercise a non-integral partition whose second tile
    starts at CTA 2, and ``k=32896`` has a 128-element tail. 68 CTAs split every N
    tile exactly in two. 132 SMs leave too few idle for Stream-K, so that tile runs
    whole.
    """
    m, n, k = 1, 2176, 32896
    test = GemmW4A16Test(m, n, k, torch.float16)
    inputs = test.gen_inputs()
    outputs = {}
    for sms in (67, 68, 132):
        config = _select_config(m, n, k, GROUP_SIZE, sms=sms)
        pinned = _pinned_w4a16(config)
        op = GemmW4A16FwdOp(kernel_map={"gemm_w4a16": pinned}, target=BUILTIN)
        outputs[sms] = op(*inputs)
        (kernel,) = op.built_kernels("gemm_w4a16").values()
        assert type(kernel) is pinned
        assert kernel.config == config
        assert config["stream_ctas"] == (0 if sms == 132 else sms)
    streamed, two_way, whole = outputs[67], outputs[68], outputs[132]
    # The FP32 partials of up to three CTAs are summed in a different order.
    compare_outputs(streamed, whole, w4a16_partition_verification())
    compare_outputs(two_way, whole, w4a16_partition_verification())
    compare_outputs(streamed, test.ref_program(*inputs), test.verification())


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
        assert torch.equal(
            nibbles(repacked[:, start : start + step]),
            nibbles(packed[:, start : start + step]),
        )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(("n", "k"), [(64, 256), (1024, 512)])
def test_w4a16_repack_kernel_matches_the_reference(n: int, k: int) -> None:
    """The kernel and the tensor-expression repack agree bit for bit."""
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    op = GemmW4A16FwdOp(target=BUILTIN)

    actual = op.repack(packed)

    (kernel,) = op.built_kernels("w4a16_repack").values()
    assert type(kernel) is W4A16RepackKernel
    assert actual.dtype == torch.uint8
    assert actual.shape == packed.shape
    assert torch.equal(actual, repack_w4a16_weight(packed))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_repack_feeds_forward() -> None:
    test = GemmW4A16Test(64, 1024, 512, torch.float16)
    activation, _, scale, zero = test.gen_inputs()
    packed = test.row_major_weight

    prepacked = GemmW4A16FwdOp().repack(packed)
    actual = GemmW4A16FwdOp()(activation, prepacked, scale, zero)

    compare_outputs(
        actual, test.ref_program(activation, prepacked, scale, zero), test.verification()
    )


@pytest.mark.smoke
def test_gemm_w4a16_repack_refuses_a_partial_k_step() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        GemmW4A16FwdOp().repack(torch.zeros((8, 96), dtype=torch.uint8, device=run_device()))
