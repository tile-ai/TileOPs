import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase, served_in_tree
from tileops.backend import BUILTIN
from tileops.kernels.gemm.bmm import BmmFP8TransposeKernel, BmmPersistentKernel
from tileops.kernels.gemm.call_spec import BmmCall
from tileops.ops import BmmFP8FwdOp, BmmFwdOp
from workloads.device import run_device
from workloads.gemm import BmmFP8Workload, BmmWorkload

# Covering the [B,K,N] path is the point of these tests, so the perf hint
# BmmFP8FwdOp emits for it is expected output, not a signal.
pytestmark = pytest.mark.filterwarnings("ignore:BmmFP8FwdOp")


class BmmTest(BmmWorkload, TestBase):
    pass


class BmmFP8Test(BmmFP8Workload, TestBase):
    pass


class BmmFixture(FixtureBase):
    PARAMS = [
        (
            "batch, m, n, k, dtype, tune",
            [
                pytest.param(
                    4,
                    128,
                    128,
                    128,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="gemm")],
                    id="smoke-fp16-b4-128",
                ),
                pytest.param(
                    4,
                    128,
                    128,
                    128,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bf16-b4-128",
                ),
                pytest.param(
                    8,
                    512,
                    512,
                    512,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-b8-512",
                ),
                pytest.param(
                    8,
                    512,
                    512,
                    512,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-b8-512",
                ),
                pytest.param(
                    16,
                    256,
                    256,
                    256,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-b16-256",
                ),
                pytest.param(
                    1,
                    1024,
                    1024,
                    1024,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-b1-1k",
                ),
                pytest.param(
                    32,
                    128,
                    512,
                    128,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-b32-mha-qk",
                ),
                pytest.param(
                    8,
                    128,
                    128,
                    2048,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-b8-mha-pv",
                ),
                pytest.param(
                    8,
                    128,
                    128,
                    2048,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-b8-mha-pv",
                ),
                pytest.param(
                    32,
                    256,
                    256,
                    1024,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-b32-moe",
                ),
                pytest.param(
                    4,
                    200,
                    300,
                    128,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-b4-mn-nonaligned",
                ),
            ],
        ),
    ]


@BmmFixture
def test_bmm(batch: int, m: int, n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    test = BmmTest(batch, m, n, k, dtype)
    op = BmmFwdOp(tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_bmm_k_not_multiple_of_16_raises() -> None:
    """The in-tree kernels refuse a K that is not a multiple of 16, which torch.bmm admits."""
    op = BmmFwdOp()
    a = torch.randn(4, 16, 24, device=run_device(), dtype=torch.float16)
    b = torch.randn(4, 24, 16, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="requires k a multiple of 16"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_persistent_calibrated_dispatch_region() -> None:
    """The persistent path claims aligned calls worth half a persistent wave on a calibrated board."""

    def call(batch=64, m=128, n=2048, *, calibration="h200"):
        return BmmCall(
            batch=batch,
            m=m,
            n=n,
            k=2048,
            dtype=torch.bfloat16,
            arch=90,
            calibration=calibration,
            sm_count=132,
        )

    assert BmmPersistentKernel.applies(call())
    assert not BmmPersistentKernel.applies(call(batch=32, m=256, n=256))
    assert not BmmPersistentKernel.applies(call(m=200, n=300))
    assert not BmmPersistentKernel.applies(call(calibration=None))
    # n is TMA-aligned and the shape is large, so only the tile count rejects it.
    assert not BmmPersistentKernel.applies(call(batch=1, m=2048, n=1024))


@pytest.mark.smoke
def test_bmm_persistent_region_holds_manifest_workloads() -> None:
    """The two manifest workloads nearest the wave threshold keep their routing."""
    for batch, m, n, k, claimed in [
        (16, 512, 512, 512, True),
        (32, 256, 256, 256, False),
    ]:
        call = BmmCall(
            batch=batch,
            m=m,
            n=n,
            k=k,
            dtype=torch.bfloat16,
            arch=90,
            calibration="h200",
            sm_count=132,
        )
        assert BmmPersistentKernel.applies(call) is claimed, call


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(16, 512, 512, 512, id="one-wave"),
        pytest.param(32, 512, 512, 512, id="two-waves"),
        # Neither output extent is a whole number of tiles.
        pytest.param(3, 1000, 1000, 1024, id="ragged-tiles"),
    ],
)
def test_bmm_runs_the_persistent_path_where_it_claims_the_call(
    batch: int, m: int, n: int, k: int, dtype: torch.dtype
) -> None:
    """A call the persistent path claims runs its template through dispatch."""
    shape = {"batch": batch, "m": m, "n": n, "k": k}
    call = BmmCall(**shape, dtype=dtype, device=torch.device(run_device()))
    if not BmmPersistentKernel.applies(call):
        pytest.skip("the persistent path serves only a calibrated board")
    test = BmmTest(*shape.values(), dtype)
    op = BmmFwdOp()
    test.check(op, *test.gen_inputs())
    if served_in_tree(op):
        (kernel,) = op.built_kernels("bmm").values()
        assert type(kernel) is BmmPersistentKernel


class BmmFP8Fixture(FixtureBase):
    PARAMS = [
        (
            "batch, m, n, k, dtype, out_dtype",
            [
                pytest.param(
                    4,
                    128,
                    128,
                    128,
                    torch.float8_e4m3fn,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-fp8-b4-per-tensor",
                ),
                pytest.param(
                    8,
                    128,
                    256,
                    128,
                    torch.float8_e4m3fn,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-fp8-b8-per-tensor",
                ),
                pytest.param(
                    16,
                    128,
                    128,
                    2048,
                    torch.float8_e4m3fn,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="full-fp8-b16-mha-pv-per-tensor",
                ),
            ],
        ),
    ]


@BmmFP8Fixture
def test_bmm_fp8(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    test = BmmFP8Test(batch, m, n, k, dtype, out_dtype=out_dtype)
    op = BmmFP8FwdOp(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    test.check(op, *inputs)


@pytest.mark.smoke
def test_bmm_fp8_rejects_e5m2() -> None:
    """BmmFP8FwdOp advertises fp8_e4m3fn only; e5m2 inputs must be rejected.

    Kept separate from the main fixture so that ``test_bmm_fp8`` carries a
    single purpose (correctness on supported dtypes) and this test carries
    the other (dtype-guard on unsupported dtypes).
    """
    test = BmmFP8Test(4, 128, 128, 128, torch.float8_e5m2)
    op = BmmFP8FwdOp(out_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"outside \['float8_e4m3fn'\]"):
        op(*test.gen_inputs())


@pytest.mark.smoke
def test_bmm_fp8_k_not_multiple_of_32_raises() -> None:
    op = BmmFP8FwdOp()
    a = torch.randn(4, 128, 48, device=run_device()).to(torch.float8_e4m3fn)
    b = torch.randn(4, 48, 128, device=run_device()).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device=run_device(), dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=run_device(), dtype=torch.float32)
    with pytest.raises(ValueError, match="multiple of 32"):
        op(a, b, scale_a, scale_b)


@pytest.mark.smoke
def test_bmm_fp8_accepts_nk_layout_when_k_ne_n() -> None:
    batch, m, n, k = 4, 128, 256, 128
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()  # b_kn is [B, K, N]
    # NK-layout: [B, N, K], K-innermost, contiguous.  ``transpose+
    # contiguous`` materialises the physical NK buffer so its shape[2]
    # unambiguously carries K.
    b_nk = b_kn.transpose(-2, -1).contiguous()
    assert b_nk.shape == (batch, n, k)
    op_kn = BmmFP8FwdOp(out_dtype=torch.bfloat16)
    op_nk = BmmFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_kn = op_kn(a, b_kn, scale_a, scale_b).clone()
    out_nk = op_nk(a, b_nk, scale_a, scale_b)
    # Numerically identical: same kernel, same buffer bits, just no
    # internal DtoD copy on the NK path.
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_nk_view_when_k_eq_n() -> None:
    batch, m, n, k = 4, 128, 128, 128  # K == N
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()  # contiguous [B, K, N]
    b_nk_view = b_kn.transpose(-2, -1)
    assert b_nk_view.shape == (batch, n, k)
    assert b_nk_view.stride(-2) == 1
    op_kn = BmmFP8FwdOp(out_dtype=torch.bfloat16)  # trans_b=False: b as [B, K, N]
    op_nk = BmmFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_kn = op_kn(a, b_kn, scale_a, scale_b).clone()
    out_nk = op_nk(a, b_nk_view, scale_a, scale_b)
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_contiguous_nk_square_when_k_eq_n() -> None:
    batch, m, n, k = 4, 128, 128, 128  # K == N
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()  # contiguous [B, K, N]
    # Physical contiguous [B, N, K]: stride is (N*K, K, 1) => stride(-2) == K.
    b_nk = b_kn.transpose(-2, -1).contiguous()
    assert b_nk.is_contiguous()
    assert b_nk.shape == (batch, n, k)
    assert b_nk.stride(-2) == k

    # Same logical B matrix, explicit layouts.  Both must yield the same d.
    op_nk = BmmFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_nk = op_nk(a, b_nk, scale_a, scale_b).clone()
    op_kn = BmmFP8FwdOp(out_dtype=torch.bfloat16)  # trans_b=False: b as [B, K, N]
    out_kn = op_kn(a, b_kn, scale_a, scale_b)
    torch.testing.assert_close(out_nk, out_kn, atol=0.0, rtol=0.0)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "block",
    [b for b in BmmFP8TransposeKernel.TILE_CANDIDATES if b != BmmFP8TransposeKernel.TILE],
)
def test_bmm_fp8_kn_transpose_tuned_tile_handles_tail(block: int) -> None:
    """Each tuned staging tile transposes a ``[B, K, N]`` operand with tails exactly.

    N leaves a tail under the tile; K is a multiple of 32, so it leaves one under
    every tile but 32.
    """

    class _Pinned(BmmFP8TransposeKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "config": {"block": block, "threads": 128}})

    batch, m, n, k = 2, 128, 2 * block + 16, block + 32
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    op = BmmFP8FwdOp(
        out_dtype=torch.bfloat16, kernel_map={"bmm_fp8_transpose": _Pinned}, target=BUILTIN
    )
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("bmm_fp8_transpose").values()
    assert type(kernel) is _Pinned


@pytest.mark.smoke
def test_bmm_fp8_kn_transpose_handles_tile_tail() -> None:
    """Extents that leave a tail under every staging tile transpose exactly."""
    batch, m, n, k = 3, 128, 80, 160
    assert all(n % tile for tile in BmmFP8TransposeKernel.TILE_CANDIDATES)
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()
    b_nk = b_kn.transpose(-2, -1).contiguous()

    op_kn = BmmFP8FwdOp(out_dtype=torch.bfloat16)
    op_nk = BmmFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_kn = op_kn(a, b_kn, scale_a, scale_b).clone()
    out_nk = op_nk(a, b_nk, scale_a, scale_b)
    if served_in_tree(op_kn):
        assert op_kn.built_kernels("bmm_fp8_transpose")
        assert not op_nk.built_kernels("bmm_fp8_transpose")
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_no_transpose_when_b_already_k_innermost() -> None:
    """What decides the copy is ``b``'s strides, not ``trans_b``."""
    batch, m, n, k = 4, 128, 256, 128
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()
    # [B, K, N] by shape, K-innermost in memory.
    b_kn_view = b_kn.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert b_kn_view.stride(-2) == 1

    op = BmmFP8FwdOp(out_dtype=torch.bfloat16)
    out_view = op(a, b_kn_view, scale_a, scale_b).clone()
    if served_in_tree(op):
        assert not op.built_kernels("bmm_fp8_transpose")

    out_kn = BmmFP8FwdOp(out_dtype=torch.bfloat16)(a, b_kn, scale_a, scale_b)
    torch.testing.assert_close(out_view, out_kn, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_persistent_default_tile_boundary() -> None:
    batch, m, n, k = 8, 64, 64, 32
    test = BmmFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()
    op = BmmFP8FwdOp(out_dtype=torch.bfloat16)

    test.check(op, a, b_kn, scale_a, scale_b)
