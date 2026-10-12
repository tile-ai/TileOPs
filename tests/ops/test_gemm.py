import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.gemm import (
    GemmFP8Block128WaveFwdKernel,
    GemmFP8Call,
    GemmFP8TensorScaleKernel,
    GemmFP81D2DFwdKernel,
    GemmW4A16Kernel,
    fp8_1d2d,
)
from tileops.kernels.gemm.heuristics import fp8_ws_config
from tileops.ops import GemmFP8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from tileops.utils import device_calibration, get_sm_count
from workloads.device import run_device
from workloads.gemm import (
    GemmFP8Workload,
    GemmW4A16BasisWorkload,
    GemmW4A16Workload,
    GemmWorkload,
    quantize_weight_int4,
    repack_w4a16_weight,
)
from workloads.numerics import compare_outputs


class GemmTest(GemmWorkload, TestBase):
    pass


class GemmFP8Test(GemmFP8Workload, TestBase):
    pass


class GemmW4A16Test(GemmW4A16Workload, TestBase):
    pass


def _fp8_call(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> GemmFP8Call:
    """The call spec ``GemmFP8FwdOp`` dispatches on for these bf16-output inputs."""
    return GemmFP8Call(
        m=a.shape[0],
        n=b.shape[0],
        k=a.shape[1],
        dtype=a.dtype,
        scale_a_shape=tuple(scale_a.shape),
        scale_b_shape=tuple(scale_b.shape),
        out_dtype=torch.bfloat16,
        has_bias=False,
        device=a.device,
    )


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
                    256,
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
                    256,
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
    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b)
    if tune:
        op.request_tune()
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


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("m, n, k", [(64, 64, 128), (128, 256, 256)])
def test_gemm_w4a16_k_tile_no_wider_than_k(m: int, n: int, k: int) -> None:
    """A short K takes a K tile and a ring no wider than it.

    The padding past K would be staged and multiplied for nothing.
    """
    test = GemmW4A16Test(m, n, k, torch.float16)
    config = GemmW4A16Kernel(m, n, k, torch.float16).config
    assert config["block_k"] <= k
    # The shallowest ring the kernel builds is two stages, whatever K.
    assert config["num_stages"] <= max(2, -(-k // config["block_k"]))
    test.check(GemmW4A16FwdOp(target=BUILTIN), *test.gen_inputs())


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


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("scale_mode", ["per_tensor", "block128", "block128x128"])
def test_gemm_fp8_without_tma_or_wgmma(scale_mode: str) -> None:
    """SM89 has FP8 tensor cores but no TMA or WGMMA, so every scale grid runs without them."""
    test = GemmFP8Test(256, 512, 1024, torch.float8_e4m3fn, scale_mode)
    test.check(GemmFP8FwdOp(), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_1d2d_shared_epilogue_matches_reference() -> None:
    """The shared-memory epilogue publishes the whole tile; the calibrated schedule for
    this decode shape takes it."""
    test = GemmFP8Test(128, 2112, 7168, torch.float8_e4m3fn, "block128x128")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, schedule",
    [
        # A 72-row M tail and an 8-column N tail in one 192-wide wave.
        pytest.param(
            200,
            200,
            2048,
            {"block_n": 192, "num_stages": 4, "group_size_m": 8, "waves": 1},
            id="one-wave-tails",
        ),
        # Two waves: the last tile's first wave holds 64 rows and its second none; a
        # 64-column N tail.
        pytest.param(
            320,
            320,
            2048,
            {"block_n": 128, "num_stages": 3, "group_size_m": 8, "waves": 2},
            id="two-wave-tails",
        ),
    ],
)
def test_gemm_fp8_1d1d_wave_matches_reference(
    m: int, n: int, k: int, schedule: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 1x128 ``scale_b`` stages per-column scales in the wave kernel's ring and folds them
    through the per-column helper. ``schedule`` is installed as the device board's schedule
    for the shape, so dispatch reaches the wave kernel on any SM90 board. ``k`` spans four
    scale groups, so each scale buffer is refilled after it is read."""
    test = GemmFP8Test(m, n, k, torch.float8_e4m3fn, "block128")
    inputs = test.gen_inputs()
    board = device_calibration(inputs[0].device.index)
    monkeypatch.setitem(
        fp8_1d2d._FP8_1D1D_WAVE_CONFIGS,
        board,
        {
            **fp8_1d2d._FP8_1D1D_WAVE_CONFIGS.get(board, {}),
            # Fewer CTAs than tiles, so a CTA stores a tail tile after a full one.
            (m, n, k): {"kernel": schedule, "sm_count": 2},
        },
    )
    op = GemmFP8FwdOp(target=BUILTIN)
    kernel = op.kernel_for("gemm_fp8", _fp8_call(*inputs))
    assert isinstance(kernel, GemmFP8Block128WaveFwdKernel)
    assert kernel.scale_b_per_row
    assert kernel.config == schedule
    test.check(op, *inputs)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, scale_mode, kernel_type, block_m, block_n",
    [
        pytest.param(
            128, 1024, 7168, "per_tensor", GemmFP8TensorScaleKernel, 64, 32, id="one-row-tile"
        ),
        pytest.param(
            128, 576, 7168, "block128x128", GemmFP81D2DFwdKernel, 64, 16, id="1d2d-one-row-tile"
        ),
    ],
)
def test_gemm_fp8_default_tile(
    m: int, n: int, k: int, scale_mode: str, kernel_type: type, block_m: int, block_n: int
) -> None:
    """The default schedule takes 64 rows on a single 128-row tile. Both shapes select
    the same tile on every SM count from 66 to 132."""
    test = GemmFP8Test(m, n, k, torch.float8_e4m3fn, scale_mode)
    op = GemmFP8FwdOp(target=BUILTIN)
    inputs = test.gen_inputs()
    kernel = op.kernel_for("gemm_fp8", _fp8_call(*inputs))
    assert isinstance(kernel, kernel_type)
    assert (kernel.config["block_m"], kernel.config["block_n"]) == (block_m, block_n)
    test.check(op, *inputs)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_1d2d_deepest_ring_launches() -> None:
    """The 64-row ring depth counts the mbarriers' static unit and each buffer's alignment.
    Counted without them, 14 stages of a 64x64 tile over 20 K-steps pass the estimate and
    exceed the shared memory a block may opt in to; 13 fit."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    # Two 64-row tiles over 32 columns an SM put 64 columns on each SM: the 64-wide tile.
    test = GemmFP8Test(128, 32 * sm_count, 2560, torch.float8_e4m3fn, "block128x128")
    op = GemmFP8FwdOp(target=BUILTIN)
    inputs = test.gen_inputs()
    kernel = op.kernel_for("gemm_fp8", _fp8_call(*inputs))
    assert isinstance(kernel, GemmFP81D2DFwdKernel)
    assert (kernel.config["block_m"], kernel.config["block_n"]) == (64, 64)
    assert kernel.config["num_stages"] == 13
    test.check(op, *inputs)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_per_tensor_192_wide_matches_reference() -> None:
    """Per-tensor scaling takes the 192-wide tile where its waves, counted whole, cover fewer
    columns than the 128-wide tile's, and stores a 64-column N tail of it."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    # Two row tiles by ``sm_count // 2`` 192-wide column tiles fill one wave; the 128-wide
    # tiles, half as many again, take two.
    test = GemmFP8Test(256, 192 * (sm_count // 2) - 128, 256, torch.float8_e4m3fn, "per_tensor")
    op = GemmFP8FwdOp(target=BUILTIN)
    inputs = test.gen_inputs()
    kernel = op.kernel_for("gemm_fp8", _fp8_call(*inputs))
    assert isinstance(kernel, GemmFP8TensorScaleKernel)
    assert (kernel.config["block_m"], kernel.config["block_n"]) == (128, 192)
    test.check(op, *inputs)


@pytest.mark.smoke
def test_fp8_ws_config_keeps_128_columns_where_192_saves_no_wave() -> None:
    """Three waves of 128-wide tiles cover 384 columns, no more than two waves of 192-wide
    ones, so per-tensor scaling keeps the 128-wide tile."""
    config = fp8_ws_config(1280, 2112, 256, 78, block_scaled=False)
    assert (config["block_m"], config["block_n"]) == (128, 128)


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmFwdOp(trans_a=False, trans_b=True)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmFwdOp(trans_a=False, trans_b=False)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_gemm_coop2_epilogue_staged_in_two_chunks() -> None:
    """A 256-wide coop2 tile at ``block_k = 64`` stages its epilogue in two 128-column
    chunks; this shape takes that schedule by default."""
    test = GemmTest(704, 5440, 128, torch.bfloat16, False, True)
    test.check(GemmFwdOp(trans_b=True, target=BUILTIN), *test.gen_inputs())


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_k_tail_shorter_than_one_k_block() -> None:
    """A K of two elements, below one ``block_k`` and TMA-misaligned, rides on the
    zero-padded K tail of the pipelined mainloop."""
    test = GemmTest(32, 64, 2, torch.bfloat16, False, True)
    test.check(GemmFwdOp(trans_b=True, target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("m", [100, 257])
def test_gemm_w4a16_kernel_predicates_a_ragged_token_count(m: int) -> None:
    test = GemmW4A16Test(m, 1024, 512, torch.float16)
    op = GemmW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "n, k",
    [
        # One-row grids too narrow for the device slice K, eight ways or two.
        pytest.param(1024, 8192, id="split-k-8", marks=pytest.mark.smoke),
        pytest.param(2176, 8192, id="split-k-2", marks=pytest.mark.smoke),
        # Tiles just past half the SM count stream K; 67 tiles cross a tile boundary, 66
        # split exactly in two, and k = 32896 leaves a 128-element K tail.
        pytest.param(4288, 32896, id="stream-k-crossing", marks=pytest.mark.full),
        pytest.param(4224, 32896, id="stream-k-two-way", marks=pytest.mark.full),
    ],
)
def test_gemm_w4a16_decode_partitions_match_the_reference(n: int, k: int) -> None:
    test = GemmW4A16Test(1, n, k, torch.float16)
    test.check(GemmW4A16FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_long_k_stages_metadata_per_tile() -> None:
    test = GemmW4A16Test(1, 64, 32896, torch.float16)
    op = GemmW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_request_tune_keeps_composite_runtime_state() -> None:
    test = GemmW4A16Test(1, 1024, 8192, torch.float16)
    inputs = test.gen_inputs()
    op = GemmW4A16FwdOp(target=BUILTIN)
    expected = op(*inputs)

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        op.request_tune()

    assert op._tune_requested is False
    assert torch.equal(op(*inputs), expected)

    next_test = GemmW4A16Test(2, 1024, 8192, torch.float16)
    next_inputs = next_test.gen_inputs()
    next_test.check(op, *next_inputs)

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        new_op = GemmW4A16FwdOp()
        new_op.request_tune()
    assert new_op._tune_requested is False


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
