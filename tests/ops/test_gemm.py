import re

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.gemm import GemmFP8Block128WaveFwdKernel
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
                    1408,
                    2112,
                    256,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-coop2-176-wide",
                ),
                pytest.param(
                    1400,
                    2112,
                    256,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-176-wide-m-tail",
                ),
                pytest.param(
                    4000,
                    2112,
                    512,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-pingpong-176",
                ),
                pytest.param(
                    2800,
                    2112,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-176-m-tail",
                ),
                pytest.param(
                    2840,
                    2080,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-192-mn-tail",
                ),
                pytest.param(
                    2048,
                    6144,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-192-four-tiles",
                ),
                pytest.param(
                    4000,
                    2090,
                    512,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-unaligned-n",
                ),
                pytest.param(
                    192,
                    6144,
                    128,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-basic-192-one-wave",
                ),
                pytest.param(
                    16,
                    24576,
                    512,
                    torch.bfloat16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-basic-192-one-wave-nn",
                ),
                pytest.param(
                    16,
                    24576,
                    512,
                    torch.bfloat16,
                    True,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-basic-192-one-wave-tn",
                ),
                pytest.param(
                    16,
                    24576,
                    512,
                    torch.bfloat16,
                    True,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-basic-192-one-wave-tt",
                ),
                pytest.param(
                    256,
                    2048,
                    6144,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-splitk-short-wave",
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
                    5,
                    200,
                    272,
                    torch.float8_e4m3fn,
                    "per_tensor",
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-per-tensor-swap-ab-tail",
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
    "m, n, k",
    [
        pytest.param(4096, 576, 7168, id="one-wave"),
        pytest.param(4096, 7168, 16384, id="two-wave"),
    ],
)
def test_gemm_fp8_1d1d_wave_matches_reference(m: int, n: int, k: int) -> None:
    """A 1x128 ``scale_b`` on the H200 shapes calibrated for the wave kernel, one with
    single-wave tiles and one with two-wave tiles: each column folds in under its own
    scale."""
    test = GemmFP8Test(m, n, k, torch.float8_e4m3fn, "block128")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(128, 32768, 512, id="one-row-tile"),
        pytest.param(4096, 32768, 512, id="two-wave"),
        pytest.param(128, 7168, 16384, id="56-wide"),
        pytest.param(4096, 2112, 7168, id="176-wide"),
    ],
)
def test_gemm_fp8_1d2d_wave_matches_reference(m: int, n: int, k: int) -> None:
    """A 128x128 ``scale_b`` on four H200 shapes calibrated for the wave kernel: one with a
    single row tile and its ``scale_a`` staged as one contiguous span, one whose 256-row
    tiles stage each 128-row wave in its own output tile, one whose 56-wide tiles can span
    two ``scale_b`` blocks, and one whose 176-wide tiles start at every multiple of 16
    columns within a block and can span three.

    The epilogue relies on the barrier TileLang places before the stmatrix writes into a
    staging tile to order them after the previous tile's TMA store; the generated source
    must keep one between the store wait and the first write.
    """
    test = GemmFP8Test(m, n, k, torch.float8_e4m3fn, "block128x128")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())
    kernel = GemmFP8Block128WaveFwdKernel(m, n, k, torch.float8_e4m3fn, torch.bfloat16)
    source = kernel.kernel(**kernel.config).get_kernel_source()
    epilogue = source[source.index("tma_store_wait") :]
    # Both consumer warpgroups, 256 threads, have to meet before the first staging write.
    assert re.search(r"__sync_thread_partial\(\d+, 256\)", epilogue[: epilogue.index("c_smem0")])


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_1d2d_wave_176_wide_tails(monkeypatch: pytest.MonkeyPatch) -> None:
    """176-wide wave tiles with an M tail and an N tail, scheduled for this shape on four
    CTAs: their six tiles start 128, 80 and 32 columns before a ``scale_b`` boundary, the
    last spanning three blocks into the N tail, and two CTAs run a second tile."""
    m, n, k = 200, 520, 512
    table = GemmFP8Block128WaveFwdKernel._FP8_1D2D_WAVE_CONFIGS
    calibration = device_calibration(torch.device(run_device()).index)
    schedule = {"kernel": {"block_n": 176, "num_stages": 4, "group_size_m": 8, "waves": 1}}
    monkeypatch.setitem(
        table, calibration, {**table.get(calibration, {}), (m, n, k): {**schedule, "sm_count": 4}}
    )
    test = GemmFP8Test(m, n, k, torch.float8_e4m3fn, "block128x128")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, scale_mode",
    [
        pytest.param(1024, "per_tensor", id="per-tensor"),
        pytest.param(576, "block128x128", id="block128x128"),
    ],
)
def test_gemm_fp8_single_row_tile_matches_reference(n: int, scale_mode: str) -> None:
    """Rows that fit one 128-row tile, run as two 64-row tiles of one consumer warpgroup
    each, under a per-tensor and a 128x128 ``scale_b`` grid."""
    test = GemmFP8Test(128, n, 7168, torch.float8_e4m3fn, scale_mode)
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_1d2d_deepest_ring_launches() -> None:
    """A 64-row block128x128 tile whose ring is sized to the block's shared memory launches
    and matches: its mbarriers and each buffer's alignment come out of the same budget."""
    test = GemmFP8Test(128, 4096, 2560, torch.float8_e4m3fn, "block128x128")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_per_tensor_176_wide_matches_reference() -> None:
    """Per-tensor scaling on a grid whose 176-wide tiles fill as many waves as the 192-wide
    ones, issued as 176-wide WGMMAs, with an M tail and a 16-column N tail."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    # Two row tiles by ``sm_count // 2`` 176-wide column tiles fill one wave, as the
    # 192-wide ones do; the 128-wide ones take two.
    test = GemmFP8Test(248, 176 * (sm_count // 2) - 16, 256, torch.float8_e4m3fn, "per_tensor")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_per_tensor_192_wide_matches_reference() -> None:
    """Per-tensor scaling on a grid whose 192-wide tiles fill one wave while the 128-wide
    ones need two, with a 64-column N tail."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    # Two row tiles by ``sm_count // 2`` 192-wide column tiles fill one wave; the 128-wide
    # tiles, half as many again, take two.
    test = GemmFP8Test(256, 192 * (sm_count // 2) - 128, 256, torch.float8_e4m3fn, "per_tensor")
    test.check(GemmFP8FwdOp(target=BUILTIN), *test.gen_inputs())


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
def test_gemm_w4a16_split_wide_tile_matches_the_reference() -> None:
    """128 rows of K 25600: a split 256-thread tile reloads its metadata every K tile."""
    test = GemmW4A16Test(128, 5120, 25600, torch.float16)
    test.check(GemmW4A16FwdOp(target=BUILTIN), *test.gen_inputs())


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
