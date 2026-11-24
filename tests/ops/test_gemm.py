from types import SimpleNamespace

import pytest
import torch

from tests.misaligned import misaligned
from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops._kernels.gemm import GEMMINT8W8A8Call, GEMMW4A8Call, GEMMW4A8PrefillKernel
from tileops._kernels.gemm.call_spec import GEMMCall
from tileops._kernels.gemm.dense import GEMMCpAsyncKernel, GEMMTMAKernel, GemvKernel
from tileops._kernels.gemm.heuristics import best_config, int8_pingpong_grid
from tileops._kernels.gemm.w4a8_decode import _select_config as select_w4a8_decode_config
from tileops._kernels.gemm.w4a16 import (
    _cross_tile_wgmma,
    _drained_ring,
    _meta_window,
    _pair_reduced,
    _select_config,
)
from tileops._utils import calibration_key, get_sm_count
from tileops.backend import BUILTIN
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from tileops.gemm import GEMMFP8FwdOp, GEMMFwdOp, GEMMINT8W8A8FwdOp, GEMMW4A8FwdOp, GEMMW4A16FwdOp
from workloads.device import run_device
from workloads.gemm import (
    GEMMFP8Workload,
    GEMMINT8W8A8Workload,
    GEMMW4A8MixedSignWorkload,
    GEMMW4A8Workload,
    GEMMW4A16BasisWorkload,
    GEMMW4A16Workload,
    GEMMWorkload,
    repack_w4a8_weight,
    repack_w4a16_weight,
)
from workloads.numerics import compare_outputs


class GEMMTest(GEMMWorkload, TestBase):
    pass


class GEMMFP8Test(GEMMFP8Workload, TestBase):
    pass


class GEMMINT8W8A8Test(GEMMINT8W8A8Workload, TestBase):
    pass


class GEMMW4A16Test(GEMMW4A16Workload, TestBase):
    pass


class GEMMW4A8Test(GEMMW4A8Workload, TestBase):
    pass


class GEMMW4A8MixedSignTest(GEMMW4A8MixedSignWorkload, TestBase):
    pass


class GEMMFixture(FixtureBase):
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
                # The layouts whose A or B tile is MN-major, on the same M / N tails.
                pytest.param(
                    1440,
                    4600,
                    512,
                    torch.bfloat16,
                    True,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-tn-mn-tail",
                ),
                pytest.param(
                    1440,
                    4600,
                    512,
                    torch.bfloat16,
                    True,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-tt-mn-tail",
                ),
                pytest.param(
                    1440,
                    4600,
                    512,
                    torch.bfloat16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-nn-mn-tail",
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
                # 12 row tiles of 11 x 160 + 2 x 176, past 64 K steps (ping-pong over
                # coop2).
                pytest.param(
                    4000,
                    2112,
                    7168,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-176-narrow-160-m-tail",
                ),
                # The last 20 row tiles start with 40 tiles 144 wide; the full rows end in
                # a 128-column tail tile, and the last K step is partial.
                pytest.param(
                    4000,
                    7168,
                    1032,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-pingpong-176-narrow-144-mnk-tail",
                ),
                # 192-wide tiles whose last 24 row tiles start with 39 tiles 160 wide and
                # end in a 160-column tail tile.
                pytest.param(
                    3000,
                    7168,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-192-narrow-160-mn-tail",
                ),
                # 256-wide coop2 tiles whose last 8 row tiles start with 16 tiles 240 wide;
                # the last K step is partial.
                pytest.param(
                    4000,
                    4096,
                    1032,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-coop2-256-narrow-240-mk-tail",
                ),
                # Every tile 144 wide: 30 narrow columns fill all 8 row tiles.
                pytest.param(
                    1000,
                    4320,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-pingpong-176-all-narrow-m-tail",
                ),
                # Every row tile narrow, each ending in 176-wide tiles and an N tail.
                pytest.param(
                    1000,
                    5504,
                    1024,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-pingpong-176-narrow-rows-only-mn-tail",
                ),
                # Every tile 240 wide on the coop2 256 grid.
                pytest.param(
                    1000,
                    6720,
                    1024,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-coop2-256-all-narrow-m-tail",
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
                    8448,
                    2048,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-simple-cluster",
                ),
                # One-wave fill tiles: a width no basic tile has, two M tiles as a cluster.
                pytest.param(
                    128,
                    7168,
                    2048,
                    torch.bfloat16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-fill-cluster",
                ),
                # A partial second M tile, an N tail and a K tail.
                pytest.param(
                    120,
                    7000,
                    1032,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-fill-tails",
                ),
            ],
        ),
    ]


class GemvBoundaryFixture(FixtureBase):
    """GEMV outputs remain correct for non-aligned free and contraction dimensions."""

    PARAMS = [
        (
            "n, k, dtype",
            [
                # lhs_row: m=1, trans_b=True — non-aligned n
                pytest.param(3000, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(3000, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                # lhs_row: non-aligned k
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                # rhs_col: n=1 — non-aligned m (mapped to gemv n param)
                pytest.param(3001, 1024, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class GEMMFP8Fixture(FixtureBase):
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
                    256,
                    torch.float8_e4m3fn,
                    "block128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128",
                ),
                pytest.param(
                    128,
                    256,
                    512,
                    torch.float8_e4m3fn,
                    "block128x128",
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp8-e4m3-block128x128",
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


class GEMMW4A16Fixture(FixtureBase):
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
                    marks=pytest.mark.full,
                    id="full-w4a16-rect",
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
                    17,
                    65,
                    256,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-small-mn-tail",
                ),
                pytest.param(
                    40,
                    4096,
                    1024,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-deferred-scale-32-rows",
                ),
            ],
        ),
    ]


class GEMMW4A8Fixture(FixtureBase):
    PARAMS = [
        (
            "m, n, k",
            [
                pytest.param(1, 256, 512, marks=pytest.mark.smoke, id="smoke-w4a8-m1"),
                pytest.param(64, 256, 512, marks=pytest.mark.smoke, id="smoke-w4a8-m64"),
                pytest.param(
                    385, 2560, 128, marks=pytest.mark.full, id="full-w4a8-m385-n2560-k128"
                ),
                # Token counts on both sides of 32 and off the 64-row tile.
                pytest.param(8, 512, 1024, marks=pytest.mark.full, id="full-w4a8-m8"),
                pytest.param(32, 512, 1024, marks=pytest.mark.full, id="full-w4a8-m32"),
                pytest.param(33, 512, 1024, marks=pytest.mark.full, id="full-w4a8-m33"),
                # 112 row tiles on 132 SMs run a tile a CTA, with a ragged last tile and an odd
                # K-group count.
                pytest.param(
                    65, 14320, 384, marks=pytest.mark.full, id="full-w4a8-m65-one-wave-ragged-odd-k"
                ),
                # Short-K model projections at small token counts: one CTA per 64-row tile,
                # its 256-K items split over two units in the CTA.
                pytest.param(16, 7168, 2304, marks=pytest.mark.full, id="full-w4a8-decode-ds-m16"),
                pytest.param(32, 7168, 2304, marks=pytest.mark.full, id="full-w4a8-decode-ds-m32"),
                # Past 2048 K a unit: two-warpgroup units on 128-row tiles split across CTAs,
                # with a short last item.
                pytest.param(
                    16,
                    4096,
                    4224,
                    marks=pytest.mark.full,
                    id="full-w4a8-decode-two-warpgroup-units",
                ),
                # A short-K TP8 down projection at 64 tokens.
                pytest.param(64, 8192, 3584, marks=pytest.mark.full, id="full-w4a8-llama-tp8-m64"),
                pytest.param(128, 256, 512, marks=pytest.mark.full, id="full-w4a8-m128"),
                pytest.param(512, 512, 1024, marks=pytest.mark.full, id="full-w4a8-m512"),
                # N with 16, 32 and 48 rows past the last 64-row block, and an odd count of
                # 128-wide K blocks, at one token count on each side of 32.
                pytest.param(1, 16, 128, marks=pytest.mark.full, id="full-w4a8-m1-n16-k128"),
                pytest.param(8, 80, 384, marks=pytest.mark.full, id="full-w4a8-m8-n80-k384"),
                pytest.param(8, 96, 384, marks=pytest.mark.full, id="full-w4a8-m8-n96-k384"),
                pytest.param(8, 112, 384, marks=pytest.mark.full, id="full-w4a8-m8-n112-k384"),
                pytest.param(33, 16, 128, marks=pytest.mark.full, id="full-w4a8-m33-n16-k128"),
                pytest.param(80, 80, 384, marks=pytest.mark.full, id="full-w4a8-m80-n80-k384"),
                pytest.param(96, 96, 640, marks=pytest.mark.full, id="full-w4a8-m96-n96-k640"),
                pytest.param(64, 112, 384, marks=pytest.mark.full, id="full-w4a8-m64-n112-k384"),
                # N a multiple of 64 but not of 128.
                pytest.param(64, 192, 384, marks=pytest.mark.full, id="full-w4a8-m64-n192-k384"),
                # A large token count on a narrow output projection.
                pytest.param(
                    4096, 2304, 128, marks=pytest.mark.full, id="full-w4a8-m4096-n2304-k128"
                ),
                # 16 tokens on the smallest weight.
                pytest.param(16, 16, 128, marks=pytest.mark.full, id="full-w4a8-m16-n16-k128"),
                # Contraction sizes with an uneven count of 128-element groups.
                pytest.param(32, 272, 6784, marks=pytest.mark.full, id="full-w4a8-m32-n272-k6784"),
                pytest.param(
                    1, 4096, 16512, marks=pytest.mark.full, id="full-w4a8-m1-n4096-k16512"
                ),
                # The K of an eight-way tensor-parallel DeepSeek-R1 down projection.
                pytest.param(64, 7168, 2304, marks=pytest.mark.full, id="full-w4a8-m64-k2304"),
            ],
        ),
    ]


@GEMMFixture
def test_gemm(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    trans_a: bool,
    trans_b: bool,
    tune: bool,
    fp32_gemm_reduction,
) -> None:
    test = GEMMTest(m, n, k, dtype, trans_a, trans_b)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_a_float32_gemm_output_rejects_one_rounded_to_bfloat16() -> None:
    workload = GEMMWorkload(4, 8, 64, torch.bfloat16, out_dtype=torch.float32)
    a, b = torch.randn(4, 64, dtype=torch.bfloat16), torch.randn(64, 8, dtype=torch.bfloat16)
    exact = workload.ref_program(a, b)
    compare_outputs(exact, exact, workload.verification(a, b))
    with pytest.raises(AssertionError):
        compare_outputs(exact.bfloat16().float(), exact, workload.verification(a, b))


@GEMMFP8Fixture
def test_gemm_fp8(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    scale_mode: str,
    out_dtype: torch.dtype,
    bias: bool,
) -> None:
    test = GEMMFP8Test(m, n, k, dtype, scale_mode, out_dtype=out_dtype, bias=bias)
    op = GEMMFP8FwdOp(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    if dtype != torch.float8_e4m3fn:
        with pytest.raises(ValueError, match=r"outside \['float8_e4m3fn'\]"):
            op(*inputs)
        return
    test.check(op, *inputs)


_INT8_WS, _INT8_PP, _INT8_SIMPLE = (
    "gemm_int8_w8a8",
    "gemm_int8_w8a8_pingpong",
    "gemm_int8_w8a8_simple",
)


class GEMMINT8W8A8Fixture(FixtureBase):
    PARAMS = [
        (
            "m, n, k, per_token, per_channel, out_dtype, bias, key",
            [
                # Each output dtype, on the 64-row kernel's 256-wide K step (m = 64).
                pytest.param(
                    64,
                    512,
                    1024,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.smoke,
                    id="smoke-bf16",
                ),
                pytest.param(
                    64,
                    512,
                    1024,
                    True,
                    True,
                    torch.float16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.smoke,
                    id="smoke-fp16",
                ),
                # Ping-pong on 132 SMs (the off-16-byte-boundary test runs it in smoke): M / N /
                # K tails, per-channel scales and a bias, and
                # 192-wide tiles whose last 8 row tiles start with 14 tiles 160 wide, 376 tiles:
                # more than two waves, so each CTA runs both consumers and hands the turn back
                # and forth.
                pytest.param(
                    2200,
                    3720,
                    1040,
                    True,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_PP,
                    marks=pytest.mark.full,
                    id="full-pingpong-tails",
                ),
                # On 132 SMs (H200) the narrowest one-wave width of these N is 80, 112 and 192,
                # each tile issued whole by T.wgmma_gemm: a column tail on each, a one-row A box
                # at m = 1, a K tail at m = 64 and a partial row tile at m = 37.
                pytest.param(
                    1,
                    10008,
                    1040,
                    False,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-80-wide",
                ),
                pytest.param(
                    64,
                    13000,
                    1040,
                    True,
                    True,
                    torch.float16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-112-wide",
                ),
                pytest.param(
                    37,
                    22000,
                    1040,
                    True,
                    False,
                    torch.bfloat16,
                    True,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-192-wide",
                ),
                # k off TMA's 16-byte unit: the fallback, which the off-16-byte-boundary test
                # runs in smoke.
                pytest.param(
                    37,
                    104,
                    200,
                    True,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_SIMPLE,
                    marks=pytest.mark.full,
                    id="full-unaligned-k",
                ),
                # The other scale granularities: one activation scale, one weight scale.
                pytest.param(
                    64,
                    512,
                    1024,
                    False,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-tensor-scale-a",
                ),
                pytest.param(
                    64,
                    512,
                    1024,
                    True,
                    False,
                    torch.bfloat16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-tensor-scale-b",
                ),
                pytest.param(
                    64,
                    512,
                    1024,
                    True,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-bias",
                ),
                # One row: a one-row TMA box and the elementwise store.
                pytest.param(
                    1,
                    512,
                    1024,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-one-row",
                ),
                # A box of fewer rows than a tile, an N tail and a K tail past the 128-wide
                # step.
                pytest.param(
                    37,
                    520,
                    1040,
                    True,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-row-box-n-k-tails",
                ),
                # Two row tiles, the second partial: the staged store's elementwise branch.
                pytest.param(
                    100,
                    512,
                    1024,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_WS,
                    marks=pytest.mark.full,
                    id="full-partial-row-tile",
                ),
                # Ping-pong on 132 SMs: fp16 and tensor scales, and 192-wide tiles whose last 8
                # row tiles start with three tiles 32 wide, 282 tiles; M / N / K tails.
                pytest.param(
                    2400,
                    2568,
                    1040,
                    False,
                    False,
                    torch.float16,
                    True,
                    _INT8_PP,
                    marks=pytest.mark.full,
                    id="full-pingpong-tensor-scales-bias",
                ),
                # Ping-pong 192-wide tiles with no narrow ones on 132 SMs, the 64-column store
                # boxes: M / N tails, a bias, and one K step, whose only stage the drain after
                # the mainloop releases.
                pytest.param(
                    1900,
                    1032,
                    16,
                    True,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_PP,
                    marks=pytest.mark.full,
                    id="full-pingpong-192-uniform-bias",
                ),
                # Ping-pong on 132 SMs, one wave of 92 tiles: 192-wide tiles whose four row
                # tiles each start with 23 tiles 96 wide; M and K tails.
                pytest.param(
                    400,
                    2208,
                    1040,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_PP,
                    marks=pytest.mark.full,
                    id="full-pingpong-narrow-96",
                ),
                # Ping-pong 128-wide tiles on 132 SMs, 392 tiles: M / N / K tails over more
                # than two waves, fp16 with tensor scales and a bias.
                pytest.param(
                    1700,
                    3464,
                    1040,
                    False,
                    False,
                    torch.float16,
                    True,
                    _INT8_PP,
                    marks=pytest.mark.full,
                    id="full-pingpong-128-tails-bias",
                ),
                # n off the output's 16-byte TMA row pitch: the fallback.
                pytest.param(
                    37,
                    101,
                    208,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_SIMPLE,
                    marks=pytest.mark.full,
                    id="full-unaligned-n",
                ),
            ],
        ),
    ]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@GEMMINT8W8A8Fixture
def test_gemm_int8_w8a8(
    m: int,
    n: int,
    k: int,
    per_token: bool,
    per_channel: bool,
    out_dtype: torch.dtype,
    bias: bool,
    key: str,
    arch: int,
) -> None:
    test = GEMMINT8W8A8Test(m, n, k, per_token, per_channel, out_dtype=out_dtype, bias=bias)
    op = GEMMINT8W8A8FwdOp(out_dtype=out_dtype)
    call = GEMMINT8W8A8Call(
        m=m,
        n=n,
        k=k,
        per_token=per_token,
        per_channel=per_channel,
        out_dtype=out_dtype,
        has_bias=bias,
        device=run_device(),
    )
    assert op.key_for("gemm_int8_w8a8", call) == (_INT8_SIMPLE if arch == 89 else key)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "m, n, grid",
    [
        # The ping-pong cases above: 160-, 96- and 32-wide narrow tiles, and 192 with none.
        pytest.param(2200, 3720, (192, 160, 8, 14), marks=pytest.mark.smoke, id="narrow-160"),
        pytest.param(400, 2208, (192, 96, 4, 23), marks=pytest.mark.smoke, id="narrow-96"),
        pytest.param(2400, 2568, (192, 32, 8, 3), marks=pytest.mark.smoke, id="narrow-32"),
        pytest.param(1900, 1032, (192, 0, 0, 0), marks=pytest.mark.smoke, id="uniform-192"),
        # Equal busiest-CTA columns, but 192-wide tiles give one consumer of a CTA more
        # columns of epilogue: 288 against 256 (the 128-wide case above), 352 against 256.
        pytest.param(1700, 3464, (128, 0, 0, 0), marks=pytest.mark.smoke, id="uniform-128"),
        pytest.param(2048, 4096, (128, 0, 0, 0), marks=pytest.mark.smoke, id="consumer-tie"),
        # 192-wide tiles give the busiest CTA more columns: 192 against 128.
        pytest.param(300, 512, (128, 0, 0, 0), marks=pytest.mark.smoke, id="cta-more"),
    ],
)
def test_int8_pingpong_grid_on_132_sms(m: int, n: int, grid: tuple) -> None:
    """The ping-pong grid's tile width and narrow tiles on a 132-SM H200, which the
    ping-pong cases above rely on to reach each path."""
    got = int8_pingpong_grid(m, n, 132, 16)
    assert (got["block_n"], got["narrow_n"], got["narrow_rows"], got["narrow_cols"]) == grid


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "shape, scales, out_dtype, bias, tile",
    [
        pytest.param(
            (128, 520, 1040),
            (True, True),
            torch.bfloat16,
            False,
            (96, 128),
            marks=pytest.mark.smoke,
            id="smoke-96-k128",
        ),
        pytest.param(
            (64, 520, 1040),
            (False, False),
            torch.float16,
            True,
            (96, 256),
            marks=pytest.mark.full,
            id="full-96-k256-tensor-scales",
        ),
        pytest.param(
            (37, 1000, 1040),
            (True, True),
            torch.bfloat16,
            True,
            (160, 128),
            marks=pytest.mark.full,
            id="full-160-k128-row-box",
        ),
        pytest.param(
            (64, 1000, 1040),
            (True, False),
            torch.bfloat16,
            False,
            (160, 256),
            marks=pytest.mark.full,
            id="full-160-k256",
        ),
    ],
)
def test_gemm_int8_w8a8_native_width_tiles(
    shape: tuple, scales: tuple, out_dtype: torch.dtype, bias: bool, tile: tuple
) -> None:
    """Pin native tiles to cover K atoms, tail stores, scale modes and bias on any SM90."""
    test = GEMMINT8W8A8Test(*shape, *scales, out_dtype=out_dtype, bias=bias)
    op = GEMMINT8W8A8FwdOp(out_dtype=out_dtype, target=BUILTIN)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    (kernel,) = [kern for kern in op.iter_kernels() if type(kern).__name__ == "GEMMINT8W8A8Kernel"]
    kernel.config = {"block_n": tile[0], "num_stages": 2, "group_size_m": 8, "block_k": tile[1]}
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    "key, n",
    [
        pytest.param(_INT8_WS, 512, marks=pytest.mark.sm90),
        pytest.param(_INT8_PP, 512, marks=pytest.mark.sm90),
        pytest.param(_INT8_SIMPLE, 512, marks=pytest.mark.sm89),
        # n off a multiple of 8, which the TMA kernels refuse.
        pytest.param(_INT8_SIMPLE, 510, marks=pytest.mark.sm90),
    ],
)
def test_gemm_int8_w8a8_operands_off_a_16_byte_boundary(key: str, n: int) -> None:
    """A contiguous view that starts one byte into its allocation: no TMA descriptor can
    address it, including the one SM90 lowers the simple kernel's copies to, and SM89's
    vectorized loads fault on it; each kernel is selected and still runs the call."""
    m = 64 if key == _INT8_WS else 300
    test = GEMMINT8W8A8Test(m, n, 1024, True, True)
    a, b, a_scale, b_scale = test.gen_inputs()
    a_off = torch.empty(a.numel() + 1, dtype=torch.int8, device=a.device)[1:].view(a.shape)
    b_off = torch.empty(b.numel() + 1, dtype=torch.int8, device=b.device)[1:].view(b.shape)
    a_off.copy_(a)
    b_off.copy_(b)
    assert a_off.is_contiguous() and a_off.data_ptr() % 16
    op = GEMMINT8W8A8FwdOp()
    call = GEMMINT8W8A8Call(
        m=m,
        n=n,
        k=1024,
        per_token=True,
        per_channel=True,
        out_dtype=torch.bfloat16,
        device=run_device(),
    )
    assert op.key_for("gemm_int8_w8a8", call) == key
    test.check(op, a_off, b_off, a_scale, b_scale)


def _h200_w4a8_call(m: int, n: int, k: int) -> GEMMW4A8Call:
    """A W4A8 call stating the facts of a 132-SM H200, the board its selection is calibrated
    for, so that selection runs on any host."""
    device = torch.device("cuda", 0)  # lint: device-literal allow: facts are stated
    return GEMMW4A8Call(
        m=m,
        n=n,
        k=k,
        group_size=128,
        arch=90,
        sm_count=132,
        calibration="h200",
        smem_budget=BLOCK_SHARED_BYTES_OPT_IN[90],
        device=device,
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, key",
    [
        pytest.param(32, 512, 1024, "gemm_w4a8_decode", id="base-decode-upper-endpoint"),
        # Past one wave of 128-row tiles, 512-K items split unevenly: the 40-token tile.
        pytest.param(33, 133 * 128, 7168, "gemm_w4a8_decode", id="one-tile-past-a-wave"),
        # 64 KiB power-of-two weight blocks, below the stride that slows the 40-token tile.
        pytest.param(40, 288 * 128, 2048, "gemm_w4a8_decode", id="past-two-waves"),
        pytest.param(33, 144 * 128, 17408, "gemm_w4a8_decode", id="non-power-of-two-k"),
        # Within one wave, a token count past the tile, ragged N, 256-K items, items that
        # divide evenly over the SMs, and power-of-two weight blocks stay with prefill.
        pytest.param(33, 131 * 128, 7168, "gemm_w4a8_prefill", id="within-one-wave"),
        pytest.param(41, 144 * 128, 7168, "gemm_w4a8_prefill", id="token-past-the-tile"),
        pytest.param(33, 144 * 128 + 64, 7168, "gemm_w4a8_prefill", id="ragged-n"),
        pytest.param(33, 144 * 128, 7296, "gemm_w4a8_prefill", id="k-off-512"),
        pytest.param(33, 231 * 128, 8192, "gemm_w4a8_prefill", id="items-divide-evenly"),
        pytest.param(33, 144 * 128, 16384, "gemm_w4a8_prefill", id="power-of-two-k"),
    ],
)
def test_w4a8_forty_token_dispatch(m: int, n: int, k: int, key: str) -> None:
    """The 40-token selection on a 132-SM H200."""
    assert GEMMW4A8FwdOp(target=BUILTIN).key_for("gemm_w4a8", _h200_w4a8_call(m, n, k)) == key


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, grid",
    [
        # One CTA per 64-row tile. Nine 256-K items over two units, five on the busier one;
        # a sixth ring stage would push the partial sums out of shared memory.
        pytest.param(32, 7168, 2304, (1, 112, 5), id="k-off-512"),
        # Eight 512-K items: 2048 K a unit at the limit, 2560 one item past it.
        pytest.param(16, 4096, 4096, (1, 64, 3), id="unit-k-at-limit"),
        pytest.param(16, 4096, 4608, (2, 132, 3), id="unit-k-past-limit"),
        pytest.param(1, 4096, 4096, (1, 64, 4), id="eight-token-tile"),
        # The 8-token tile meets the same limit: a rule exempting it fails only this case.
        pytest.param(8, 4096, 4608, (1, 132, 4), id="eight-token-unit-k-past-limit"),
        # 132 row tiles are one wave; 133 are not, and split tiles across CTAs.
        pytest.param(32, 132 * 64, 2048, (1, 132, 2), id="one-wave-of-row-tiles"),
        pytest.param(32, 133 * 64, 2048, (1, 132, 2), id="row-tiles-past-a-wave"),
    ],
)
def test_w4a8_decode_exchange_free_grid(m: int, n: int, k: int, grid: tuple[int, int, int]) -> None:
    """One CTA per 64-row tile with the partial sums in shared memory; ``grid`` is the
    dispatched unit warpgroups, CTAs and ring stages on a 132-SM H200."""
    call = _h200_w4a8_call(m, n, k)
    assert GEMMW4A8FwdOp(target=BUILTIN).key_for("gemm_w4a8", call) == "gemm_w4a8_decode"
    cfg = select_w4a8_decode_config(m, n, k, call.smem_budget, call.sm_count)
    assert (cfg["unit_warpgroups"], cfg["num_ctas"], cfg["num_stages"]) == grid


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, stream_k",
    [
        # 99 of 132 SMs hold a tile (exactly three quarters), 8 idle K groups an SM.
        pytest.param(33, 99 * 128, 4096, 0, id="three-quarters-full"),
        pytest.param(33, 98 * 128, 4096, 2, id="under-three-quarters"),
        # 99 tiles: 64 K groups leave exactly 16 idle groups an SM, 65 leave 16.25.
        pytest.param(33, 99 * 128, 64 * 128, 0, id="idle-groups-at-16"),
        pytest.param(33, 99 * 128, 65 * 128, 2, id="idle-groups-over-16"),
        pytest.param(33, 132 * 128, 4096, 2, id="whole-wave"),
    ],
)
def test_gemm_w4a8_prefill_runs_a_full_one_wave_grid_a_tile_a_cta(
    m: int, n: int, k: int, stream_k: int
) -> None:
    """The cases count row tiles and idle K groups against the 132 SMs of an H200, the board
    the schedule selection is calibrated for."""
    call = _h200_w4a8_call(m, n, k)
    assert GEMMW4A8FwdOp(target=BUILTIN).key_for("gemm_w4a8", call) == "gemm_w4a8_prefill"
    # The default config reads only the shape and the SM count; building the kernel compiles it.
    kernel = object.__new__(GEMMW4A8PrefillKernel)
    kernel.m, kernel.n, kernel.k = m, n, k
    kernel.build_target = SimpleNamespace(sm_count=call.sm_count)
    assert kernel.default_config["stream_k"] == stream_k


@GEMMW4A16Fixture
def test_gemm_w4a16(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = GEMMW4A16Test(m, n, k, dtype)
    op = GEMMW4A16FwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.sm90
def test_gemm_w4a16_a_published_tile_height_pads_m_for_itself() -> None:
    """The M padding follows the published tile: a whole-tile height after the narrow decode
    tile pads M to its own extent, and a return to the narrow tile runs on logical M."""
    test = GEMMW4A16Test(8, 256, 512, torch.float16)
    op = GEMMW4A16FwdOp()
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    (kernel,) = [k for k in op.iter_kernels() if type(k).__name__ == "GEMMW4A16Kernel"]
    narrow = dict(kernel.config)
    assert narrow["block_m"] == 8
    for block_m, padded in ((64, 64), (8, 8)):
        kernel.config = {**narrow, "block_m": block_m}
        test.check(op, *inputs)
        ((_, m_pad),) = kernel._generation.programs.values()
        assert m_pad == padded


@pytest.mark.parametrize(
    "k_index",
    [pytest.param(0, marks=pytest.mark.smoke)]
    + [pytest.param(i, marks=pytest.mark.full) for i in (1, 127, 128, 200, 383)],
)
def test_gemm_w4a16_is_exact_on_a_basis_vector(k_index: int) -> None:
    """A basis vector selects the exact signed-INT4 value, group scale and zero point."""
    workload = GEMMW4A16BasisWorkload(k_index)
    TestBase.check(workload, GEMMW4A16FwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_gemm_fp8_block128_single_k_block_uses_block_kernel() -> None:
    test = GEMMFP8Test(128, 256, 128, torch.float8_e4m3fn, "block128")
    op = GEMMFP8FwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.sm89
@pytest.mark.parametrize(
    "scale_mode",
    [
        pytest.param("per_tensor", marks=pytest.mark.smoke),
        pytest.param("block128", marks=pytest.mark.full),
        pytest.param("block128x128", marks=pytest.mark.full),
    ],
)
def test_gemm_fp8_without_tma_or_wgmma(scale_mode: str) -> None:
    """SM89 has FP8 tensor cores but no TMA or WGMMA, so every scale grid runs without them."""
    test = GEMMFP8Test(256, 512, 1024, torch.float8_e4m3fn, scale_mode)
    test.check(GEMMFP8FwdOp(), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_fp8_1d2d_shared_epilogue_matches_reference() -> None:
    """Per-block scales preserve the complete output of a narrow projection."""
    test = GEMMFP8Test(128, 2112, 7168, torch.float8_e4m3fn, "block128x128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(4096, 576, 7168, marks=pytest.mark.smoke, id="one-wave"),
        pytest.param(4096, 7168, 16384, marks=pytest.mark.full, id="two-wave"),
    ],
)
def test_gemm_fp8_1d1d_wave_matches_reference(m: int, n: int, k: int) -> None:
    """Per-row weight scales preserve every output on tall projections."""
    test = GEMMFP8Test(m, n, k, torch.float8_e4m3fn, "block128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(128, 32768, 512, marks=pytest.mark.smoke, id="one-row-tile"),
        pytest.param(4096, 32768, 512, marks=pytest.mark.full, id="two-wave"),
        pytest.param(128, 7168, 16384, marks=pytest.mark.full, id="k-split-pair"),
        pytest.param(4096, 2112, 7168, marks=pytest.mark.full, id="176-wide"),
    ],
)
def test_gemm_fp8_1d2d_wave_matches_reference(m: int, n: int, k: int) -> None:
    """Per-block weight scales preserve every output on tall and wide projections."""
    test = GEMMFP8Test(m, n, k, torch.float8_e4m3fn, "block128x128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "n, k",
    [
        pytest.param(4096, 7168, marks=pytest.mark.smoke, id="column-run"),
        pytest.param(24576, 1536, marks=pytest.mark.full, id="scale-span"),
    ],
)
def test_gemm_fp8_1d2d_wave_column_run_matches_reference(n: int, k: int) -> None:
    """Column-scale blocks and reused row-scale spans preserve every output."""
    test = GEMMFP8Test(4096, n, k, torch.float8_e4m3fn, "block128x128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "n, scale_mode",
    [
        pytest.param(1024, "per_tensor", marks=pytest.mark.smoke, id="per-tensor"),
        pytest.param(576, "block128x128", marks=pytest.mark.full, id="block128x128"),
    ],
)
def test_gemm_fp8_single_row_tile_matches_reference(n: int, scale_mode: str) -> None:
    """Per-tensor and per-block scale grids both preserve narrow-row projections."""
    test = GEMMFP8Test(128, n, 7168, torch.float8_e4m3fn, scale_mode)
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_fp8_1d2d_deepest_ring_launches() -> None:
    """Repeated scale groups preserve every output across the contraction."""
    test = GEMMFP8Test(128, 4096, 2560, torch.float8_e4m3fn, "block128x128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_fp8_per_tensor_176_wide_matches_reference() -> None:
    """Per-tensor scaling preserves ragged row and column boundaries."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    test = GEMMFP8Test(248, 176 * (sm_count // 2) - 16, 256, torch.float8_e4m3fn, "per_tensor")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_fp8_per_tensor_192_wide_matches_reference() -> None:
    """Per-tensor scaling preserves a projection with a partial column boundary."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    test = GEMMFP8Test(256, 192 * (sm_count // 2) - 128, 256, torch.float8_e4m3fn, "per_tensor")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("m", "n", "k", "scale_mode", "out_dtype", "tune"),
    [
        # Blocks short of the SMs and a long K: K split into slices and reduced into an output
        # whose rows TMA cannot store; 12 rows of a padded to 16.
        pytest.param(
            12,
            1003,
            3072,
            "block128x128",
            torch.float16,
            False,
            marks=pytest.mark.smoke,
            id="split",
        ),
        # More 64-row blocks than SMs: CTAs stream several blocks each and the last CTA's
        # last one is past N; a partial last block and 128-row scale block; 16 rows of a.
        pytest.param(
            16, None, 512, "block128x128", torch.bfloat16, False, marks=pytest.mark.full, id="units"
        ),
        # One row of a over one K step: a's block scale is (1, 1), b's tell the grids apart;
        # a ring of one stage.
        pytest.param(
            1,
            256,
            128,
            "block128x128",
            torch.bfloat16,
            False,
            marks=pytest.mark.full,
            id="one-step",
        ),
        # The same blocks over a K too short to split: the unsplit tile stored row by row.
        pytest.param(
            12,
            1003,
            1024,
            "block128x128",
            torch.bfloat16,
            False,
            marks=pytest.mark.full,
            id="masked",
        ),
        # Per-tensor scales on every K slice of a split call, tuned over two ring depths: the
        # deepest ring holds fewer K steps than a CTA streams.
        pytest.param(
            5,
            1280,
            16384,
            "per_tensor",
            torch.bfloat16,
            True,
            marks=pytest.mark.full,
            id="split-pt",
        ),
        # Scales staged for a long K hold the ring to its shallowest depth.
        pytest.param(
            16,
            64,
            393216,
            "block128x128",
            torch.bfloat16,
            False,
            marks=pytest.mark.full,
            id="scales-fill-smem",
        ),
        # Scales staged for a longer K leave no room for the ring: another kernel serves it.
        pytest.param(
            16,
            64,
            524288,
            "block128x128",
            torch.bfloat16,
            False,
            marks=pytest.mark.full,
            id="scales-past-smem",
        ),
    ],
)
def test_gemm_fp8_short_call_streams_b_in_64_row_blocks(
    m: int, n: int | None, k: int, scale_mode: str, out_dtype: torch.dtype, tune: bool
) -> None:
    """A call of few rows runs as c^T = b @ a^T: WGMMA's 64-row M axis takes b's rows and its
    N axis, a multiple of 8, takes a's."""
    sm_count = get_sm_count(torch.device(run_device()).index)
    n = 64 * (sm_count + 9) - 24 if n is None else n
    test = GEMMFP8Test(m, n, k, torch.float8_e4m3fn, scale_mode, out_dtype=out_dtype)
    op = GEMMFP8FwdOp(out_dtype=out_dtype)
    if tune:
        op.request_tune()
        # The first call tunes and returns the untuned result; the check reads the tuned one.
        op(*test.gen_inputs())
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, arch: int) -> None:
    """A single activation row supports non-aligned weight rows and contraction sizes."""
    test = GEMMTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GEMMFwdOp(trans_a=False, trans_b=True)
    test.check(op, *test.gen_inputs())
    assert all(isinstance(entry, GemvKernel) for entry in op._entries_built("gemm").values())


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, arch: int) -> None:
    """A single output column supports non-aligned activation rows and contraction sizes."""
    m = n
    test = GEMMTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GEMMFwdOp(trans_a=False, trans_b=False)
    test.check(op, *test.gen_inputs())
    assert all(isinstance(entry, GemvKernel) for entry in op._entries_built("gemm").values())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gemm_coop2_epilogue_staged_in_two_chunks() -> None:
    """A wide output preserves every column across the epilogue."""
    test = GEMMTest(704, 5440, 128, torch.bfloat16, False, True)
    test.check(GEMMFwdOp(trans_b=True, target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("dtype", "trans_a", "trans_b", "block_n", "options"),
    [
        pytest.param(torch.bfloat16, True, False, 160, {}, marks=pytest.mark.smoke),
        pytest.param(torch.float16, True, False, 160, {}, marks=pytest.mark.smoke),
        # Raster groups of 4 row blocks.
        pytest.param(torch.bfloat16, True, True, 128, {"group_size_m": 4}, marks=pytest.mark.full),
        pytest.param(torch.bfloat16, False, False, 128, {}, marks=pytest.mark.full),
        pytest.param(torch.bfloat16, False, True, 160, {}, marks=pytest.mark.full),
        # Row blocks of 6 wide and 1 narrow tile, then of 4 wide and 3 narrow.
        pytest.param(
            torch.bfloat16,
            True,
            False,
            160,
            {"narrow_n": 128, "row_mix": [2, 6, 1, 4, 3]},
            marks=pytest.mark.full,
        ),
        pytest.param(
            torch.bfloat16,
            False,
            False,
            160,
            {"narrow_n": 128, "row_mix": [2, 6, 1, 4, 3]},
            marks=pytest.mark.full,
        ),
        # The same runs in raster groups of 2: the 3-block runs end in a partial group.
        pytest.param(
            torch.bfloat16,
            True,
            False,
            160,
            {"narrow_n": 128, "row_mix": [2, 6, 1, 4, 3], "group_size_m": 2},
            marks=pytest.mark.full,
        ),
        # Narrow tiles of whole 64-column swizzle atoms in a 128-wide stage.
        pytest.param(
            torch.bfloat16,
            True,
            False,
            128,
            {"narrow_n": 64, "row_mix": [2, 7, 2, 6, 4]},
            marks=pytest.mark.full,
        ),
    ],
)
def test_gemm_coop2_swap_matches_reference(
    dtype: torch.dtype, trans_a: bool, trans_b: bool, block_n: int, options: dict
) -> None:
    """The transposed 256-row tile stores every output past M, N and K tails, in each layout,
    with and without narrow tiles."""
    test = GEMMTest(1000, 1032, 1048, dtype, trans_a, trans_b)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b, target=BUILTIN)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    (kernel,) = [k for k in op.iter_kernels() if type(k).__name__ == "GEMMTMAKernel"]
    kernel.config = {
        "coop2_swap": True,
        "block_n": block_n,
        "num_stages": 4,
        "group_size_m": 1,
        **options,
    }
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("m", "n", "k", "trans_a", "trans_b", "dtype", "uniform_192"),
    [
        # A wider tile removes a scheduling wave on 132-SM H200.
        pytest.param(4224, 1528, 512, True, False, torch.bfloat16, True, marks=pytest.mark.smoke),
        pytest.param(4224, 1528, 512, True, False, torch.float16, True, marks=pytest.mark.smoke),
        pytest.param(4224, 1528, 512, False, False, torch.bfloat16, True, marks=pytest.mark.full),
        pytest.param(4224, 1528, 512, True, True, torch.bfloat16, True, marks=pytest.mark.full),
        # Long K must retain uniform tiles.
        pytest.param(2112, 3072, 1048, True, False, torch.bfloat16, True, marks=pytest.mark.full),
        # Partial row tiles must exclude this candidate.
        pytest.param(4200, 1528, 512, True, False, torch.bfloat16, False, marks=pytest.mark.full),
    ],
)
def test_gemm_192_wide_transposed_tiles_match_reference(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: torch.dtype,
    uniform_192: bool,
) -> None:
    """Hopper supports these tiles; only 132-SM H200 has the asserted calibrated ranking."""
    test = GEMMTest(m, n, k, dtype, trans_a, trans_b)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b, target=BUILTIN)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    (kernel,) = [kern for kern in op.iter_kernels() if type(kern).__name__ == "GEMMTMAKernel"]
    cfg = kernel.config
    runs_192 = bool(cfg.get("coop2_swap")) and cfg["block_n"] == 192
    device = torch.device(run_device())
    board = calibration_key(torch.cuda.get_device_name(device))
    if board == "h200" and get_sm_count(device.index) == 132:
        assert runs_192 == uniform_192
    if not uniform_192:
        assert not runs_192, "the 192-wide candidate needs whole row tiles on every board"
    if runs_192:
        assert cfg["num_stages"] == 3 and "narrow_n" not in cfg
    if uniform_192:
        kernel.config = {
            "coop2_swap": True,
            "block_n": 192,
            "num_stages": 3,
            "group_size_m": 1,
        }
        test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_coop2_swap_rejects_a_narrow_tile_splitting_a_swizzle_atom() -> None:
    """With trans_a, a 128-wide stage holds 64-column atoms, so a 96-wide tile is refused."""
    test = GEMMTest(1000, 1032, 1048, torch.bfloat16, True, False)
    op = GEMMFwdOp(trans_a=True, trans_b=False, target=BUILTIN)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    (kernel,) = [k for k in op.iter_kernels() if type(k).__name__ == "GEMMTMAKernel"]
    kernel.config = {
        "coop2_swap": True,
        "block_n": 128,
        "num_stages": 4,
        "group_size_m": 1,
        "narrow_n": 96,
        "row_mix": [2, 7, 2, 7, 2],
    }
    with pytest.raises(ValueError, match="multiple of 64"):
        op(*inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gemm_k_tail_shorter_than_one_k_block() -> None:
    """A contraction of two elements preserves every output without alignment padding."""
    test = GEMMTest(32, 64, 2, torch.bfloat16, False, True)
    test.check(GEMMFwdOp(trans_b=True, target=BUILTIN), *test.gen_inputs())


_WS = {"block_m": 128, "block_n": 128, "block_k": 64, "num_stages": 4, "panel_size": 16}
_COOP2 = {"block_k": 64, "num_stages": 3, "group_size_m": 8}
_PINGPONG = {"pingpong": True, "block_k": 64, "num_stages": 4, "group_size_m": 8}
_COOP2_SWAP = {"coop2_swap": True, "block_n": 160, "num_stages": 4}


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("m", "n", "k", "trans_a", "trans_b", "dtype", "config"),
    [
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            torch.bfloat16,
            {"coop2": True, "block_n": 176, **_COOP2},
            marks=pytest.mark.smoke,
            id="coop2-176",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            torch.float16,
            {**_PINGPONG, "block_n": 176, "stage_n": 16, "epi_bufs": 2},
            marks=pytest.mark.smoke,
            id="pingpong-176",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            True,
            False,
            torch.bfloat16,
            {**_COOP2_SWAP, "group_size_m": 1},
            marks=pytest.mark.full,
            id="coop2-swap",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            torch.float16,
            {**_WS, "split_k": 1},
            marks=pytest.mark.full,
            id="ws-tma-store",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            torch.bfloat16,
            {
                "block_m": 64,
                "block_n": 112,
                "block_k": 64,
                "num_stages": 4,
                "panel_size": 0,
                "split_k": 1,
                "cluster_m": 2,
            },
            marks=pytest.mark.full,
            id="ws-fill-cluster",
        ),
        pytest.param(
            1000,
            1001,
            1048,
            False,
            True,
            torch.bfloat16,
            {**_WS, "split_k": 1},
            marks=pytest.mark.full,
            id="ws-row-store",
        ),
        pytest.param(
            1000,
            1032,
            1024,
            False,
            True,
            torch.bfloat16,
            {**_WS, "split_k": 2},
            marks=pytest.mark.full,
            id="split-k",
        ),
        pytest.param(
            1000,
            1032,
            1024,
            False,
            True,
            torch.bfloat16,
            {"coop2_splitk": True, "block_n": 64, "block_k": 128, "num_stages": 4, "split_k": 2},
            marks=pytest.mark.full,
            id="coop2-split-k",
        ),
        pytest.param(
            1024,
            1024,
            1024,
            False,
            True,
            torch.bfloat16,
            {"simple": True, "block_m": 64, "block_n": 128, "block_k": 64, "num_stages": 3},
            marks=pytest.mark.full,
            id="simple",
        ),
        pytest.param(
            8,
            1032,
            1048,
            False,
            True,
            torch.bfloat16,
            {"swap_ab": True, "block_nn": 64, "block_k": 128, "num_stages": 4},
            marks=pytest.mark.full,
            id="swap-ab",
        ),
        pytest.param(
            1024,
            1024,
            1024,
            False,
            False,
            torch.bfloat16,
            {"coop2s": True, "block_n": 64, "block_k": 64, "num_stages": 4},
            marks=pytest.mark.full,
            id="coop2s",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            False,
            torch.bfloat16,
            {"coop2": True, "block_n": 256, **_COOP2},
            marks=pytest.mark.full,
            id="coop2",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            torch.bfloat16,
            {
                "coop2": True,
                "block_n": 256,
                **_COOP2,
                "stage_n": 16,
                "narrow_n": 240,
                "narrow_rows": 2,
                "narrow_cols": 2,
            },
            marks=pytest.mark.full,
            id="coop2-narrow",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            torch.bfloat16,
            {**_PINGPONG, "block_n": 128},
            marks=pytest.mark.full,
            id="pingpong",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            True,
            False,
            torch.bfloat16,
            {**_COOP2_SWAP, "group_size_m": 2, "narrow_n": 128, "row_mix": [2, 6, 1, 4, 3]},
            marks=pytest.mark.full,
            id="coop2-swap-narrow",
        ),
    ],
)
def test_gemm_bias_on_each_structure(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: torch.dtype, config: dict
) -> None:
    """Each structure adds a column's bias to every row of it, past M, N and K tails."""
    test = GEMMTest(m, n, k, dtype, trans_a, trans_b, bias=True)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b, target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = [k for k in op.iter_kernels() if type(k).__name__ == "GEMMTMAKernel"]
    kernel.config = config
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("out_dtype", [None, torch.float32], ids=["bf16-out", "float32-out"])
@pytest.mark.parametrize(
    ("m", "n", "k", "trans_b", "kernel"),
    [
        pytest.param(1, 4608, 3584, True, GemvKernel, marks=pytest.mark.smoke, id="lhs-row"),
        pytest.param(1000, 1, 1000, False, GemvKernel, marks=pytest.mark.full, id="rhs-col"),
        # Rows TMA cannot address.
        pytest.param(
            1000, 1001, 1003, True, GEMMCpAsyncKernel, marks=pytest.mark.full, id="cp-async"
        ),
    ],
)
def test_gemm_bias_on_selected_kernels(
    m: int, n: int, k: int, trans_b: bool, kernel: type, out_dtype: torch.dtype | None
) -> None:
    """The kernel selected for a shape adds the bias, a single one over a one-column output,
    to the float32 sum, and stores it in the output dtype."""
    test = GEMMTest(m, n, k, torch.bfloat16, False, trans_b, bias=True, out_dtype=out_dtype)
    op = GEMMFwdOp(trans_b=trans_b, out_dtype=out_dtype, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    assert all(isinstance(entry, kernel) for entry in op.iter_kernels())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("m", "n", "k", "trans_a", "trans_b", "bias", "config"),
    [
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            True,
            {**_WS, "split_k": 1},
            marks=pytest.mark.smoke,
            id="ws-tma-store",
        ),
        # Rows of 1001 float32 values are no whole number of 16-byte units.
        pytest.param(
            1000,
            1001,
            1048,
            False,
            True,
            False,
            {**_WS, "split_k": 1},
            marks=pytest.mark.full,
            id="ws-row-store",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            True,
            {
                "block_m": 64,
                "block_n": 112,
                "block_k": 64,
                "num_stages": 4,
                "panel_size": 0,
                "split_k": 1,
                "cluster_m": 2,
            },
            marks=pytest.mark.full,
            id="ws-fill-cluster",
        ),
        pytest.param(
            1000,
            1032,
            1024,
            False,
            True,
            True,
            {**_WS, "split_k": 2},
            marks=pytest.mark.full,
            id="split-k",
        ),
        pytest.param(
            1024,
            1024,
            1024,
            False,
            True,
            True,
            {"simple": True, "block_m": 64, "block_n": 128, "block_k": 64, "num_stages": 3},
            marks=pytest.mark.full,
            id="simple",
        ),
        pytest.param(
            8,
            1032,
            1048,
            False,
            True,
            True,
            {"swap_ab": True, "block_nn": 64, "block_k": 128, "num_stages": 4},
            marks=pytest.mark.full,
            id="swap-ab",
        ),
        # A 64-column staging chunk leaves the ring its three stages beside two float32 halves.
        pytest.param(
            1000,
            1032,
            1048,
            False,
            False,
            True,
            {"coop2": True, "block_n": 256, **_COOP2, "stage_n": 64},
            marks=pytest.mark.full,
            id="coop2",
        ),
        # Twenty-four rows of A, a WGMMA width 256 does not divide, run the helper's own.
        pytest.param(
            24,
            8456,
            1040,
            False,
            True,
            True,
            {"stream": True, "block_k": 128, "num_stages": 8, "blocks": 2},
            marks=pytest.mark.full,
            id="stream-helper",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            True,
            # The ring best_config selects for a float32 output: five stages and two
            # staging buffers fill all but a few KiB of the shared memory.
            {**_PINGPONG, "block_n": 176, "num_stages": 5, "stage_n": 16, "epi_bufs": 2},
            marks=pytest.mark.full,
            id="pingpong-176",
        ),
        pytest.param(
            1000,
            1032,
            1024,
            False,
            True,
            False,
            {"coop2_splitk": True, "block_n": 64, "block_k": 128, "num_stages": 4, "split_k": 2},
            marks=pytest.mark.full,
            id="coop2-split-k",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            False,
            {"coop2": True, "block_n": 176, **_COOP2, "stage_n": 88},
            marks=pytest.mark.full,
            id="coop2-176",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            True,
            {
                "coop2": True,
                "block_n": 256,
                **_COOP2,
                "stage_n": 16,
                "narrow_n": 240,
                "narrow_rows": 2,
                "narrow_cols": 2,
            },
            marks=pytest.mark.full,
            id="coop2-narrow",
        ),
        pytest.param(
            32,
            8456,
            1040,
            False,
            True,
            False,
            {"stream": True, "block_k": 128, "num_stages": 8, "blocks": 2},
            marks=pytest.mark.full,
            id="stream",
        ),
        pytest.param(
            1000,
            1032,
            1048,
            False,
            True,
            False,
            {**_PINGPONG, "block_n": 128},
            marks=pytest.mark.full,
            id="pingpong",
        ),
        pytest.param(
            2200,
            3720,
            1048,
            False,
            True,
            False,
            {
                **_PINGPONG,
                "block_n": 192,
                "stage_n": 32,
                "epi_bufs": 1,
                "narrow_n": 160,
                "narrow_rows": 8,
                "narrow_cols": 14,
            },
            marks=pytest.mark.full,
            id="pingpong-192-narrow",
        ),
        pytest.param(
            1024,
            1024,
            1024,
            False,
            False,
            True,
            # The eight-stage ring best_config selects, beside two float32 staging tiles.
            {"coop2s": True, "block_n": 64, "block_k": 64, "num_stages": 8},
            marks=pytest.mark.full,
            id="coop2s",
        ),
    ],
)
def test_gemm_float32_output_on_each_structure(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, bias: bool, config: dict
) -> None:
    """Each structure writing a float32 output stores the unrounded float32 sum of bf16
    products, past M, N and K tails."""
    test = GEMMTest(m, n, k, torch.bfloat16, trans_a, trans_b, bias=bias, out_dtype=torch.float32)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b, out_dtype=torch.float32, target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = [k for k in op.iter_kernels() if type(k).__name__ == "GEMMTMAKernel"]
    kernel.config = config
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_coop2_swap_refuses_a_float32_output() -> None:
    """The coop2_swap tile, whose transposed store writes 16-bit elements only, refuses a
    float32 output when it is built, rather than rounding it."""
    test = GEMMTest(1000, 1032, 1048, torch.bfloat16, True, False, out_dtype=torch.float32)
    op = GEMMFwdOp(trans_a=True, trans_b=False, out_dtype=torch.float32, target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = op.iter_kernels()
    kernel.config = {**_COOP2_SWAP, "group_size_m": 1}
    with pytest.raises(ValueError, match="coop2_swap"):
        op(*inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("m", "n"),
    [
        # DeepSeek-V4-Flash's router and LM head over 64 decode tokens.
        pytest.param(64, 256, marks=pytest.mark.smoke, id="router-decode"),
        pytest.param(64, 129280, marks=pytest.mark.full, id="lm-head-decode"),
        # Its LM head over one decode token, a GEMV.
        pytest.param(1, 129280, marks=pytest.mark.full, id="lm-head-gemv"),
        # Its LM head over 24 decode tokens, which streams the weight 64 rows at a time.
        pytest.param(24, 129280, marks=pytest.mark.full, id="lm-head-stream"),
        # Its router and KV compressors over an 8192-token prefill.
        pytest.param(8192, 256, marks=pytest.mark.full, id="router-prefill"),
        pytest.param(8192, 512, marks=pytest.mark.full, id="compressor-512"),
        pytest.param(8192, 1024, marks=pytest.mark.full, id="compressor-1024"),
    ],
)
def test_gemm_float32_output_on_pilot_model_rows(m: int, n: int) -> None:
    """The tile selected for a pilot model's float32 logits, whose staging tile takes twice
    the shared memory of a 16-bit one, builds and stores the unrounded float32 sum."""
    test = GEMMTest(m, n, 4096, torch.bfloat16, False, True, out_dtype=torch.float32)
    op = GEMMFwdOp(out_dtype=torch.float32, target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_bias_on_a_tile_wider_than_its_producer() -> None:
    """On H200 a short, wide NT call runs the one-wave 64 x 192 tile, whose 128 producer
    threads stage its bias in two passes; every column adds its bias past M, N and K tails."""
    test = GEMMTest(76, 8456, 1040, torch.bfloat16, False, True, bias=True)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    if calibration_key(torch.cuda.get_device_name(torch.device(run_device()))) != "h200":
        return
    (kernel,) = op.iter_kernels()
    assert kernel.config["block_n"] > 128 and kernel.config["split_k"] == 1
    assert not kernel.config.get("simple")


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_bias_on_a_two_tile_fill_ring() -> None:
    """On H200 a biased NT call whose rows fill two 64-row tiles runs the clustered fill
    tile, whose ring the shared memory holds beside its staged bias row."""
    test = GEMMTest(128, 5280, 4096, torch.bfloat16, False, True, bias=True)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    if calibration_key(torch.cuda.get_device_name(torch.device(run_device()))) != "h200":
        return
    (kernel,) = op.iter_kernels()
    assert kernel.config["cluster_m"] == 2 and kernel.config["block_n"] not in (64, 128, 192, 256)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("m", "n", "k", "dtype", "bias"),
    [
        # A 40-wide tile, which TileLang would issue as n8 WGMMAs, over output rows TMA
        # cannot store.
        pytest.param(
            36, 8455, 1048, torch.bfloat16, True, marks=pytest.mark.smoke, id="native-width"
        ),
        # A 32-wide tile TileLang issues as one WGMMA, stored by TMA.
        pytest.param(30, 8456, 1048, torch.float16, True, marks=pytest.mark.smoke, id="tma-store"),
        # The widest tile offered, through the helper and the TMA store.
        pytest.param(70, 8456, 1048, torch.float16, True, marks=pytest.mark.full, id="widest"),
        # A 24-wide tile, whose ring and two staged tiles fill the shared memory; no bias.
        pytest.param(20, 8456, 1048, torch.bfloat16, False, marks=pytest.mark.full, id="full-smem"),
    ],
)
def test_gemm_short_wide_call_streams_b_in_64_row_blocks(
    m: int, n: int, k: int, dtype: torch.dtype, bias: bool
) -> None:
    """On a 132-SM H200 an NT call this short and wide runs the stream tile with two 64-row
    blocks of B a CTA, the last CTA's second block past N; M, N and K all end inside a
    tile."""
    test = GEMMTest(m, n, k, dtype, False, True, bias=bias)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    call = GEMMCall(m=m, n=n, k=k, dtype=dtype, trans_b=True, has_bias=bias, device=run_device())
    if call.calibration != "h200" or call.sm_count != 132:
        return
    (kernel,) = op.iter_kernels()
    assert kernel.config.get("stream") and kernel.config["blocks"] == 2


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_short_grid_runs_the_whole_smem_ring() -> None:
    """On H200 the basic tile a short NT grid runs takes every ring stage the shared memory
    holds, with a bias."""
    test = GEMMTest(96, 4096, 12288, torch.bfloat16, False, True, bias=True)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    if calibration_key(torch.cuda.get_device_name(torch.device(run_device()))) != "h200":
        return
    (kernel,) = op.iter_kernels()
    cfg = kernel.config
    assert "block_m" in cfg and not any(cfg.get(f) for f in GEMMTMAKernel._STRUCTURE_FLAGS)
    stage = (cfg["block_m"] + cfg["block_n"]) * cfg["block_k"] * 2
    staging = cfg["block_m"] * cfg["block_n"] * 2
    assert (cfg["num_stages"] + 1) * stage + staging > BLOCK_SHARED_BYTES_OPT_IN[90]


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("out_dtype", [None, torch.float32], ids=["bf16-out", "float32-out"])
def test_gemm_bias_on_two_row_gemv(out_dtype: torch.dtype | None) -> None:
    """Both rows of the two-row GEMV band, which only H200 is calibrated to take, add the bias,
    in either output dtype."""
    test = GEMMTest(2, 256, 4096, torch.bfloat16, False, True, bias=True, out_dtype=out_dtype)
    op = GEMMFwdOp(trans_b=True, out_dtype=out_dtype, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    if calibration_key(torch.cuda.get_device_name(torch.device(run_device()))) == "h200":
        (kernel,) = op.iter_kernels()
        assert isinstance(kernel, GemvKernel) and kernel.band == "lhs_rows"


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "n, k, dtype",
    [
        # Two K steps, the second a tail, on the 6-deep ring.
        pytest.param(1536, 1000, torch.float16, marks=pytest.mark.smoke, id="ring-past-k"),
        # The manifest's two-row shape: the narrowest weight the calibration still gives the
        # GEMV, so a lower GEMV rate loses it to a tile grid.
        pytest.param(2112, 7168, torch.bfloat16, marks=pytest.mark.smoke, id="crossover"),
    ],
)
def test_gemm_two_rows_run_the_gemv_where_it_streams_faster(
    n: int, k: int, dtype: torch.dtype
) -> None:
    """A two-row NT call runs the GEMV on a 6-deep ring where the H200 calibration ranks it
    above every tile grid."""
    test = GEMMTest(2, n, k, dtype, False, True)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    if calibration_key(torch.cuda.get_device_name(torch.device(run_device()))) != "h200":
        return
    (kernel,) = op.iter_kernels()
    assert isinstance(kernel, GemvKernel) and kernel.band == "lhs_rows"
    assert kernel.config["num_stages"] == 6


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_two_rows_of_a_wide_weight_run_a_tile_grid() -> None:
    """A two-row NT call over a wide weight, which the operand-swapped tile grid streams
    faster, runs that grid on H200."""
    test = GEMMTest(2, 3584, 6144, torch.bfloat16, False, True)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    if calibration_key(torch.cuda.get_device_name(torch.device(run_device()))) != "h200":
        return
    (kernel,) = op.iter_kernels()
    assert isinstance(kernel, GEMMTMAKernel)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k, dtype, bias, width, stages",
    [
        # An M tail, a 16-column N tail and a K tail; 17 K steps wrap the 8-deep ring.
        pytest.param(
            40,
            9016,
            1032,
            torch.bfloat16,
            True,
            72,
            8,
            marks=pytest.mark.smoke,
            id="bf16-bias-tails",
        ),
        # Whole 80-wide tiles; 11 stages fit, so the ring is 8 deep.
        pytest.param(
            64, 10560, 1032, torch.float16, False, 80, 8, marks=pytest.mark.smoke, id="fp16-ring-8"
        ),
    ],
)
def test_gemm_rows_of_one_tile_run_a_one_wave_column_grid(
    m: int, n: int, k: int, dtype: torch.dtype, bias: bool, width: int, stages: int
) -> None:
    """Rows that fit one 64-row tile run, a CTA a tile, the narrowest column tiles whose grid
    fits one wave, on the deepest power-of-two ring the shared memory holds. The cases count
    tiles against the 132 SMs of an H200."""
    call = GEMMCall(m=m, n=n, k=k, dtype=dtype, trans_b=True, has_bias=bias, device=run_device())
    if call.calibration != "h200" or call.sm_count != 132:
        pytest.skip("the one-wave column grids above are counted on a 132-SM H200")
    test = GEMMTest(m, n, k, dtype, False, True, bias=bias)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    config = op.kernel_for("gemm", lambda: call).config
    assert (config["block_n"], config["num_stages"], config.get("cluster_m")) == (width, stages, 1)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k, dtype, out_dtype, bias, tile",
    [
        # At M = 8150 the last of the 64 row tiles holds 86 rows: both 64-row consumers have rows.
        # 64 row tiles take three 128-wide column tiles, 192 CTAs, or two 144 wide, 128. With one
        # tile per CTA, a float32 tile with 16-byte rows is staged in two chunks or more; other
        # tiles keep the widest chunk that leaves the deepest ring.
        pytest.param(
            8150,
            280,
            1032,
            torch.bfloat16,
            torch.float32,
            True,
            (144, 48),
            marks=pytest.mark.smoke,
            id="f32-144-bias-n-tail",
        ),
        pytest.param(
            8150,
            288,
            1032,
            torch.float16,
            torch.float32,
            False,
            (144, 48),
            marks=pytest.mark.smoke,
            id="fp16-f32-144",
        ),
        # Rows of 281 float32 values are no whole number of 16-byte units: no TMA store.
        pytest.param(
            8150,
            281,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (144, 144),
            marks=pytest.mark.full,
            id="f32-144-unaligned-rows",
        ),
        pytest.param(
            8150,
            288,
            1032,
            torch.bfloat16,
            None,
            False,
            (144, 144),
            marks=pytest.mark.full,
            id="bf16-144",
        ),
        # The other widths the helper issues, each the narrowest whose two column tiles fit
        # one wave. Rows of 332 values take no ping-pong tile.
        pytest.param(
            8150,
            320,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (160, 32),
            marks=pytest.mark.full,
            id="f32-160",
        ),
        pytest.param(
            8150,
            332,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (176, 16),
            marks=pytest.mark.full,
            id="f32-176-n-tail",
        ),
        # Rows of 330 float32 values are no whole number of 16-byte units: no TMA store, and
        # the tile keeps the chunk that leaves the deepest ring.
        pytest.param(
            8150,
            330,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (176, 88),
            marks=pytest.mark.full,
            id="f32-176-unaligned-rows",
        ),
        pytest.param(
            8150,
            440,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (224, 32),
            marks=pytest.mark.full,
            id="f32-224",
        ),
        pytest.param(
            8150,
            472,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (240, 80),
            marks=pytest.mark.full,
            id="f32-240",
        ),
        # Two 128-wide column tiles, 128 CTAs: a float32 tile stages 64-column chunks beside a
        # six-stage ring, one stage deeper than the 128 x 128 basic tile's beside its whole
        # staged tile.
        pytest.param(
            8150,
            256,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (128, 64),
            marks=pytest.mark.full,
            id="f32-128-deep-ring",
        ),
        # Rows of 250 float32 values leave through predicated row stores.
        pytest.param(
            8150,
            250,
            1032,
            torch.bfloat16,
            torch.float32,
            True,
            (128, 64),
            marks=pytest.mark.full,
            id="f32-128-bias-unaligned-rows",
        ),
        # Two 256-wide column tiles, 128 CTAs, where 128-wide ones take 256.
        pytest.param(
            8150,
            500,
            1032,
            torch.bfloat16,
            torch.float32,
            False,
            (256, 64),
            marks=pytest.mark.full,
            id="f32-256-n-tail",
        ),
    ],
)
def test_gemm_one_wave_grid_runs_the_narrowest_coop2_tile(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype | None,
    bias: bool,
    tile: tuple,
) -> None:
    """A call whose 128-row tiles fit one wave at a coop2 width runs coop2 at the narrowest such
    width, staged in the columns *tile* names, past M, N and K tails. At 128 wide the coop2 tile
    replaces the basic tile with a deeper ring. The cases count tiles against the 132 SMs of an
    H200."""
    call = GEMMCall(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        trans_b=True,
        has_bias=bias,
        out_dtype=out_dtype,
        device=run_device(),
    )
    if call.calibration != "h200" or call.sm_count != 132:
        pytest.skip("the one-wave grids above are counted on a 132-SM H200")
    test = GEMMTest(m, n, k, dtype, False, True, bias=bias, out_dtype=out_dtype)
    op = GEMMFwdOp(trans_b=True, out_dtype=out_dtype, target=BUILTIN)
    config = op.kernel_for("gemm", lambda: call).config
    staged = config.get("stage_n") or config["block_n"]
    assert (config.get("coop2"), config["block_n"], staged) == (True, *tile)
    test.check(op, *test.gen_inputs())


# An H200's L2, which sets only the raster group of a persistent tile.
_H200_L2_BYTES = 60 * 2**20


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, out_bytes, basic",
    [
        # The second of two row tiles holds 64 rows: one coop2 consumer would compute none.
        pytest.param(192, 10240, 1032, 2, None, id="half-empty-last-row-tile"),
        # The narrowest one-wave grid, 48 row tiles by two 256-wide column tiles, fills 96 SMs.
        pytest.param(6144, 512, 1032, 4, None, id="under-15-16-of-a-wave"),
        # One row tile of 65 K steps: 128 columns of 224 fit one wave, and the 128 x 128 basic
        # tile scores best of the others.
        pytest.param(96, 28672, 4160, 2, (128, 128), id="one-row-tile-past-64-k-steps"),
        # A bfloat16 128-wide coop2 tile holds no deeper ring than the basic tile.
        pytest.param(8150, 256, 1032, 2, (128, 128), id="bf16-128-equal-ring"),
        # Grids of one row tile, or of one column tile, at 128 wide.
        pytest.param(96, 16032, 2048, 4, (128, 128), id="f32-128-one-row-tile"),
        pytest.param(16350, 128, 1032, 4, (128, 128), id="f32-128-one-column-tile"),
    ],
)
def test_gemm_one_wave_grid_keeps_other_tiles_where_coop2_gains_nothing(
    m: int, n: int, k: int, out_bytes: int, basic: tuple | None
) -> None:
    """On a 132-SM H200 the one-wave coop2 tile is not offered where its last row tile leaves a
    64-row consumer without rows, or where its grid fills less than 15/16 of the wave. In one
    row tile past 64 K steps the 128 x 128 basic tile, which streams B at the DRAM rate as the
    one-wave tile does, keeps the call where it scores best of the others. At 128 wide the
    128 x 128 basic tile keeps the call where the coop2 ring is no deeper, and on grids of one
    tile along an axis."""
    cfg = best_config(
        m, n, k, False, True, 132, "h200", l2_bytes=_H200_L2_BYTES, out_bytes=out_bytes
    )
    assert not cfg.get("coop2")
    if basic is not None:
        assert (cfg["block_m"], cfg["block_n"]) == basic


@pytest.mark.smoke
def test_gemm_coop2_several_tiles_a_cta_stage_the_widest_chunk_of_the_deepest_ring() -> None:
    """A coop2 grid whose CTAs each run several tiles, 2 row tiles by 108 column tiles 176 wide
    on 132 SMs, hides each tile's epilogue under the next tile's mainloop, so it stages the
    widest chunk that leaves the deepest ring."""
    cfg = best_config(
        256, 19008, 1032, False, True, 132, "h200", l2_bytes=_H200_L2_BYTES, out_bytes=4
    )
    assert (cfg.get("coop2"), cfg["block_n"], cfg.get("stage_n")) == (True, 176, 88)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("m", "n", "k", "trans_b", "tile"),
    [
        # Split-K over 128-wide K steps: 100 rows load as 104, two swizzle atoms a step.
        pytest.param(
            100, 2112, 7168, True, (128, 128, 4), marks=pytest.mark.smoke, id="splitk-k128"
        ),
        # Split-K over 32-wide K steps, narrower than one swizzle atom: 12 rows load as 16.
        pytest.param(12, 1032, 512, False, (64, 32, 4), marks=pytest.mark.full, id="splitk-k32"),
    ],
)
def test_gemm_one_tile_grid_loads_only_the_rows_of_a(
    m: int, n: int, k: int, trans_b: bool, tile: tuple
) -> None:
    """A split-K grid of one M tile taller than M loads A's rows rounded up to a whole 8-row
    atom; the tile's other rows hold stale data, and every stored output matches the
    reference."""
    call = GEMMCall(m=m, n=n, k=k, dtype=torch.bfloat16, trans_b=trans_b, device=run_device())
    test = GEMMTest(m, n, k, torch.bfloat16, False, trans_b)
    op = GEMMFwdOp(trans_b=trans_b, target=BUILTIN)
    config = op.kernel_for("gemm", lambda: call).config
    if call.calibration == "h200" and call.sm_count == 132:
        assert (config["block_m"], config["block_k"], config["split_k"]) == tile
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_rows_of_half_a_tile_keep_tiles_at_least_64_wide() -> None:
    """Rows filling half of a 64-row tile run no one-wave fill tile, whose width is none of
    the basic widths, though the scorer would rank its 72-wide grid first here."""
    m, n, k = 32, 9016, 1032
    call = GEMMCall(m=m, n=n, k=k, dtype=torch.bfloat16, trans_b=True, device=run_device())
    if call.calibration != "h200" or call.sm_count != 132:
        pytest.skip("the one-wave column grid is counted on a 132-SM H200")
    test = GEMMTest(m, n, k, torch.bfloat16, False, True)
    op = GEMMFwdOp(trans_b=True, target=BUILTIN)
    assert op.kernel_for("gemm", lambda: call).config.get("block_n", 64) in (64, 128, 192, 256)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "m, n, dtype, bias, trans_a",
    [
        # Rows filling one 64-row tile, and rows filling two, each served a one-wave grid.
        pytest.param(
            48, 4616, torch.bfloat16, True, False, marks=pytest.mark.smoke, id="bf16-one-tile"
        ),
        pytest.param(
            128, 7168, torch.float16, False, False, marks=pytest.mark.smoke, id="fp16-two-tiles"
        ),
        # TMA cannot address 63 transposed rows, so every architecture takes the cp.async kernel.
        pytest.param(63, 65, torch.float16, True, True, marks=pytest.mark.full, id="cp-async"),
    ],
)
def test_gemm_empty_contraction_is_the_bias(
    m: int, n: int, dtype: torch.dtype, bias: bool, trans_a: bool
) -> None:
    """A product over K = 0 sums nothing: each element is its column's bias, or zero."""
    test = GEMMTest(m, n, 0, dtype, trans_a, True, bias=bias)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("m", "n", "k", "trans_b"),
    [
        pytest.param(1, 129, 1024, True, marks=pytest.mark.smoke, id="lhs-row"),
        pytest.param(129, 1, 1024, False, marks=pytest.mark.full, id="rhs-col"),
        pytest.param(2, 256, 4096, True, marks=pytest.mark.full, id="lhs-rows"),
    ],
)
def test_gemv_bias_survives_cancelling_products(m: int, n: int, k: int, trans_b: bool) -> None:
    """The GEMV bands add the bias after the K reduction, where cancelling products cannot
    absorb it."""
    device = torch.device(run_device())
    a = torch.full((m, k), 4096, dtype=torch.float16, device=device)
    signs = torch.ones(k, dtype=torch.float16, device=device)
    signs[k // 2 :] = -1
    b = (4096 * signs).expand(n, k) if trans_b else (4096 * signs)[:, None].expand(k, n)
    bias = torch.ones(n, dtype=torch.float16, device=device)
    op = GEMMFwdOp(trans_b=trans_b, target=BUILTIN)
    out = op(a, b.contiguous(), bias)
    torch.testing.assert_close(out, torch.ones(m, n, dtype=torch.float16, device=device))
    if m == 1 or n == 1:
        assert all(isinstance(entry, GemvKernel) for entry in op.iter_kernels())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize("out_dtype", [None, torch.float32], ids=["bf16-out", "float32-out"])
def test_gemm_cp_async_split_k_adds_the_bias_in_the_reduction(
    out_dtype: torch.dtype | None,
) -> None:
    """The pipelined kernel's split slices leave the bias to their reduction, which writes the
    output dtype."""

    class Pinned(GEMMCpAsyncKernel):
        general = False
        preferred_over = frozenset(GEMMFwdOp.kernel_types)

    state = registry.snapshot()
    try:
        register_kernel_type("GEMMFwdOp", "gemm_cp_async_pinned", Pinned)
        op = GEMMFwdOp(out_dtype=out_dtype, target=BUILTIN)
    finally:
        registry.restore(state)
    test = GEMMTest(256, 512, 1024, torch.bfloat16, False, True, bias=True, out_dtype=out_dtype)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = op.iter_kernels()
    assert isinstance(kernel, Pinned)
    kernel.config = {**kernel.default_config, "split_k": 2}
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m", [pytest.param(100, marks=pytest.mark.smoke), pytest.param(257, marks=pytest.mark.full)]
)
def test_gemm_w4a16_kernel_predicates_a_ragged_token_count(m: int) -> None:
    test = GEMMW4A16Test(m, 1024, 512, torch.float16)
    op = GEMMW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "n, k",
    [
        # A one-row projection on two warpgroups split six ways unevenly.
        pytest.param(2176, 8192, id="paired-split-k-6-uneven", marks=pytest.mark.smoke),
        # A long contraction with a final 128-element group, on two warpgroups split four
        # ways.
        pytest.param(4224, 32896, id="paired-split-k-4-k-tail", marks=pytest.mark.full),
        # Stream-K is not offered to a two-warpgroup tile; the cost model would stream this
        # one.
        pytest.param(12288, 12288, id="two-warpgroups-unstreamed", marks=pytest.mark.full),
    ],
)
def test_gemm_w4a16_decode_partitions_match_the_reference(n: int, k: int) -> None:
    test = GEMMW4A16Test(1, n, k, torch.float16)
    test.check(GEMMW4A16FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a16_long_k_stages_metadata_per_tile() -> None:
    test = GEMMW4A16Test(1, 64, 32896, torch.float16)
    op = GEMMW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())


def _w4a16_config(
    block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, split_k: int
) -> dict:
    warp_specialized = threads > 128
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "threads": threads,
        "producer_reg": 32 if warp_specialized else 0,
        "consumer_reg": 224 if warp_specialized else 0,
        "split_k": split_k,
        "stream_ctas": 0,
    }


def _check_w4a16_config(m: int, n: int, k: int, config: dict) -> None:
    """Check the selected config, then ``config``, against the reference."""
    test = GEMMW4A16Test(m, n, k, torch.float16)
    op = GEMMW4A16FwdOp(target=BUILTIN)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    (kernel,) = [kern for kern in op.iter_kernels() if type(kern).__name__ == "GEMMW4A16Kernel"]
    kernel.config = config
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a16_two_stage_ring_refills_after_the_last_sub_step() -> None:
    """Repeated K-tile refills preserve every sub-step's input until its MMA completes."""
    _check_w4a16_config(384, 2048, 4096, _w4a16_config(64, 128, 512, 2, 256, 1))


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k, config",
    [
        # Split slices stage their own groups; the last K tile is partial.
        pytest.param(
            64,
            1024,
            14464,
            _w4a16_config(64, 128, 256, 3, 256, 4),
            id="k-tail-split-staged",
            marks=pytest.mark.smoke,
        ),
        # A two-way split runs as a cluster pair: uneven slices over a partial last K tile,
        # even slices on a 64-row tile, and a deferred-scale tile.
        pytest.param(
            64,
            1024,
            14464,
            _w4a16_config(128, 128, 256, 2, 256, 2),
            id="pair-split-uneven-k-tail",
            marks=pytest.mark.full,
        ),
        # The cross-tile mainloop paired over uneven slices (odd and even K-tile counts) and
        # padded rows.
        pytest.param(
            72,
            1024,
            1408,
            _w4a16_config(128, 128, 128, 3, 256, 2),
            id="cross-tile-pair-uneven",
            marks=pytest.mark.full,
        ),
        pytest.param(
            40,
            2048,
            8192,
            _w4a16_config(64, 128, 128, 4, 256, 2),
            id="pair-split-64-rows",
            marks=pytest.mark.full,
        ),
        pytest.param(
            40,
            1024,
            14464,
            _w4a16_config(32, 128, 512, 2, 256, 2),
            id="pair-split-deferred-scale",
            marks=pytest.mark.full,
        ),
        # The cross-tile mainloop whole over padded rows.
        pytest.param(
            96,
            2048,
            4096,
            _w4a16_config(128, 128, 128, 4, 256, 1),
            id="cross-tile",
            marks=pytest.mark.full,
        ),
        pytest.param(
            1,
            96,
            8192,
            _w4a16_config(8, 64, 512, 4, 128, 8),
            id="decode-split-staged",
            marks=pytest.mark.full,
        ),
        pytest.param(
            20,
            4096,
            1024,
            _w4a16_config(16, 64, 512, 2, 128, 1),
            id="deferred-scale-16-rows",
            marks=pytest.mark.full,
        ),
        pytest.param(
            40,
            2048,
            14336,
            _w4a16_config(16, 64, 512, 4, 128, 4),
            id="deferred-scale-16-rows-split-k",
            marks=pytest.mark.full,
        ),
        pytest.param(
            96,
            2048,
            14336,
            _w4a16_config(32, 64, 512, 4, 128, 4),
            id="deferred-scale-32-rows-split-k",
            marks=pytest.mark.full,
        ),
        # Stream-K over exactly two CTAs per N tile: 66 N tiles on 132 CTAs. The selector picks
        # it only where N tiles numbering half the SMs do not pair up, which no 132-SM board has.
        pytest.param(
            1,
            4224,
            32896,
            {**_w4a16_config(8, 64, 512, 4, 128, 1), "stream_ctas": 132},
            id="stream-k-two-way",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gemm_w4a16_pinned_tile_matches_the_reference(m: int, n: int, k: int, config: dict) -> None:
    """Each pinned tile matches the reference."""
    _check_w4a16_config(m, n, k, config)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k",
    [
        # One K tile, so the output is staged in the stage it used; M pads to the tile.
        pytest.param(300, 8192, 128, id="one-k-tile", marks=pytest.mark.smoke),
        # Three stages, the output staged in the second; the last N tile is half outside N.
        pytest.param(2048, 1088, 384, id="three-stages-n-tail", marks=pytest.mark.full),
        # A two-stage ring refilled 30 times.
        pytest.param(300, 8192, 4096, id="refilled-ring", marks=pytest.mark.full),
    ],
)
def test_gemm_w4a16_drained_ring_serves_whole_k_two_warpgroup_tiles(m: int, n: int, k: int) -> None:
    """On 132 SMs a two-warpgroup tile of 128-wide K tiles over whole K builds the drained
    ring, which matches the reference."""
    if get_sm_count(torch.device(run_device()).index) != 132:
        pytest.skip("the op selects on the device's SMs, and the programs checked count 132")
    test = GEMMW4A16Test(m, n, k, torch.float16)
    op = GEMMW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = [kern for kern in op.iter_kernels() if type(kern).__name__ == "GEMMW4A16Kernel"]
    cfg = kernel.config
    source = kernel._builder(kernel._padded_m(cfg))(**cfg).get_kernel_source()
    assert "void drained_main_kernel(" in source


@pytest.mark.smoke
def test_gemm_w4a16_drained_ring_keeps_split_unaligned_and_one_stage_tiles_off() -> None:
    """The 256-row two-warpgroup tile keeps its other program split, with N off a multiple of 8,
    which the ring's TMA store cannot write, or on a one-stage ring, whose last K tile refills
    the stage the output is staged in."""
    fields = ("block_m", "block_n", "block_k", "num_stages", "threads")
    # Split four ways on 132 SMs.
    cfg = _select_config(129, 4096, 4096, 128, 132)
    assert cfg["split_k"] == 4
    assert not _drained_ring(*(cfg[f] for f in fields), 4096, 4096, 128, cfg["split_k"])
    # N = 1090; the same tile at N = 1088 takes the ring.
    cfg = _select_config(2048, 1090, 384, 128, 132)
    assert not _drained_ring(*(cfg[f] for f in fields), 1090, 384, 128, 1)
    assert _drained_ring(*(cfg[f] for f in fields), 1088, 384, 128, 1)
    # Arguments: block_m, block_n, block_k, num_stages, threads, N, K, group, split_k.
    assert not _drained_ring(256, 128, 128, 1, 256, 8192, 4096, 128, 1)


@pytest.mark.smoke
def test_gemm_w4a16_selector_moves_128_row_tiles_onto_the_cross_tile_ring() -> None:
    """A 128-row tile split at most two ways takes 128-wide K tiles on a four-stage ring; one
    whose slice metadata does not fit beside that ring stays on the pipelined mainloop."""
    for m, n, k, split_k in ((128, 8192, 8192, 2), (96, 14336, 4096, 1)):
        cfg = _select_config(m, n, k, 128, 132)
        tile = (cfg["block_m"], cfg["block_k"], cfg["num_stages"], cfg["split_k"])
        assert tile == (128, 128, 4, split_k)
    cfg = _select_config(65, 14336, 25600, 128, 132)
    assert not _cross_tile_wgmma(
        cfg["block_m"],
        cfg["block_n"],
        cfg["block_k"],
        cfg["num_stages"],
        cfg["threads"],
        25600,
        128,
        cfg["split_k"],
    )


def _leave_nan_in_free_blocks() -> None:
    """Free blocks the caching allocator hands out next for outputs under 1 MiB, holding NaN:
    an output element no kernel writes then reads NaN, not a fresh block's zeros."""
    blocks = [torch.full((1 << 17,), float("nan"), device=run_device()) for _ in range(16)]
    del blocks


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, tile",
    [
        # Kept: a doubled split leaves a slice under two K tiles; the paired tile does not
        # fit in shared memory.
        pytest.param(1, 1024, 8192, (64, 128, 8, False), id="kept-short-slice"),
        pytest.param(32, 8192, 8192, (64, 128, 1, False), id="kept-no-fit"),
        pytest.param(1, 8192, 8192, (128, 256, 2, False), id="paired-split-1"),
        pytest.param(1, 4096, 14336, (128, 256, 4, False), id="paired-split-2"),
        # A ragged token count on the 16-row tile.
        pytest.param(11, 8192, 8192, (128, 256, 2, False), id="paired-m11"),
        # Paired slices the four-stage ring holds. A cluster pair is kept where the pick's
        # own slice of four K tiles fits the ring, and where its 384 CTAs take more than one
        # wave; it pairs where the pick's eight refill the ring in one wave.
        pytest.param(16, 8192, 2048, (64, 128, 1, False), id="kept-cluster-in-ring"),
        pytest.param(16, 24576, 4096, (64, 128, 1, False), id="kept-cluster-waves"),
        pytest.param(16, 6144, 4096, (128, 256, 2, False), id="paired-cluster"),
        # A split is kept where it adds K tiles to the longest slice (56 K tiles over 8
        # slices pair into slices of three and four), and pairs where it adds none.
        pytest.param(16, 1024, 28672, (64, 128, 8, False), id="kept-uneven-in-ring"),
        pytest.param(16, 4096, 4096, (128, 256, 4, False), id="paired-even-in-ring"),
        # 50 K tiles: slices of 17 pair into slices of 9, which still refill the ring.
        pytest.param(8, 5120, 25600, (128, 256, 6, False), id="paired-uneven-refills"),
        # Stream-K picks: the 132 streamed CTAs become 64 paired N tiles split two ways (128
        # CTAs); 67 N tiles do not pair up and keep the stream.
        pytest.param(1, 8192, 81920, (128, 256, 2, False), id="paired-stream-k"),
        pytest.param(1, 4288, 32896, (64, 128, 1, True), id="kept-odd-n-tiles"),
        # Kept streams whose N tiles two or three CTAs share.
        pytest.param(1, 4416, 32896, (64, 128, 1, True), id="kept-stream-k-tail"),
        pytest.param(1, 4288, 81920, (64, 128, 1, True), id="kept-stream-whole-k"),
        pytest.param(8, 5056, 32896, (64, 128, 1, True), id="kept-stream-eight-rows"),
    ],
)
def test_gemm_w4a16_pairs_deferred_scale_tiles(m: int, n: int, k: int, tile: tuple) -> None:
    """On 132 SMs a deferred-scale tile on one consumer warpgroup pairs into twice its weight
    rows on two warpgroups, or keeps its tile; each case's comment names why."""
    cfg = _select_config(m, n, k, 128, 132)
    assert (cfg["block_n"], cfg["threads"], cfg["split_k"], cfg["stream_ctas"] > 0) == tile


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(1, 1024, 8192, id="kept-short-slice", marks=pytest.mark.smoke),
        pytest.param(1, 8192, 8192, id="paired-split-1", marks=pytest.mark.full),
        pytest.param(11, 8192, 8192, id="paired-m11", marks=pytest.mark.full),
        pytest.param(1, 8192, 81920, id="paired-stream-k", marks=pytest.mark.full),
        pytest.param(1, 4288, 32896, id="kept-odd-n-tiles", marks=pytest.mark.full),
        # The slots no CTA of a streamed tile writes are zeroed: 69 tiles with a partial
        # last K tile, 67 tiles of whole K tiles.
        pytest.param(1, 4416, 32896, id="kept-stream-k-tail", marks=pytest.mark.full),
        pytest.param(1, 4288, 81920, id="kept-stream-whole-k", marks=pytest.mark.full),
        pytest.param(8, 5056, 32896, id="kept-stream-eight-rows", marks=pytest.mark.full),
    ],
)
def test_gemm_w4a16_deferred_scale_tiles_match_the_reference(m: int, n: int, k: int) -> None:
    """On 132 SMs the paired two-way split, the kept split, the kept and paired stream-K picks
    and ragged rows of the deferred-scale selection match the reference."""
    if get_sm_count(torch.device(run_device()).index) != 132:
        pytest.skip("the op selects on the device's SMs, and the programs checked count 132")
    test = GEMMW4A16Test(m, n, k, torch.float16)
    inputs = test.gen_inputs()
    _leave_nan_in_free_blocks()
    test.check(GEMMW4A16FwdOp(target=BUILTIN), *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k, paired",
    [
        # One consumer warpgroup split two ways, on at most 72 CTAs: a deferred-scale 16-row
        # tile over padded rows, a partial last N tile and a partial last K tile.
        pytest.param(9, 2248, 1792, True, id="pair-deferred-scale", marks=pytest.mark.smoke),
        # A scale-in-decode 64-row tile over padded rows and the same tails.
        pytest.param(48, 2248, 1792, True, id="pair-64-rows", marks=pytest.mark.full),
        # The narrow tile at a logical M under its eight rows.
        pytest.param(1, 1672, 1792, True, id="pair-narrow", marks=pytest.mark.full),
        # Deferred-scale 32-row and scale-in-decode 64-row pairs whose 256-wide K tiles copy
        # metadata as four-group windows, starting at groups 0 and 2 of a window.
        pytest.param(33, 2048, 1024, True, id="pair-two-group-tiles", marks=pytest.mark.full),
        pytest.param(33, 3072, 1536, True, id="pair-scale-in-decode", marks=pytest.mark.full),
        # A split whose landing area does not fit beside its ring keeps the reduction launch,
        # on 96 CTAs so that the landing alone decides it.
        pytest.param(128, 3072, 4096, False, id="landing-unfit", marks=pytest.mark.full),
    ],
)
def test_gemm_w4a16_one_warpgroup_split_matches_the_reference(
    m: int, n: int, k: int, paired: bool
) -> None:
    """A one-warpgroup tile split two ways sums its K slices over a two-CTA cluster, the
    second CTA sending its whole partial to the first, which stores every row of the tile."""
    # On 132 SMs each case picks a one-warpgroup two-way split and runs the program `paired`
    # names: the cluster pair, or the reduction launch.
    cfg = _select_config(m, n, k, 128, 132)
    assert (cfg["block_n"], cfg["threads"], cfg["split_k"]) == (64, 128, 2)
    m_pad = m if cfg["block_m"] == 8 else -(-m // cfg["block_m"]) * cfg["block_m"]
    assert _pair_reduced(m_pad, n, k, 128, 132, cfg) == paired
    if get_sm_count(torch.device(run_device()).index) != 132:
        pytest.skip("the op selects on the device's SMs, and the program checked counts 132")
    test = GEMMW4A16Test(m, n, k, torch.float16)
    inputs = test.gen_inputs()
    _leave_nan_in_free_blocks()
    test.check(GEMMW4A16FwdOp(target=BUILTIN), *inputs)


@pytest.mark.smoke
def test_gemm_w4a16_pairs_a_one_warpgroup_split_in_one_wave() -> None:
    """On 132 SMs a one-warpgroup tile split two ways sums its slices over a cluster pair
    where every cluster launches in the first wave, and keeps the reduction launch past it."""
    narrow = _select_config(1, 2176, 1024, 128, 132)
    assert _pair_reduced(1, 2176, 1024, 128, 132, narrow)
    # 141 N tiles split two ways: 282 CTAs.
    assert not _pair_reduced(1, 9000, 1024, 128, 132, narrow)


@pytest.mark.smoke
def test_gemm_w4a16_keeps_plain_metadata_copies_off_the_unrefilled_one_warpgroup_ring() -> None:
    """A per-K-tile metadata copy stays plain on two consumer warpgroups, on a refilled or
    streamed ring, where K's groups are not a multiple of four, where the slice is staged once
    and where a K tile's groups do not divide the window."""
    # Arguments: block_m, block_n, block_k, num_stages, threads, split_k, stream_ctas, K, group.
    # Two consumer warpgroups, slices of four 512-wide K tiles on a four-stage ring.
    assert _meta_window(16, 128, 512, 4, 256, 2, 0, 4096, 128) == 0
    # Slices of six 256-wide K tiles refill a three-stage ring.
    assert _meta_window(64, 64, 256, 3, 128, 2, 0, 3072, 128) == 0
    # A streamed ring, whose slice of four 256-wide K tiles would otherwise fit its four stages.
    assert _meta_window(8, 64, 256, 4, 128, 1, 132, 1024, 128) == 0
    # 14 groups: a window at a multiple of four is not 4-byte aligned in every row.
    assert _meta_window(64, 64, 256, 4, 128, 2, 0, 1792, 128) == 0
    # 512-wide K tiles of a 128-thread tile stage their small slice once before the ring.
    assert _meta_window(32, 64, 512, 3, 128, 1, 0, 1536, 128) == 0
    # 384-wide K tiles of an explicit config: three groups a K tile do not divide the window.
    assert _meta_window(8, 64, 384, 4, 128, 24, 0, 36864, 128) == 0


@pytest.mark.smoke
@pytest.mark.parametrize("m, n, k", [(33, 2048, 1024), (33, 3072, 1536)])
def test_gemm_w4a16_copies_metadata_windows_where_the_one_warpgroup_ring_holds_the_slice(
    m: int, n: int, k: int
) -> None:
    """On 132 SMs a one-warpgroup tile whose ring holds its slice copies each K tile's metadata
    as a four-group window, on the 32-row deferred-scale tile and on the 64-row tile."""
    cfg = _select_config(m, n, k, 128, 132)
    fields = ("block_m", "block_n", "block_k", "num_stages", "threads", "split_k", "stream_ctas")
    assert _meta_window(*(cfg[field] for field in fields), k, 128) == 4


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a16_split_wide_tile_matches_the_reference() -> None:
    """Uneven contraction partitions preserve affine group scales and zero points."""
    test = GEMMW4A16Test(128, 5120, 25600, torch.float16)
    test.check(GEMMW4A16FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a16_request_tune_keeps_composite_runtime_state() -> None:
    test = GEMMW4A16Test(1, 1024, 8192, torch.float16)
    inputs = test.gen_inputs()
    op = GEMMW4A16FwdOp(target=BUILTIN)
    expected = op(*inputs)

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        op.request_tune()

    assert op._tune_requested is False
    assert torch.equal(op(*inputs), expected)

    next_test = GEMMW4A16Test(2, 1024, 8192, torch.float16)
    next_inputs = next_test.gen_inputs()
    next_test.check(op, *next_inputs)

    with pytest.warns(UserWarning, match="does not support generic autotuning"):
        new_op = GEMMW4A16FwdOp()
        new_op.request_tune()
    assert new_op._tune_requested is False


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("n", "k"),
    [
        pytest.param(64, 256, marks=pytest.mark.smoke),
        pytest.param(1024, 512, marks=pytest.mark.full),
    ],
)
def test_w4a16_repack_kernel_matches_the_reference(n: int, k: int) -> None:
    """The kernel and the tensor-expression repack agree bit for bit."""
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=run_device())
    op = GEMMW4A16FwdOp(target=BUILTIN)

    actual = op.repack(packed)

    assert actual.dtype == torch.uint8
    assert actual.shape == packed.shape
    assert torch.equal(actual, repack_w4a16_weight(packed))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gemm_w4a16_repack_feeds_forward() -> None:
    test = GEMMW4A16Test(64, 1024, 512, torch.float16)
    activation, _, scale, zero = test.gen_inputs()
    packed = test.row_major_weight

    prepacked = GEMMW4A16FwdOp().repack(packed)
    actual = GEMMW4A16FwdOp()(activation, prepacked, scale, zero)

    compare_outputs(
        actual, test.ref_program(activation, prepacked, scale, zero), test.verification()
    )


@pytest.mark.smoke
def test_gemm_w4a16_repack_refuses_a_partial_k_step() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        GEMMW4A16FwdOp().repack(torch.zeros((8, 96), dtype=torch.uint8, device=run_device()))


def _repacked(op: GEMMW4A8FwdOp):
    """The op as a caller runs it on a row-major weight: ``repack``, then ``forward``."""

    def run(activation, packed_weight, *scales):
        return op(activation, op.repack(packed_weight), *scales)

    return run


@pytest.mark.sm90
@GEMMW4A8Fixture
def test_gemm_w4a8(m: int, n: int, k: int) -> None:
    test = GEMMW4A8Test(m, n, k, 128)
    op = GEMMW4A8FwdOp()
    test.check(op, *test.gen_inputs(), runs=_repacked(op))


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gemm_w4a8_decode_publishes_a_grid_with_the_workspace_it_sizes() -> None:
    """A published config with more CTAs launches with partial sums and flags for each, and
    a graph captured before it still replays on the workspace it captured."""
    import gc
    import weakref

    from tileops._kernels.gemm.w4a8_decode import GEMMW4A8DecodeKernel

    # One CTA per 64-row tile by default; every SM then splits tiles across CTAs, and a
    # sixth stage pushes the units' partial sums out of shared memory.
    test = GEMMW4A8Test(32, 7168, 2304, 128)
    op = GEMMW4A8FwdOp()
    inputs = test.gen_inputs()
    run = _repacked(op)
    test.check(op, *inputs, runs=run)
    activation, weight, *scales = inputs
    repacked = op.repack(weight)
    expected = op(activation, repacked, *scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(activation, repacked, *scales)
    (kernel,) = [k for k in op.iter_kernels() if isinstance(k, GEMMW4A8DecodeKernel)]
    ((_, *workspace),) = kernel._generation.programs.values()
    captured_workspace = [weakref.ref(t) for t in workspace]
    del workspace
    ctas = get_sm_count(kernel.device_index)
    assert kernel.config["num_ctas"] < ctas
    kernel.config = {**kernel.config, "num_ctas": ctas, "num_stages": 6}
    test.check(op, *inputs, runs=run)
    ((_, _, flags, _),) = kernel._generation.programs.values()
    assert flags.numel() == ctas * kernel.config["unit_warpgroups"]
    # More CTAs than tiles hand partial sums across CTAs; each reader clears its ready flag,
    # so a repeat or a graph replay waits for fresh partial sums.
    assert not flags.any()
    gc.collect()
    assert all(ref() is not None for ref in captured_workspace)
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(8, 80, 384, marks=pytest.mark.smoke, id="m8"),
        pytest.param(64, 80, 384, marks=pytest.mark.smoke, id="m64"),
    ],
)
def test_gemm_w4a8_takes_group_scales_of_either_sign(m: int, n: int, k: int) -> None:
    test = GEMMW4A8MixedSignTest(m, n, k, 128)
    op = GEMMW4A8FwdOp()
    test.check(op, *test.gen_inputs(), runs=_repacked(op))


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "m, n, k",
    [
        # Ragged N and odd K require partial transport; the last stmatrix group has
        # only its first eight-token half live.
        pytest.param(33, 272, 384, marks=pytest.mark.smoke, id="single-pass-half-group"),
        # Both halves of the final live group are present, with a whole group to zero.
        pytest.param(105, 272, 384, marks=pytest.mark.full, id="single-pass-whole-group"),
        # A later pass uses pass-relative scales and a global accumulator offset.
        pytest.param(129, 272, 7296, marks=pytest.mark.full, id="three-pass-half-group"),
        pytest.param(137, 272, 7296, marks=pytest.mark.full, id="three-pass-whole-group"),
        # Four passes alternate output buffers without the odd-pass wraparound.
        pytest.param(193, 272, 7296, marks=pytest.mark.full, id="four-pass-half-group"),
    ],
)
def test_gemm_w4a8_prefill_prefix_edges(m: int, n: int, k: int) -> None:
    """Clipped accumulator groups produce fresh, scaled results through the public op."""
    test = GEMMW4A8MixedSignTest(m, n, k, 128)
    op = GEMMW4A8FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs(), runs=_repacked(op))


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "run_mode",
    ["repeat", pytest.param("graph", marks=pytest.mark.cuda_graph(op="GEMMW4A8FwdOp"))],
)
@pytest.mark.parametrize(
    "token_counts, n, k",
    [
        pytest.param((129,), 272, 384, marks=pytest.mark.smoke, id="ragged-n"),
        pytest.param(
            (385, 4097, 385),
            2560,
            2176,
            marks=pytest.mark.full,
            id="short-k-repeated-token-count",
        ),
        pytest.param(
            (129, 192, 193, 129),
            7168,
            7296,
            marks=pytest.mark.full,
            id="first-token-tile-edges-odd-k",
        ),
        # Resume the same cached entry after a multi-M-tile call. Odd K and ragged N
        # retain split-K contributors and their physical workspace pitch.
        pytest.param((129, 257, 129), 272, 7296, marks=pytest.mark.full, id="split-k-entry-reuse"),
        # More output tiles than resident CTAs reuse buffers across waves, with an
        # odd/even number of passes per tile respectively. N follows the real device.
        pytest.param((129,), None, 128, marks=pytest.mark.full, id="odd-pass-wave-reuse"),
        pytest.param((193,), None, 128, marks=pytest.mark.full, id="even-pass-wave-reuse"),
        # On a 132-SM H200, 2-CTA cluster pairs split a partial wave four ways a tile,
        # each CTA of a pair with its own counter.
        pytest.param((4096,), 2304, 1024, marks=pytest.mark.full, id="cluster-stream-k"),
        # Reuse a small-token entry after serving another token count. On a 132-SM H200
        # this exercises the decode kernel's independent flag workspace.
        pytest.param(
            (33, 40, 33), 18432, 7168, marks=pytest.mark.full, id="small-token-entry-reuse"
        ),
        # A 132-SM H200 runs these on the 40-token tile with each CTA's item run spanning
        # three tiles.
        pytest.param(
            (33, 40), 280 * 128, 2048, marks=pytest.mark.full, id="forty-token-past-two-waves"
        ),
    ],
)
def test_gemm_w4a8_reuses_prefill_state_with_changed_inputs(
    token_counts: tuple[int, ...],
    n: int | None,
    k: int,
    run_mode: str,
) -> None:
    """Changed inputs survive entry reuse, persistent waves and poisoned graph replays."""
    op = GEMMW4A8FwdOp(target=BUILTIN)
    if n is None:
        group_size = op.group_size
        n = (2 * get_sm_count(torch.device(run_device()).index) + 1) * group_size
    repeats = 20 if len(token_counts) == 1 else 3
    for m in token_counts:
        test = GEMMW4A8MixedSignTest(m, n, k, 128)
        raw = test.gen_inputs()
        test.check(op, *raw, runs=_repacked(op))
        alternate = (-raw[0].float()).to(raw[0].dtype)
        logical_inputs = (raw, (alternate, *raw[1:]))
        references = tuple(test.ref_program(*args) for args in logical_inputs)
        evidence = tuple(test.verification(*args) for args in logical_inputs)
        activation = raw[0].clone()
        inputs = (activation, op.repack(raw[1]), *raw[2:])
        compare_outputs(op(*inputs), references[0], evidence[0])
        torch.cuda.synchronize()
        if run_mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = op(*inputs)
        for iteration in range(repeats):
            index = (iteration + 1) % 2
            activation.copy_(logical_inputs[index][0])
            if run_mode == "graph":
                # A skipped store must not pass because an old result remained.
                captured.fill_(float("nan"))
                graph.replay()
                actual = captured
            else:
                actual = op(*inputs)
            torch.cuda.synchronize()
            compare_outputs(actual, references[index], evidence[index])


@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a8_request_tune_keeps_results() -> None:
    """A tuned op serves new token counts and serves a repeated one again."""
    op = GEMMW4A8FwdOp()
    op.request_tune()
    for m in (1, 64, 1):
        test = GEMMW4A8Test(m, 256, 512, 128)
        test.check(op, *test.gen_inputs(), runs=_repacked(op))


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "n, k",
    [
        pytest.param(16, 128, marks=pytest.mark.smoke, id="n16-k128"),
        pytest.param(112, 384, marks=pytest.mark.full, id="n112-k384"),
        pytest.param(7184, 2304, marks=pytest.mark.full, id="n7184-k2304"),
    ],
)
def test_w4a8_repack_kernel_matches_the_reference(n: int, k: int) -> None:
    """The op's repack and the tensor-expression repack agree bit for bit."""
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")

    actual = GEMMW4A8FwdOp(target=BUILTIN).repack(packed)

    assert actual.dtype == torch.uint8
    assert actual.shape == packed.shape
    assert torch.equal(actual, repack_w4a8_weight(packed))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, k", [pytest.param(24, 128, id="n24"), pytest.param(16, 192, id="k192")]
)
def test_gemm_w4a8_repack_refuses_a_partial_unit(n: int, k: int) -> None:
    with pytest.raises(ValueError, match="multiple of"):
        GEMMW4A8FwdOp().repack(torch.zeros((n, k // 2), dtype=torch.uint8, device=run_device()))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, rule",
    [
        pytest.param(0, 16, 128, "M > 0", id="m0"),
        pytest.param(1, 0, 128, "N > 0", id="n0"),
        pytest.param(1, 16, 0, "K > 0", id="k0"),
        pytest.param(1, 15, 128, "N % 16", id="n15"),
        pytest.param(1, 16, 127, "K % 128", id="k127"),
    ],
)
def test_gemm_w4a8_rejects_a_shape_outside_the_signature(m: int, n: int, k: int, rule: str) -> None:
    device = run_device()
    inputs = (
        torch.zeros((m, k), device=device).to(torch.float8_e4m3fn),
        torch.zeros((n, k // 2), dtype=torch.uint8, device=device),
        torch.ones((n, k // 128), device=device).to(torch.float8_e4m3fn),
        torch.ones(n, device=device),
        torch.ones(m, device=device),
    )
    with pytest.raises(ValueError, match=rule):
        GEMMW4A8FwdOp()(*inputs)


@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a8_refuses_a_group_other_than_128() -> None:
    """``group_size`` 64 is in the signature, and no in-tree kernel serves it."""
    test = GEMMW4A8Test(1, 256, 512, 64)
    op = GEMMW4A8FwdOp(group_size=64, target=BUILTIN)
    with pytest.raises(ValueError, match="group_size"):
        op(*test.gen_inputs())
    assert not op._entries_built("gemm_w4a8")


@pytest.mark.smoke
@pytest.mark.parametrize("n, k", [(0, 128), (16, 0)])
def test_gemm_w4a8_repack_refuses_an_empty_axis(n: int, k: int) -> None:
    op = GEMMW4A8FwdOp(target=BUILTIN)
    with pytest.raises(ValueError, match="positive N and K"):
        op.repack(torch.empty(n, k // 2, dtype=torch.uint8, device=run_device()))
    assert not op._entries_built("w4a8_repack")


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "m, n, dtype",
    [
        pytest.param(64, 64, torch.float16, marks=pytest.mark.smoke),
        pytest.param(64, 64, torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(63, 65, torch.float16, marks=pytest.mark.full),
        pytest.param(65, 63, torch.float16, marks=pytest.mark.full),
    ],
)
def test_gemm_tn_single_contraction(m: int, n: int, dtype: torch.dtype) -> None:
    """A narrow contraction still has wide physical rows in TN layout."""
    a = (torch.arange(m, device=run_device()) % 7 - 3).to(dtype).reshape(1, m)
    b = (torch.arange(n, device=run_device()) % 5 - 2).to(dtype).reshape(1, n)
    op = GEMMFwdOp(trans_a=True, trans_b=False, target=BUILTIN)
    actual = op(a, b)
    torch.testing.assert_close(actual, a.T @ b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k,ta,tb,short",
    [
        pytest.param(64, 64, 1, False, False, "k", marks=pytest.mark.smoke),
        pytest.param(64, 64, 1, False, True, "k", marks=pytest.mark.full),
        pytest.param(64, 64, 1, True, True, "k", marks=pytest.mark.full),
        pytest.param(1, 64, 1, True, False, "m", marks=pytest.mark.full),
        pytest.param(64, 1, 1, True, False, "n", marks=pytest.mark.full),
        pytest.param(64, 64, 1, True, False, None, marks=pytest.mark.full),
        pytest.param(1, 64, 2, True, False, None, marks=pytest.mark.full),
        pytest.param(64, 1, 2, True, False, None, marks=pytest.mark.full),
        pytest.param(64, 64, 0, True, False, None, marks=pytest.mark.full),
    ],
)
def test_gemm_narrow_eligibility(m, n, k, ta, tb, short) -> None:
    """A refusal names the operand row too short to load; ``None`` is admitted. The rule
    reads only the element size, which float16 and bfloat16 share."""
    call = GEMMCall(
        arch=89, sm_count=128, m=m, n=n, k=k, dtype=torch.float16, trans_a=ta, trans_b=tb
    )
    reason = GEMMCpAsyncKernel.refusal(call)
    if short is None:
        assert reason is None
    else:
        assert f"{short} must span at least one" in reason


def _gemm_offset_case():
    a = torch.randn(256, 512, device=run_device()).bfloat16()
    b = torch.randn(384, 512, device=run_device()).bfloat16()
    return GEMMFwdOp(trans_b=True), [a, b]


def _gemm_fp8_block_scaled_offset_case():
    a = torch.randn(256, 512, device=run_device()).to(torch.float8_e4m3fn)
    b = torch.randn(384, 512, device=run_device()).to(torch.float8_e4m3fn)
    a_scale = torch.rand(256, 4, device=run_device()) + 0.5
    b_scale = torch.rand(3, 4, device=run_device()) + 0.5
    return GEMMFP8FwdOp(), [a, b, a_scale, b_scale]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "case, shifted",
    [
        pytest.param(_gemm_offset_case, 0, id="gemm-a"),
        pytest.param(_gemm_offset_case, 1, id="gemm-b"),
        pytest.param(_gemm_fp8_block_scaled_offset_case, 2, id="gemm-fp8-a_scale"),
    ],
)
def test_gemm_offset_operand_runs_as_an_aligned_one(case, shifted: int) -> None:
    """An operand whose storage starts off a 16-byte boundary gives the aligned result."""
    op, inputs = case()
    expected = op(*inputs)
    inputs[shifted] = misaligned(inputs[shifted])
    torch.testing.assert_close(op(*inputs), expected, rtol=0, atol=0)


@pytest.mark.sm89
@pytest.mark.parametrize(
    "reduce_threads, block_n, dtype",
    [
        pytest.param(64, 16, torch.float16, marks=pytest.mark.smoke),
        pytest.param(64, 16, torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(128, 8, torch.float16, marks=pytest.mark.full),
        pytest.param(256, 4, torch.float16, marks=pytest.mark.full),
    ],
)
def test_gemv_largest_tuning_ring(reduce_threads: int, block_n: int, dtype: torch.dtype) -> None:
    """The largest admitted staged ring compiles and preserves GEMV outputs."""
    test = GEMMWorkload(1, 256, 16384, dtype, trans_b=True)
    inputs = test.gen_inputs()
    op = GEMMFwdOp(trans_b=True)
    op(*inputs)
    entries = tuple(op._entries_built("gemm").values())
    assert len(entries) == 1 and isinstance(entries[0], GemvKernel)
    kernel = entries[0]
    budget = torch.cuda.get_device_properties(kernel.device_index).shared_memory_per_block_optin
    vector_bytes = 16
    assert all(
        cfg["block_n"] * cfg["reduce_threads"] * vector_bytes * cfg["num_stages"] <= budget
        for cfg in kernel.autotune_configs
    )
    config = {"block_n": block_n, "reduce_threads": reduce_threads, "num_stages": 6}
    assert config in kernel.autotune_configs
    kernel.init_config(config)
    compare_outputs(op(*inputs), test.ref_program(*inputs), test.verification(*inputs))


@pytest.mark.sm89
@pytest.mark.parametrize(
    "lhs_row, operand, dtype",
    [
        pytest.param(True, 0, torch.float16, marks=pytest.mark.smoke),
        pytest.param(True, 0, torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(True, 1, torch.float16, marks=pytest.mark.full),
        pytest.param(False, 0, torch.float16, marks=pytest.mark.full),
        pytest.param(False, 1, torch.float16, marks=pytest.mark.full),
    ],
)
def test_gemv_offset_storage(lhs_row: bool, operand: int, dtype: torch.dtype) -> None:
    """Contiguous inputs with an element offset retain the public GEMV contract."""
    m, n = (1, 3001) if lhs_row else (3001, 1)
    test = GEMMWorkload(m, n, 1024, dtype, trans_b=lhs_row)
    inputs = test.gen_inputs()
    source = inputs[operand]
    backing = torch.empty(source.numel() + 2, dtype=dtype, device=source.device)
    backing.fill_(17)
    shifted = backing[1:-1].view_as(source)
    shifted.copy_(source)
    assert shifted.is_contiguous() and shifted.storage_offset() == 1
    actual_inputs = tuple(shifted if i == operand else value for i, value in enumerate(inputs))
    before = backing.clone()
    op = GEMMFwdOp(trans_b=lhs_row)
    output = op(*actual_inputs)
    entries = tuple(op._entries_built("gemm").values())
    assert entries and all(isinstance(entry, GemvKernel) for entry in entries)
    compare_outputs(output, test.ref_program(*actual_inputs), test.verification(*actual_inputs))
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
