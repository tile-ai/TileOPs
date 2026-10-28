import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.gemm import GEMMINT8W8A8Call, GEMMW4A8Call
from tileops._kernels.gemm.dense import GemvKernel
from tileops._utils import get_sm_count
from tileops.backend import BUILTIN
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
    """GEMV outputs remain correct for non-aligned free and contraction dimensions."""

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
                pytest.param(
                    20,
                    4096,
                    1024,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-deferred-scale-16-rows",
                ),
                pytest.param(
                    40,
                    4096,
                    1024,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-deferred-scale-32-rows",
                ),
                pytest.param(
                    40,
                    2048,
                    14336,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-deferred-scale-16-rows-split-k",
                ),
                pytest.param(
                    96,
                    2048,
                    14336,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-w4a16-deferred-scale-32-rows-split-k",
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
                # Short-K model projections at small token counts.
                pytest.param(16, 7168, 2304, marks=pytest.mark.full, id="full-w4a8-decode-ds-m16"),
                pytest.param(32, 7168, 2304, marks=pytest.mark.full, id="full-w4a8-decode-ds-m32"),
                pytest.param(
                    32, 7680, 3456, marks=pytest.mark.full, id="full-w4a8-decode-negative-guard"
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    test = GEMMTest(m, n, k, dtype, trans_a, trans_b)
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


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
                    marks=pytest.mark.smoke,
                    id="smoke-tensor-scale-a",
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
                    marks=pytest.mark.smoke,
                    id="smoke-tensor-scale-b",
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
                    marks=pytest.mark.smoke,
                    id="smoke-bias",
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
                    marks=pytest.mark.smoke,
                    id="smoke-one-row",
                ),
                # Ping-pong: M / N / K tails, and 297 tiles, more than two waves of 132 SMs,
                # so each CTA runs both consumers and hands the turn back and forth.
                pytest.param(
                    1100,
                    4104,
                    1040,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    _INT8_PP,
                    marks=pytest.mark.smoke,
                    id="smoke-pingpong-tails",
                ),
                # k off TMA's 16-byte unit: the fallback.
                pytest.param(
                    37,
                    104,
                    200,
                    True,
                    True,
                    torch.bfloat16,
                    True,
                    _INT8_SIMPLE,
                    marks=pytest.mark.smoke,
                    id="smoke-unaligned-k",
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
                pytest.param(
                    1100,
                    4104,
                    1040,
                    False,
                    False,
                    torch.float16,
                    True,
                    _INT8_PP,
                    marks=pytest.mark.full,
                    id="full-pingpong-tensor-scales-bias",
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


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k, key",
    [
        pytest.param(32, 512, 1024, "gemm_w4a8_decode", id="base-decode-upper-endpoint"),
        # Both implementations accept these extension calls; decode must take priority.
        pytest.param(33, 17024, 7168, "gemm_w4a8_decode", id="lower-endpoints"),
        pytest.param(40, 28800, 8192, "gemm_w4a8_decode", id="upper-endpoints"),
        pytest.param(33, 18432, 7680, "gemm_w4a8_decode", id="interior-k"),
        # Reachable H200 neighbors retain prefill.
        pytest.param(41, 18432, 7168, "gemm_w4a8_prefill", id="token-upper-neighbor"),
        pytest.param(33, 18432, 8064, "gemm_w4a8_prefill", id="k-remainder"),
        pytest.param(33, 18432, 6656, "gemm_w4a8_prefill", id="k-lower-neighbor"),
        pytest.param(33, 18432, 8704, "gemm_w4a8_prefill", id="k-upper-neighbor"),
        pytest.param(33, 16896, 7168, "gemm_w4a8_prefill", id="n-lower-neighbor"),
        pytest.param(33, 28928, 7168, "gemm_w4a8_prefill", id="n-upper-neighbor"),
        pytest.param(33, 18496, 7168, "gemm_w4a8_prefill", id="n-remainder"),
    ],
)
def test_w4a8_extended_decode_calibrated_band(m: int, n: int, k: int, key: str) -> None:
    """The real H200 calibration selects decode in its extension and prefill outside it."""
    op = GEMMW4A8FwdOp(target=BUILTIN)
    call = GEMMW4A8Call(m=m, n=n, k=k, group_size=op.group_size, device=run_device())
    if call.calibration != "h200" or call.sm_count != 132:
        pytest.skip("the extended decode selection is calibrated on a 132-SM H200")
    assert op.key_for("gemm_w4a8", call) == key


@GEMMW4A16Fixture
def test_gemm_w4a16(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = GEMMW4A16Test(m, n, k, dtype)
    op = GEMMW4A16FwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
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


@pytest.mark.smoke
@pytest.mark.parametrize("k_index", [0, 1, 127, 128, 200, 383])
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
@pytest.mark.smoke
@pytest.mark.parametrize("scale_mode", ["per_tensor", "block128", "block128x128"])
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
@pytest.mark.smoke
@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(4096, 576, 7168, id="one-wave"),
        pytest.param(4096, 7168, 16384, id="two-wave"),
    ],
)
def test_gemm_fp8_1d1d_wave_matches_reference(m: int, n: int, k: int) -> None:
    """Per-row weight scales preserve every output on tall projections."""
    test = GEMMFP8Test(m, n, k, torch.float8_e4m3fn, "block128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
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
    """Per-block weight scales preserve every output on tall and wide projections."""
    test = GEMMFP8Test(m, n, k, torch.float8_e4m3fn, "block128x128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, k",
    [
        pytest.param(4096, 7168, id="column-run"),
        pytest.param(24576, 1536, id="scale-span"),
    ],
)
def test_gemm_fp8_1d2d_wave_column_run_matches_reference(n: int, k: int) -> None:
    """Column-scale blocks and reused row-scale spans preserve every output."""
    test = GEMMFP8Test(4096, n, k, torch.float8_e4m3fn, "block128x128")
    test.check(GEMMFP8FwdOp(target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, scale_mode",
    [
        pytest.param(1024, "per_tensor", id="per-tensor"),
        pytest.param(576, "block128x128", id="block128x128"),
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


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool, arch: int) -> None:
    """A single activation row supports non-aligned weight rows and contraction sizes."""
    test = GEMMTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GEMMFwdOp(trans_a=False, trans_b=True)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())
    assert all(isinstance(entry, GemvKernel) for entry in op._entries_built("gemm").values())


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool, arch: int) -> None:
    """A single output column supports non-aligned activation rows and contraction sizes."""
    m = n
    test = GEMMTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GEMMFwdOp(trans_a=False, trans_b=False)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())
    assert all(isinstance(entry, GemvKernel) for entry in op._entries_built("gemm").values())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gemm_coop2_epilogue_staged_in_two_chunks() -> None:
    """A wide output preserves every column across the epilogue."""
    test = GEMMTest(704, 5440, 128, torch.bfloat16, False, True)
    test.check(GEMMFwdOp(trans_b=True, target=BUILTIN), *test.gen_inputs())


@pytest.mark.in_tree_kernels
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


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("m", [100, 257])
def test_gemm_w4a16_kernel_predicates_a_ragged_token_count(m: int) -> None:
    test = GEMMW4A16Test(m, 1024, 512, torch.float16)
    op = GEMMW4A16FwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "n, k",
    [
        # One-row projections with different output widths.
        pytest.param(1024, 8192, id="split-k-8", marks=pytest.mark.smoke),
        pytest.param(2176, 8192, id="split-k-3-uneven", marks=pytest.mark.smoke),
        # Long contractions with a final 128-element group.
        pytest.param(4288, 32896, id="stream-k-crossing", marks=pytest.mark.full),
        pytest.param(4224, 32896, id="stream-k-two-way", marks=pytest.mark.full),
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


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gemm_w4a16_two_stage_ring_refills_after_the_last_sub_step() -> None:
    """Repeated K-tile refills preserve every sub-step's input until its MMA completes."""
    test = GEMMW4A16Test(384, 2048, 4096, torch.float16)
    test.check(GEMMW4A16FwdOp(target=BUILTIN), *test.gen_inputs())


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
@pytest.mark.smoke
@pytest.mark.parametrize(("n", "k"), [(64, 256), (1024, 512)])
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
    from tileops._utils import get_sm_count

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
    kernel.config = {**kernel.config, "num_ctas": ctas}
    test.check(op, *inputs, runs=run)
    ((_, _, flags, _),) = kernel._generation.programs.values()
    assert flags.numel() == ctas * kernel.config["unit_warpgroups"]
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
        # Reuse a small-token entry after serving another token count. Calibrated
        # boards can also exercise the decode kernel's independent flag workspace.
        pytest.param(
            (33, 40, 33), 18432, 7168, marks=pytest.mark.full, id="small-token-entry-reuse"
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
@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, k",
    [
        pytest.param(16, 128, id="n16-k128"),
        pytest.param(112, 384, id="n112-k384"),
        pytest.param(7184, 2304, id="n7184-k2304"),
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
