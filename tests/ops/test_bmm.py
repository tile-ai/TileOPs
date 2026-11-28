import pytest
import torch

from tests.cuda_graph_contract import CapturedGraph, check_replay
from tests.misaligned import misaligned
from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops._kernels.gemm import BMMCall, BMMW8A16Call
from tileops.backend import OpNotAvailableError
from tileops.gemm import BMMFP8FwdOp, BMMFwdOp, BMMW8A16FwdOp
from workloads.device import run_device
from workloads.gemm import (
    BMMBatchStridedWorkload,
    BMMFP8Workload,
    BMMW8A16BasisWorkload,
    BMMW8A16BatchStridedWorkload,
    BMMW8A16Workload,
    BMMWorkload,
)
from workloads.numerics import compare_outputs

# Covering the [B,K,N] path is the point of these tests, so the perf hint
# BMMFP8FwdOp emits for it is expected output, not a signal.
pytestmark = pytest.mark.filterwarnings("ignore:BMMFP8FwdOp")


class BMMTest(BMMWorkload, TestBase):
    pass


class BMMFP8Test(BMMFP8Workload, TestBase):
    pass


class BMMBatchStridedTest(BMMBatchStridedWorkload, TestBase):
    pass


def _skip_off_a_132_sm_h200(call: BMMCall) -> None:
    if call.calibration != "h200" or call.sm_count != 132:
        pytest.skip("the persistent and ping-pong paths are calibrated on a 132-SM H200")


class BMMFixture(FixtureBase):
    PARAMS = [
        (
            "batch, m, n, k, dtype",
            [
                pytest.param(
                    4,
                    128,
                    128,
                    128,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="gemm")],
                    id="smoke-fp16-b4-128",
                ),
                pytest.param(
                    4,
                    128,
                    128,
                    128,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-bf16-b4-128",
                ),
                pytest.param(
                    8,
                    512,
                    512,
                    512,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-fp16-b8-512",
                ),
                pytest.param(
                    32,
                    256,
                    256,
                    1024,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="full-bf16-b32-moe",
                ),
                pytest.param(
                    4,
                    200,
                    300,
                    128,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="full-fp16-b4-mn-nonaligned",
                ),
            ],
        ),
    ]


@BMMFixture
def test_bmm(batch: int, m: int, n: int, k: int, dtype: torch.dtype) -> None:
    test = BMMTest(batch, m, n, k, dtype)
    test.check(BMMFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_bmm_k_not_multiple_of_16_raises() -> None:
    """The in-tree kernels refuse a K that is not a multiple of 16, which torch.bmm admits."""
    op = BMMFwdOp()
    a = torch.randn(4, 16, 24, device=run_device(), dtype=torch.float16)
    b = torch.randn(4, 24, 16, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="requires k a multiple of 16"):
        op(a, b)


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    [
        # K past the ping-pong band keeps these on the persistent 128x256 tile.
        pytest.param(16, 512, 512, 1280, torch.bfloat16, marks=pytest.mark.smoke, id="one-wave"),
        pytest.param(
            16, 512, 512, 1280, torch.float16, marks=pytest.mark.smoke, id="one-wave-fp16"
        ),
        pytest.param(32, 512, 512, 1280, torch.bfloat16, marks=pytest.mark.full, id="two-waves"),
        # CTAs run unequal tile counts: the last held tile stores after the wave loop.
        pytest.param(36, 512, 512, 256, torch.bfloat16, marks=pytest.mark.full, id="uneven-waves"),
        # K shorter than the ring: each tile stores as it finishes.
        pytest.param(
            32, 128, 2048, 128, torch.bfloat16, marks=pytest.mark.full, id="short-k-two-waves"
        ),
        # Neither output extent is a whole number of tiles.
        pytest.param(
            3, 1000, 1000, 1024, torch.bfloat16, marks=pytest.mark.full, id="ragged-tiles"
        ),
    ],
)
def test_bmm_wave_filling_shapes_match_the_reference(
    batch: int, m: int, n: int, k: int, dtype: torch.dtype
) -> None:
    """Shapes whose whole tiles fill a persistent wave on the calibrated board."""
    test = BMMTest(batch, m, n, k, dtype)
    op = BMMFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "batch, m, n, k, key",
    [
        pytest.param(8, 1024, 1024, 1024, "bmm_pingpong", id="two-waves"),
        pytest.param(8, 1024, 1024, 384, "bmm_pingpong", id="k-lower-endpoint"),
        pytest.param(8, 1024, 1024, 376, "bmm_persistent", id="k-lower-neighbor"),
        pytest.param(8, 1024, 1024, 1032, "bmm_persistent", id="k-upper-neighbor"),
        pytest.param(9, 1024, 1024, 1024, "bmm_persistent", id="waves-upper-neighbor"),
        # 132 tiles of 128x128 give no CTA a second; 133 give one CTA two.
        pytest.param(132, 128, 128, 512, "bmm_persistent", id="one-tile-a-cta"),
        pytest.param(133, 128, 128, 512, "bmm_pingpong", id="one-cta-two-tiles"),
        pytest.param(4, 1024, 1088, 1024, "bmm_persistent", id="n-partial-tile"),
    ],
)
def test_bmm_pingpong_calibrated_band(batch: int, m: int, n: int, k: int, key: str) -> None:
    """The H200 calibration selects ping-pong in its band and the persistent tile outside it."""
    device = torch.device("cuda", 0)  # lint: device-literal allow: facts are stated
    call = BMMCall(
        batch=batch,
        m=m,
        n=n,
        k=k,
        dtype=torch.bfloat16,
        arch=90,
        sm_count=132,
        calibration="h200",
        smem_budget=BLOCK_SHARED_BYTES_OPT_IN[90],
        device=device,
    )
    assert BMMFwdOp().key_for("bmm", call) == key


@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(16, 512, 512, 1280, marks=pytest.mark.smoke),
        pytest.param(8, 1024, 1024, 1024, marks=pytest.mark.full),
    ],
)
def test_bmm_template_runs_the_config_its_kernel_publishes(monkeypatch, batch, m, n, k):
    """The template launch follows the published config: the description of the first
    launch compiles the same instantiation again, and another config compiles its own."""
    import tileops._kernels.gemm.persistent.template as template

    call = BMMCall(batch=batch, m=m, n=n, k=k, dtype=torch.bfloat16, device=run_device())
    _skip_off_a_132_sm_h200(call)
    built = []
    build = template._grouped_gemm_kernel

    def recording(spec):
        built.append(spec)
        return build(spec)

    monkeypatch.setattr(template, "_grouped_gemm_kernel", recording)
    test = BMMTest(batch, m, n, k, torch.bfloat16)
    inputs = test.gen_inputs()
    op = BMMFwdOp()
    test.check(op, *inputs)
    test.check(op, *inputs)
    first, again = built[0], built[-1]
    assert again == first
    (kernel,) = op.iter_kernels()
    kernel.config = {**kernel.config, "num_stages": first.num_stages - 1}
    test.check(op, *inputs)
    assert built[-1].num_stages == first.num_stages - 1


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    [
        # 512 tiles over 128 CTAs: each runs four.
        pytest.param(
            8, 1024, 1024, 1024, torch.bfloat16, marks=pytest.mark.smoke, id="two-waves-bf16"
        ),
        pytest.param(
            8, 1024, 1024, 1024, torch.float16, marks=pytest.mark.smoke, id="two-waves-fp16"
        ),
        # 320 tiles over every SM: CTAs run two or three, so a CTA's groups end apart.
        pytest.param(
            5, 1024, 1024, 1024, torch.bfloat16, marks=pytest.mark.full, id="uneven-tiles"
        ),
        # K is not a whole number of 64-wide blocks.
        pytest.param(4, 1024, 1024, 528, torch.bfloat16, marks=pytest.mark.full, id="k-tail"),
    ],
)
def test_bmm_pingpong_matches_the_reference(
    batch: int, m: int, n: int, k: int, dtype: torch.dtype
) -> None:
    """Calls in the ping-pong band, each group of a CTA running whole 128x128 tiles in turn."""
    op = BMMFwdOp()
    call = BMMCall(batch=batch, m=m, n=n, k=k, dtype=dtype, device=run_device())
    _skip_off_a_132_sm_h200(call)
    assert op.key_for("bmm", call) == "bmm_pingpong"
    test = BMMTest(batch, m, n, k, dtype)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "tune, batch, m, n, k, key",
    [
        pytest.param(False, 4, 128, 128, 128, "bmm", marks=pytest.mark.smoke, id="two-panels"),
        pytest.param(
            False, 16, 512, 512, 1280, "bmm_persistent", marks=pytest.mark.smoke, id="persistent"
        ),
        pytest.param(True, 4, 128, 128, 128, "bmm", marks=pytest.mark.full, id="two-panels-tuned"),
        pytest.param(
            False, 8, 64, 1024, 4096, "bmm", marks=pytest.mark.full, id="deepseek-wo-a-64"
        ),
        pytest.param(
            False, 8, 1024, 1024, 1024, "bmm_pingpong", marks=pytest.mark.full, id="pingpong"
        ),
        pytest.param(
            False, 3, 1000, 1000, 1024, "bmm_persistent", marks=pytest.mark.full, id="ragged-tiles"
        ),
    ],
)
@pytest.mark.in_tree_kernels
def test_bmm_reads_and_writes_batch_strided_views(
    tune: bool, batch: int, m: int, n: int, k: int, key: str
) -> None:
    """DeepSeek-V4-Flash's grouped output projection: ``a`` and ``out`` are the batch-major
    views of ``[M, B, K]`` and ``[M, B, N]`` tensors, read and written where they lie, the
    product equal bit for bit to the contiguous call's; with ``tune``, on an op in tuned mode,
    where TileLang's tuner first times the general kernel's candidate configurations."""
    op = BMMFwdOp()
    call = BMMCall(batch=batch, m=m, n=n, k=k, dtype=torch.bfloat16, device=run_device())
    if key != "bmm":
        _skip_off_a_132_sm_h200(call)
    assert op.key_for("bmm", call) == key
    if tune:
        op.request_tune()
    test = BMMBatchStridedTest(batch, m, n, k, torch.bfloat16)
    test.check(op, *test.gen_inputs())
    a, b = test.gen_inputs()
    rows = torch.full((m, batch, n), float("nan"), dtype=a.dtype, device=a.device)
    out = rows.transpose(0, 1)
    assert op(a, b, out=out) is out
    assert torch.equal(out, op(a.contiguous(), b))


@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(4, 128, 128, 128, marks=pytest.mark.smoke, id="general"),
        pytest.param(32, 512, 512, 1280, marks=pytest.mark.full, id="persistent-two-waves"),
    ],
)
def test_bmm_takes_operands_of_any_strides(batch: int, m: int, n: int, k: int):
    """``a`` and ``out`` whose rows lie four elements past their width, 8 bytes, which the
    general kernel reads and writes where they lie and the persistent one, whose TMA row
    pitch and asynchronous stores take whole 16-byte steps, through contiguous copies; ``a``
    of every other element, which the op copies; and ``out`` transposed in its last two axes.
    Each gives the contiguous call's product."""
    test = BMMTest(batch, m, n, k, torch.bfloat16)
    a, b = test.gen_inputs()
    op = BMMFwdOp()
    expected = op(a, b)
    a_rows = torch.zeros(batch, m, k + 4, dtype=a.dtype, device=a.device)
    a_rows[..., :k] = a
    out_rows = torch.full((batch, m, n + 4), float("nan"), dtype=a.dtype, device=a.device)
    out = out_rows[..., :n]
    assert op(a_rows[..., :k], b, out=out) is out
    assert torch.equal(out, expected)
    assert out_rows[..., n:].isnan().all()
    a_wide = torch.zeros(batch, m, 2 * k, dtype=a.dtype, device=a.device)
    a_wide[..., ::2] = a
    out = torch.full((batch, n, m), float("nan"), dtype=a.dtype, device=a.device).mT
    assert op(a_wide[..., ::2], b, out=out) is out
    assert torch.equal(out, expected)


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="BMMFwdOp")
@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(
            1, 64, 4096 // 4, 128, marks=pytest.mark.smoke,
            id="deepseek-v4-flash-indexer-scores",
        ),
        pytest.param(
            1, 64, 4128 // 4, 128, marks=pytest.mark.full,
            id="deepseek-v4-flash-indexer-scores-4128",
        ),
        pytest.param(
            1, 1, 4096 // 4, 64, marks=pytest.mark.full, id="deepseek-v4-flash-indexer-logits"
        ),
        pytest.param(
            1, 1, 4128 // 4, 64, marks=pytest.mark.full,
            id="deepseek-v4-flash-indexer-logits-4128",
        ),
    ],
)  # fmt: skip
def test_bmm_graph_replay(batch: int, m: int, n: int, k: int) -> None:
    """DeepSeek-V4-Flash's decode-step indexer scores and logits over the compressed entries of
    max_seq_len 4096 and 4128."""
    check_replay(BMMTest(batch, m, n, k, torch.bfloat16), BMMFwdOp())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="BMMFwdOp")
@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(4, 128, 128, 128, marks=pytest.mark.smoke, id="two-panels"),
        pytest.param(
            8, 1, 1024, 64 * 512 // 8, marks=pytest.mark.full, id="deepseek-v4-flash-wo-a"
        ),
        pytest.param(16, 512, 512, 1280, marks=pytest.mark.full, id="persistent"),
    ],
)
def test_bmm_graph_replay_into_batch_strided_views(batch: int, m: int, n: int, k: int) -> None:
    """A replay reading ``a`` and writing ``out`` as batch-strided views, as DeepSeek-V4-Flash's
    decode step calls its grouped output projection: ``a`` the group-major view of the
    attention output, ``out`` that of the token-major buffer ``wo_b`` reads."""
    test = BMMBatchStridedTest(batch, m, n, k, torch.bfloat16)
    op = BMMFwdOp()
    a, b = test.gen_inputs()
    out = torch.empty(m, batch, n, dtype=a.dtype, device=a.device).transpose(0, 1)
    graph = CapturedGraph((lambda *inputs: op(*inputs, out=out), (a, b)))
    assert graph.outputs[0] is out
    fresh = test.gen_inputs()
    graph.renew(fresh)
    graph.replay()
    compare_outputs(out, test.ref_program(*fresh), test.verification(*fresh))


class BMMFP8Fixture(FixtureBase):
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


@BMMFP8Fixture
def test_bmm_fp8(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    test = BMMFP8Test(batch, m, n, k, dtype, out_dtype=out_dtype)
    op = BMMFP8FwdOp(out_dtype=out_dtype, trans_b=False)
    inputs = test.gen_inputs()
    test.check(op, *inputs)


@pytest.mark.smoke
def test_bmm_fp8_rejects_e5m2() -> None:
    """BMMFP8FwdOp advertises fp8_e4m3fn only; e5m2 inputs must be rejected."""
    test = BMMFP8Test(4, 128, 128, 128, torch.float8_e5m2)
    op = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)
    with pytest.raises(ValueError, match=r"outside \['float8_e4m3fn'\]"):
        op(*test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_bmm_fp8_k_not_multiple_of_32_raises() -> None:
    op = BMMFP8FwdOp(trans_b=False)
    a = torch.randn(4, 128, 48, device=run_device()).to(torch.float8_e4m3fn)
    b = torch.randn(4, 48, 128, device=run_device()).to(torch.float8_e4m3fn)
    a_scale = torch.tensor(1.0, device=run_device(), dtype=torch.float32)
    b_scale = torch.tensor(1.0, device=run_device(), dtype=torch.float32)
    with pytest.raises(ValueError, match="multiple of 32"):
        op(a, b, a_scale, b_scale)


@pytest.mark.smoke
def test_bmm_fp8_nk_view_when_k_eq_n() -> None:
    batch, m, n, k = 4, 128, 128, 128
    test = BMMFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, a_scale, b_scale = test.gen_inputs()
    b_nk_view = b_kn.transpose(-2, -1)
    assert b_nk_view.shape == (batch, n, k)
    assert b_nk_view.stride(-2) == 1
    op_kn = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)
    op_nk = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_kn = op_kn(a, b_kn, a_scale, b_scale).clone()
    out_nk = op_nk(a, b_nk_view, a_scale, b_scale)
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_contiguous_nk_square_when_k_eq_n() -> None:
    batch, m, n, k = 4, 128, 128, 128
    test = BMMFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, a_scale, b_scale = test.gen_inputs()
    b_nk = b_kn.transpose(-2, -1).contiguous()
    assert b_nk.is_contiguous()
    assert b_nk.shape == (batch, n, k)
    assert b_nk.stride(-2) == k
    op_nk = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_nk = op_nk(a, b_nk, a_scale, b_scale).clone()
    op_kn = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)
    out_kn = op_kn(a, b_kn, a_scale, b_scale)
    torch.testing.assert_close(out_nk, out_kn, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_kn_transpose_handles_tile_tail() -> None:
    """A contiguous [B, N, K] b gives the [B, K, N] result, on extents that leave a tail
    under every staging tile."""
    batch, m, n, k = 3, 128, 80, 160
    test = BMMFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, a_scale, b_scale = test.gen_inputs()
    b_nk = b_kn.transpose(-2, -1).contiguous()

    op_kn = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)
    op_nk = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    out_kn = op_kn(a, b_kn, a_scale, b_scale).clone()
    out_nk = op_nk(a, b_nk, a_scale, b_scale)
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_no_transpose_when_b_already_k_innermost() -> None:
    """What decides the copy is ``b``'s strides, not ``trans_b``."""
    batch, m, n, k = 4, 128, 256, 128
    test = BMMFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, a_scale, b_scale = test.gen_inputs()
    # [B, K, N] by shape, K-innermost in memory.
    b_kn_view = b_kn.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert b_kn_view.stride(-2) == 1

    op = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)
    out_view = op(a, b_kn_view, a_scale, b_scale).clone()

    out_kn = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)(a, b_kn, a_scale, b_scale)
    torch.testing.assert_close(out_view, out_kn, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_one_32_wide_k_step() -> None:
    batch, m, n, k = 8, 64, 64, 32
    test = BMMFP8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, a_scale, b_scale = test.gen_inputs()
    op = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)

    test.check(op, a, b_kn, a_scale, b_scale)


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_bmm_fp8_under_tuning(tune: bool) -> None:
    test = BMMFP8Test(2, 128, 144, 96, torch.float8_e4m3fn)
    op = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=False)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("shifted", [0, 1], ids=["a", "b"])
def test_bmm_offset_operand_runs_as_an_aligned_one(shifted: int) -> None:
    """An operand whose storage starts off a 16-byte boundary gives the aligned result."""
    inputs = [
        torch.randn(4, 256, 512, device=run_device()).bfloat16(),
        torch.randn(4, 512, 384, device=run_device()).bfloat16(),
    ]
    op = BMMFwdOp()
    expected = op(*inputs)
    inputs[shifted] = misaligned(inputs[shifted])
    torch.testing.assert_close(op(*inputs), expected, rtol=0, atol=0)


class BMMW8A16Test(BMMW8A16Workload, TestBase):
    pass


class BMMW8A16BatchStridedTest(BMMW8A16BatchStridedWorkload, TestBase):
    pass


class BMMW8A16BasisTest(BMMW8A16BasisWorkload, TestBase):
    pass


class BMMW8A16Fixture(FixtureBase):
    PARAMS = [
        (
            "batch, m, n, k",
            [
                pytest.param(
                    8,
                    1,
                    1024,
                    4096,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="gemm")],
                    id="ds-v4-flash-wo-a-decode",
                ),
                pytest.param(2, 3, 136, 384, marks=pytest.mark.full, id="more-k-parts-than-steps"),
                pytest.param(2, 64, 192, 512, marks=pytest.mark.full, id="widest-split-k-tile"),
                pytest.param(3, 17, 200, 384, marks=pytest.mark.full, id="ragged-rows-and-n"),
                pytest.param(2, 130, 520, 640, marks=pytest.mark.full, id="few-64-row-units"),
                pytest.param(2, 260, 4232, 256, marks=pytest.mark.full, id="two-weight-warpgroups"),
                pytest.param(1, 300, 72, 256, marks=pytest.mark.full, id="n-in-one-block"),
                pytest.param(7, 2, 2432, 1408, marks=pytest.mark.full, id="split-k-units-per-cta"),
                pytest.param(
                    1,
                    2**21 + 128,
                    1024,
                    128,
                    marks=pytest.mark.nightly,
                    id="output-past-2-31-elements",
                ),
            ],
        ),
    ]


@BMMW8A16Fixture
def test_bmm_w8a16(batch: int, m: int, n: int, k: int) -> None:
    test = BMMW8A16Test(batch, m, n, k)
    test.check(BMMW8A16FwdOp(), *test.gen_inputs())


@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(8, 1, 1024, 4096, marks=pytest.mark.smoke, id="ds-v4-flash-wo-a-decode"),
        pytest.param(2, 64, 192, 512, marks=pytest.mark.full, id="split-k-tile"),
        pytest.param(2, 200, 520, 1024, marks=pytest.mark.full, id="64-row-tiles"),
        pytest.param(2, 260, 4232, 256, marks=pytest.mark.full, id="two-weight-warpgroups"),
        pytest.param(1, 512, 128, 16512, marks=pytest.mark.full, id="k-past-128-scale-steps"),
    ],
)
def test_bmm_w8a16_dequantizes_each_weight_exactly(batch: int, m: int, n: int, k: int) -> None:
    """Basis-vector rows select single weights, so each output is one weight as the reference
    dequantizes it, the code times its block's scale rounded to bfloat16, bit for bit: every
    e4m3 code, every 16-byte chunk of a K step, scales that are not powers of two."""
    test = BMMW8A16BasisTest(batch, m, n, k)
    test.check(BMMW8A16FwdOp(), *test.gen_inputs())


@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(8, 1, 1024, 4096, marks=pytest.mark.smoke, id="ds-v4-flash-wo-a-decode"),
        pytest.param(2, 260, 4232, 256, marks=pytest.mark.full, id="two-weight-warpgroups"),
    ],
)
def test_bmm_w8a16_reads_and_writes_batch_strided_views(batch: int, m: int, n: int, k: int) -> None:
    """DeepSeek-V4-Flash's grouped output projection: ``activation`` and ``out`` are the
    batch-major views of ``[M, B, K]`` and ``[M, B, N]`` tensors, read and written where they
    lie, the product equal bit for bit to the contiguous call's."""
    test = BMMW8A16BatchStridedTest(batch, m, n, k)
    op = BMMW8A16FwdOp()
    test.check(op, *test.gen_inputs())
    a, weight, scale = test.gen_inputs()
    rows = torch.full((m, batch, n), float("nan"), dtype=a.dtype, device=a.device)
    out = rows.transpose(0, 1)
    assert op(a, weight, scale, out=out) is out
    assert torch.equal(out, op(a.contiguous(), weight, scale))


@pytest.mark.smoke
def test_bmm_w8a16_takes_operands_of_any_strides() -> None:
    """Operands at the decode shape: ``activation`` whose rows lie four elements past their
    width, 8 bytes, which TMA cannot address, and ``out`` likewise; ``out`` whose rows lie eight
    elements past their width, which the kernel writes in place; ``activation`` of every other
    element; ``out`` transposed in its last two axes. Each gives the contiguous call's product,
    and every padding element stays untouched."""
    batch, m, n, k = 8, 1, 1024, 4096
    test = BMMW8A16Test(batch, m, n, k)
    a, weight, scale = test.gen_inputs()
    op = BMMW8A16FwdOp()
    expected = op(a, weight, scale)
    a_rows = torch.zeros(batch, m, k + 4, dtype=a.dtype, device=a.device)
    a_rows[..., :k] = a
    for pad in (4, 8):
        out_rows = torch.full((batch, m, n + pad), float("nan"), dtype=a.dtype, device=a.device)
        out = out_rows[..., :n]
        assert op(a_rows[..., :k], weight, scale, out=out) is out
        assert torch.equal(out, expected)
        assert out_rows[..., n:].isnan().all()
    a_wide = torch.zeros(batch, m, 2 * k, dtype=a.dtype, device=a.device)
    a_wide[..., ::2] = a
    out = torch.full((batch, n, m), float("nan"), dtype=a.dtype, device=a.device).mT
    assert op(a_wide[..., ::2], weight, scale, out=out) is out
    assert torch.equal(out, expected)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_bmm_w8a16_k_past_its_ring_is_refused() -> None:
    """The in-tree kernel refuses a K whose staged scales leave shared memory no ring stage,
    even with K unsplit."""
    call = BMMW8A16Call(batch=1, m=64, n=1024, k=128 * 60000, device=run_device())
    with pytest.raises(OpNotAvailableError, match="ring stages"):
        BMMW8A16FwdOp().key_for("bmm_w8a16", call)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "m",
    [
        pytest.param(1, marks=pytest.mark.smoke, id="ds-v4-flash-wo-a-decode"),
        pytest.param(40, marks=pytest.mark.full, id="ds-v4-flash-wo-a-40"),
        pytest.param(64, marks=pytest.mark.full, id="ds-v4-flash-wo-a-64"),
    ],
)
def test_bmm_w8a16_ring_gives_each_slot_one_k_part(m: int) -> None:
    """A ring that wraps within a unit holds a whole number of K parts' steps, so a part never
    waits on a slot whose previous fill another part owns: its parity wait would return while
    that fill's copy is in flight."""
    test = BMMW8A16Test(8, m, 1024, 4096)
    op = BMMW8A16FwdOp()
    test.check(op, *test.gen_inputs())
    (kernel,) = op.iter_kernels()
    cfg = kernel.config
    assert cfg["num_stages"] >= 4096 // 128 or cfg["num_stages"] % cfg["ksplit"] == 0


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_bmm_w8a16_k_zero_is_refused() -> None:
    """The in-tree kernel refuses K = 0, which the signature admits and the reference answers
    with zeros."""
    test = BMMW8A16Test(1, 4, 128, 0)
    with pytest.raises(OpNotAvailableError, match="positive multiple"):
        BMMW8A16FwdOp()(*test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_bmm_w8a16_n_not_multiple_of_8_is_refused() -> None:
    """The in-tree kernel refuses an ``n`` that is not a multiple of 8, which the signature
    admits."""
    test = BMMW8A16Test(1, 4, 132, 256)
    with pytest.raises(OpNotAvailableError, match="not a multiple of 8"):
        BMMW8A16FwdOp()(*test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="BMMW8A16FwdOp")
@pytest.mark.parametrize(
    "batch, m, n, k",
    [
        pytest.param(8, 1, 1024, 4096, marks=pytest.mark.smoke, id="ds-v4-flash-wo-a-decode"),
        pytest.param(2, 260, 4232, 256, marks=pytest.mark.full, id="two-weight-warpgroups"),
    ],
)
def test_bmm_w8a16_graph_replay_into_batch_strided_views(
    batch: int, m: int, n: int, k: int
) -> None:
    """A replay reading ``activation`` and writing ``out`` as batch-strided views, as
    DeepSeek-V4-Flash's decode step calls ``wo_a``, on fresh activations, weights and scales."""
    test = BMMW8A16BatchStridedTest(batch, m, n, k)
    op = BMMW8A16FwdOp()
    a, weight, scale = test.gen_inputs()
    out = torch.empty(m, batch, n, dtype=a.dtype, device=a.device).transpose(0, 1)
    graph = CapturedGraph((lambda *inputs: op(*inputs, out=out), (a, weight, scale)))
    assert graph.outputs[0] is out
    fresh = test.gen_inputs()
    graph.renew(fresh)
    graph.replay()
    compare_outputs(out, test.ref_program(*fresh), test.verification(*fresh))
