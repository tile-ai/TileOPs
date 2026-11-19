"""Varlen GQA tests against a pure-PyTorch reference."""

import re

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._perf.formulas import visible_scores
from tileops.attention import GQAVarlenFwdOp
from tileops.backend import OpNotAvailableError
from workloads.attention.gqa.varlen import GQAVarlenFwdWorkload


class GQAVarlenFwdTest(GQAVarlenFwdWorkload, TestBase):
    pass


def _padded(inputs: tuple) -> tuple:
    """A workload's inputs padded to the op's eleven forward arguments."""
    return tuple(inputs) + (None,) * (11 - len(inputs))


def _launched_rows(op: GQAVarlenFwdOp, inputs: tuple) -> int:
    """Query rows a tile of the program the warp-specialized kernel launched for *inputs*:
    64 a consumer warpgroup, read from the program's thread count."""
    kernel = op._get_kernel(_padded(inputs))
    program = kernel.program(kernel._compile, *kernel._program_facts(inputs[0]))
    threads = re.search(r"__launch_bounds__\((\d+)", program.jit.get_kernel_source())
    return 64 * (int(threads.group(1)) // 128 - 1)


class GQAVarlenFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype",
            [
                # Prefill: seqlen_q == seqlen_k (offset=0)
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),  # causal
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                ),  # causal bf16
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    128,
                    -1,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),  # causal + wl
                pytest.param(
                    1,
                    [128],
                    [128],
                    8,
                    2,
                    128,
                    True,
                    64,
                    -1,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),  # D=128 uses the two-stage sliding pipeline on SM90
                pytest.param(
                    1,
                    [6],
                    [2],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),  # leading queries have no visible key
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    False,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # bidirectional
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    False,
                    64,
                    64,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # window
                # KV-cache: seqlen_k > seqlen_q (offset > 0)
                pytest.param(
                    2,
                    [64, 128],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # causal kvcache
                pytest.param(
                    2,
                    [64, 128],
                    [256, 512],
                    8,
                    2,
                    64,
                    True,
                    128,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # causal+wl kvcache
                pytest.param(
                    2,
                    [64, 128],
                    [256, 512],
                    8,
                    2,
                    64,
                    False,
                    64,
                    64,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # window kvcache
                # bfloat16
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    False,
                    64,
                    64,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                ),  # window bf16
                # GQA ratios
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    8,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # MHA 1:1
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    16,
                    1,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # ratio 16:1
                # Mixed lengths within batch
                pytest.param(
                    3,
                    [128, 256, 384],
                    [128, 256, 384],
                    8,
                    2,
                    64,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),
                # Right window only
                pytest.param(
                    2,
                    [256, 512],
                    [256, 512],
                    8,
                    2,
                    64,
                    False,
                    -1,
                    64,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # right window
                # wl=0 boundary
                pytest.param(
                    2,
                    [128, 256],
                    [128, 256],
                    8,
                    2,
                    64,
                    True,
                    0,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # wl=0
                # Regression: a window at D=256 reached the sliding-window kernel, whose
                # tiles need 394240 bytes of shared memory, and failed at launch; the
                # general kernel serves it
                pytest.param(
                    2,
                    [300, 129],
                    [300, 129],
                    8,
                    2,
                    256,
                    True,
                    64,
                    -1,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                ),  # D=256 causal+wl
                # D=512 stages the score tile through shared memory: two warpgroups where
                # shared memory allows it, a 32-row tile on one warpgroup where it does not
                pytest.param(
                    2,
                    [300, 700],
                    [300, 700],
                    8,
                    2,
                    512,
                    True,
                    -1,
                    -1,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                ),  # D=512 causal, partial tiles
                pytest.param(
                    2,
                    [100, 200],
                    [300, 700],
                    8,
                    2,
                    512,
                    True,
                    -1,
                    -1,
                    torch.float16,
                    marks=pytest.mark.full,
                ),  # D=512 causal kvcache
            ],
        ),
    ]


@GQAVarlenFwdFixture
def test_gqa_varlen_fwd_op(
    batch: int,
    seqlens_q: list[int],
    seqlens_k: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
) -> None:
    test = GQAVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    op = GQAVarlenFwdOp(
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
    )
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_varlen_int32_max_right_window_is_no_window() -> None:
    """A right window of INT32_MAX keys hides none; added to a query position it overflows
    int32, so the output must equal the unwindowed reference."""
    test = GQAVarlenFwdTest(2, [1024, 512], [1024, 600], 8, 2, 128, False, -1, -1, torch.float16)
    test.check(GQAVarlenFwdOp(window_size_right=2**31 - 1), *test.gen_inputs())


@pytest.mark.smoke
def test_varlen_reuses_one_op_across_dynamic_packed_totals() -> None:
    op = GQAVarlenFwdOp(is_causal=True)
    for q_lens, kv_lens in (([31, 65], [63, 129]), ([127, 3], [255, 7])):
        test = GQAVarlenFwdTest(2, q_lens, kv_lens, 8, 2, 64, True, -1, -1, torch.float16)
        test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
def test_varlen_tuned_window_large_batch() -> None:
    """Hopper window attention must retain valid packed inputs when tuning large batches."""
    batch = 449
    test = GQAVarlenFwdTest(batch, [2] * batch, [4] * batch, 2, 1, 64, True, 64, -1, torch.float16)
    op = GQAVarlenFwdOp(is_causal=True, window_size_left=64)
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "scale,cap",
    [
        pytest.param(0.125, 5.0, marks=pytest.mark.smoke, id="positive-softcap"),
    ],
)
def test_varlen_regular_forwards_scale_and_softcap(scale, cap) -> None:
    """A scale and a softcap preserve causal masking."""
    test = GQAVarlenFwdTest(
        2,
        [65, 127],
        [129, 255],
        8,
        2,
        64,
        True,
        -1,
        -1,
        torch.float16,
        sm_scale=scale,
        softcap=cap,
    )
    op = GQAVarlenFwdOp(
        is_causal=True,
        sm_scale=scale,
        softcap=cap,
    )
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "q_lens, kv_lens",
    [([0, 65], [0, 129]), ([4, 4], [0, 8])],
)
def test_varlen_handles_empty_requests_and_per_request_kv(
    q_lens: list[int], kv_lens: list[int]
) -> None:
    test = GQAVarlenFwdTest(2, q_lens, kv_lens, 8, 2, 64, True, -1, -1, torch.float16)
    op = GQAVarlenFwdOp(is_causal=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize(
    "q_lens, kv_lens, is_causal, options, key",
    [
        pytest.param(
            [129, 65, 6, 0, 300],
            [33, 131, 2, 0, 400],
            True,
            {},
            "gqa_varlen_ws",
            marks=pytest.mark.smoke,
            id="causal-ragged",
        ),
        pytest.param(
            [65, 129, 128],
            [129, 65, 0],
            False,
            {"sm_scale": 0.125, "softcap": 5.0},
            "gqa_varlen_ws",
            marks=pytest.mark.full,
            id="bidirectional-softcap-empty-kv",
        ),
        # No KV at all: TMA needs an extent, so the general kernel serves.
        pytest.param(
            [4, 128], [0, 0], True, {}, "gqa_varlen", marks=pytest.mark.full, id="all-kv-empty"
        ),
        # More requests than the warp-specialized kernel's shared prefix holds.
        pytest.param(
            [1] * 449, [1] * 449, True, {}, "gqa_varlen", marks=pytest.mark.full, id="batch-449"
        ),
        pytest.param(
            ([0, 1, 17, 130] * 16)[:64],
            ([0, 200, 17, 130] * 16)[:64],
            True,
            {},
            "gqa_varlen_ws",
            marks=pytest.mark.full,
            id="batch-64",
        ),
        pytest.param(
            ([0, 1, 17, 130] * 17)[:65],
            ([0, 200, 17, 130] * 17)[:65],
            True,
            {},
            "gqa_varlen_ws",
            marks=pytest.mark.full,
            id="batch-65",
        ),
        # A rotated width of 10 rotates an odd number of pairs per half.
        pytest.param(
            [256, 130],
            [300, 130],
            True,
            {"pos_encoding_mode": "rope", "rotary_dim": 10},
            "gqa_varlen_ws",
            marks=pytest.mark.full,
            id="rope-odd-half",
        ),
    ],
)
def test_varlen_ws_dims_serve_ragged_requests_on_sm90(
    q_lens: list[int], kv_lens: list[int], is_causal: bool, options: dict, key: str, dim: int
) -> None:
    """Partial tiles, q_len > kv_len, and empty requests on the one-head warp-specialized
    kernel: a group of 3 does not divide a warpgroup's rows, so no packed program serves."""
    test = GQAVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, 6, 2, dim, is_causal, -1, -1, torch.float16, **options
    )
    op = GQAVarlenFwdOp(is_causal=is_causal, **options)
    inputs = test.gen_inputs()
    assert op.key_for("gqa_varlen", op._varlen_call(_padded(inputs))) == key
    test.check(op, *inputs)


@pytest.mark.smoke
@pytest.mark.sm90
def test_varlen_ws_kernel_claims_work_across_calls() -> None:
    """More work items than SMs; a later call must see the counter the first one reset."""
    test = GQAVarlenFwdTest(1, [1152], [1152], 16, 8, 128, True, -1, -1, torch.bfloat16)
    op = GQAVarlenFwdOp(is_causal=True)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    first = op(*inputs)
    assert torch.equal(op(*inputs), first)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "q_lens, kv_lens, heads, heads_kv, dim, is_causal, wl, wr, dtype, options",
    [
        # One KV head a query head and no mask, as vision encoders attend: partial query and
        # key tiles, q_len > kv_len, and an empty request.
        pytest.param(
            [129, 65, 6, 0, 300],
            [33, 131, 2, 0, 400],
            4,
            4,
            72,
            False,
            -1,
            -1,
            torch.bfloat16,
            {},
            marks=pytest.mark.smoke,
            id="d72-bidirectional",
        ),
        # A group of 2 divides a warpgroup's rows: the packed program, whose tiles hold whole
        # 64-channel boxes, must leave the call to the one-head program.
        pytest.param(
            [129, 65, 6, 0, 300],
            [33, 131, 2, 0, 400],
            4,
            2,
            72,
            True,
            -1,
            -1,
            torch.float16,
            {},
            marks=pytest.mark.smoke,
            id="d72-causal",
        ),
        pytest.param(
            [130, 70],
            [200, 70],
            6,
            2,
            72,
            False,
            64,
            32,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="d72-window",
        ),
        # Under 64 channels: one 48-channel tile.
        pytest.param(
            [129, 65],
            [200, 65],
            6,
            2,
            40,
            False,
            -1,
            -1,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="d40",
        ),
        # Fused RoPE on a head padded to one 48-channel tile.
        pytest.param(
            [129, 65],
            [200, 65],
            6,
            2,
            40,
            True,
            -1,
            -1,
            torch.float16,
            {"pos_encoding_mode": "rope"},
            marks=pytest.mark.full,
            id="d40-rope",
        ),
        # A 32-channel edge (24 channels and 8 zeros) beside the 64-channel boxes.
        pytest.param(
            [129, 65],
            [200, 65],
            6,
            2,
            88,
            True,
            -1,
            -1,
            torch.float16,
            {},
            marks=pytest.mark.full,
            id="d88",
        ),
        # A multiple of 16 that 64 does not divide: a 32-channel edge with no padding.
        pytest.param(
            [129, 65],
            [200, 65],
            6,
            2,
            96,
            True,
            -1,
            -1,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="d96",
        ),
        # A 64-channel edge (56 channels and 8 zeros) under a 128-byte swizzle.
        pytest.param(
            [129, 65],
            [200, 65],
            6,
            2,
            120,
            False,
            -1,
            -1,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="d120",
        ),
    ],
)
def test_varlen_ws_serves_head_dims_off_the_contraction_step_on_sm90(
    q_lens: list[int],
    kv_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
    options: dict,
) -> None:
    """A TMA row is a multiple of 16 bytes, a 16-bit WGMMA contracts 16 channels a step, and
    TMA fills a box past a tensor's last channel with zeros: on SM90 the warp-specialized
    kernel serves head dims of 8-channel steps on tiles padded to the WGMMA step."""
    test = GQAVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, heads, heads_kv, dim, is_causal, wl, wr, dtype, **options
    )
    op = GQAVarlenFwdOp(is_causal=is_causal, window_size_left=wl, window_size_right=wr, **options)
    inputs = test.gen_inputs()
    assert op.key_for("gqa_varlen", op._varlen_call(_padded(inputs))) == "gqa_varlen_ws"
    test.check(op, *inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "q_lens, kv_lens, heads, heads_kv, dim, dtype, options",
    [
        # Requests that fill whole 192-row tiles and partial ones, q_len > kv_len, and an empty
        # request, on a 64-channel block and a 16-channel edge (8 channels and 8 zeros).
        pytest.param(
            [700, 65, 0, 450],
            [500, 900, 0, 450],
            4,
            4,
            72,
            torch.bfloat16,
            {},
            marks=pytest.mark.smoke,
            id="d72-bidirectional",
        ),
        # The causal ranking claims items of 192 rows; a group of 2 still leaves the call to
        # the one-head program, as its head has an edge.
        pytest.param(
            [700, 300],
            [900, 300],
            4,
            2,
            72,
            torch.float16,
            {"is_causal": True},
            marks=pytest.mark.smoke,
            id="d72-causal",
        ),
        # Windows long enough that a row's maximum rises after its first key tiles, which the
        # running partial sums must take as a correction.
        pytest.param(
            [600, 330],
            [700, 330],
            4,
            4,
            72,
            torch.bfloat16,
            {"window_size_left": 100, "window_size_right": 50},
            marks=pytest.mark.full,
            id="d72-window",
        ),
        pytest.param(
            [600, 330],
            [700, 330],
            4,
            4,
            72,
            torch.bfloat16,
            {"has_sinks": True},
            marks=pytest.mark.full,
            id="d72-sinks",
        ),
        # A whole 64-channel block, no edge; RoPE rotates each warpgroup's 64 rows.
        pytest.param(
            [700, 65, 450],
            [500, 900, 450],
            4,
            4,
            64,
            torch.float16,
            {"pos_encoding_mode": "rope"},
            marks=pytest.mark.full,
            id="d64-rope",
        ),
        # One part padded to 64 channels (56 and 8 zeros).
        pytest.param(
            [700, 450],
            [500, 450],
            4,
            4,
            56,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="d56",
        ),
        # An edge of 16 channels with no padding.
        pytest.param(
            [700, 450],
            [500, 450],
            4,
            4,
            80,
            torch.bfloat16,
            {"is_causal": True},
            marks=pytest.mark.full,
            id="d80-causal",
        ),
    ],
)
def test_varlen_ws_runs_three_consumer_warpgroups_on_long_requests_on_sm90(
    q_lens: list[int],
    kv_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    dtype: torch.dtype,
    options: dict,
) -> None:
    """SM90's register file leaves three consumer warpgroups too few registers for P beside the
    scores and the output, so with three consumer warpgroups each reads P from shared memory
    through an SS WGMMA."""
    options = dict(options)
    is_causal = options.pop("is_causal", False)
    wl = options.pop("window_size_left", -1)
    wr = options.pop("window_size_right", -1)
    has_sinks = options.pop("has_sinks", False)
    test = GQAVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, heads, heads_kv, dim, is_causal, wl, wr, dtype,
        has_sinks=has_sinks, **options,
    )  # fmt: skip
    op = GQAVarlenFwdOp(is_causal=is_causal, window_size_left=wl, window_size_right=wr, **options)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    assert _launched_rows(op, inputs) == 192


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "q_lens, kv_lens, dim, rows",
    [
        # Short requests keep 128-row tiles, which pad them less: the d72 bf16 smoke shape.
        pytest.param(
            [129, 65, 6, 0, 300],
            [33, 131, 2, 0, 400],
            72,
            128,
            marks=pytest.mark.smoke,
            id="d72-narrow",
        ),
    ],
)
def test_varlen_ws_keeps_two_consumers_for_short_requests_on_sm90(
    q_lens: list[int], kv_lens: list[int], dim: int, rows: int
) -> None:
    """Tiles of three 64-row consumer warpgroups pad short requests more than tiles of two, so a
    call of short requests keeps two."""
    test = GQAVarlenFwdTest(len(q_lens), q_lens, kv_lens, 4, 4, dim, False, -1, -1, torch.bfloat16)
    op = GQAVarlenFwdOp()
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    assert _launched_rows(op, inputs) == rows


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "q_lens, kv_lens, options, reason",
    [
        # Fused RoPE rotates one query tile, and head dim 72 spans a box and an edge.
        pytest.param(
            [4, 128],
            [4, 128],
            {"pos_encoding_mode": "rope", "rotary_dim": 72},
            "fuses RoPE only into a head held in one tile",
            marks=pytest.mark.smoke,
            id="rope",
        ),
        # No key in the call: TMA has no extent to load.
        pytest.param(
            [4, 128],
            [0, 0],
            {},
            "does not serve an empty KV sequence",
            marks=pytest.mark.full,
            id="no-keys",
        ),
        # More requests than the warp-specialized kernel's shared prefix holds.
        pytest.param(
            [1] * 449,
            [1] * 449,
            {},
            "serves at most 448 sequences",
            marks=pytest.mark.full,
            id="batch-449",
        ),
    ],
)
def test_varlen_refuses_head_dim_72_where_the_ws_kernel_cannot_before_building(
    q_lens: list[int], kv_lens: list[int], options: dict, reason: str
) -> None:
    """The general kernel contracts the head dim 16 channels at a time, so a head dim-72 call
    the warp-specialized kernel refuses has no kernel."""
    test = GQAVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, 4, 4, 72, True, -1, -1, torch.bfloat16, **options
    )
    op = GQAVarlenFwdOp(is_causal=True, **options)
    with pytest.raises(OpNotAvailableError, match=reason):
        op(*test.gen_inputs())
    assert not op._entries_built("gqa_varlen")


@pytest.mark.parametrize(
    "q_lens, kv_lens, dim, rotary_dim, rope_layout, window_size_left",
    [
        pytest.param(
            [256, 130],
            [300, 130],
            128,
            10,
            "neox",
            -1,
            marks=pytest.mark.smoke,
            id="odd-half-packed",
        ),
        pytest.param(
            [256, 130],
            [300, 130],
            256,
            14,
            "neox",
            -1,
            marks=pytest.mark.full,
            id="odd-half-general",
        ),
        # Head dim 256 reaches the general kernel on SM90 too.
        pytest.param(
            [256, 130],
            [300, 130],
            256,
            10,
            "neox",
            -1,
            marks=pytest.mark.full,
            id="odd-half-general-split",
        ),
        pytest.param(
            [256, 128], [256, 128], 128, None, "neox", -1, marks=pytest.mark.full, id="neox-full"
        ),
        pytest.param(
            [256, 128],
            [256, 128],
            64,
            None,
            "interleaved",
            -1,
            marks=pytest.mark.full,
            id="interleaved-full",
        ),
        # The channels past a partial rotary width pass through unrotated.
        pytest.param(
            [1, 0, 130],
            [200, 0, 130],
            128,
            64,
            "neox",
            -1,
            marks=pytest.mark.full,
            id="ragged-partial",
        ),
        pytest.param(
            [130, 70],
            [130, 70],
            128,
            None,
            "interleaved",
            64,
            marks=pytest.mark.full,
            id="windowed",
        ),
        pytest.param(
            [70, 40], [200, 40], 512, None, "neox", -1, marks=pytest.mark.full, id="dim-512"
        ),
        pytest.param(
            [1, 0, 130],
            [200, 0, 130],
            128,
            12,
            "neox",
            -1,
            marks=pytest.mark.full,
            id="narrow-neox",
        ),
        pytest.param(
            [130, 70],
            [130, 70],
            64,
            14,
            "interleaved",
            -1,
            marks=pytest.mark.full,
            id="narrow-interleaved",
        ),
    ],
)
def test_varlen_rope_rotates_at_per_request_positions(
    q_lens: list[int],
    kv_lens: list[int],
    dim: int,
    rotary_dim: int | None,
    rope_layout: str,
    window_size_left: int,
) -> None:
    """Query token i of a request sits at kv_len - q_len + i, key token j at j."""
    test = GQAVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, 8, 2, dim, True, window_size_left, -1, torch.float16,
        pos_encoding_mode="rope", rotary_dim=rotary_dim, rope_layout=rope_layout,
    )  # fmt: skip
    op = GQAVarlenFwdOp(
        is_causal=True,
        window_size_left=window_size_left,
        pos_encoding_mode="rope",
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    test.check(op, *test.gen_inputs())


# ----------------------------------------------------------------------
# Visible-score accounting for the packed-varlen and sliding-window GQA rooflines.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q_len", "kv_len", "is_causal", "left", "visible"),
    [
        # A square causal request sees the lower triangle.
        (8, 8, True, -1, 36),
        # Without a causal mask every query sees every key.
        (8, 8, False, -1, 64),
        # Bottom-right aligned: query i of 2 against 6 keys sees keys 0..i+4.
        (2, 6, True, -1, 11),
        # The first four of 6 queries against 2 keys see nothing, never a negative count.
        (6, 2, True, -1, 3),
        # A left window of 3 caps each row at 4 keys.
        (2, 6, True, 3, 8),
        (8, 8, True, 3, 26),
    ],
)
@pytest.mark.smoke
def test_visible_scores_follow_alignment_mask_and_window(
    q_len: int, kv_len: int, is_causal: bool, left: int, visible: int
) -> None:
    assert visible_scores(q_len, kv_len, is_causal, left, -1) == visible


@pytest.mark.parametrize(
    "q_lens, kv_lens, is_causal, wl, wr",
    [
        pytest.param(
            [300, 700],
            [300, 700],
            True,
            100,
            -1,
            marks=pytest.mark.smoke,
            id="causal-left-partial-tiles",
        ),
        pytest.param([256, 512], [256, 512], False, 64, 64, marks=pytest.mark.full, id="both"),
        pytest.param(
            [64, 128], [256, 512], False, -1, 64, marks=pytest.mark.full, id="right-kvcache"
        ),
    ],
)
def test_varlen_sliding_window_with_fused_rope(
    q_lens: list[int], kv_lens: list[int], is_causal: bool, wl: int, wr: int
) -> None:
    """Window bounds hold when the call also rotates Q and K."""
    test = GQAVarlenFwdTest(
        len(q_lens),
        q_lens,
        kv_lens,
        8,
        2,
        64,
        is_causal,
        wl,
        wr,
        torch.float16,
        pos_encoding_mode="rope",
    )
    op = GQAVarlenFwdOp(is_causal, wl, wr, pos_encoding_mode="rope")
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    "q_lens, keys, heads, heads_kv, key",
    [
        # A group dividing 64 rows runs the packed program.
        pytest.param(
            [1] * 63 + [1024], 4096, 32, 8, "gqa_varlen_packed_ws", id="decodes-and-chunk"
        ),
        # A group of 3 does not divide a warpgroup's rows; one head a tile.
        pytest.param([1] * 8, 4096, 24, 8, "gqa_varlen_ws", id="group-3"),
    ],
)
def test_varlen_selects_the_packed_program_for_groups_dividing_a_warpgroup(
    q_lens: list[int], keys: int, heads: int, heads_kv: int, key: str
) -> None:
    """On SM90 a call whose group divides a warpgroup's 64 rows runs the program that packs a
    KV head's group into its tiles; other calls keep one head a tile."""
    q = torch.empty(sum(q_lens), heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(keys, heads_kv, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.zeros(len(q_lens) + 1, device="cuda", dtype=torch.int32)
    op = GQAVarlenFwdOp(is_causal=True)
    call = op._varlen_call((q, k, k, cu, cu, *([None] * 6)))
    assert op.key_for("gqa_varlen", call) == key


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_varlen_one_op_selects_packed_programs_by_call() -> None:
    """One op moves between the packed program's variants as its calls change: decodes may
    split by chunk and run as short requests; long requests do neither."""
    op = GQAVarlenFwdOp(is_causal=True)
    calls = (([1, 1], [3000, 2000], (True, False)), ([512, 512], [1024, 2048], (False, True)))
    for q_lens, kv_lens, facts in (*calls, calls[0]):
        test = GQAVarlenFwdTest(2, q_lens, kv_lens, 64, 8, 64, True, -1, -1, torch.bfloat16)
        inputs = test.gen_inputs()
        test.check(op, *inputs)
        kernel = op._get_kernel(_padded(inputs))
        kernel = getattr(kernel, "__self__", kernel)
        assert kernel._program_facts(inputs[0]) == facts
    assert {key[1:] for key in kernel._generation.programs} == {facts for *_, facts in calls}


@pytest.mark.sm90
@pytest.mark.parametrize(
    "q_lens, kv_lens, heads, heads_kv, dim, dtype, options",
    [
        # Narrow, wide packed (17 positions of 8 heads) and one-head tiles side by side, with
        # rows that see no key.
        pytest.param(
            [17, 1, 4, 300, 6],
            [600, 900, 130, 300, 2],
            16,
            2,
            128,
            torch.float16,
            {},
            marks=pytest.mark.smoke,
            id="mixed-group-8",
        ),
        # Fewer items than SMs: each request's keys split into parts that the last one merges,
        # on one warpgroup's rows or, at 24 positions of 4 heads, on both.
        pytest.param(
            [1, 3, 24],
            [5000, 9000, 9000],
            32,
            8,
            128,
            torch.bfloat16,
            {},
            marks=pytest.mark.smoke,
            id="split-keys",
        ),
        # Empty requests pad the batch; the one decode still splits its keys.
        pytest.param(
            [0] * 16 + [1],
            [0] * 16 + [9000],
            32,
            8,
            128,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="padded-split-keys",
        ),
        # Few requests, but their unsplit items fill the SMs: no request splits.
        pytest.param(
            [300] + [1] * 7,
            [300] + [2000] * 7,
            32,
            8,
            128,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="saturated-no-split",
        ),
        # More requests than the ranking serves: claims in request order.
        pytest.param(
            ([1, 3, 0, 40] * 33)[:130],
            ([500, 40, 7, 90] * 33)[:130],
            8,
            2,
            64,
            torch.float16,
            {},
            marks=pytest.mark.full,
            id="unranked-dim-64",
        ),
        # A head under one 64-channel box packs in one box of its own width.
        pytest.param(
            [1, 3, 24, 5],
            [900, 300, 24, 5],
            32,
            8,
            48,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="dim-48",
        ),
        pytest.param(
            [1, 2, 5],
            [300, 70, 5],
            16,
            1,
            128,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="group-16",
        ),
        pytest.param(
            [1, 2, 40, 5],
            [300, 70, 90, 5],
            4,
            2,
            64,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="group-2",
        ),
        pytest.param(
            [1, 4, 30],
            [700, 300, 30],
            16,
            2,
            128,
            torch.float16,
            {"window_size_left": 100},
            marks=pytest.mark.full,
            id="window",
        ),
        pytest.param(
            [1, 4, 30],
            [700, 300, 30],
            16,
            2,
            64,
            torch.float16,
            {"is_causal": False, "sm_scale": 0.125, "softcap": 5.0},
            marks=pytest.mark.full,
            id="bidirectional-softcap",
        ),
        # Bidirectional at 128 channels: the scan of warpgroup 1, idle on the decode, stops at
        # its one tile though the keys fill several.
        pytest.param(
            [1, 4],
            [512, 300],
            16,
            2,
            128,
            torch.bfloat16,
            {"is_causal": False},
            marks=pytest.mark.full,
            id="bidirectional-dim-128",
        ),
        pytest.param(
            [17, 1, 3],
            [300, 500, 40],
            16,
            2,
            128,
            torch.bfloat16,
            {"pos_encoding_mode": "rope", "rope_layout": "interleaved"},
            marks=pytest.mark.full,
            id="rope-wide",
        ),
        pytest.param(
            [1, 2, 4],
            [3000, 7000, 1500],
            16,
            2,
            64,
            torch.float16,
            {"pos_encoding_mode": "rope", "has_sinks": True},
            marks=pytest.mark.full,
            id="split-keys-rope-sinks",
        ),
        # A large score scale: rows that see no key stay at zero weight.
        pytest.param(
            [6, 1, 20],
            [2, 300, 20],
            16,
            4,
            128,
            torch.bfloat16,
            {"sm_scale": 4.0},
            marks=pytest.mark.full,
            id="empty-rows-large-scale",
        ),
        # A rotated width off the 16-byte runs takes the general rotation; the decode's
        # positions sit below the table's last row.
        pytest.param(
            [1, 2, 40],
            [100, 300, 50],
            8,
            2,
            128,
            torch.float16,
            {"pos_encoding_mode": "rope", "rotary_dim": 12},
            marks=pytest.mark.full,
            id="rope-narrow-odd-width",
        ),
        # More queries than keys: the rows before the first key rotate at a clamped position.
        pytest.param(
            [6, 1],
            [2, 300],
            16,
            4,
            128,
            torch.float16,
            {"pos_encoding_mode": "rope", "rotary_dim": 12},
            marks=pytest.mark.full,
            id="rope-odd-width-empty-rows",
        ),
    ],
)
def test_varlen_packs_short_requests_on_sm90(
    q_lens: list[int], kv_lens: list[int], heads: int, heads_kv: int, dim: int,
    dtype: torch.dtype, options: dict,
) -> None:  # fmt: skip
    """Calls of short requests: decodes, verifications and a prefill chunk among them."""
    options = options.copy()
    is_causal = options.pop("is_causal", True)
    wl = options.pop("window_size_left", -1)
    has_sinks = options.pop("has_sinks", False)
    test = GQAVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, heads, heads_kv, dim, is_causal, wl, -1, dtype,
        has_sinks=has_sinks, **options,
    )  # fmt: skip
    op = GQAVarlenFwdOp(is_causal=is_causal, window_size_left=wl, **options)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="GQAVarlenFwdOp")
@pytest.mark.parametrize(
    "q_lens, kv_lens",
    [
        pytest.param([1, 24, 3], [5000, 9000, 700], marks=pytest.mark.smoke, id="split-keys"),
        # Items fill the SMs beside fewer narrow items than SMs: the last wide claims scan
        # their keys in two parts; tail items under eight key tiles run whole.
        pytest.param(
            [1500, 130, 1, 1, 3],
            [2200, 130, 3000, 900, 40],
            marks=pytest.mark.full,
            id="tail-split",
        ),
    ],
)
def test_varlen_split_keys_merge_across_calls_streams_and_graphs(
    q_lens: list[int], kv_lens: list[int]
) -> None:
    """The last part of a split item leaves its merge counter at zero: a later call, a
    concurrent call on another stream and a replayed graph each merge their parts afresh."""
    test = GQAVarlenFwdTest(len(q_lens), q_lens, kv_lens, 32, 8, 128, True, -1, -1, torch.bfloat16)
    op = GQAVarlenFwdOp(is_causal=True)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    test.check(op, *inputs)
    side = torch.cuda.Stream()

    def two_streams(*args):
        side.wait_stream(torch.cuda.current_stream())
        first = op(*args)
        with torch.cuda.stream(side):
            second = op(*args)
        torch.cuda.current_stream().wait_stream(side)
        torch.testing.assert_close(first, second, atol=1e-2, rtol=1e-2)
        return second

    test.check(op, *inputs, runs=two_streams)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    inputs[0].mul_(0.5)
    for _ in range(2):
        output.fill_(float("nan"))
        graph.replay()
        test.check(op, *inputs, runs=lambda *args: output)
