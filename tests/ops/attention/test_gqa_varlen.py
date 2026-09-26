"""Varlen GQA tests against a pure-PyTorch reference."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, served_in_tree
from tileops.ops import (
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.perf.formulas import visible_scores
from workloads.attention.gqa import (
    GroupedQueryAttentionVarlenFwdWorkload,
)


class GroupedQueryAttentionVarlenFwdTest(GroupedQueryAttentionVarlenFwdWorkload, TestBase):
    pass


class GroupedQueryAttentionVarlenFwdFixture(FixtureBase):
    # Parameters: (batch, seqlens_q, seqlens_k, heads, heads_kv, dim,
    #              is_causal, wl, wr, dtype, tune)
    PARAMS = [
        (
            "batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype, tune",
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
                    False,
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
                    False,
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
                    False,
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
                    False,
                    marks=pytest.mark.smoke,
                ),  # D=128 uses the two-stage sliding pipeline
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
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
                    False,
                    marks=pytest.mark.full,
                ),  # wl=0
            ],
        ),
    ]


@GroupedQueryAttentionVarlenFwdFixture
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
    tune: bool,
) -> None:
    test = GroupedQueryAttentionVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    op = GroupedQueryAttentionVarlenFwdOp(
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
    )
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_legacy_varlen_ops_remain_implemented_during_migration() -> None:
    regular = GroupedQueryAttentionVarlenFwdTest(
        2, [65, 127], [129, 255], 8, 2, 64, True, -1, -1, torch.float16
    )
    regular_op = GroupedQueryAttentionPrefillVarlenFwdOp(127, 255, is_causal=True)
    regular.check(regular_op, *regular.gen_inputs(), atol=1e-3, rtol=1e-3)

    windowed = GroupedQueryAttentionVarlenFwdTest(
        2, [65, 127], [129, 255], 8, 2, 64, True, 64, -1, torch.float16
    )
    windowed_op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
        127, is_causal=True, window_size_left=64
    )
    windowed.check(windowed_op, *windowed.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_varlen_reuses_one_op_across_dynamic_packed_totals() -> None:
    op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)
    for q_lens, kv_lens in (([31, 65], [63, 129]), ([127, 3], [255, 7])):
        test = GroupedQueryAttentionVarlenFwdTest(
            2, q_lens, kv_lens, 8, 2, 64, True, -1, -1, torch.float16
        )
        test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_varlen_regular_forwards_scale_and_softcap() -> None:
    test = GroupedQueryAttentionVarlenFwdTest(
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
        sm_scale=0.125,
        softcap=5.0,
    )
    op = GroupedQueryAttentionVarlenFwdOp(
        is_causal=True,
        sm_scale=0.125,
        softcap=5.0,
    )
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "q_lens, kv_lens",
    [([0, 65], [0, 129]), ([4, 4], [0, 8])],
)
def test_varlen_handles_empty_requests_and_per_request_kv(
    q_lens: list[int], kv_lens: list[int]
) -> None:
    test = GroupedQueryAttentionVarlenFwdTest(
        2, q_lens, kv_lens, 8, 2, 64, True, -1, -1, torch.float16
    )
    op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.parametrize(
    "q_lens, kv_lens, is_causal, scores, kernel",
    [
        pytest.param(
            [129, 65, 6, 0, 300],
            [33, 131, 2, 0, 400],
            True,
            {},
            "GQAPrefillVarlenWSFwdKernel",
            id="causal-ragged",
        ),
        pytest.param(
            [65, 129, 128],
            [129, 65, 0],
            False,
            {"sm_scale": 0.125, "softcap": 5.0},
            "GQAPrefillVarlenWSFwdKernel",
            id="bidirectional-softcap-empty-kv",
        ),
        # No KV at all: TMA needs an extent, so the general kernel serves.
        pytest.param([4, 128], [0, 0], True, {}, "GQAPrefillVarlenFwdKernel", id="all-kv-empty"),
        # More requests than the warp-specialized kernel's shared prefix holds.
        pytest.param([1] * 449, [1] * 449, True, {}, "GQAPrefillVarlenFwdKernel", id="batch-449"),
    ],
)
def test_varlen_dim128_serves_ragged_requests_on_sm90(
    q_lens: list[int], kv_lens: list[int], is_causal: bool, scores: dict, kernel: str
) -> None:
    """Partial tiles, q_len > kv_len, and empty requests on the warp-specialized kernel."""
    test = GroupedQueryAttentionVarlenFwdTest(
        len(q_lens), q_lens, kv_lens, 8, 2, 128, is_causal, -1, -1, torch.float16, **scores
    )
    op = GroupedQueryAttentionVarlenFwdOp(is_causal=is_causal, **scores)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-3, rtol=1e-3)
    if served_in_tree(op):
        assert type(op._get_kernel((*inputs, None, None, None, None, None))).__name__ == kernel


@pytest.mark.smoke
@pytest.mark.sm90
def test_varlen_ws_kernel_claims_work_across_calls() -> None:
    """More work items than SMs; a later call must see the counter the first one reset."""
    test = GroupedQueryAttentionVarlenFwdTest(
        1, [1152], [1152], 16, 8, 128, True, -1, -1, torch.bfloat16
    )
    op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-2, rtol=1e-2)
    first = op(*inputs)
    assert torch.equal(op(*inputs), first)


@pytest.mark.smoke
def test_varlen_rejects_invalid_cumulative_lengths_contract() -> None:
    test = GroupedQueryAttentionVarlenFwdTest(
        2, [8, 8], [16, 16], 8, 2, 64, True, -1, -1, torch.float16
    )
    q, k, v, cu_q, cu_kv = test.gen_inputs()
    op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)
    with pytest.raises(ValueError, match="int32"):
        op(q, k, v, cu_q.to(torch.int64), cu_kv)
    with pytest.raises(ValueError, match="same shape"):
        op(q, k, v, cu_q, cu_kv[:-1])

    checked = GroupedQueryAttentionVarlenFwdOp(is_causal=True, validate_inputs=True)
    with pytest.raises(ValueError, match=r"cu_seqlens_q\[-1\] must equal"):
        checked(q[:-1], k, v, cu_q, cu_kv)
    with pytest.raises(ValueError, match=r"cu_seqlens_kv\[-1\] must equal"):
        checked(q, k[:-1], v[:-1], cu_q, cu_kv)
    with pytest.raises(ValueError, match="cu_seqlens_q must be non-decreasing"):
        checked(q, k, v, torch.tensor([0, 17, 16], device=q.device, dtype=torch.int32), cu_kv)


@pytest.mark.smoke
def test_varlen_compatibility_validates_lengths_and_dtype() -> None:
    test = GroupedQueryAttentionVarlenFwdTest(
        2, [8, 8], [16, 16], 8, 2, 64, True, -1, -1, torch.float16
    )
    q, k, v, cu_q, cu_kv = test.gen_inputs()
    old = GroupedQueryAttentionPrefillVarlenFwdOp(7, 16, validate_inputs=True)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        old(q, k, v, cu_q, cu_kv)

    new = GroupedQueryAttentionVarlenFwdOp()
    with pytest.raises(ValueError, match="float16, bfloat16, or float8_e4m3fn"):
        new(q.float(), k.float(), v.float(), cu_q, cu_kv)


@pytest.mark.smoke
@pytest.mark.parametrize("parameter", [{"sm_scale": 0.125}, {"softcap": 5.0}])
def test_varlen_rejects_unimplemented_window_score_combinations(parameter: dict) -> None:
    with pytest.raises(ValueError, match="windowed Varlen GQA does not yet support"):
        GroupedQueryAttentionVarlenFwdOp(window_size_left=32, **parameter)


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
