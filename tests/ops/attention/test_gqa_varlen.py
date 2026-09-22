"""Varlen GQA tests against a pure-PyTorch reference."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.manifest import load_workloads
from tileops.ops import (
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.perf.formulas import (
    gqa_prefill_varlen_fwd_roofline,
    gqa_sliding_window_varlen_fwd_roofline,
    gqa_varlen_fwd_roofline,
)
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
        2,
        8,
        2,
        64,
        127,
        is_causal=True,
        window_size_left=64,
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


@pytest.mark.parametrize("parameter", [{"sm_scale": 0.125}, {"softcap": 5.0}])
def test_varlen_rejects_unimplemented_window_score_combinations(parameter: dict) -> None:
    with pytest.raises(ValueError, match="windowed Varlen GQA does not yet support"):
        GroupedQueryAttentionVarlenFwdOp(window_size_left=32, **parameter)


# ----------------------------------------------------------------------
# Visible-score accounting for the packed-varlen and sliding-window GQA rooflines.
# ----------------------------------------------------------------------


BATCH, SEQ, HEADS, HEADS_KV, DIM = 4, 8, 32, 8, 128
ELEM_BYTES = 2


def _varlen_kwargs(**overrides: object) -> dict:
    """Payload shaped like ``GroupedQueryAttentionVarlenFwdOp.eval_roofline``."""
    kwargs = {
        "q_shape": (BATCH * SEQ, HEADS, DIM),
        "k_shape": (BATCH * SEQ, HEADS_KV, DIM),
        "batch": BATCH,
        "total_q": BATCH * SEQ,
        "total_k": BATCH * SEQ,
        "heads": HEADS,
        "heads_kv": HEADS_KV,
        "dim": DIM,
        "max_seqlen_q": SEQ,
        "max_seqlen_kv": SEQ,
        "q_lens": [SEQ] * BATCH,
        "kv_lens": [SEQ] * BATCH,
        "is_causal": True,
        "dtype": torch.float16,
    }
    kwargs.update(overrides)
    return kwargs


def _varlen_flops(visible: int) -> int:
    return 4 * HEADS * visible * DIM


@pytest.mark.smoke
def test_varlen_causal_counts_lower_triangle_per_request() -> None:
    """Four square requests of length 8 see 36 scores each."""
    flops, _ = gqa_prefill_varlen_fwd_roofline(**_varlen_kwargs())

    assert flops == _varlen_flops(4 * 36)


@pytest.mark.smoke
def test_varlen_non_causal_counts_full_product() -> None:
    """Without a causal mask every query sees every key."""
    flops, _ = gqa_prefill_varlen_fwd_roofline(**_varlen_kwargs(is_causal=False))

    assert flops == _varlen_flops(4 * 8 * 8)


@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "visible"),
    [
        # Short query run against a longer key run: bottom-right aligned, so
        # query i sees keys 0..i+4: 5 + 6 = 11 scores.
        ([2], [6], 11),
        # Long query run against a short key run: the first four queries see
        # nothing, the last two see 1 and 2 keys: 3 scores.
        ([6], [2], 3),
        # Mixed batch: 1 + 11 + 3 = 15.
        ([1, 2, 6], [1, 6, 2], 15),
    ],
)
@pytest.mark.smoke
def test_varlen_causal_handles_asymmetric_request_lengths(
    q_lens: list[int], kv_lens: list[int], visible: int
) -> None:
    """Queries past the end of the key run contribute no scores, never negative ones."""
    payload = _varlen_kwargs(
        q_shape=(sum(q_lens), HEADS, DIM),
        k_shape=(sum(kv_lens), HEADS_KV, DIM),
        batch=len(q_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        q_lens=q_lens,
        kv_lens=kv_lens,
    )

    flops, _ = gqa_prefill_varlen_fwd_roofline(**payload)

    assert flops == _varlen_flops(visible)


@pytest.mark.smoke
def test_varlen_bytes_count_q_kv_and_cu_seqlens() -> None:
    """Byte traffic is Q + 2*KV + O plus the two cu_seqlens vectors."""
    _, nbytes = gqa_prefill_varlen_fwd_roofline(**_varlen_kwargs())

    q_elems = BATCH * SEQ * HEADS * DIM
    kv_elems = BATCH * SEQ * HEADS_KV * DIM
    expected = (2 * q_elems + 2 * kv_elems) * ELEM_BYTES + 2 * (BATCH + 1) * 4
    assert nbytes == expected


@pytest.mark.smoke
def test_varlen_derives_lengths_from_cu_seqlens() -> None:
    """``cu_seqlens_*`` tensors are differenced into per-request lengths."""
    cu = torch.tensor([0, 1, 3, 9], dtype=torch.int32)
    payload = _varlen_kwargs(
        q_shape=(9, HEADS, DIM),
        k_shape=(9, HEADS_KV, DIM),
        batch=3,
        max_seqlen_q=6,
        max_seqlen_kv=6,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
    )
    payload.pop("q_lens")
    payload.pop("kv_lens")

    flops, _ = gqa_prefill_varlen_fwd_roofline(**payload)

    # Lengths [1, 2, 6] against themselves: 1 + 3 + 21 = 25.
    assert flops == _varlen_flops(25)


@pytest.mark.smoke
def test_varlen_fills_requests_to_max_len_when_lengths_absent() -> None:
    """Without lengths, ``_distribute_total`` fills early requests to max_seqlen."""
    payload = _varlen_kwargs(
        q_shape=(10, HEADS, DIM),
        k_shape=(10, HEADS_KV, DIM),
        batch=4,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    payload.pop("q_lens")
    payload.pop("kv_lens")

    flops, _ = gqa_prefill_varlen_fwd_roofline(**payload)

    # Fill is [4, 4, 1, 1], not an even [3, 3, 2, 2]: 10 + 10 + 1 + 1 = 22.
    assert flops == _varlen_flops(22)


@pytest.mark.parametrize("window_size_left", [-1, 3])
@pytest.mark.smoke
def test_varlen_roofline_accepts_manifest_op_call(window_size_left: int) -> None:
    """The manifest calls its roofline function with the bound Op instance."""
    payload = _varlen_kwargs(window_size_left=window_size_left, window_size_right=-1)
    op = GroupedQueryAttentionVarlenFwdOp.__new__(GroupedQueryAttentionVarlenFwdOp)
    op._roofline_kwargs = payload
    expected = (
        gqa_sliding_window_varlen_fwd_roofline(**payload)
        if window_size_left != -1
        else gqa_prefill_varlen_fwd_roofline(**payload)
    )

    assert gqa_varlen_fwd_roofline(op) == expected


@pytest.mark.smoke
def test_sliding_window_varlen_offsets_short_queries_to_sequence_end() -> None:
    """A 2-query, 6-key request aligns bottom-right: rows [4, 4], 8 scores."""
    flops, _ = gqa_sliding_window_varlen_fwd_roofline(
        batch=1,
        heads=HEADS,
        heads_kv=HEADS_KV,
        dim=DIM,
        q_lens=[2],
        k_lens=[6],
        is_causal=True,
        window_size_left=3,
        dtypes=["float16"],
    )

    assert flops == 4 * HEADS * 8 * DIM


@pytest.mark.smoke
def test_sliding_window_varlen_windows_each_request_separately() -> None:
    """Per-request windows do not leak across the packed batch: 26 + 8 = 34 scores."""
    flops, _ = gqa_sliding_window_varlen_fwd_roofline(
        batch=2,
        heads=HEADS,
        heads_kv=HEADS_KV,
        dim=DIM,
        q_lens=[8, 2],
        k_lens=[8, 6],
        is_causal=True,
        window_size_left=3,
        dtypes=["float16"],
    )

    assert flops == 4 * HEADS * 34 * DIM


@pytest.mark.smoke
def test_sliding_window_varlen_counts_cu_seqlens_bytes() -> None:
    """Sliding-window traffic includes both cumulative-length arrays."""
    _, nbytes = gqa_sliding_window_varlen_fwd_roofline(
        batch=BATCH,
        heads=HEADS,
        heads_kv=HEADS_KV,
        dim=DIM,
        q_lens=[SEQ] * BATCH,
        kv_lens=[SEQ] * BATCH,
        is_causal=True,
        window_size_left=3,
        dtype=torch.float16,
    )

    q_elems = BATCH * SEQ * HEADS * DIM
    kv_elems = BATCH * SEQ * HEADS_KV * DIM
    expected = (2 * q_elems + 2 * kv_elems) * ELEM_BYTES + 2 * (BATCH + 1) * 4
    assert nbytes == expected


@pytest.mark.smoke
def test_varlen_manifest_workloads_are_evaluable() -> None:
    """Every declared workload binds to its formula without a missing key."""
    for workload in load_workloads("GroupedQueryAttentionVarlenFwdOp"):
        flops, nbytes = gqa_varlen_fwd_roofline(**workload)
        assert flops > 0, workload["label"]
        assert nbytes > 0, workload["label"]
