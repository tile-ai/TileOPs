"""Visible-score accounting for the packed-varlen and sliding-window GQA rooflines.

These three formulas are only reachable through ``Op.eval_roofline()``, which
requires a prior ``forward()`` on CUDA. The tests here call them with the same
keyword payload ``_record_roofline`` builds, so the score-counting loops are
checked without a device.

Expected score totals are written as literals, counted by hand from the
attention mask, rather than recomputed from the production bounds arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tileops.manifest import load_workloads
from tileops.perf.formulas import (
    gqa_prefill_varlen_fwd_roofline,
    gqa_sliding_window_fwd_roofline,
    gqa_sliding_window_varlen_fwd_roofline,
)

pytestmark = pytest.mark.smoke

BATCH, SEQ, HEADS, HEADS_KV, DIM = 4, 8, 32, 8, 128
ELEM_BYTES = 2


def _varlen_kwargs(**overrides: object) -> dict:
    """Payload shaped like ``GroupedQueryAttentionPrefillVarlenFwdOp.eval_roofline``."""
    kwargs = {
        "q_shape": (BATCH * SEQ, HEADS, DIM),
        "k_shape": (BATCH * SEQ, HEADS_KV, DIM),
        "batch": BATCH,
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


def test_varlen_causal_counts_lower_triangle_per_request() -> None:
    """Four square requests of length 8 see 36 scores each."""
    flops, _ = gqa_prefill_varlen_fwd_roofline(**_varlen_kwargs())

    assert flops == _varlen_flops(4 * 36)


def test_varlen_non_causal_counts_full_product() -> None:
    """Without a causal mask every query sees every key."""
    flops, _ = gqa_prefill_varlen_fwd_roofline(**_varlen_kwargs(is_causal=False))

    assert flops == _varlen_flops(4 * 8 * 8)


@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "visible"),
    [
        # Short query run against a longer key run: bottom-right aligned, so
        # query i sees keys 0..i+4 — 5 + 6 = 11 scores.
        ([2], [6], 11),
        # Long query run against a short key run: the first four queries see
        # nothing, the last two see 1 and 2 keys — 3 scores.
        ([6], [2], 3),
        # Mixed batch: 1 + 11 + 3 = 15.
        ([1, 2, 6], [1, 6, 2], 15),
    ],
)
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


def test_varlen_bytes_count_q_kv_and_cu_seqlens() -> None:
    """Byte traffic is Q + 2*KV + O plus the two cu_seqlens vectors."""
    _, nbytes = gqa_prefill_varlen_fwd_roofline(**_varlen_kwargs())

    q_elems = BATCH * SEQ * HEADS * DIM
    kv_elems = BATCH * SEQ * HEADS_KV * DIM
    expected = (2 * q_elems + 2 * kv_elems) * ELEM_BYTES + 2 * (BATCH + 1) * 4
    assert nbytes == expected


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


def test_varlen_unwraps_roofline_kwargs_from_op() -> None:
    """A bound Op carries its payload under ``_roofline_kwargs``."""
    op = SimpleNamespace(_roofline_kwargs=_varlen_kwargs())

    flops, _ = gqa_prefill_varlen_fwd_roofline(op)

    assert flops == _varlen_flops(4 * 36)


def test_sliding_window_caps_attended_scores_at_window() -> None:
    """Causal rows over S=8 with a left window of 3 are [1,2,3,4,4,4,4,4] — 26 scores."""
    flops, nbytes = gqa_sliding_window_fwd_roofline(
        q_shape=(BATCH, SEQ, HEADS, DIM),
        kv_shape=(BATCH, SEQ, HEADS_KV, DIM),
        is_causal=True,
        window_size_left=3,
        dtypes=["float16"],
    )

    assert flops == 4 * BATCH * HEADS * 26 * DIM
    assert nbytes == 2 * BATCH * SEQ * (HEADS + HEADS_KV) * DIM * ELEM_BYTES


def test_sliding_window_unbounded_left_counts_lower_triangle() -> None:
    """``window_size_left=-1`` degrades to plain causal: 36 scores over S=8."""
    flops, _ = gqa_sliding_window_fwd_roofline(
        q_shape=(BATCH, SEQ, HEADS, DIM),
        kv_shape=(BATCH, SEQ, HEADS_KV, DIM),
        is_causal=True,
        window_size_left=-1,
        dtypes=["float16"],
    )

    assert flops == 4 * BATCH * HEADS * 36 * DIM


def test_sliding_window_non_causal_uses_right_window() -> None:
    """Bidirectional rows with windows (2, 1) are [2,3,4,4,4,4,4,3] — 28 scores."""
    flops, _ = gqa_sliding_window_fwd_roofline(
        q_shape=(BATCH, SEQ, HEADS, DIM),
        kv_shape=(BATCH, SEQ, HEADS_KV, DIM),
        is_causal=False,
        window_size_left=2,
        window_size_right=1,
        dtypes=["float16"],
    )

    assert flops == 4 * BATCH * HEADS * 28 * DIM


def test_sliding_window_varlen_offsets_short_queries_to_sequence_end() -> None:
    """A 2-query, 6-key request aligns bottom-right: rows [4, 4] — 8 scores."""
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


@pytest.mark.parametrize(
    "op_name",
    [
        "GroupedQueryAttentionSlidingWindowFwdOp",
        "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
    ],
)
def test_sliding_window_manifest_workloads_are_evaluable(op_name: str) -> None:
    """Every declared workload binds to its formula without a missing key."""
    formula = (
        gqa_sliding_window_fwd_roofline
        if op_name.endswith("SlidingWindowFwdOp")
        else gqa_sliding_window_varlen_fwd_roofline
    )

    for workload in load_workloads(op_name):
        flops, nbytes = formula(**workload)
        assert flops > 0, workload["label"]
        assert nbytes > 0, workload["label"]
