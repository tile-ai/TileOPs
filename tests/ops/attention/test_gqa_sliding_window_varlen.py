"""Tests for GroupedQueryAttentionSlidingWindowVarlenFwdOp against a pure-PyTorch reference."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionSlidingWindowVarlenFwdOp
from tileops.perf.formulas import (
    gqa_prefill_varlen_fwd_roofline,
    gqa_sliding_window_fwd_roofline,
    gqa_sliding_window_varlen_fwd_roofline,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionSlidingWindowVarlenFwdWorkload,
)


class GroupedQueryAttentionSlidingWindowVarlenFwdTest(GroupedQueryAttentionSlidingWindowVarlenFwdWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Pure-PyTorch reference: per-sample masked softmax attention.

        offset = seqlen_k - seqlen_q aligns the causal mask bottom-right
        (FA3 convention).  When seqlen_q == seqlen_k, offset=0 and the mask
        reduces to the standard causal mask.
        """
        groups = self.heads // self.heads_kv
        scale = self.dim ** -0.5
        outputs = []

        for i in range(self.batch):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            kv_start = cu_seqlens_k[i].item()
            kv_end = cu_seqlens_k[i + 1].item()

            q_i = q[q_start:q_end]          # [seqlen_q, heads,    dim]
            k_i = k[kv_start:kv_end]        # [seqlen_k, heads_kv, dim]
            v_i = v[kv_start:kv_end]

            seqlen_q = q_end - q_start
            seqlen_k = kv_end - kv_start
            # offset: aligns causal mask to bottom-right corner
            offset = seqlen_k - seqlen_q

            # Expand KV for GQA
            k_exp = k_i.repeat_interleave(groups, dim=1).float()  # [sk, H, D]
            v_exp = v_i.repeat_interleave(groups, dim=1).float()

            # [H, seqlen_q, seqlen_k]
            scores = torch.matmul(
                q_i.float().transpose(0, 1),              # [H, sq, D]
                k_exp.transpose(0, 1).transpose(-2, -1),  # [H, D, sk]
            ) * scale

            # Build attention mask
            q_pos = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
            k_pos = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
            mask = torch.zeros(seqlen_q, seqlen_k,
                               dtype=torch.bool, device=q.device)
            if self.is_causal:
                mask = mask | (k_pos > q_pos + offset)
            if self.wl >= 0:
                mask = mask | (k_pos < q_pos + offset - self.wl)
            if self.wr >= 0:
                mask = mask | (k_pos > q_pos + offset + self.wr)

            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            probs = torch.softmax(scores, dim=-1).nan_to_num()
            out_i = torch.matmul(probs, v_exp.transpose(0, 1))  # [H, sq, D]
            outputs.append(out_i.transpose(0, 1).to(q.dtype))   # [sq, H, D]

        return torch.cat(outputs, dim=0)  # [total_q, H, D]


class GroupedQueryAttentionSlidingWindowVarlenFwdFixture(FixtureBase):
    # Parameters: (batch, seqlens_q, seqlens_k, heads, heads_kv, dim,
    #              is_causal, wl, wr, dtype, tune)
    PARAMS = [
        ("batch, seqlens_q, seqlens_k, heads, heads_kv, dim,"
         " is_causal, wl, wr, dtype, tune", [
             # ── Prefill: seqlen_q == seqlen_k (offset=0) ─────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, True,  -1,  -1, torch.float16,  False, marks=pytest.mark.smoke),   # causal
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, True,  -1,  -1, torch.bfloat16, False, marks=pytest.mark.smoke),   # causal bf16
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, True, 128,  -1, torch.float16,  False, marks=pytest.mark.full),    # causal + wl
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # bidirectional
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, 64,  64, torch.float16,  False, marks=pytest.mark.full),    # window
             # ── KV-cache: seqlen_k > seqlen_q (offset > 0) ───────────────────
             pytest.param(2, [64, 128],  [256, 512], 8, 2, 64, True,  -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # causal kvcache
             pytest.param(2, [64, 128],  [256, 512], 8, 2, 64, True, 128,  -1, torch.float16,  False, marks=pytest.mark.full),    # causal+wl kvcache
             pytest.param(2, [64, 128],  [256, 512], 8, 2, 64, False, 64,  64, torch.float16,  False, marks=pytest.mark.full),    # window kvcache
             # ── bfloat16 ─────────────────────────────────────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, 64,  64, torch.bfloat16, False, marks=pytest.mark.full),    # window bf16
             # ── GQA ratios ───────────────────────────────────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 8, 64, True,  -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # MHA 1:1
             pytest.param(2, [256, 512], [256, 512], 16, 1, 64, True, -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # ratio 16:1
             # ── Mixed lengths within batch ────────────────────────────────────
             pytest.param(3, [128, 256, 384], [128, 256, 384], 8, 2, 64, True, -1, -1, torch.float16, False, marks=pytest.mark.full),
             # ── Right window only ─────────────────────────────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, -1,  64, torch.float16,  False, marks=pytest.mark.full),    # right window
             # ── wl=0 boundary ────────────────────────────────────────────────
             pytest.param(2, [128, 256], [128, 256], 8, 2, 64, True,   0,  -1, torch.float16,  False, marks=pytest.mark.full),    # wl=0
         ]),
    ]


@GroupedQueryAttentionSlidingWindowVarlenFwdFixture
def test_gqa_sliding_window_varlen_fwd_op(
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
    test = GroupedQueryAttentionSlidingWindowVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim,
        is_causal, wl, wr, dtype)
    op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
        batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
        is_causal=is_causal, window_size_left=wl, window_size_right=wr,
        tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])


# ----------------------------------------------------------------------
# Visible-score accounting for the packed-varlen and sliding-window GQA rooflines.
# ----------------------------------------------------------------------


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
        # query i sees keys 0..i+4 — 5 + 6 = 11 scores.
        ([2], [6], 11),
        # Long query run against a short key run: the first four queries see
        # nothing, the last two see 1 and 2 keys — 3 scores.
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.parametrize(
    "op_name",
    [
        "GroupedQueryAttentionSlidingWindowFwdOp",
        "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
    ],
)
@pytest.mark.smoke
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
