"""Paged attention sinks with ragged queries, RoPE and score controls."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload


def _check_sink_normalization(dtype):
    """Sink normalization is supported on every paged-attention architecture."""
    w = GQAPagedFwdWorkload(8, 2, 64, [32], [256], 64, 4, 4, dtype, has_sinks=True)
    inputs = w.gen_inputs()
    *_, sinks = inputs
    sinks.fill_(5.0)
    TestBase.check(w, GQAPagedFwdOp(), *inputs)


def _check_ragged_features(dtype, dim, group, rope, layout, left, right, causal, cap):
    """Hopper supports the TMA/WGMMA path used by these ragged attention calls."""
    semantics = dict(
        pos_encoding_mode="rope" if rope else "none",
        rotary_dim=32 if rope else None,
        rope_layout=layout,
        window_size_left=left,
        window_size_right=right,
        is_causal=causal,
        softcap=cap,
    )
    w = GQAPagedFwdWorkload(
        group * 2,
        2,
        dim,
        [513 if group == 1 else 257, 1, 0, 31],
        [513, 97, 0, 129],
        48,
        11,
        44,
        dtype,
        has_sinks=True,
        **semantics,
    )
    inputs = w.gen_inputs()
    q, k_pages, v_pages, page_table, cache_seqlens, cu_q, _qs, _ks, _vs, _cos, _sin, sinks = inputs
    # Reuse one physical page at different logical positions; never rotate in place.
    page_table[1, 0] = page_table[0, 1]
    page = int(page_table[0, 513 // 48])
    k_pages[page, 513 % 48 :] = float("nan")
    v_pages[page, 513 % 48 :] = float("inf")
    sinks[0] = -float("inf")
    sinks[1] = 80
    op = GQAPagedFwdOp(**semantics)
    TestBase.check(w, op, *inputs)
    if rope and layout == "neox":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(*inputs)
        q.mul_(0.5)
        sinks.fill_(1.5)
        cache_seqlens[0] = 510
        cu_q.copy_(torch.tensor([0, 256, 258, 258, 289], device=q.device, dtype=torch.int32))
        page_table[1, 0] = page_table[0, 2]
        for _ in range(2):
            graph.replay()
            TestBase.check(w, op, *inputs, runs=lambda *args: output)

        # Reuse the Op with different packed Q lengths and page-pool capacities.
        for total_q, pages in ((161, 44), (161, 48), (289, 48)):
            changed = GQAPagedFwdWorkload(
                w.heads,
                w.heads_kv,
                dim,
                [total_q - 32, 1, 0, 31],
                w.cache_lens,
                w.page_size,
                w.max_pages_per_req,
                pages,
                dtype,
                has_sinks=True,
                **semantics,
            )
            TestBase.check(changed, op, *changed.gen_inputs())


@pytest.mark.parametrize(
    "q_lens,cache_lens",
    [
        pytest.param([2, 1], [257, 33], id="short-queries"),
        pytest.param([1024, 1], [1025, 33], id="long-query"),
    ],
)
@pytest.mark.parametrize(
    "scale,features",
    [
        pytest.param(0.125, False, marks=pytest.mark.smoke),
        pytest.param(0.125, True, marks=pytest.mark.full),
        pytest.param(0.0, True, marks=pytest.mark.full),
        pytest.param(-0.125, True, marks=pytest.mark.full),
    ],
)
def test_paged_sink_score_controls(q_lens, cache_lens, scale, features):
    """Short and long queries preserve sink normalization and score controls."""
    page_size = 37
    width = (max(cache_lens) + page_size - 1) // page_size
    w = GQAPagedFwdWorkload(
        16,
        4,
        64,
        q_lens,
        cache_lens,
        page_size,
        width,
        len(q_lens) * width,
        torch.bfloat16,
        sm_scale=scale,
        softcap=2.0 if features and scale == 0.125 else None,
        has_sinks=True,
        pos_encoding_mode="rope" if features else "none",
        rotary_dim=32 if features else None,
        window_size_left=17 if features else -1,
    )
    op = GQAPagedFwdOp(
        sm_scale=w.sm_scale,
        softcap=w.softcap,
        pos_encoding_mode=w.pos_encoding_mode,
        rotary_dim=w.rotary_dim,
        window_size_left=w.window_size_left,
    )
    inputs = w.gen_inputs()
    q, k_pages, v_pages, page_table, cache_seqlens, cu_q, _qs, _ks, _vs, _cos, _sin, sinks = inputs
    if features:
        sinks[0] = -float("inf")
        sinks[1] = float("inf")
        sinks[2] = 80
    else:
        sinks.fill_(5.0)
    TestBase.check(w, op, *inputs)


def _check_window_negative_logits(left):
    """A masked first tile must not pin the running maximum above negative logits."""
    w = GQAPagedFwdWorkload(
        16,
        4,
        64,
        [256, 256],
        [1024, 1024],
        64,
        16,
        32,
        torch.bfloat16,
        window_size_left=left,
        sm_scale=1.0,
        pos_encoding_mode="rope",
    )
    inputs = w.gen_inputs()
    q, k_pages, _v_pages, _table, _lengths, _cu, _qs, _ks, _vs, cos, sin, _sinks = inputs
    q.fill_(1)
    k_pages.fill_(-1)
    cos.fill_(1)
    sin.zero_()
    TestBase.check(
        w, GQAPagedFwdOp(window_size_left=left, sm_scale=1.0, pos_encoding_mode="rope"), *inputs
    )


def _check_rope_scratch_budget():
    """An oversized page-table capacity falls back without expanding the KV pool."""
    from tileops.kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    w = GQAPagedFwdWorkload(
        16,
        4,
        64,
        [128, 128],
        [256, 256],
        64,
        4,
        8,
        torch.float16,
        pos_encoding_mode="rope",
    )
    inputs = w.gen_inputs()
    q, k_pages, _v_pages, page_table, _lengths, _cu_q, _qs, _ks, _vs, cos, _sin, sinks = inputs
    page_table = page_table.repeat(1, 16384)
    inputs = (q, k_pages, _v_pages, page_table, _lengths, _cu_q, _qs, _ks, _vs, cos, _sin, sinks)
    op = GQAPagedFwdOp(pos_encoding_mode="rope")
    assert "scratch budget" in GQAPagedFwdWSKernel.refusal(
        op.paged_call(q, k_pages, page_table, cos, sinks)
    )
    TestBase.check(w, op, *inputs)


@pytest.mark.parametrize(
    "check,args",
    [
        pytest.param(
            _check_sink_normalization, (torch.float16,), marks=pytest.mark.smoke, id="sink-fp16"
        ),
        pytest.param(
            _check_sink_normalization, (torch.bfloat16,), marks=pytest.mark.smoke, id="sink-bf16"
        ),
        pytest.param(
            _check_ragged_features,
            (torch.float16, 128, 4, True, "neox", -1, -1, True, None),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="ragged-neox",
        ),
        pytest.param(
            _check_ragged_features,
            (torch.bfloat16, 64, 8, True, "interleaved", 17, 3, False, 2.0),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="ragged-window-rope",
        ),
        pytest.param(
            _check_ragged_features,
            (torch.float16, 64, 1, False, "neox", 0, 0, True, None),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="self-window",
        ),
        pytest.param(
            _check_ragged_features,
            (torch.bfloat16, 128, 16, False, "neox", 128, -1, True, 5.0),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="window-softcap",
        ),
        pytest.param(
            _check_ragged_features,
            (torch.float16, 128, 4, True, "interleaved", -1, 0, False, None),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="right-window-rope",
        ),
        pytest.param(
            _check_ragged_features,
            (torch.bfloat16, 64, 4, False, "neox", 31, 0, False, None),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="two-sided-window",
        ),
        pytest.param(
            _check_window_negative_logits,
            (17,),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="negative-window-17",
        ),
        pytest.param(
            _check_window_negative_logits,
            (256,),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="negative-window-256",
        ),
        pytest.param(
            _check_rope_scratch_budget,
            (),
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="rope-scratch-budget",
        ),
    ],
)
def test_paged_features(check, args):
    """Keep a sink-only smoke tier; Hopper feature checks have their own setup routines."""
    check(*args)
