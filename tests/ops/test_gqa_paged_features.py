"""Paged attention sinks with ragged queries, RoPE and score controls."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload


@pytest.mark.sm90
@pytest.mark.parametrize(
    "dtype,dim,group,rope,layout,left,right,causal,cap,basic",
    [
        pytest.param(
            torch.float16, 64, 4, False, "neox", -1, -1, True, None, True, marks=pytest.mark.smoke
        ),
        pytest.param(
            torch.bfloat16, 64, 4, False, "neox", -1, -1, True, None, True, marks=pytest.mark.smoke
        ),
        pytest.param(
            torch.float16, 128, 4, True, "neox", -1, -1, True, None, False, marks=pytest.mark.full
        ),
        pytest.param(
            torch.bfloat16,
            64,
            8,
            True,
            "interleaved",
            17,
            3,
            False,
            2.0,
            False,
            marks=pytest.mark.full,
        ),
        pytest.param(
            torch.float16, 64, 1, False, "neox", 0, 0, True, None, False, marks=pytest.mark.full
        ),
        pytest.param(
            torch.bfloat16,
            128,
            16,
            False,
            "neox",
            128,
            -1,
            True,
            5.0,
            False,
            marks=pytest.mark.full,
        ),
        pytest.param(
            torch.float16,
            128,
            4,
            True,
            "interleaved",
            -1,
            0,
            False,
            None,
            False,
            marks=pytest.mark.full,
        ),
        pytest.param(
            torch.bfloat16, 64, 4, False, "neox", 31, 0, False, None, False, marks=pytest.mark.full
        ),
    ],
)
def test_paged_ragged_features(dtype, dim, group, rope, layout, left, right, causal, cap, basic):
    """Hopper supports the TMA/WGMMA path used by these ragged attention calls."""
    if basic:
        w = GQAPagedFwdWorkload(8, 2, 64, [32], [256], 64, 4, 4, dtype, has_sinks=True)
        inputs = list(w.gen_inputs())
        inputs[11].fill_(5.0)
        op = GQAPagedFwdOp()
        TestBase.check(w, op, *inputs)
        return
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
    inputs = list(w.gen_inputs())
    # Reuse one physical page at different logical positions; never rotate in place.
    inputs[3][1, 0] = inputs[3][0, 1]
    page = int(inputs[3][0, 513 // 48])
    inputs[1][page, 513 % 48 :] = float("nan")
    inputs[2][page, 513 % 48 :] = float("inf")
    inputs[11][0] = -float("inf")
    inputs[11][1] = 80
    op = GQAPagedFwdOp(**semantics)
    TestBase.check(w, op, *inputs)
    if rope and layout == "neox":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(*inputs)
        inputs[0].mul_(0.5)
        inputs[11].fill_(1.5)
        inputs[4][0] = 510
        inputs[5].copy_(
            torch.tensor([0, 256, 258, 258, 289], device=inputs[0].device, dtype=torch.int32)
        )
        inputs[3][1, 0] = inputs[3][0, 2]
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
    inputs = list(w.gen_inputs())
    if features:
        inputs[11][0] = -float("inf")
        inputs[11][1] = float("inf")
        inputs[11][2] = 80
    else:
        inputs[11].fill_(5.0)
    TestBase.check(w, op, *inputs)
