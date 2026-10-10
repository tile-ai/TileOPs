"""Feature combinations on the persistent paged pipeline and split fallback."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.attention.gqa.paged import GQAPagedFwdKernel
from tileops.kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel
from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "dtype,dim,group,rope,layout,left,right,causal,cap",
    [
        (torch.float16, 128, 4, True, "neox", -1, -1, True, None),
        (torch.bfloat16, 64, 8, True, "interleaved", 17, 3, False, 2.0),
        (torch.float16, 64, 1, False, "neox", 0, 0, True, None),
        (torch.bfloat16, 128, 16, False, "neox", 128, -1, True, 5.0),
        (torch.float16, 128, 4, True, "interleaved", -1, 0, False, None),
        (torch.bfloat16, 64, 4, False, "neox", 31, 0, False, None),
    ],
)
def test_paged_ws_features(dtype, dim, group, rope, layout, left, right, causal, cap):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 persistent pipeline")
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
    assert any(isinstance(k, GQAPagedFwdWSKernel) for k in op._dispatched.values())
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


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("scale", [0.125, 0.0, -0.125])
def test_paged_sink_split_fallback(splits, scale):
    class FixedSplit(GQAPagedFwdKernel):
        @property
        def default_config(self):
            return {**super().default_config, "num_split": splits}

    w = GQAPagedFwdWorkload(
        16,
        4,
        64,
        [2, 1],
        [257, 33],
        37,
        7,
        14,
        torch.bfloat16,
        sm_scale=scale,
        softcap=2.0 if scale == 0.125 else None,
        has_sinks=True,
        pos_encoding_mode="rope",
        rotary_dim=32,
        window_size_left=17,
    )
    op = GQAPagedFwdOp(
        sm_scale=w.sm_scale,
        softcap=w.softcap,
        pos_encoding_mode="rope",
        rotary_dim=32,
        window_size_left=17,
        kernel_map={"gqa_paged_varlen_kernel": FixedSplit},
    )
    inputs = list(w.gen_inputs())
    inputs[11][0] = -float("inf")
    inputs[11][1] = float("inf")
    inputs[11][2] = 80
    TestBase.check(w, op, *inputs)
