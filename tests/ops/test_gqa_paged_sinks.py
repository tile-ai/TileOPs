"""Paged sink correctness across the builtin packed and decode paths."""

import pytest
import torch

from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload
from workloads.numerics import compare_outputs


def _workload(q_lens, cache_lens, page, dim, dtype, **semantics):
    width = max(1, -(-max(cache_lens) // page))
    return GQAPagedFwdWorkload(
        32,
        4,
        dim,
        q_lens,
        cache_lens,
        page,
        width,
        len(q_lens) * width,
        dtype,
        has_sinks=True,
        **semantics,
    )


@pytest.mark.parametrize(
    "q_lens,cache_lens,page,dim,dtype,semantics",
    [
        pytest.param(
            [1],
            [511],
            256,
            128,
            torch.float16,
            {},
            marks=[pytest.mark.smoke, pytest.mark.sm90],
            id="bs1-short",
        ),
        pytest.param(
            [1],
            [4097],
            64,
            128,
            torch.float16,
            {},
            marks=[pytest.mark.smoke, pytest.mark.sm90],
            id="bs1-split",
        ),
        pytest.param(
            [1],
            [63],
            16,
            64,
            torch.float16,
            {"sm_scale": -0.125},
            marks=pytest.mark.smoke,
            id="packed-short-negative",
        ),
        pytest.param(
            [1],
            [2049],
            16,
            64,
            torch.bfloat16,
            {"softcap": 3.0},
            marks=pytest.mark.smoke,
            id="packed-split-softcap",
        ),
        pytest.param(
            [5, 1],
            [130, 63],
            64,
            128,
            torch.float16,
            {},
            marks=pytest.mark.smoke,
            id="packed-split-tail",
        ),
        pytest.param(
            [160, 160],
            [700, 450],
            256,
            128,
            torch.bfloat16,
            {},
            marks=pytest.mark.smoke,
            id="packed-unsplit-tail",
        ),
        pytest.param(
            [65, 0, 17],
            [129, 0, 33],
            7,
            64,
            torch.bfloat16,
            {"is_causal": False, "window_size_left": 8, "window_size_right": 3},
            marks=pytest.mark.smoke,
            id="odd-page-window",
        ),
        pytest.param(
            [1, 3],
            [0, 2],
            16,
            64,
            torch.float16,
            {"sm_scale": 0.0},
            marks=pytest.mark.smoke,
            id="empty-masked-zero",
        ),
        pytest.param(
            [129, 5],
            [513, 127],
            48,
            128,
            torch.float16,
            {
                "pos_encoding_mode": "rope",
                "rotary_dim": 64,
                "rope_layout": "interleaved",
                "window_size_left": 127,
            },
            marks=pytest.mark.smoke,
            id="rope-partial",
        ),
        pytest.param(
            [1],
            [1025],
            64,
            128,
            torch.float16,
            {"sm_scale": 0.0},
            marks=pytest.mark.smoke,
            id="bs1-zero-scale",
        ),
        pytest.param(
            [257, 17],
            [2049, 1025],
            64,
            256,
            torch.bfloat16,
            {"is_causal": False},
            marks=pytest.mark.full,
            id="wide-noncausal",
        ),
    ],
)
def test_gqa_paged_sinks(q_lens, cache_lens, page, dim, dtype, semantics):
    """SM90 also selects the warp-specialized batch-one decode path."""
    torch.manual_seed(0)
    workload = _workload(q_lens, cache_lens, page, dim, dtype, **semantics)
    inputs = list(workload.gen_inputs())
    inputs[3] = inputs[3].flip(-1).contiguous()
    # Stale rows in the last page must never participate in PV, even at zero scale.
    for b, length in enumerate(cache_lens):
        slots = torch.arange(length, inputs[3].shape[1] * page, device=inputs[0].device)
        stale = (inputs[3][b, slots // page], slots % page)
        inputs[1][stale] = float("nan")
        inputs[2][stale] = float("nan")
    inputs[-1][0] = -float("inf")
    inputs[-1][-1] = 1000.0
    op = GQAPagedFwdOp(**semantics)
    compare_outputs(op(*inputs), workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.smoke
def test_gqa_paged_sinks_reuse():
    """Presence, noncontiguous head logits and device cache lengths can change on one Op."""
    workload = _workload([1], [2049], 64, 128, torch.bfloat16)
    inputs = workload.gen_inputs()
    sinks = inputs[-1].repeat_interleave(2)[::2]
    op = GQAPagedFwdOp()
    for length in (2049, 63, 0):
        inputs[4].fill_(length)
        for value in (None, sinks, -sinks, None):
            call = (*inputs[:-1], value)
            compare_outputs(op(*call), workload.ref_program(*call), workload.verification(*call))


@pytest.mark.smoke
@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_gqa_paged_sinks_invalid(invalid):
    workload = _workload([1], [17], 16, 64, torch.float16)
    inputs = workload.gen_inputs()
    sinks = inputs[-1][:2] if invalid == "shape" else inputs[-1].half()
    with pytest.raises(ValueError, match="sinks"):
        GQAPagedFwdOp()(*inputs[:-1], sinks)
