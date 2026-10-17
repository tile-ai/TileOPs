"""Attention state merge with adapted and native-call baseline timings."""

import math

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLASHINFER_TAG, VLLM_TAG, flashinfer_op, vllm_op
from tileops.attention import MergeAttentionStatesFwdOp


def _flashinfer_inputs(v_a, s_a, v_b, s_b):
    # merge_state takes log2 LSE, this op natural-log LSE. At D=128 it runs 16 threads a
    # head, so H=128 exceeds the CTA thread limit; rows are independent, so a view regroups
    # the heads without a copy.
    heads = math.gcd(v_a.shape[1], 64)
    return (
        v_a.view(-1, heads, v_a.shape[2]),
        s_a.view(-1, heads) / math.log(2),
        v_b.view(-1, heads, v_b.shape[2]),
        s_b.view(-1, heads) / math.log(2),
    )


def _flashinfer_merge():
    merge = flashinfer_op("merge_state", "cascade")

    def run(v_a, s_a, v_b, s_b):
        v, s = merge(*_flashinfer_inputs(v_a, s_a, v_b, s_b))
        return v.view_as(v_a), (s * math.log(2)).view_as(s_a)

    return run


def _vllm_inputs(v_a, s_a, v_b, s_b):
    return v_a, s_a.T.contiguous(), v_b, s_b.T.contiguous()


def _vllm_merge(inputs):
    merge = vllm_op("merge_attn_states")
    output = torch.empty_like(inputs[0])
    # vLLM keeps LSE head-major: the adapted call transposes it, inside the timer.
    output_lse = torch.empty(
        (inputs[1].shape[1], inputs[1].shape[0]), dtype=torch.float32, device=inputs[1].device
    )

    def run(v_a, s_a, v_b, s_b):
        merge(output, *_vllm_inputs(v_a, s_a, v_b, s_b), output_lse)
        return output, output_lse.T.contiguous()

    return run


def _flashinfer_native(inputs):
    """Time the log2 API with inputs already in its native contract."""
    merge = flashinfer_op("merge_state", "cascade")
    return bench.Implementation(
        run=merge,
        args=_flashinfer_inputs(*inputs),
        noncomparable_reason="native log2 LSE; input and output base conversions excluded",
    )


def _vllm_native(inputs):
    """Time the head-major API with the layout preparation outside timing."""
    merge = vllm_op("merge_attn_states")
    v_a, a, v_b, b = _vllm_inputs(*inputs)
    output, lse = torch.empty_like(v_a), torch.empty_like(a)

    def run():
        merge(output, v_a, a, v_b, b, lse)
        return output, lse

    return bench.Implementation(
        run=run,
        args=(),
        noncomparable_reason="native head-major LSE; input and output layout conversions excluded",
    )


@pytest.mark.parametrize("case", bench.cases(MergeAttentionStatesFwdOp), ids=lambda case: case.id)
def test_merge_attention_states_bench(case):
    bench.Runner(MergeAttentionStatesFwdOp(**case.arguments), case).compare(
        {
            FLASHINFER_TAG: _flashinfer_merge(),
            VLLM_TAG: _vllm_merge(case.inputs),
            f"{FLASHINFER_TAG}-native": _flashinfer_native(case.inputs),
            f"{VLLM_TAG}-native": _vllm_native(case.inputs),
        }
    )
