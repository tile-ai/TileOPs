"""Attention state merge with adapted and native-call baseline timings."""

import math

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import flashinfer_op, vllm_op
from tileops.attention import MergeAttentionStatesFwdOp


def _flashinfer_merge():
    merge = flashinfer_op("merge_state", "cascade")

    def run(v_a, s_a, v_b, s_b):
        # FlashInfer's state uses log2 LSE; our public contract uses natural logs.
        # At D=128, it launches 16 threads/head: H=128 exceeds the CTA limit.
        # Rows are independent, so regroup heads with views (no data copies).
        heads = math.gcd(v_a.shape[1], 64)
        v, s = merge(
            v_a.view(-1, heads, v_a.shape[2]),
            s_a.view(-1, heads) / math.log(2),
            v_b.view(-1, heads, v_b.shape[2]),
            s_b.view(-1, heads) / math.log(2),
        )
        return v.view_as(v_a), (s * math.log(2)).view_as(s_a)

    return run


def _vllm_merge(inputs):
    merge = vllm_op("merge_attn_states")
    output = torch.empty_like(inputs[0])
    # vLLM stores its LSE head-major. Converting live inputs belongs inside timing.
    output_lse = torch.empty(
        (inputs[1].shape[1], inputs[1].shape[0]), dtype=torch.float32, device=inputs[1].device
    )

    def run(v_a, s_a, v_b, s_b):
        merge(output, v_a, s_a.T.contiguous(), v_b, s_b.T.contiguous(), output_lse)
        return output, output_lse.T.contiguous()

    return run


def _flashinfer_native(inputs):
    """Time the log2 API with inputs already in its native contract."""
    merge = flashinfer_op("merge_state", "cascade")
    v_a, s_a, v_b, s_b = inputs
    heads = math.gcd(v_a.shape[1], 64)
    args = (
        v_a.view(-1, heads, v_a.shape[2]),
        s_a.view(-1, heads) / math.log(2),
        v_b.view(-1, heads, v_b.shape[2]),
        s_b.view(-1, heads) / math.log(2),
    )
    return bench.Implementation(
        run=merge,
        args=args,
        noncomparable_reason="native log2 LSE; input and output base conversions excluded",
    )


def _vllm_native(inputs):
    """Time the head-major API with the layout preparation outside timing."""
    merge = vllm_op("merge_attn_states")
    v_a, s_a, v_b, s_b = inputs
    a, b = s_a.T.contiguous(), s_b.T.contiguous()
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
            "flashinfer-adapted": _flashinfer_merge(),
            "vllm-adapted": _vllm_merge(case.inputs),
            "flashinfer-native": _flashinfer_native(case.inputs),
            "vllm-native": _vllm_native(case.inputs),
        }
    )
