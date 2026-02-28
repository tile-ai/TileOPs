from typing import Optional

import pytest
import torch

from tests.ops.test_deepseek_nsa_gqa_window_sliding import (
    GqaWindowSlidingFixture,
    GqaWindowSlidingTest,
)
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GQAWindowSlidingOp


class GqaWindowSlidingBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        total_flops = 2.0 * t.heads * t.uq * t.ukv * t.dim * 2
        if t.is_causal:
            total_flops *= 0.5
        return total_flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        head_kv = t.heads // t.groups
        q_memory = t.uq * t.heads * t.dim * t.dtype.itemsize
        k_memory = t.ukv * head_kv * t.dim * t.dtype.itemsize
        v_memory = t.ukv * head_kv * t.dim * t.dtype.itemsize
        output_memory = t.uq * t.heads * t.dim * t.dtype.itemsize
        return q_memory + k_memory + v_memory + output_memory


def _baseline_gqa_window_sliding(test: GqaWindowSlidingTest):
    """Return FA3 varlen forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        return None

    def baseline_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            window_size=(test.window_size_left, test.window_size_right),
            causal=test.is_causal,
        )

    return baseline_fn


@GqaWindowSlidingFixture
def test_gqa_window_sliding_bench(
    batch_size: int,
    groups: int,
    uq: int,
    ukv: int,
    heads: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GqaWindowSlidingTest(batch_size, groups, uq, ukv, heads, dim, is_causal,
                                window_size_left, window_size_right, dtype, accum_dtype)
    bm = GqaWindowSlidingBenchmark(test)
    inputs = test.gen_inputs()

    op = GQAWindowSlidingOp(
        batch_size=batch_size, groups=groups, uq=uq, ukv=ukv, heads=heads, dim=dim,
        is_causal=is_causal, window_size_left=window_size_left,
        window_size_right=window_size_right, dtype=dtype, accum_dtype=accum_dtype, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gqa_window_sliding", locals(), result, tag="tileops")

    baseline_fn = _baseline_gqa_window_sliding(test)
    if baseline_fn is not None:
        result_bl = bm.profile(baseline_fn, *inputs)
        BenchmarkReport.record("gqa_window_sliding", locals(), result_bl, tag="FA3")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
