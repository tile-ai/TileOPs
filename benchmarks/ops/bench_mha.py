from typing import Optional

import pytest
import torch

from tests.ops.test_mha import (
    MhaFwdFixture,
    MhaBwdFixture,
    MhaFwdTest,
    MhaBwdTest,
)
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp


class MhaFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len * t.seq_len * t.dim
        flops = flops_per_matmul * 2
        return flops / 2 if t.is_causal else flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return 4 * t.batch * t.heads * t.seq_len * t.dim * t.dtype.itemsize


class MhaBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len * t.seq_len * t.dim
        flops = flops_per_matmul * 5
        return flops / 2 if t.is_causal else flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return 7 * t.batch * t.heads * t.seq_len * t.dim * t.dtype.itemsize


def _baseline_mha_fwd(test: MhaFwdTest):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        import flash_attn_interface
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        return flash_attn_interface.flash_attn_func(
            q, k, v, softmax_scale=None, causal=test.is_causal)

    return baseline_fn


def _baseline_mha_bwd(test: MhaBwdTest):
    """Return FA3 backward baseline callable, or None if not installed."""
    try:
        import flash_attn_interface
    except ImportError:
        return None

    softmax_scale = test.dim**(-0.5)

    def baseline_fn(q, k, v, o, grad_output, lse):
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq, dk, dv, _ = flash_attn_interface._flash_attn_backward(
            grad_output, q, k, v, o, lse, None, None, None, None, None, None, dq, dk, dv,
            softmax_scale, test.is_causal)
        return dq, dk, dv

    return baseline_fn


@MhaFwdFixture
def test_mha_fwd_bench(batch: int, seq_len: int, heads: int, dim: int, causal: bool,
                       dtype: torch.dtype, tune: bool) -> None:
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    bm = MhaFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mha_fwd", locals(), result, tag="tileops")

    baseline_fn = _baseline_mha_fwd(test)
    if baseline_fn is not None:
        result_bl = bm.profile(baseline_fn, *inputs)
        BenchmarkReport.record("mha_fwd", locals(), result_bl, tag="FA3")


@MhaBwdFixture
def test_mha_bwd_bench(batch: int, seq_len: int, heads: int, dim: int, causal: bool,
                       dtype: torch.dtype, tune: bool) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    bm = MhaBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mha_bwd", locals(), result, tag="tileops")

    baseline_fn = _baseline_mha_bwd(test)
    if baseline_fn is not None:
        result_bl = bm.profile(baseline_fn, *inputs)
        BenchmarkReport.record("mha_bwd", locals(), result_bl, tag="FA3")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
