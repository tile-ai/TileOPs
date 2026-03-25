from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_mha import (
    MhaBwdTest,
    MhaFwdTest,
)
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


def _torch_mha_fwd(test):
    """Torch SDPA forward baseline."""
    def fn(q, k, v):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=test.is_causal)
        return out.transpose(1, 2)
    return fn


def _torch_mha_bwd(test):
    """Torch SDPA backward baseline (includes forward recompute)."""
    @torch.enable_grad()
    def fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=test.is_causal)
        out.transpose(1, 2).contiguous().backward(grad_output)
        return q.grad, k.grad, v.grad
    return fn


_MHA_FWD_BENCH_PARAMS = [
    pytest.param(1, 1024, 8, 64, False, torch.float16, True, id="prefill-fp16"),
    pytest.param(16, 2048, 16, 128, False, torch.float16, True, id="throughput-fp16"),
    pytest.param(4, 4096, 16, 128, False, torch.bfloat16, True, id="long-seq-bf16"),
]


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", _MHA_FWD_BENCH_PARAMS)
def test_mha_fwd_bench(batch: int, seq_len: int, heads: int, dim: int, causal: bool,
                       dtype: torch.dtype, tune: bool) -> None:
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    bm = MhaFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    baseline_fn = _baseline_mha_fwd(test)
    if baseline_fn is not None:
        result_bl = bm.profile(baseline_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(_torch_mha_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


_MHA_BWD_BENCH_PARAMS = _MHA_FWD_BENCH_PARAMS


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", _MHA_BWD_BENCH_PARAMS)
def test_mha_bwd_bench(batch: int, seq_len: int, heads: int, dim: int, causal: bool,
                       dtype: torch.dtype, tune: bool) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    bm = MhaBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    baseline_fn = _baseline_mha_bwd(test)
    if baseline_fn is not None:
        result_bl = bm.profile(baseline_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(_torch_mha_bwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
