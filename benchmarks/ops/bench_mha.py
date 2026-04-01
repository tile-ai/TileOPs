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


def _fa3_mha_fwd(test: MhaFwdTest):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        return flash_attn_func(q, k, v, causal=test.is_causal)

    return baseline_fn


def _fa3_mha_bwd(test: MhaBwdTest):
    """Return FA3 backward baseline callable, or None if not installed."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    @torch.enable_grad()
    def baseline_fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        out = flash_attn_func(q, k, v, causal=test.is_causal)
        out.backward(grad_output)
        return q.grad, k.grad, v.grad

    return baseline_fn


def _flashinfer_mha_fwd(test: MhaFwdTest, q, k, v):
    """Set up FlashInfer batched prefill wrapper. Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    B, S, H, D = q.shape
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * S

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens, kv_indptr=cu_seqlens,
        num_qo_heads=H, num_kv_heads=H, head_dim_qk=D,
        causal=test.is_causal,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D), k.reshape(-1, H, D), v.reshape(-1, H, D),
        ).reshape(B, S, H, D)

    return run_fn


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

    fa3_fn = _fa3_mha_fwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(_torch_mha_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")

    fi_fn = _flashinfer_mha_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")


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

    fa3_fn = _fa3_mha_bwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(_torch_mha_bwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
