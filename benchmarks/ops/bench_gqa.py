from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa import (
    GqaBwdTest,
    GqaFwdTest,
)
from tileops.ops import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp


class GqaFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len * t.seq_len * t.dim
        flops = flops_per_matmul * 2
        return flops / 2 if t.is_causal else flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        query_size = t.batch * t.seq_len * t.heads * t.dim
        kv_size = t.batch * t.seq_len * t.heads_kv * t.dim
        return 2 * (query_size + kv_size) * t.dtype.itemsize


class GqaBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len * t.seq_len * t.dim
        flops = flops_per_matmul * 5
        return flops / 2 if t.is_causal else flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        total_heads = (3 * t.heads + 4 * t.heads_kv)
        return t.batch * total_heads * t.seq_len * t.dim * t.dtype.itemsize


def _fa3_gqa_fwd(test: GqaFwdTest):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        return flash_attn_func(q, k, v, causal=test.is_causal)

    return baseline_fn


def _fa3_gqa_bwd(test: GqaBwdTest):
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


def _flashinfer_gqa_fwd(test: GqaFwdTest, q, k, v):
    """Set up FlashInfer batched prefill wrapper. Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    B, S, H, D = q.shape
    Hkv = k.shape[2]
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * S

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens, kv_indptr=cu_seqlens,
        num_qo_heads=H, num_kv_heads=Hkv, head_dim_qk=D,
        causal=test.is_causal,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D), k.reshape(-1, Hkv, D), v.reshape(-1, Hkv, D),
        ).reshape(B, S, H, D)

    return run_fn


def _torch_gqa_fwd(test):
    """Torch SDPA forward baseline."""
    def fn(q, k, v):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=test.is_causal, enable_gqa=True)
        return out.transpose(1, 2)
    return fn


def _torch_gqa_bwd(test):
    """Torch SDPA backward baseline (includes forward recompute)."""
    @torch.enable_grad()
    def fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=test.is_causal, enable_gqa=True)
        out.transpose(1, 2).contiguous().backward(grad_output)
        return q.grad, k.grad, v.grad
    return fn


_GQA_FWD_BENCH_PARAMS = [
    pytest.param(1, 1024, 8, 4, 64, False, torch.float16, True, id="prefill-fp16"),
    pytest.param(4, 2048, 64, 4, 128, False, torch.float16, True, id="throughput-fp16"),
    pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, True, id="throughput-bf16"),
]


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_FWD_BENCH_PARAMS,
)
def test_gqa_fwd_bench(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int,
                       causal: bool, dtype: torch.dtype, tune: bool) -> None:
    test = GqaFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    bm = GqaFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_gqa_fwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(_torch_gqa_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")

    fi_fn = _flashinfer_gqa_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")


_GQA_BWD_BENCH_PARAMS = _GQA_FWD_BENCH_PARAMS


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_BWD_BENCH_PARAMS,
)
def test_gqa_bwd_bench(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int,
                       causal: bool, dtype: torch.dtype, tune: bool) -> None:
    test = GqaBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    bm = GqaBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_gqa_bwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(_torch_gqa_bwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
