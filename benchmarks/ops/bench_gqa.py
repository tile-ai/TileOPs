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
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(q, k, v, causal=test.is_causal)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _fa3_gqa_bwd(test: GqaBwdTest):
    """Return FA3 backward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    @torch.enable_grad()
    def baseline_fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        out = flash_attn_func(q, k, v, causal=test.is_causal)
        out = out[0] if isinstance(out, tuple) else out
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


# GQA forward benchmark parameters.
#
# Three head profiles cover the mainstream LLM GQA configurations:
#   small  (32:8:128) — Llama-3.1-8B, Qwen3-8B, Mistral-24B
#   medium (64:8:128) — Llama-3.1-70B, Qwen3-32B, Qwen2.5-72B
#   large  (128:8:128) — Llama-3.1-405B
# head_dim=128 and kv_heads=8 are near-universal across Llama, Qwen3, and Mistral.
#
# Inference prefill (fp16): seq_len from 1K to 128K covers short chat to
# full-context workloads.  B=1 because prefill is single-request in practice.
#
# Training (bf16): seq_len 2K-8K covers SFT (2K) and pretraining (4K-8K).
# B=1-2 reflects typical micro-batch sizes.  No long-context training configs
# since >90% of pretraining compute is at 4K-8K.
_GQA_FWD_BENCH_PARAMS = [
    # ── Inference prefill: B=1, causal, fp16 ──
    # Short chat prompt
    pytest.param(1, 1024, 32, 8, 128, True, torch.float16, True, id="llama8b-1k"),
    # Document QA / code completion
    pytest.param(1, 4096, 32, 8, 128, True, torch.float16, True, id="llama8b-4k"),
    # RAG context / Llama-4 chunk_size
    pytest.param(1, 8192, 32, 8, 128, True, torch.float16, True, id="llama8b-8k"),
    # Long-document summarization
    pytest.param(1, 32768, 32, 8, 128, True, torch.float16, True, id="llama8b-32k"),
    # Full-context (Llama-3.1 max)
    pytest.param(1, 131072, 32, 8, 128, True, torch.float16, True, id="llama8b-128k"),
    # 70B-class single-request prefill
    pytest.param(1, 4096, 64, 8, 128, True, torch.float16, True, id="llama70b-4k"),
    # 405B-class single-request prefill
    pytest.param(1, 4096, 128, 8, 128, True, torch.float16, True, id="llama405b-4k"),
    # ── Training: bf16, fwd benchmarked here, bwd below ──
    # Pretraining main phase (8B-class, micro-batch=2)
    pytest.param(2, 4096, 32, 8, 128, True, torch.bfloat16, True, id="train-8b-4k"),
    # Pretraining longer sequences (8B-class)
    pytest.param(1, 8192, 32, 8, 128, True, torch.bfloat16, True, id="train-8b-8k"),
    # Pretraining 70B-class
    pytest.param(1, 4096, 64, 8, 128, True, torch.bfloat16, True, id="train-70b-4k"),
    # Pretraining 405B-class
    pytest.param(1, 4096, 128, 8, 128, True, torch.bfloat16, True, id="train-405b-4k"),
    # SFT / LoRA fine-tuning (shorter sequences, micro-batch=2)
    pytest.param(2, 2048, 32, 8, 128, True, torch.bfloat16, True, id="sft-8b"),
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

    fi_fn = _flashinfer_gqa_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(_torch_gqa_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


# GQA backward benchmark parameters (training only).
# Backward is only used during training — extract the training subset from
# _GQA_FWD_BENCH_PARAMS by ID prefix to avoid manual duplication.
_GQA_BWD_BENCH_PARAMS = [
    p for p in _GQA_FWD_BENCH_PARAMS
    if p.id.startswith(("train", "sft"))
]


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
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
