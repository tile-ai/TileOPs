from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.kernels.attention import (
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAFwdWsPersistentKernel,
)
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionPrefillWithKVCacheFwdOp,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdTest,
    GroupedQueryAttentionFwdTest,
)
from workloads.attention.gqa_prefill import (
    GQAPrefillFwdTest,
    GQAPrefillPagedWithKVCacheFwdTest,
    GQAPrefillVarlenFwdTest,
    GQAPrefillWithKVCacheFwdTest,
)


class GroupedQueryAttentionFwdBenchmark(BenchmarkBase[GroupedQueryAttentionFwdTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len * t.seq_len * t.dim
        flops = flops_per_matmul * 2
        return flops / 2 if t.is_causal else flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        query_size = t.batch * t.seq_len * t.heads * t.dim
        kv_size = t.batch * t.seq_len * t.heads_kv * t.dim
        return 2 * (query_size + kv_size) * t.dtype.itemsize


class GroupedQueryAttentionBwdBenchmark(BenchmarkBase[GroupedQueryAttentionBwdTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len * t.seq_len * t.dim
        flops = flops_per_matmul * 5
        return flops / 2 if t.is_causal else flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        total_heads = (3 * t.heads + 4 * t.heads_kv)
        return t.batch * total_heads * t.seq_len * t.dim * t.dtype.itemsize


class GQAPrefillFwdBenchmark(BenchmarkBase[GQAPrefillFwdTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        visible = (
            t.seq_len_q * t.seq_len_kv - t.seq_len_q * (t.seq_len_q - 1) / 2
            if t.is_causal else t.seq_len_q * t.seq_len_kv)
        return 4.0 * t.batch * t.heads * visible * t.dim

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        query_size = t.batch * t.seq_len_q * t.heads * t.dim
        kv_size = t.batch * t.seq_len_kv * t.heads_kv * t.dim
        output_size = query_size
        return (query_size + 2 * kv_size + output_size) * t.dtype.itemsize


class GQAPrefillVarlenFwdBenchmark(BenchmarkBase[GQAPrefillVarlenFwdTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        visible = 0
        for q_len, kv_len in zip(t.q_lens, t.kv_lens, strict=True):
            visible += (
                q_len * kv_len - q_len * (q_len - 1) / 2
                if t.is_causal else q_len * kv_len)
        return 4.0 * t.heads * visible * t.dim

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        query_size = sum(t.q_lens) * t.heads * t.dim
        kv_size = sum(t.kv_lens) * t.heads_kv * t.dim
        output_size = query_size
        cu_seqlens_size = 2 * (t.batch + 1)
        return (
            (query_size + 2 * kv_size + output_size) * t.dtype.itemsize
            + cu_seqlens_size * torch.int32.itemsize
        )


class GQAPrefillWithKVCacheFwdBenchmark(BenchmarkBase[GQAPrefillWithKVCacheFwdTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        old_len = t.seq_len_cap - t.seq_len_new
        visible = (
            t.seq_len_new * old_len + t.seq_len_new * (t.seq_len_new + 1) / 2
            if t.is_causal else t.seq_len_new * t.seq_len_cap)
        return 4.0 * t.batch * t.heads * visible * t.dim

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        query_size = t.batch * t.seq_len_new * t.heads * t.dim
        old_len = t.seq_len_cap - t.seq_len_new
        old_kv_size = 2 * t.batch * old_len * t.heads_kv * t.dim
        new_kv_size = 2 * t.batch * t.seq_len_new * t.heads_kv * t.dim
        append_kv_size = new_kv_size
        output_size = query_size
        cache_seqlens_size = t.batch
        return (
            (query_size + old_kv_size + new_kv_size + append_kv_size + output_size)
            * t.dtype.itemsize
            + cache_seqlens_size * torch.int32.itemsize
        )


class GQAPrefillPagedWithKVCacheFwdBenchmark(
        BenchmarkBase[GQAPrefillPagedWithKVCacheFwdTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        visible = 0
        for q_len, old_len in zip(t.q_lens, t.cache_lens, strict=True):
            visible += (
                q_len * old_len + q_len * (q_len + 1) / 2
                if t.is_causal else q_len * (old_len + q_len))
        return 4.0 * t.heads * visible * t.dim

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        query_size = t.total_q * t.heads * t.dim
        old_kv_size = 2 * sum(t.cache_lens) * t.heads_kv * t.dim
        new_kv_size = 2 * t.total_q * t.heads_kv * t.dim
        append_kv_size = new_kv_size
        output_size = query_size
        metadata_size = (
            (t.batch + 1 + t.batch + t.batch * t.max_pages_per_req) * torch.int32.itemsize)
        return (
            (query_size + old_kv_size + new_kv_size + append_kv_size + output_size)
            * t.dtype.itemsize
            + metadata_size
        )


def _fa3_gqa_fwd(test: GroupedQueryAttentionFwdTest):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(q, k, v, causal=test.is_causal)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _fa3_gqa_bwd(test: GroupedQueryAttentionBwdTest):
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


def _flashinfer_gqa_fwd(test: GroupedQueryAttentionFwdTest, q, k, v):
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


def _torch_gqa_prefill_ref(test: GQAPrefillFwdTest):
    """Materialized torch reference for dense prefill with bottom-right causal mask."""
    def fn(q, k, v):
        groups = test.heads // test.heads_kv
        q_bhsd = q.transpose(1, 2).float()
        k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * (test.dim**-0.5)
        if test.is_causal:
            offset = test.seq_len_kv - test.seq_len_q
            q_pos = torch.arange(test.seq_len_q, device=q.device)[:, None] + offset
            k_pos = torch.arange(test.seq_len_kv, device=q.device)[None, :]
            mask = k_pos <= q_pos
            scores = scores.masked_fill(
                ~mask.view(1, 1, test.seq_len_q, test.seq_len_kv), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v_bhsd).transpose(1, 2).to(q.dtype).contiguous()
    return fn


def _torch_gqa_prefill_varlen_ref(test: GQAPrefillVarlenFwdTest):
    """Materialized torch reference for packed-varlen prefill."""
    def fn(q, k, v, cu_seqlens_q, cu_seqlens_kv):
        groups = test.heads // test.heads_kv
        outputs = []
        for b in range(test.batch):
            q_start = int(cu_seqlens_q[b].item())
            q_end = int(cu_seqlens_q[b + 1].item())
            kv_start = int(cu_seqlens_kv[b].item())
            kv_end = int(cu_seqlens_kv[b + 1].item())
            q_i = q[q_start:q_end].transpose(0, 1).float()
            k_i = k[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            v_i = v[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * (test.dim**-0.5)
            if test.is_causal:
                offset = kv_len - q_len
                q_pos = torch.arange(q_len, device=q.device)[:, None] + offset
                kv_pos = torch.arange(kv_len, device=q.device)[None, :]
                mask = kv_pos <= q_pos
                scores = scores.masked_fill(~mask.view(1, q_len, kv_len), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(probs, v_i).transpose(0, 1).to(q.dtype).contiguous())
        return torch.cat(outputs, dim=0)
    return fn


def _torch_gqa_prefill_with_kv_cache_ref(test: GQAPrefillWithKVCacheFwdTest):
    """Materialized torch reference for contiguous-cache prefill."""
    def fn(q, k_new, v_new, k_cache, v_cache, cache_seqlens):
        groups = test.heads // test.heads_kv
        outputs = []
        for b in range(test.batch):
            old_len = int(cache_seqlens[b].item())
            k_all = torch.cat([k_cache[b, :old_len], k_new[b]], dim=0)
            v_all = torch.cat([v_cache[b, :old_len], v_new[b]], dim=0)
            q_bhsd = q[b].transpose(0, 1).float()
            k_bhsd = k_all.repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            v_bhsd = v_all.repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            total_len = old_len + test.seq_len_new
            scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * (test.dim**-0.5)
            if test.is_causal:
                q_pos = torch.arange(test.seq_len_new, device=q.device)[:, None] + old_len
                k_pos = torch.arange(total_len, device=q.device)[None, :]
                mask = k_pos <= q_pos
                scores = scores.masked_fill(~mask.view(1, test.seq_len_new, total_len),
                                            float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            outputs.append(
                torch.matmul(probs, v_bhsd).transpose(0, 1).to(q.dtype).contiguous())
        return torch.stack(outputs, dim=0)
    return fn


def _tileops_gqa_variant(op: GroupedQueryAttentionFwdOp) -> str:
    kernel = op.kernel
    if isinstance(kernel, GQAFwdWsPersistentCausalKernel):
        return "ws_causal"
    if isinstance(kernel, GQAFwdWsPersistentKernel):
        return "ws_noncausal"
    if isinstance(kernel, GQAFwdWgmmaPipelinedKernel):
        return "wgmma_pipelined"
    if isinstance(kernel, GQAFwdKernel):
        return "legacy"
    return kernel.__class__.__name__


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
    # ── WS/anchor validation cases on H200 ──
    pytest.param(4, 512, 64, 4, 128, False, torch.float16, True, id="ws-noncausal-4x512"),
    pytest.param(4, 512, 64, 4, 128, True, torch.float16, True, id="ws-causal-4x512"),
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
    test = GroupedQueryAttentionFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    bm = GroupedQueryAttentionFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    tileops_variant = _tileops_gqa_variant(op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag=f"tileops_{tileops_variant}")

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
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    bm = GroupedQueryAttentionBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
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


_GQA_PREFILL_FWD_BENCH_PARAMS = [
    pytest.param(1, 128, 128, 8, 2, 64, True, torch.float16, False,
                 id="prefill-dense-q-eq-kv-fp16"),
    pytest.param(1, 128, 256, 8, 2, 64, True, torch.float16, False,
                 id="prefill-dense-q-lt-kv-fp16"),
    pytest.param(1, 128, 256, 8, 2, 64, True, torch.bfloat16, False,
                 id="prefill-dense-q-lt-kv-bf16"),
]


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_PREFILL_FWD_BENCH_PARAMS,
)
def test_gqa_prefill_fwd_bench(batch: int, seq_len_q: int, seq_len_kv: int, heads: int,
                               heads_kv: int, dim: int, causal: bool, dtype: torch.dtype,
                               tune: bool) -> None:
    test = GQAPrefillFwdTest(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, causal, dtype)
    bm = GQAPrefillFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillFwdOp(
        batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_gqa_prefill_ref(test), *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


_GQA_PREFILL_VARLEN_FWD_BENCH_PARAMS = [
    pytest.param(4, [512, 512, 512, 512], [1024, 1024, 1024, 1024], 32, 8, 128, True,
                 torch.float16, False, id="llama-3.1-8b-prefill-varlen-uniform-fp16"),
    pytest.param(4, [128, 256, 640, 512], [512, 768, 1280, 1024], 32, 8, 128, True,
                 torch.float16, False, id="llama-3.1-8b-prefill-varlen-mixed-fp16"),
    pytest.param(2, [512, 512], [1024, 2048], 64, 8, 128, True, torch.bfloat16, False,
                 id="llama-3.1-70b-prefill-varlen-q-lt-kv-bf16"),
]


@pytest.mark.parametrize(
    "batch, q_lens, kv_lens, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_PREFILL_VARLEN_FWD_BENCH_PARAMS,
)
def test_gqa_prefill_varlen_fwd_bench(batch: int, q_lens: list[int], kv_lens: list[int],
                                      heads: int, heads_kv: int, dim: int, causal: bool,
                                      dtype: torch.dtype, tune: bool) -> None:
    test = GQAPrefillVarlenFwdTest(batch, heads, heads_kv, q_lens, kv_lens, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch, heads, heads_kv, dim, test.max_seqlen_q, test.max_seqlen_kv, causal, dtype,
        tune=tune)
    bm = GQAPrefillVarlenFwdBenchmark(test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_gqa_prefill_varlen_ref(test), *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


_GQA_PREFILL_WITH_KV_CACHE_FWD_BENCH_PARAMS = [
    pytest.param(1, 64, 256, 8, 2, 64, True, torch.float16, False, False, None, None,
                 id="prefill-contig-cache-fp16"),
    pytest.param(2, 64, 320, 16, 4, 128, True, torch.float16, False, False, None, None,
                 id="prefill-contig-cache-batch2-fp16"),
    pytest.param(1, 64, 256, 8, 2, 64, True, torch.bfloat16, False, False, None, None,
                 id="prefill-contig-cache-bf16"),
    pytest.param(1, 1024, 33792, 32, 8, 256, True, torch.float16, False, True, 64, None,
                 id="qwen35-9b-prefill-contig-fullattn-prefix32k-chunk1k-partial-rope64-fp16"),
]


@pytest.mark.parametrize(
    "batch, seq_len_new, seq_len_cap, heads, heads_kv, dim, causal, dtype, tune, fuse_rope, "
    "rotary_dim, softcap",
    _GQA_PREFILL_WITH_KV_CACHE_FWD_BENCH_PARAMS,
)
def test_gqa_prefill_with_kv_cache_fwd_bench(batch: int, seq_len_new: int,
                                             seq_len_cap: int, heads: int, heads_kv: int,
                                             dim: int, causal: bool, dtype: torch.dtype,
                                             tune: bool, fuse_rope: bool,
                                             rotary_dim: Optional[int],
                                             softcap: Optional[float]) -> None:
    test = GQAPrefillWithKVCacheFwdTest(
        batch, heads, heads_kv, seq_len_new, seq_len_cap, dim, causal, dtype,
        fuse_rope=fuse_rope, rotary_dim=rotary_dim, softcap=softcap)
    bm = GQAPrefillWithKVCacheFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_new, seq_len_cap, dim, causal, dtype,
        softcap=softcap, tune=tune, fuse_rope=fuse_rope,
        max_position=seq_len_cap if fuse_rope else None, rotary_dim=rotary_dim)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if not fuse_rope and softcap is None:
        result_bl = bm.profile(_torch_gqa_prefill_with_kv_cache_ref(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_BENCH_PARAMS = [
    pytest.param(
        8, [1024] * 8, [32768] * 8, 32, 8, 64, 256, True, torch.float16, False, True, 64, None,
        id="qwen35-9b-prefill-paged-fullattn-b8-prefix32k-chunk1k-p64-partial-rope64-fp16"),
    pytest.param(
        8,
        [256, 512, 768, 1024, 384, 640, 896, 128],
        [4096, 8192, 16384, 32768, 12288, 24576, 30720, 2048],
        32, 8, 64, 256, True, torch.float16, False, True, 64, None,
        id="qwen35-9b-prefill-paged-fullattn-mixed-b8-p64-partial-rope64-fp16"),
    pytest.param(
        8, [512] * 8, [4096] * 8, 32, 8, 64, 128, True, torch.float16, False, True, None, None,
        id="llama31-8b-prefill-paged-b8-prefix4k-chunk512-p64-full-rope-fp16"),
    pytest.param(
        4, [512] * 4, [4096] * 4, 8, 2, 64, 64, True, torch.float16, False, False, None, 50.0,
        id="gqa-prefill-paged-softcap50-b4-prefix4k-chunk512-p64-fp16"),
]


@pytest.mark.parametrize(
    "batch, q_lens, cache_lens, heads, heads_kv, page_size, dim, causal, dtype, tune, "
    "fuse_rope, rotary_dim, softcap",
    _GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_BENCH_PARAMS,
)
def test_gqa_prefill_paged_with_kv_cache_fwd_bench(
    batch: int,
    q_lens: list[int],
    cache_lens: list[int],
    heads: int,
    heads_kv: int,
    page_size: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
    fuse_rope: bool,
    rotary_dim: Optional[int],
    softcap: Optional[float],
) -> None:
    test = GQAPrefillPagedWithKVCacheFwdTest(
        batch, heads, heads_kv, q_lens, cache_lens, page_size, dim, causal, dtype,
        fuse_rope=fuse_rope, rotary_dim=rotary_dim, softcap=softcap)
    bm = GQAPrefillPagedWithKVCacheFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=test.max_pages_per_req,
        page_size=page_size,
        dim=dim,
        is_causal=causal,
        dtype=dtype,
        softcap=softcap,
        tune=tune,
        fuse_rope=fuse_rope,
        max_position=test.max_total_len if fuse_rope else None,
        rotary_dim=rotary_dim,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
