from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import (
    gqa_prefill_args,
    gqa_prefill_paged_args,
    gqa_qkv_args,
    manifest_params,
)
from tileops.kernels.attention import (
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAFwdWsPersistentKernel,
    GQAPrefillFwdKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
)
from tileops.manifest import load_workloads
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdTest,
    GroupedQueryAttentionFwdTest,
)
from workloads.attention.gqa_prefill import (
    GQAPrefillFwdTest,
    GQAPrefillPagedWithKVCacheFwdTest,
    GQAPrefillVarlenFwdTest,
)

_GQA_FWD_OP = "GroupedQueryAttentionFwdOp"
_GQA_BWD_OP = "GroupedQueryAttentionBwdOp"
_GQA_PREFILL_FWD_OP = "GroupedQueryAttentionPrefillFwdOp"
_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_OP = "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp"


class GQAPrefillVarlenFwdBenchmark(BenchmarkBase[GQAPrefillVarlenFwdTest]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        visible = 0
        for q_len, kv_len in zip(t.q_lens, t.kv_lens, strict=True):
            visible += q_len * kv_len - q_len * (q_len - 1) / 2 if t.is_causal else q_len * kv_len
        return 4.0 * t.heads * visible * t.dim

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        query_size = sum(t.q_lens) * t.heads * t.dim
        kv_size = sum(t.kv_lens) * t.heads_kv * t.dim
        output_size = query_size
        cu_seqlens_size = 2 * (t.batch + 1)
        return (
            query_size + 2 * kv_size + output_size
        ) * t.dtype.itemsize + cu_seqlens_size * torch.int32.itemsize


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


def _uniform_packed_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    batch, seq_len_q, _, _ = q.shape
    _, seq_len_kv, heads_kv, _ = k.shape
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * seq_len_q
    cu_kv = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * seq_len_kv
    q_scale = torch.ones((batch, heads_kv), device=q.device, dtype=torch.float32)
    return (
        q.reshape(batch * seq_len_q, q.shape[2], q.shape[3]).contiguous(),
        k.reshape(batch * seq_len_kv, heads_kv, k.shape[3]).contiguous(),
        v.reshape(batch * seq_len_kv, heads_kv, v.shape[3]).contiguous(),
        cu_q,
        cu_kv,
        q_scale,
        torch.ones_like(q_scale),
        torch.ones_like(q_scale),
    )


def _flashinfer_gqa_fwd(test, q, k, v):
    """FlashInfer ragged-prefill baseline. Handles seq_len_q != seq_len_kv (square is
    the seq_len_q == seq_len_kv case). Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    Hkv = k.shape[2]
    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * Sq
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * Skv

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=H,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        causal=test.is_causal,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D),
            k.reshape(-1, Hkv, D),
            v.reshape(-1, Hkv, D),
        ).reshape(B, Sq, H, D)

    return run_fn


def _torch_gqa_fwd(test):
    """Torch SDPA forward baseline."""

    def fn(q, k, v):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=test.is_causal,
            enable_gqa=True,
        )
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
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=test.is_causal,
            enable_gqa=True,
        )
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
                ~mask.view(1, 1, test.seq_len_q, test.seq_len_kv), float("-inf")
            )
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


def _tileops_gqa_variant(op: GroupedQueryAttentionFwdOp) -> str:
    kernel = op.kernel
    if isinstance(kernel, GQAPrefillFwdWsPersistentCausalKernel):
        return "prefill_ws_causal"
    if isinstance(kernel, GQAPrefillFwdKernel):
        return "prefill"
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
_GQA_FWD_BENCH_PARAMS = manifest_params(load_workloads(_GQA_FWD_OP), gqa_qkv_args)


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_FWD_BENCH_PARAMS,
)
def test_gqa_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    bm = ManifestBenchmark(_GQA_FWD_OP, op, test)
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
_GQA_BWD_BENCH_PARAMS = manifest_params(load_workloads(_GQA_BWD_OP), gqa_qkv_args)


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_BWD_BENCH_PARAMS,
)
def test_gqa_bwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    bm = ManifestBenchmark(_GQA_BWD_OP, op, test)
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


_GQA_PREFILL_FWD_BENCH_PARAMS = manifest_params(
    [workload for workload in load_workloads(_GQA_PREFILL_FWD_OP) if workload.get("backend") != "fp8"],
    gqa_prefill_args,
    tune=False,
)


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_PREFILL_FWD_BENCH_PARAMS,
)
def test_gqa_prefill_fwd_bench(
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GQAPrefillFwdTest(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, causal, dtype)
    inputs = test.gen_inputs()
    packed_inputs = _uniform_packed_prefill_inputs(*inputs)

    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=causal,
        dtype=dtype,
        tune=tune,
    )
    bm = ManifestBenchmark(_GQA_PREFILL_FWD_OP, op, test)
    result = bm.profile(op, *packed_inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_gqa_prefill_ref(test), *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")

    fi_fn = _flashinfer_gqa_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")


_GQA_PREFILL_VARLEN_FWD_BENCH_PARAMS = [
    pytest.param(
        4,
        [512, 512, 512, 512],
        [1024, 1024, 1024, 1024],
        32,
        8,
        128,
        True,
        torch.float16,
        False,
        id="llama-3.1-8b-prefill-varlen-uniform-fp16",
    ),
    pytest.param(
        4,
        [128, 256, 640, 512],
        [512, 768, 1280, 1024],
        32,
        8,
        128,
        True,
        torch.float16,
        False,
        id="llama-3.1-8b-prefill-varlen-mixed-fp16",
    ),
    pytest.param(
        2,
        [512, 512],
        [1024, 2048],
        64,
        8,
        128,
        True,
        torch.bfloat16,
        False,
        id="llama-3.1-70b-prefill-varlen-q-lt-kv-bf16",
    ),
]


@pytest.mark.parametrize(
    "batch, q_lens, kv_lens, heads, heads_kv, dim, causal, dtype, tune",
    _GQA_PREFILL_VARLEN_FWD_BENCH_PARAMS,
)
def test_gqa_prefill_varlen_fwd_bench(
    batch: int,
    q_lens: list[int],
    kv_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GQAPrefillVarlenFwdTest(batch, heads, heads_kv, q_lens, kv_lens, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch, heads, heads_kv, dim, test.max_seqlen_q, test.max_seqlen_kv, causal, dtype, tune=tune
    )
    bm = GQAPrefillVarlenFwdBenchmark(test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_gqa_prefill_varlen_ref(test), *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")



def _fp8_paged_cache_inputs(
    test: GQAPrefillPagedWithKVCacheFwdTest,
) -> tuple[torch.Tensor, ...]:
    q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table, max_seqlen_q = (
        test.gen_inputs()
    )
    k_scale = torch.full((1,), 0.01, dtype=torch.float32, device=q.device)
    v_scale = torch.full((1,), 0.01, dtype=torch.float32, device=q.device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    k_pages = (k_pages / k_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
    v_pages = (v_pages / v_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
    return (
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
        max_seqlen_q,
    )


_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_BENCH_PARAMS = manifest_params(
    load_workloads(_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_OP),
    gqa_prefill_paged_args,
    tune=False,
)


@pytest.mark.parametrize(
    "batch, q_lens, cache_lens, heads, heads_kv, page_size, dim, causal, fuse_rope, "
    "rotary_dim, softcap, cache_dtype, dtype, tune",
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
    fuse_rope: bool,
    rotary_dim: Optional[int],
    softcap: Optional[float],
    cache_dtype: Optional[torch.dtype],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if cache_dtype == fp8_dtype and fp8_dtype is not None:
        if fuse_rope or rotary_dim is not None:
            pytest.skip("FP8 paged KV cache benchmark does not support fused RoPE")
    elif cache_dtype is not None and fp8_dtype is None:
        pytest.skip("torch fp8 is unavailable")
    test = GQAPrefillPagedWithKVCacheFwdTest(
        batch,
        heads,
        heads_kv,
        q_lens,
        cache_lens,
        page_size,
        dim,
        causal,
        dtype,
        fuse_rope=fuse_rope,
        rotary_dim=rotary_dim,
        softcap=softcap,
    )
    if cache_dtype == fp8_dtype and fp8_dtype is not None:
        inputs = _fp8_paged_cache_inputs(test)
    else:
        q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table, max_seqlen_q = (
            test.gen_inputs()
        )
        k_scale = torch.ones((1,), dtype=torch.float32, device=q.device)
        v_scale = torch.ones((1,), dtype=torch.float32, device=q.device)
        inputs = (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_seqlen_q,
        )

    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=test.max_pages_per_req,
        page_size=page_size,
        dim=dim,
        is_causal=causal,
        dtype=dtype,
        cache_dtype=cache_dtype,
        softcap=softcap,
        tune=tune,
        fuse_rope=fuse_rope,
        max_position=test.max_total_len if fuse_rope else None,
        rotary_dim=rotary_dim,
    )
    op.total_q = test.total_q
    op.q_lens = q_lens
    op.cache_lens = cache_lens
    op.max_seqlen_q = test.max_seqlen_q
    bm = ManifestBenchmark(_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
