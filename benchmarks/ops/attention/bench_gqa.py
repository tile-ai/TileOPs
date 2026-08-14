from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    backward_of,
)
from benchmarks.ops.attention.manifest_params import (
    gqa_fwd_args,
    gqa_prefill_paged_args,
    gqa_prefill_varlen_args,
    gqa_qkv_args,
    manifest_params,
)
from tileops.kernels.attention import (
    GQAFwdWsPersistentCausalKernel,
    GQAPrefillFwdKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
)
from tileops.manifest import load_workloads
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionPrefillDenseFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from workloads.attention.gqa import (
    GQAPrefillPagedWithKVCacheFwdWorkload,
    GQAPrefillVarlenFwdWorkload,
    GroupedQueryAttentionBwdWorkload,
    GroupedQueryAttentionFwdWorkload,
)

_GQA_FWD_OP = "GroupedQueryAttentionPrefillDenseFwdOp"
_GQA_BWD_OP = "GroupedQueryAttentionBwdOp"
_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_OP = "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp"


def _fa3_gqa_fwd(test: GroupedQueryAttentionFwdWorkload):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(
            q,
            k,
            v,
            causal=test.is_causal,
            softcap=test.softcap or 0.0,
            window_size=(test.window_size_left, test.window_size_right),
        )
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _fa3_gqa_bwd(test: GroupedQueryAttentionBwdWorkload):
    """Return FA3 backward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        return None

    @torch.enable_grad()
    def baseline_fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        raw = flash_attn_func(q, k, v, causal=test.is_causal)
        outputs = raw if isinstance(raw, tuple) else (raw,)
        return backward_of(outputs[0])(grad_output, *(None,) * (len(outputs) - 1))

    return baseline_fn


def _flashinfer_gqa_fwd(test, q, k, v):
    """FlashInfer ragged-prefill baseline. Handles seq_len_q != seq_len_kv (square is
    the seq_len_q == seq_len_kv case). Returns callable or None."""
    if test.window_size_right >= 0:
        return None
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
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
        logits_soft_cap=getattr(test, "softcap", None) or 0.0,
        sm_scale=getattr(test, "sm_scale", None),
        window_left=test.window_size_left,
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
    """Materialized reference covering scale, softcap, and window semantics."""

    def fn(q, k, v):
        groups = test.heads // test.heads_kv
        q_bhsd = q.transpose(1, 2).float()
        k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        scale = test.dim**-0.5 if test.sm_scale is None else test.sm_scale
        scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
        if test.softcap is not None and test.softcap > 0:
            scores = test.softcap * torch.tanh(scores / test.softcap)
        offset = test.seq_len_kv - test.seq_len
        q_pos = torch.arange(test.seq_len, device=q.device)[:, None] + offset
        k_pos = torch.arange(test.seq_len_kv, device=q.device)[None, :]
        visible = torch.ones_like(q_pos + k_pos, dtype=torch.bool)
        if test.is_causal:
            visible &= k_pos <= q_pos
        if test.window_size_left >= 0:
            visible &= k_pos >= q_pos - test.window_size_left
        if test.window_size_right >= 0:
            visible &= k_pos <= q_pos + test.window_size_right
        scores.masked_fill_(~visible.view(1, 1, test.seq_len, test.seq_len_kv), float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v_bhsd).transpose(1, 2).to(q.dtype)

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
        # Transposing grad_output into SDPA's layout is a view, so the baseline
        # measures SDPA's backward alone.
        return backward_of(out)(grad_output.transpose(1, 2))

    return fn


def _torch_gqa_prefill_varlen_ref(test: GQAPrefillVarlenFwdWorkload):
    """Materialized torch reference for packed-varlen prefill."""

    def fn(q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale):
        groups = test.heads // test.heads_kv
        outputs = []
        for b in range(test.batch):
            q_start = int(cu_seqlens_q[b].item())
            q_end = int(cu_seqlens_q[b + 1].item())
            kv_start = int(cu_seqlens_kv[b].item())
            kv_end = int(cu_seqlens_kv[b + 1].item())
            q_values = q[q_start:q_end].float()
            k_values = k[kv_start:kv_end].float()
            v_values = v[kv_start:kv_end].float()
            if q.dtype == getattr(torch, "float8_e4m3fn", None):
                q_values *= q_scale[b].repeat_interleave(groups).view(1, test.heads, 1)
                k_values *= k_scale[b].view(1, test.heads_kv, 1)
                v_values *= v_scale[b].view(1, test.heads_kv, 1)
            q_i = q_values.transpose(0, 1)
            k_i = k_values.repeat_interleave(groups, dim=1).permute(1, 0, 2)
            v_i = v_values.repeat_interleave(groups, dim=1).permute(1, 0, 2)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            scale = test.dim**-0.5 if test.sm_scale is None else test.sm_scale
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
            if test.softcap is not None and test.softcap > 0:
                scores = test.softcap * torch.tanh(scores / test.softcap)
            offset = kv_len - q_len
            center = torch.arange(q_len, device=q.device)[:, None] + offset
            kv_pos = torch.arange(kv_len, device=q.device)[None, :]
            visible = torch.ones((q_len, kv_len), device=q.device, dtype=torch.bool)
            if test.is_causal:
                visible &= kv_pos <= center
            if test.window_size_left >= 0:
                visible &= kv_pos >= center - test.window_size_left
            if test.window_size_right >= 0:
                visible &= kv_pos <= center + test.window_size_right
            scores = scores.masked_fill(~visible.view(1, q_len, kv_len), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            outputs.append(
                torch.matmul(probs, v_i).transpose(0, 1).to(test.output_dtype).contiguous()
            )
        return torch.cat(outputs, dim=0)

    return fn


def _tileops_gqa_variant(op: GroupedQueryAttentionPrefillDenseFwdOp, dtype: torch.dtype) -> str:
    kernel = op._get_kernel(dtype)
    if isinstance(kernel, GQAPrefillFwdWsPersistentCausalKernel):
        return "prefill_ws_causal"
    if isinstance(kernel, GQASlidingWindowFwdWgmmaPipelinedKernel):
        return "dense_sliding"
    if isinstance(kernel, GQAPrefillFwdKernel):
        return "prefill"
    if isinstance(kernel, GQAFwdWsPersistentCausalKernel):
        return "ws_causal"
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
_GQA_FWD_BENCH_PARAMS = manifest_params(
    [w for w in load_workloads(_GQA_FWD_OP) if w.get("input_dtype") is None],
    gqa_fwd_args,
)


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, heads, heads_kv, dim, causal, sm_scale, softcap, "
    "window_size_left, window_size_right, fuse_rope, rotary_dim, rope_layout, dtype, tune",
    _GQA_FWD_BENCH_PARAMS,
)
def test_gqa_fwd_bench(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    sm_scale: Optional[float],
    softcap: Optional[float],
    window_size_left: int,
    window_size_right: int,
    fuse_rope: bool,
    rotary_dim: Optional[int],
    rope_layout: str,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionFwdWorkload(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        causal,
        dtype,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        seq_len_kv,
    )
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        causal,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        seq_len_kv=seq_len_kv,
        fuse_rope=fuse_rope,
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
        tune=tune,
    )
    bm = ManifestBenchmark(_GQA_FWD_OP, op, test)
    tileops_variant = _tileops_gqa_variant(op, dtype)
    functors = {f"tileops_{tileops_variant}": op}

    fa3_fn = _fa3_gqa_fwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    fi_fn = _flashinfer_gqa_fwd(test, *inputs)
    if fi_fn is not None:
        functors["flashinfer"] = fi_fn

    if fa3_fn is None and fi_fn is None:
        functors["torch-sdpa"] = _torch_gqa_fwd(test)

    bm.compare(functors, *inputs, record_as=op, params=locals())


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
    test = GroupedQueryAttentionBwdWorkload(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, tune=tune)
    bm = ManifestBenchmark(_GQA_BWD_OP, op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_bwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_gqa_bwd(test)

    bm.compare(functors, *inputs, record_as=op, params=locals())
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


_GQA_PREFILL_VARLEN_FWD_BENCH_PARAMS = manifest_params(
    load_workloads("GroupedQueryAttentionPrefillVarlenFwdOp"),
    gqa_prefill_varlen_args,
    tune=False,
)


@pytest.mark.parametrize(
    "batch, q_lens, kv_lens, heads, heads_kv, dim, causal, sm_scale, softcap, "
    "window_size_left, window_size_right, fuse_rope, rotary_dim, rope_layout, "
    "output_dtype, dtype, tune",
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
    sm_scale: Optional[float],
    softcap: Optional[float],
    window_size_left: int,
    window_size_right: int,
    fuse_rope: bool,
    rotary_dim: Optional[int],
    rope_layout: str,
    output_dtype: Optional[torch.dtype],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GQAPrefillVarlenFwdWorkload(
        batch,
        heads,
        heads_kv,
        q_lens,
        kv_lens,
        dim,
        causal,
        dtype,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        output_dtype,
    )
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch,
        heads,
        heads_kv,
        dim,
        test.max_seqlen_q,
        test.max_seqlen_kv,
        causal,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        dtype=output_dtype,
        tune=tune,
        fuse_rope=fuse_rope,
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    bm = ManifestBenchmark("GroupedQueryAttentionPrefillVarlenFwdOp", op, test)

    bm.compare(
        {"tileops": op, "torch-ref": _torch_gqa_prefill_varlen_ref(test)},
        *inputs,
        record_as=op,
        params=locals(),
    )


def _fp8_paged_cache_inputs(
    test: GQAPrefillPagedWithKVCacheFwdWorkload,
) -> tuple[torch.Tensor, ...]:
    q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table, max_seqlen_q = (
        test.gen_inputs()
    )
    scale_shape = (test.batch, test.heads_kv)
    native_fp8 = test.dtype == torch.float8_e4m3fn
    q_value = 0.05 if native_fp8 else 1.0
    kv_value = 0.05 if native_fp8 else 0.01
    q_scale = torch.full(scale_shape, q_value, dtype=torch.float32, device=q.device)
    k_scale = torch.full(scale_shape, kv_value, dtype=torch.float32, device=q.device)
    v_scale = torch.full(scale_shape, kv_value, dtype=torch.float32, device=q.device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    k_pages = (k_pages / kv_value).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
    v_pages = (v_pages / kv_value).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
    if native_fp8:
        groups = test.heads // test.heads_kv
        cu_q = cu_seqlens_q.tolist()
        q_parts, k_parts, v_parts = [], [], []
        for b in range(test.batch):
            q_start, q_end = cu_q[b], cu_q[b + 1]
            q_head_scale = q_scale[b].repeat_interleave(groups).view(1, test.heads, 1)
            kv_head_scale = k_scale[b].view(1, test.heads_kv, 1)
            q_parts.append((q[q_start:q_end].float() / q_head_scale).clamp(-fp8_max, fp8_max))
            k_parts.append((k_new[q_start:q_end].float() / kv_head_scale).clamp(-fp8_max, fp8_max))
            v_parts.append((v_new[q_start:q_end].float() / kv_head_scale).clamp(-fp8_max, fp8_max))
        q = torch.cat(q_parts).to(torch.float8_e4m3fn).contiguous()
        k_new = torch.cat(k_parts).to(torch.float8_e4m3fn).contiguous()
        v_new = torch.cat(v_parts).to(torch.float8_e4m3fn).contiguous()
    return (
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        q_scale,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )


_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_BENCH_PARAMS = manifest_params(
    load_workloads(_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_OP),
    gqa_prefill_paged_args,
    tune=False,
)


@pytest.mark.parametrize(
    "batch, q_lens, cache_lens, heads, heads_kv, page_size, dim, causal, fuse_rope, "
    "rotary_dim, rope_layout, sm_scale, softcap, window_size_left, window_size_right, append_kv, "
    "cache_dtype, output_dtype, dtype, tune",
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
    rope_layout: str,
    sm_scale: Optional[float],
    softcap: Optional[float],
    window_size_left: int,
    window_size_right: int,
    append_kv: bool,
    cache_dtype: Optional[torch.dtype],
    output_dtype: Optional[torch.dtype],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if cache_dtype is not None and fp8_dtype is None:
        pytest.skip("torch fp8 is unavailable")
    test = GQAPrefillPagedWithKVCacheFwdWorkload(
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
        rope_layout=rope_layout,
        sm_scale=sm_scale,
        softcap=softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        append_kv=append_kv,
        cache_dtype=cache_dtype,
        output_dtype=output_dtype,
    )
    if cache_dtype == fp8_dtype and fp8_dtype is not None:
        inputs = _fp8_paged_cache_inputs(test)
    else:
        (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_seqlen_q,
        ) = test.gen_inputs()
        scale_shape = (batch, heads_kv)
        q_scale = torch.ones(scale_shape, dtype=torch.float32, device=q.device)
        k_scale = torch.ones(scale_shape, dtype=torch.float32, device=q.device)
        v_scale = torch.ones(scale_shape, dtype=torch.float32, device=q.device)
        inputs = (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            q_scale,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )
    if fuse_rope:
        inputs = (*inputs, *test.rope_tables())

    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=test.max_pages_per_req,
        page_size=page_size,
        dim=dim,
        max_seqlen_q=test.max_seqlen_q,
        is_causal=causal,
        cache_dtype=cache_dtype,
        sm_scale=sm_scale,
        softcap=softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        append_kv=append_kv,
        dtype=output_dtype,
        tune=tune,
        fuse_rope=fuse_rope,
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    op.total_q = test.total_q
    op.q_lens = q_lens
    op.cache_lens = cache_lens
    op.max_seqlen_q = test.max_seqlen_q
    bm = ManifestBenchmark(_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
