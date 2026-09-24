from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    reference_tolerance,
)
from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    backward_of,
    then_dtype,
    workload_params,
)
from benchmarks.ops.attention.workload_args import (
    GQADensePrefillCase,
    gqa_dense_decode_args,
    gqa_dense_prefill_args,
    gqa_prefill_paged_args,
    gqa_qkv_args,
    gqa_varlen_args,
)
from tileops.manifest import load_workloads
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa import (
    GQAPrefillPagedWithKVCacheFwdWorkload,
    GroupedQueryAttentionBwdWorkload,
    GroupedQueryAttentionDenseDecodeWorkload,
    GroupedQueryAttentionDensePrefillWorkload,
    GroupedQueryAttentionVarlenFwdWorkload,
)


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


# GQA backward benchmark parameters (training only).
# Backward is only used during training.
_GQA_BWD_BENCH_PARAMS = workload_params(
    load_workloads(GroupedQueryAttentionBwdOp), then_dtype(gqa_qkv_args, tune=True)
)


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
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_bwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_gqa_bwd(test)

    bm.compare(functors, *inputs)
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


def _fa3_gqa_dense_decode(test: GroupedQueryAttentionDenseDecodeWorkload):
    """Return the contiguous FA3 decode baseline where its defaults match."""
    if test.sm_scale != test.dim**-0.5 or test.softcap != 0.0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    cache_seqlens = torch.full((test.batch,), test.seq_len_kv, dtype=torch.int32, device="cuda")

    def baseline_fn(q, k, v):
        out = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_gqa_dense_decode(
    test: GroupedQueryAttentionDenseDecodeWorkload,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """Set up FlashInfer's contiguous or synthetic-paged decode baseline."""
    if test.sm_scale != test.dim**-0.5 or test.softcap != 0.0:
        return None
    if test.heads // test.heads_kv > 8:
        return None

    batch, _, heads, dim = q.shape
    heads_kv = k.shape[2]
    if batch == 1:
        try:
            from flashinfer.decode import single_decode_with_kv_cache
        except ImportError:
            return None

        def run_fn(q, k, v):
            out = single_decode_with_kv_cache(
                q[0, 0],
                k[0],
                v[0],
                kv_layout="NHD",
                use_tensor_cores=True,
            )
            return out.view(1, 1, heads, dim)

        return run_fn

    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    except ImportError:
        return None

    seq_len_kv = k.shape[1]
    page_size = 256
    if seq_len_kv % page_size != 0:
        return None
    pages_per_seq = seq_len_kv // page_size
    total_pages = batch * pages_per_seq
    indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=q.device) * pages_per_seq
    indices = torch.arange(total_pages, dtype=torch.int32, device=q.device)
    last_page_len = torch.full((batch,), page_size, dtype=torch.int32, device=q.device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=heads,
        num_kv_heads=heads_kv,
        head_dim=dim,
        page_size=page_size,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        k_pages = k.view(total_pages, page_size, heads_kv, dim)
        v_pages = v.view(total_pages, page_size, heads_kv, dim)
        return wrapper.run(q.squeeze(1), (k_pages, v_pages)).unsqueeze(1)

    return run_fn


# A Dense row carrying any of these is a prefill case with FP8 dequantization
# or fused RoPE; a row carrying none is a plain decode.
_DENSE_OPTIONAL_SHAPE_KEYS = (
    "q_scale_shape",
    "k_scale_shape",
    "v_scale_shape",
    "rope_cos_shape",
    "rope_sin_shape",
)


def _dense_rows(*, optional_inputs: bool) -> list[dict]:
    return [
        w
        for w in load_workloads(GroupedQueryAttentionDenseFwdOp)
        if any(key in w for key in _DENSE_OPTIONAL_SHAPE_KEYS) is optional_inputs
    ]


_GQA_DENSE_DECODE_BENCH_PARAMS = workload_params(
    _dense_rows(optional_inputs=False),
    then_dtype(gqa_dense_decode_args),
)

_GQA_DENSE_PREFILL_BENCH_PARAMS = workload_params(
    _dense_rows(optional_inputs=True),
    gqa_dense_prefill_args,
)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, sm_scale, softcap, dtype",
    _GQA_DENSE_DECODE_BENCH_PARAMS,
)
def test_gqa_dense_decode_bench(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_kv: int,
    dim: int,
    sm_scale: float | None,
    softcap: float | None,
    dtype: torch.dtype,
) -> None:
    test = GroupedQueryAttentionDenseDecodeWorkload(
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionDenseFwdOp(sm_scale=sm_scale, softcap=softcap)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_dense_decode(test)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, op, *inputs, **reference_tolerance(dtype))
        functors["fa3"] = fa3_fn

    flashinfer_fn = _flashinfer_gqa_dense_decode(test, *inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, op, *inputs, **reference_tolerance(dtype))
        functors["flashinfer"] = flashinfer_fn

    if fa3_fn is None and flashinfer_fn is None:
        assert_matches_reference(op, test.ref_program, *inputs, **reference_tolerance(dtype))
        functors["torch-ref"] = test.ref_program

    bm.compare(functors, *inputs)


@pytest.mark.parametrize("case", _GQA_DENSE_PREFILL_BENCH_PARAMS)
def test_gqa_dense_prefill_bench(case: GQADensePrefillCase) -> None:
    """Dense prefill through the op's optional inputs: FP8 scales, fused RoPE.

    Read against the reference and its compiled form: FA3 and FlashInfer fuse
    neither the per-KV-head dequantization nor a caller-supplied cos/sin table.
    """
    if case.dtype == torch.float8_e4m3fn and get_sm_version() != 90:
        pytest.skip("native FP8 Dense GQA requires SM90")
    test = GroupedQueryAttentionDensePrefillWorkload(
        case.batch,
        case.seq_len_q,
        case.seq_len_kv,
        case.heads,
        case.heads_kv,
        case.dim,
        case.dtype,
        out_dtype=case.out_dtype,
        is_causal=case.is_causal,
        sm_scale=case.sm_scale,
        softcap=case.softcap,
        rotary_dim=case.rotary_dim,
        rope_layout=case.rope_layout,
    )
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionDenseFwdOp(
        is_causal=case.is_causal,
        sm_scale=case.sm_scale,
        softcap=case.softcap,
        out_dtype=case.out_dtype,
        pos_encoding_mode="rope" if case.rotary_dim is not None else "none",
        rotary_dim=case.rotary_dim,
        rope_layout=case.rope_layout,
    )
    bm = ManifestBenchmark(op, test)
    # FP8 is held to the tolerance tests/ops/attention/test_gqa.py uses: no
    # per-dtype one covers dequantization against a 16-bit reference.
    assert_matches_reference(
        op,
        test.ref_program,
        *inputs,
        **(
            {"atol": 8e-2, "rtol": 2e-2}
            if case.dtype == torch.float8_e4m3fn
            else reference_tolerance(case.dtype)
        ),
    )

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )


def _fa3_gqa_varlen(
    test: GroupedQueryAttentionVarlenFwdWorkload,
    window_size_left: int,
    window_size_right: int,
):
    """FlashAttention-3 over the same packed-varlen layout."""
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        return None

    def _run(q, k, v, cu_seqlens_q, cu_seqlens_kv):
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            test.max_seqlen_q,
            test.max_seqlen_kv,
            causal=test.is_causal,
            window_size=(window_size_left, window_size_right),
        )
        return out[0] if isinstance(out, tuple) else out

    return _run


def _flashinfer_gqa_varlen(
    test: GroupedQueryAttentionVarlenFwdWorkload,
    window_size_left: int,
    window_size_right: int,
    *inputs: torch.Tensor,
):
    """FlashInfer ragged-prefill baseline over the same packed-varlen layout."""
    if window_size_right >= 0:
        return None
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
    except ImportError:
        return None

    q, _k, _v, cu_seqlens_q, cu_seqlens_kv = inputs
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens_q,
        kv_indptr=cu_seqlens_kv,
        num_qo_heads=test.heads,
        num_kv_heads=test.heads_kv,
        head_dim_qk=test.dim,
        causal=test.is_causal,
        window_left=window_size_left,
        q_data_type=q.dtype,
    )

    def _run(q, k, v, _cu_seqlens_q, _cu_seqlens_kv):
        return wrapper.run(q, k, v)

    return _run

_GQA_VARLEN_FWD_BENCH_PARAMS = workload_params(
    load_workloads(GroupedQueryAttentionVarlenFwdOp),
    then_dtype(gqa_varlen_args, tune=False),
)


@pytest.mark.parametrize(
    "batch, q_lens, kv_lens, heads, heads_kv, dim, causal, window_size_left, window_size_right, dtype, tune",
    _GQA_VARLEN_FWD_BENCH_PARAMS,
)
def test_gqa_varlen_fwd_bench(
    batch: int,
    q_lens: list[int],
    kv_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionVarlenFwdWorkload(
        batch,
        q_lens,
        kv_lens,
        heads,
        heads_kv,
        dim,
        causal,
        window_size_left,
        window_size_right,
        dtype,
    )
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionVarlenFwdOp(
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    if window_size_left == -1 and window_size_right == -1:
        op.plan(q_lens, kv_lens, device=inputs[0].device)
    bm = ManifestBenchmark(op, test)

    functors = {
        "tileops": op,
        "torch-ref": test.ref_program,
    }
    assert_matches_reference(op, test.ref_program, *inputs, **reference_tolerance(dtype))
    fa3_fn = _fa3_gqa_varlen(test, window_size_left, window_size_right)
    if fa3_fn is not None:
        assert_matches_reference(
            fa3_fn, functors["torch-ref"], *inputs, **reference_tolerance(dtype)
        )
        functors["fa3"] = fa3_fn
    flashinfer_fn = _flashinfer_gqa_varlen(test, window_size_left, window_size_right, *inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(
            flashinfer_fn, functors["torch-ref"], *inputs, **reference_tolerance(dtype)
        )
        functors["flashinfer"] = flashinfer_fn
    bm.compare(functors, *inputs)


def _fa3_gqa_prefill_paged(test, cache_dtype, fuse_rope, softcap):
    """FlashAttention-3 over the same paged cache, or None where it cannot serve the row.

    It reads the pages, appends the new KV in place and applies the softcap in one launch,
    so it times the work the op does rather than a materialized reference.
    """
    if fuse_rope or cache_dtype is not None:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    shape = (test.batch * test.max_pages_per_req, test.page_size, test.heads_kv, test.dim)

    def _run(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, cu_q, seqlens, table):
        del k_scale, v_scale
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_pages.view(shape),
            v_cache=v_pages.view(shape),
            k=k_new,
            v=v_new,
            cache_seqlens=seqlens,
            page_table=table,
            cu_seqlens_q=cu_q,
            cu_seqlens_k_new=cu_q,
            max_seqlen_q=test.max_seqlen_q,
            causal=test.is_causal,
            softcap=float(softcap or 0.0),
        )
        return out[0] if isinstance(out, tuple) else out

    return _run


def _fp8_paged_cache_inputs(
    test: GQAPrefillPagedWithKVCacheFwdWorkload,
) -> tuple[torch.Tensor, ...]:
    q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table = test.gen_inputs()
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
    )


_GQA_PREFILL_PAGED_WITH_KV_CACHE_FWD_BENCH_PARAMS = workload_params(
    load_workloads(GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp),
    then_dtype(
        gqa_prefill_paged_args,
        tune=False,
    ),
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
        softcap=softcap,
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
        ) = test.gen_inputs()
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
        )

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
        softcap=softcap,
        tune=tune,
        fuse_rope=fuse_rope,
        max_position=test.max_total_len if fuse_rope else None,
        rotary_dim=rotary_dim,
    )
    op.total_q = test.total_q
    op.q_lens = q_lens
    op.cache_lens = cache_lens
    bm = ManifestBenchmark(op, test)
    fa3_fn = _fa3_gqa_prefill_paged(test, cache_dtype, fuse_rope, softcap)
    if fa3_fn is None:
        # FIXME(staged-rollout): this row records no baseline.
        #
        # Broken invariant: every benchmark records >=1 non-tileops baseline.
        # Why: flash_attn_with_kvcache is the only installed implementation that attends
        #   over a paged cache in place, and it takes neither a fused-RoPE row, where the
        #   op builds its own rotary table, nor an fp8 cache, which needs q's dtype.
        # Cleanup: reach those rows too, or a second paged implementation.
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, bm.case_params(), result, tag="tileops")
        return

    assert_matches_reference(op, fa3_fn, *inputs, **reference_tolerance(dtype))
    bm.compare({"tileops": op, "fa3": fa3_fn}, *inputs)
