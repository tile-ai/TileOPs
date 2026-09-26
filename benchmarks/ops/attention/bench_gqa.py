import pytest
import torch
from torch.nn import functional as F

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flashinfer_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    backward_of,
    manifest_calls,
)
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa import (
    GQAPrefillPagedWithKVCacheFwdCall,
    GroupedQueryAttentionBwdCall,
    GroupedQueryAttentionDenseDecodeCall,
    GroupedQueryAttentionDensePrefillCall,
    GroupedQueryAttentionVarlenCall,
)


def _fa3_gqa_bwd(test: GroupedQueryAttentionBwdCall):
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


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionBwdOp))
def test_gqa_bwd_bench(call) -> None:
    """Backward is timed in training, so the kernels tune."""
    test = GroupedQueryAttentionBwdCall(call)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionBwdOp(**test.arguments(), tune=True)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_bwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_gqa_bwd(test)

    bm.compare(functors, *inputs)
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


def _bench_packed(op_cls, call, cu_kv: str) -> None:
    """Time a packed GQA op against its reference and FA3 over the same layout."""
    test = GroupedQueryAttentionVarlenCall(call, cu_kv)
    inputs = test.gen_inputs()
    op = op_cls(**test.arguments())
    bm = ManifestBenchmark(op, test)
    tolerance = reference_tolerance(test.dtype)
    assert_matches_reference(op, test.ref_program, *inputs, **tolerance)
    functors = {"tileops": op, "torch-ref": test.ref_program}
    fa3_fn = _fa3_gqa_varlen(test, test.wl, test.wr)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, test.ref_program, *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionPrefillVarlenFwdOp))
def test_gqa_prefill_varlen_fwd_bench(call) -> None:
    _bench_packed(GroupedQueryAttentionPrefillVarlenFwdOp, call, "cu_seqlens_kv")


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionSlidingWindowVarlenFwdOp))
def test_gqa_sliding_window_varlen_fwd_bench(call) -> None:
    _bench_packed(GroupedQueryAttentionSlidingWindowVarlenFwdOp, call, "cu_seqlens_k")


def _fa3_gqa_dense_decode(test: GroupedQueryAttentionDenseDecodeCall):
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
    test: GroupedQueryAttentionDenseDecodeCall,
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


def _dense_calls(*, optional_inputs: bool) -> list:
    """The Dense calls that pass FP8 scales or RoPE tables, or those that pass neither."""
    return [
        param
        for param in manifest_calls(GroupedQueryAttentionDenseFwdOp)
        if (param.values[0].present("q_scale") or param.values[0].present("rope_cos"))
        is optional_inputs
    ]


@pytest.mark.parametrize("call", _dense_calls(optional_inputs=False))
def test_gqa_dense_decode_bench(call) -> None:
    test = GroupedQueryAttentionDenseDecodeCall(call)
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionDenseFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}
    tolerance = reference_tolerance(test.dtype)

    fa3_fn = _fa3_gqa_dense_decode(test)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, op, *inputs, **tolerance)
        functors["fa3"] = fa3_fn

    flashinfer_fn = _flashinfer_gqa_dense_decode(test, *inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, op, *inputs, **tolerance)
        functors["flashinfer"] = flashinfer_fn

    if fa3_fn is None and flashinfer_fn is None:
        assert_matches_reference(op, test.ref_program, *inputs, **tolerance)
        functors["torch-ref"] = test.ref_program

    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", _dense_calls(optional_inputs=True))
def test_gqa_dense_prefill_bench(call) -> None:
    """Dense prefill through the op's optional inputs: FP8 scales, fused RoPE.

    Read against the reference and its compiled form: FA3 and FlashInfer fuse
    neither the per-KV-head dequantization nor a caller-supplied cos/sin table.
    """
    test = GroupedQueryAttentionDensePrefillCall(call)
    if test.dtype == torch.float8_e4m3fn and get_sm_version() != 90:
        pytest.skip("native FP8 Dense GQA requires SM90")
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionDenseFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    # FP8 is held to the tolerance tests/ops/attention/test_gqa.py uses: no
    # per-dtype one covers dequantization against a 16-bit reference.
    assert_matches_reference(
        op,
        test.ref_program,
        *inputs,
        **(
            {"atol": 8e-2, "rtol": 2e-2}
            if test.dtype == torch.float8_e4m3fn
            else reference_tolerance(test.dtype)
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
    test: GroupedQueryAttentionVarlenCall,
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
    test: GroupedQueryAttentionVarlenCall,
    window_size_left: int,
    window_size_right: int,
    *inputs: torch.Tensor,
):
    """FlashInfer ragged prefill over the same packed layout; it has no right window."""
    if window_size_right >= 0:
        return None
    q, _k, _v, cu_seqlens_q, cu_seqlens_kv = inputs
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer_op("prefill.BatchPrefillWithRaggedKVCacheWrapper")(
        workspace, kv_layout="NHD"
    )
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


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionVarlenFwdOp))
def test_gqa_varlen_fwd_bench(call) -> None:
    test = GroupedQueryAttentionVarlenCall(call)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionVarlenFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    tolerance = reference_tolerance(test.dtype)

    functors = {
        "tileops": op,
        "torch-ref": test.ref_program,
    }
    assert_matches_reference(op, test.ref_program, *inputs, **tolerance)
    fa3_fn = _fa3_gqa_varlen(test, test.wl, test.wr)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, functors["torch-ref"], *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    flashinfer_fn = _flashinfer_gqa_varlen(test, test.wl, test.wr, *inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, functors["torch-ref"], *inputs, **tolerance)
        functors[FLASHINFER_TAG] = flashinfer_fn
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


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp))
def test_gqa_prefill_paged_with_kv_cache_fwd_bench(call) -> None:
    test = GQAPrefillPagedWithKVCacheFwdCall(call)
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    cache_dtype = None if test.cache_dtype == test.dtype else test.cache_dtype
    fa3_fn = _fa3_gqa_prefill_paged(test, cache_dtype, test.fuse_rope, test.softcap)
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

    assert_matches_reference(op, fa3_fn, *inputs, **reference_tolerance(test.dtype))
    bm.compare({"tileops": op, "fa3": fa3_fn}, *inputs)
