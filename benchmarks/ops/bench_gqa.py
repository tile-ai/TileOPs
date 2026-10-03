"""Benchmark the TileOPs grouped-query attention ops, one case per manifest call, against FA3, FlashInfer and torch."""

import math
from itertools import accumulate

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
    ManifestBenchmark,
    backward_of,
    manifest_calls,
)
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPagedFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa import (
    GQAPrefillPagedWithKVCacheFwdCall,
    GroupedQueryAttentionBwdCall,
    GroupedQueryAttentionDenseDecodeCall,
    GroupedQueryAttentionDensePrefillCall,
    GroupedQueryAttentionPagedCall,
    GroupedQueryAttentionVarlenCall,
    GroupedQueryAttentionVarlenScaledCall,
)
from workloads.device import run_device


def _fa3_gqa_bwd(workload: GroupedQueryAttentionBwdCall, lse: torch.Tensor):
    """Return FA3's backward alone as a callable, or None if FA3 is not installed.

    ``flash_attn_func`` would run FA3's forward inside the timed call; its backward
    entry takes the forward's output and LSE directly, so only the backward is timed.
    """
    try:
        from flash_attn_interface import _flash_attn_backward
    except ImportError:
        return None

    # The workload's LSE is base 2; FA3 takes the natural logarithm.
    lse_natural = lse * math.log(2.0)

    def baseline_fn(q, k, v, o, grad_output, lse):
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            grad_output, q, k, v, o, lse_natural, dq=dq, dk=dk, dv=dv, is_causal=workload.is_causal
        )
        return dq, dk, dv

    return baseline_fn


def _torch_gqa_bwd(workload, q, k, v):
    """Torch SDPA's backward alone: the forward runs once here, outside the timed call."""
    with torch.enable_grad():
        q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=workload.is_causal,
            enable_gqa=True,
        )
    node = backward_of(out)

    def fn(q, k, v, o, grad_output, lse):
        # Transposing grad_output into SDPA's layout is a view.
        return node(grad_output.transpose(1, 2))

    return fn


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionBwdOp))
def test_gqa_bwd_bench(call) -> None:
    """Backward is timed in training, so the kernels tune."""
    workload = GroupedQueryAttentionBwdCall(call)
    inputs = workload.gen_inputs()

    op = GroupedQueryAttentionBwdOp(**workload.arguments(), tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_bwd(workload, inputs[5])
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_gqa_bwd(workload, *inputs[:3])

    bm.compare(functors, *inputs)
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


def _fa3_gqa_dense_decode(workload: GroupedQueryAttentionDenseDecodeCall):
    """Return the contiguous FA3 decode baseline where its defaults match."""
    if workload.sm_scale != workload.dim**-0.5 or workload.softcap != 0.0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    cache_seqlens = torch.full(
        (workload.batch,), workload.seq_len_kv, dtype=torch.int32, device=run_device()
    )

    def baseline_fn(q, k, v):
        out = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_gqa_dense_decode(
    workload: GroupedQueryAttentionDenseDecodeCall,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """Set up FlashInfer's contiguous or synthetic-paged decode baseline."""
    if workload.sm_scale != workload.dim**-0.5 or workload.softcap != 0.0:
        return None
    if workload.heads // workload.heads_kv > 8:
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
    workload = GroupedQueryAttentionDenseDecodeCall(call)
    inputs = workload.gen_inputs()
    op = GroupedQueryAttentionDenseFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}
    tolerance = reference_tolerance(workload.dtype)

    fa3_fn = _fa3_gqa_dense_decode(workload)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, op, *inputs, **tolerance)
        functors["fa3"] = fa3_fn

    flashinfer_fn = _flashinfer_gqa_dense_decode(workload, *inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, op, *inputs, **tolerance)
        functors[FLASHINFER_TAG] = flashinfer_fn

    if fa3_fn is None and flashinfer_fn is None:
        assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
        functors["torch-ref"] = workload.ref_program

    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", _dense_calls(optional_inputs=True))
def test_gqa_dense_prefill_bench(call) -> None:
    """Dense prefill through the op's optional inputs: FP8 scales, fused RoPE.

    Read against the reference and its compiled form: FA3 and FlashInfer fuse
    neither the per-KV-head dequantization nor a caller-supplied cos/sin table.
    """
    workload = GroupedQueryAttentionDensePrefillCall(call)
    if workload.dtype == torch.float8_e4m3fn and get_sm_version() != 90:
        pytest.skip("native FP8 Dense GQA requires SM90")
    inputs = workload.gen_inputs()
    op = GroupedQueryAttentionDenseFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    # FP8 is held to the tolerance tests/ops/test_gqa.py uses: no
    # per-dtype one covers dequantization against a 16-bit reference.
    assert_matches_reference(
        op,
        workload.ref_program,
        *inputs,
        **(
            {"atol": 8e-2, "rtol": 2e-2}
            if workload.dtype == torch.float8_e4m3fn
            else reference_tolerance(workload.dtype)
        ),
    )

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )


def _varlen_rope(workload: GroupedQueryAttentionVarlenCall, *inputs: torch.Tensor):
    """Rotate packed Q and K as a caller must before a kernel that does not fuse RoPE.

    Returns ``None`` when the row carries no tables. FlashInfer rotates a query and a key
    tensor of one packed length together, while Q and K here differ in length and in head
    count, so each takes its own call with a scratch tensor standing in for the side that
    call leaves alone. Query token ``i`` of a request sits at position ``kv_len - q_len + i``
    and key token ``j`` at position ``j``.
    """
    if workload.pos_encoding_mode != "rope":
        return None
    from flashinfer.rope import apply_rope_with_cos_sin_cache

    q, k, cos, sin = inputs[0], inputs[1], inputs[8], inputs[9]
    device = q.device
    cos_sin = torch.cat([cos.float(), sin.float()], dim=-1).contiguous()
    pos_k = torch.cat([torch.arange(kv, device=device) for kv in workload.seqlens_k]).int()
    is_neox = workload.rope_layout == "neox"
    dim = workload.dim

    if workload.seqlens_q == workload.seqlens_k:
        # Query and key tokens share their positions here, so one call rotates both.
        def rotate(q, k):
            q_rot, k_rot = apply_rope_with_cos_sin_cache(
                pos_k, q.view(q.shape[0], -1), k.view(k.shape[0], -1), dim, cos_sin, is_neox
            )
            return q_rot.view_as(q), k_rot.view_as(k)

        return rotate

    pos_q = torch.cat(
        [
            torch.arange(kv - qn, kv, device=device)
            for qn, kv in zip(workload.seqlens_q, workload.seqlens_k, strict=True)
        ]
    ).int()
    # The entry point rotates a query and a key tensor of one packed length together, and
    # these differ in length and head count, so each takes its own call against a one-head
    # stand-in on the side that call leaves alone.
    scratch_q = torch.empty(k.shape[0], dim, dtype=q.dtype, device=device)
    scratch_k = torch.empty(q.shape[0], dim, dtype=k.dtype, device=device)

    def rotate(q, k):
        q_rot, _ = apply_rope_with_cos_sin_cache(
            pos_q, q.view(q.shape[0], -1), scratch_k, dim, cos_sin, is_neox
        )
        _, k_rot = apply_rope_with_cos_sin_cache(
            pos_k, scratch_q, k.view(k.shape[0], -1), dim, cos_sin, is_neox
        )
        return q_rot.view_as(q), k_rot.view_as(k)

    return rotate


def _fa3_gqa_varlen(
    workload: GroupedQueryAttentionVarlenCall,
    window_size_left: int,
    window_size_right: int,
    rotate=None,
):
    """FlashAttention-3 over the same packed-varlen layout; it has no kernel above head dim 256.

    An FP8 call hands it the same ``[batch, heads_kv]`` descales the op takes, which its
    varlen entry dequantizes with.
    """
    if workload.dim > 256:
        return None
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        return None

    def _run(q, k, v, cu_seqlens_q, cu_seqlens_kv, *optional):
        # The op's optional inputs arrive in signature order: the three FP8 scales,
        # then the rotation tables, which this baseline applies itself.
        q_scale, k_scale, v_scale = (optional + (None,) * 3)[:3]
        if rotate is not None:
            q, k = rotate(q, k)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            workload.max_seqlen_q,
            workload.max_seqlen_kv,
            causal=workload.is_causal,
            window_size=(window_size_left, window_size_right),
            q_descale=q_scale,
            k_descale=k_scale,
            v_descale=v_scale,
        )
        return out[0] if isinstance(out, tuple) else out

    return _run


def _flashinfer_gqa_varlen(
    workload: GroupedQueryAttentionVarlenCall,
    window_size_left: int,
    window_size_right: int,
    *inputs: torch.Tensor,
    rotate=None,
):
    """FlashInfer ragged prefill over the same packed layout; it has no right window."""
    if window_size_right >= 0:
        return None
    q, _k, _v, cu_seqlens_q, cu_seqlens_kv = inputs[:5]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer_op("prefill.BatchPrefillWithRaggedKVCacheWrapper")(
        workspace, kv_layout="NHD"
    )
    wrapper.plan(
        qo_indptr=cu_seqlens_q,
        kv_indptr=cu_seqlens_kv,
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads_kv,
        head_dim_qk=workload.dim,
        causal=workload.is_causal,
        window_left=window_size_left,
        q_data_type=q.dtype,
    )

    def _run(q, k, v, _cu_seqlens_q, _cu_seqlens_kv, *_tables):
        if rotate is not None:
            q, k = rotate(q, k)
        return wrapper.run(q, k, v)

    return _run


def _varlen_calls(*, scaled: bool) -> list:
    """The packed-varlen calls that pass the FP8 scales, or those that do not.

    A rotating 16-bit call stays with the calls that pass neither: its rotation is a
    baseline concern, which ``_varlen_rope`` covers there, not a different reference.
    """
    return [
        param
        for param in manifest_calls(GroupedQueryAttentionVarlenFwdOp)
        if param.values[0].present("q_scale") is scaled
    ]


@pytest.mark.parametrize("call", _varlen_calls(scaled=False))
def test_gqa_varlen_fwd_bench(call) -> None:
    workload = GroupedQueryAttentionVarlenCall(call)
    inputs = workload.gen_inputs()

    op = GroupedQueryAttentionVarlenFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    tolerance = reference_tolerance(workload.dtype)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
    }
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
    rotate = _varlen_rope(workload, *inputs)
    fa3_fn = _fa3_gqa_varlen(workload, workload.wl, workload.wr, rotate)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, functors["torch-ref"], *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    flashinfer_fn = _flashinfer_gqa_varlen(
        workload, workload.wl, workload.wr, *inputs, rotate=rotate
    )
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, functors["torch-ref"], *inputs, **tolerance)
        functors[FLASHINFER_TAG] = flashinfer_fn
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", _varlen_calls(scaled=True))
def test_gqa_varlen_scaled_bench(call) -> None:
    """Packed varlen over FP8 Q/K/V, dequantized by one scale per request and KV head.

    FlashAttention-3 dequantizes the same per-request scales inside its own kernel, so it
    is read against the same reference and timed on the same call. FlashInfer's ragged
    prefill takes no FP8 query, and the per-request reference reads its offsets on the
    host, so neither a FlashInfer nor a torch-compile tag can express the row.
    """
    workload = GroupedQueryAttentionVarlenScaledCall(call)
    if workload.dtype == torch.float8_e4m3fn and get_sm_version() != 90:
        pytest.skip("FP8 packed-varlen GQA requires SM90")
    inputs = workload.gen_inputs()
    op = GroupedQueryAttentionVarlenFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    # FP8 is held to the tolerance tests/ops/test_gqa.py uses: no per-dtype one
    # covers dequantization against a 16-bit reference.
    tolerance = (
        {"atol": 8e-2, "rtol": 2e-2}
        if workload.dtype == torch.float8_e4m3fn
        else reference_tolerance(workload.dtype)
    )
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)

    functors = {"tileops": op, "torch-ref": workload.ref_program}
    fa3_fn = _fa3_gqa_varlen(workload, workload.wl, workload.wr)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, workload.ref_program, *inputs[:8], **tolerance)
        functors["fa3"] = (fa3_fn, inputs[:8])
    bm.compare(functors, *inputs)


def _fa3_gqa_prefill_paged(workload, cache_dtype, fuse_rope, softcap):
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

    shape = (
        workload.batch * workload.max_pages_per_req,
        workload.page_size,
        workload.heads_kv,
        workload.dim,
    )

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
            max_seqlen_q=workload.max_seqlen_q,
            causal=workload.is_causal,
            softcap=float(softcap or 0.0),
        )
        return out[0] if isinstance(out, tuple) else out

    return _run


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp))
def test_gqa_prefill_paged_with_kv_cache_fwd_bench(call) -> None:
    workload = GQAPrefillPagedWithKVCacheFwdCall(call)
    inputs = workload.gen_inputs()
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    cache_dtype = None if workload.cache_dtype == workload.dtype else workload.cache_dtype
    tolerance = reference_tolerance(workload.dtype)
    # Every tag writes k_new and v_new into the slots past cache_seqlens, and no tag's result
    # depends on what those slots held, so every tag shares the pages.
    functors = {"tileops": op, "torch-ref": workload.ref_program}
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
    fa3_fn = _fa3_gqa_prefill_paged(workload, cache_dtype, workload.fuse_rope, workload.softcap)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, workload.ref_program, *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    bm.compare(functors, *inputs)


def _fa3_gqa_paged(workload):
    """FA3 over the same pages, window, scale and softcap, or None when FA3 is not installed.

    It takes the packed queries directly through ``cu_seqlens_q``, so one call serves a
    uniform and a ragged row alike, and it reads a page table of any page size.
    """
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    window = (workload.window_size_left, workload.window_size_right)
    max_seqlen_q = max(workload.q_lens)

    def baseline_fn(q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q, *_unused):
        out = flash_attn_with_kvcache(
            q,
            k_pages,
            v_pages,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=workload.sm_scale,
            causal=workload.is_causal,
            window_size=window,
            softcap=float(workload.softcap or 0.0),
        )
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_gqa_paged_prefill(workload, inputs):
    """FlashInfer's packed-query paged attention, or None where it cannot serve the row.

    Its prefill wrapper takes ragged queries over a paged cache with a left window; it has no
    right window, so a row restricting one drops the tag.
    """
    if workload.window_size_right >= 0:
        return None
    q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q = inputs[:6]
    page_size = workload.page_size
    pages_per_request = ((cache_seqlens + page_size - 1) // page_size).tolist()
    indptr = torch.tensor([0, *accumulate(pages_per_request)], dtype=torch.int32, device=q.device)
    indices = torch.cat([page_table[b, :n] for b, n in enumerate(pages_per_request)])
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer_op("prefill.BatchPrefillWithPagedKVCacheWrapper")(
        workspace, kv_layout="NHD"
    )
    wrapper.plan(
        qo_indptr=cu_seqlens_q,
        paged_kv_indptr=indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=(cache_seqlens - 1) % page_size + 1,
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads_kv,
        head_dim_qk=workload.dim,
        page_size=page_size,
        causal=workload.is_causal,
        sm_scale=workload.sm_scale,
        window_left=workload.window_size_left,
        logits_soft_cap=workload.softcap,
        q_data_type=workload.dtype,
    )

    def run_fn(q, k_pages, v_pages, *_unused):
        return wrapper.run(q, (k_pages, v_pages))

    return run_fn


def _flashinfer_gqa_paged_decode(workload, inputs):
    """FlashInfer paged decode planned with the row's window, scale and softcap, or None where
    it cannot serve the row: its decode kernel takes one query token per request, a
    query-to-KV head ratio up to 8, and no right window."""
    if workload.heads // workload.heads_kv > 8 or set(workload.q_lens) != {1}:
        return None
    if workload.window_size_right >= 0:
        return None
    q, k_pages, v_pages, page_table, cache_seqlens = inputs[:5]
    page_size = workload.page_size
    pages_per_request = ((cache_seqlens + page_size - 1) // page_size).tolist()
    indptr = torch.tensor([0, *accumulate(pages_per_request)], dtype=torch.int32, device=q.device)
    indices = torch.cat([page_table[b, :n] for b, n in enumerate(pages_per_request)])
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer_op("decode.BatchDecodeWithPagedKVCacheWrapper")(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=(cache_seqlens - 1) % page_size + 1,
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads_kv,
        head_dim=workload.dim,
        page_size=page_size,
        q_data_type=workload.dtype,
        sm_scale=workload.sm_scale,
        window_left=workload.window_size_left,
        logits_soft_cap=workload.softcap,
    )

    def run_fn(q, k_pages, v_pages, *_unused):
        return wrapper.run(q, (k_pages, v_pages))

    return run_fn


def _flashinfer_gqa_paged(workload, inputs):
    """FlashInfer's own entry for this row's shape: its decode wrapper where every request
    carries one query token, its packed-query prefill wrapper otherwise."""
    return _flashinfer_gqa_paged_decode(workload, inputs) or _flashinfer_gqa_paged_prefill(
        workload, inputs
    )


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionPagedFwdOp))
def test_gqa_paged_fwd_bench(call) -> None:
    workload = GroupedQueryAttentionPagedCall(call)
    inputs = workload.gen_inputs()
    op = GroupedQueryAttentionPagedFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    tolerance = reference_tolerance(workload.dtype)
    functors = {"tileops": op, "torch-ref": workload.ref_program}
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
    fa3_fn = _fa3_gqa_paged(workload)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, workload.ref_program, *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    flashinfer_fn = _flashinfer_gqa_paged(workload, inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, workload.ref_program, *inputs, **tolerance)
        functors[FLASHINFER_TAG] = flashinfer_fn
    bm.compare(functors, *inputs)
