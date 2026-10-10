"""Benchmark the TileOPs grouped-query attention ops, one case per manifest call, against FA3, FlashInfer and torch."""

from itertools import accumulate

import pytest
import torch
from torch.nn import functional as F

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    backward_of,
    compiled_reference,
    flashinfer_op,
    private_inputs,
)
from tileops.ops import (
    GQABwdOp,
    GQADenseFwdOp,
    GQAPagedFwdOp,
    GQAPrefillPagedWithKVCacheFwdOp,
    GQAVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa.bwd import GQABwdCall
from workloads.attention.gqa.dense import (
    GQADenseDecodeCall,
    GQADensePrefillCall,
)
from workloads.attention.gqa.prefill_paged_kv_append import paged_prefill_result
from workloads.attention.gqa.varlen import (
    GQAVarlenCall,
    GQAVarlenScaledCall,
)
from workloads.device import run_device


def _fa3_gqa_bwd(workload: GQABwdCall, inputs: tuple):
    """Time FA3's backward with its own forward state prepared outside timing."""
    try:
        from flash_attn_interface import _flash_attn_backward, flash_attn_func
    except ImportError:
        return None

    with torch.no_grad():
        saved_out, saved_lse = flash_attn_func(
            *inputs[:3], causal=workload.is_causal, return_attn_probs=True
        )

    def baseline_fn(q, k, v, o, grad_output, lse):
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            grad_output,
            q,
            k,
            v,
            saved_out,
            saved_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            is_causal=workload.is_causal,
        )
        return dq, dk, dv

    return baseline_fn


def _torch_gqa_bwd(workload, q, k, v):
    """Torch SDPA's backward alone: the forward runs once here, outside the timed call."""
    with torch.enable_grad():
        q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
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
        return tuple(grad.transpose(1, 2) for grad in node(grad_output.transpose(1, 2)))

    return fn


@pytest.mark.parametrize("case", bench.cases(GQABwdOp), ids=lambda case: case.id)
def test_gqa_bwd_bench(case) -> None:
    """Backward is timed in training, so the kernels tune."""
    workload = case.workload
    inputs = case.inputs

    op = GQABwdOp(**case.arguments, tune=True)
    implementations = {}

    fa3_fn = _fa3_gqa_bwd(workload, inputs)
    if fa3_fn is not None:
        implementations["fa3"] = fa3_fn
    else:
        implementations["torch-sdpa"] = _torch_gqa_bwd(workload, *inputs[:3])

    bench.Runner(op, case).compare(implementations)
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


def _fa3_gqa_dense_decode(workload: GQADenseDecodeCall):
    """FA3 decode with the same score scale and soft cap."""
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    cache_seqlens = torch.full(
        (workload.batch,), workload.seq_len_kv, dtype=torch.int32, device=run_device()
    )

    def baseline_fn(q, k, v):
        out = flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=cache_seqlens,
            softmax_scale=workload.sm_scale,
            softcap=workload.softcap,
        )
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_gqa_dense_decode(
    workload: GQADenseDecodeCall,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """Set up FlashInfer's contiguous or synthetic-paged decode baseline."""
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
                sm_scale=workload.sm_scale,
                logits_soft_cap=workload.softcap,
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
    workspace = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=heads,
        num_kv_heads=heads_kv,
        head_dim=dim,
        page_size=page_size,
        q_data_type=q.dtype,
        sm_scale=workload.sm_scale,
        logits_soft_cap=workload.softcap,
    )

    def run_fn(q, k, v):
        k_pages = k.view(total_pages, page_size, heads_kv, dim)
        v_pages = v.view(total_pages, page_size, heads_kv, dim)
        return wrapper.run(q.squeeze(1), (k_pages, v_pages)).unsqueeze(1)

    return run_fn


def _gqa_dense_decode_baselines(workload: GQADenseDecodeCall, inputs: tuple, reference) -> dict:
    implementations = {}

    fa3_fn = _fa3_gqa_dense_decode(workload)
    if fa3_fn is not None:
        implementations["fa3"] = fa3_fn

    flashinfer_fn = _flashinfer_gqa_dense_decode(workload, *inputs)
    if flashinfer_fn is not None:
        implementations[FLASHINFER_TAG] = flashinfer_fn

    if fa3_fn is None and flashinfer_fn is None:
        implementations["torch-ref"] = reference
    return implementations


def _gqa_dense_prefill_baselines(workload: GQADensePrefillCall, inputs: tuple, reference) -> dict:
    """Dense prefill implementations, with the FP8 scales and rotary tables the call passes."""
    from flash_attn_interface import flash_attn_func

    rotate = flashinfer_op("rope.apply_rope_with_cos_sin_cache")
    q, k, *_ = inputs
    q_positions = (
        torch.arange(k.shape[1] - q.shape[1], k.shape[1], device=q.device).int().repeat(q.shape[0])
    )
    k_positions = torch.arange(k.shape[1], device=q.device).int().repeat(k.shape[0])
    q_scratch = torch.empty(
        (k_positions.numel(), workload.dim), device=q.device, dtype=workload.out_dtype
    )
    k_scratch = torch.empty(
        (q_positions.numel(), workload.dim), device=q.device, dtype=workload.out_dtype
    )

    def fa3_fn(q, k, v, q_scale, k_scale, v_scale, cos, sin):
        if q_scale is not None:
            q = (
                q.float()
                * q_scale.repeat_interleave(workload.heads // workload.heads_kv, 1)[
                    :, None, :, None
                ]
            ).to(workload.out_dtype)
            k = (k.float() * k_scale[:, None, :, None]).to(workload.out_dtype)
            v = (v.float() * v_scale[:, None, :, None]).to(workload.out_dtype)
        if cos is not None:
            cache = torch.cat((cos, sin), -1).float()
            q_rot, _ = rotate(
                q_positions,
                q.reshape(q_positions.numel(), -1),
                k_scratch,
                workload.dim,
                cache,
                workload.rope_layout == "neox",
            )
            _, k_rot = rotate(
                k_positions,
                q_scratch,
                k.reshape(k_positions.numel(), -1),
                workload.dim,
                cache,
                workload.rope_layout == "neox",
            )
            q, k = (q_rot.reshape_as(q), k_rot.reshape_as(k))
        return flash_attn_func(
            q,
            k,
            v,
            causal=workload.is_causal,
            softmax_scale=workload.sm_scale,
            softcap=workload.softcap,
        )

    return {
        "fa3": fa3_fn,
        "torch-ref": reference,
        TORCH_COMPILE_TAG: compiled_reference(reference),
    }


@pytest.mark.parametrize("case", bench.cases(GQADenseFwdOp), ids=lambda case: case.id)
def test_gqa_dense_fwd_bench(case) -> None:
    """Decode against FA3 and FlashInfer; prefill against FA3 and torch."""
    workload = case.workload
    if isinstance(workload, GQADensePrefillCall):
        if workload.dtype == torch.float8_e4m3fn and get_sm_version() != 90:
            pytest.skip("native FP8 Dense GQA requires SM90")
        implementations = _gqa_dense_prefill_baselines(workload, case.inputs, case.reference)
    else:
        implementations = _gqa_dense_decode_baselines(workload, case.inputs, case.reference)
    op = GQADenseFwdOp(**case.arguments)
    bench.Runner(op, case).compare(implementations)


def _varlen_rope(workload: GQAVarlenCall, *inputs: torch.Tensor):
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
    workload: GQAVarlenCall,
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
    workload: GQAVarlenCall,
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


@pytest.mark.parametrize("case", bench.cases(GQAVarlenFwdOp), ids=lambda case: case.id)
def test_gqa_varlen_fwd_bench(case) -> None:
    """Packed varlen GQA, including FP8 Q/K/V dequantized by one scale per request and KV head.

    FlashAttention-3 dequantizes the same per-request scales inside its own kernel, so a
    scaled call reads it against the same reference and times it on the same call.
    FlashInfer's ragged prefill takes no FP8 query, and the per-request reference reads its
    offsets on the host, so neither a FlashInfer nor a torch-compile tag can express a
    scaled call.
    """
    workload = case.workload
    scaled = isinstance(workload, GQAVarlenScaledCall)
    if scaled and workload.dtype == torch.float8_e4m3fn and get_sm_version() != 90:
        pytest.skip("FP8 packed-varlen GQA requires SM90")
    inputs = case.inputs
    implementations = {"torch-ref": case.reference}
    if scaled:
        fa3_fn = _fa3_gqa_varlen(workload, workload.wl, workload.wr)
        if fa3_fn is not None:
            implementations["fa3"] = bench.Implementation(run=fa3_fn, args=inputs[:8])
    else:
        rotate = _varlen_rope(workload, *inputs)
        fa3_fn = _fa3_gqa_varlen(workload, workload.wl, workload.wr, rotate)
        if fa3_fn is not None:
            implementations["fa3"] = fa3_fn
        flashinfer_fn = _flashinfer_gqa_varlen(
            workload, workload.wl, workload.wr, *inputs, rotate=rotate
        )
        if flashinfer_fn is not None:
            implementations[FLASHINFER_TAG] = flashinfer_fn
    op = GQAVarlenFwdOp(**case.arguments)
    bench.Runner(op, case).compare(implementations)


def _fa3_gqa_prefill_paged(workload, inputs):
    """FA3 paged append, including rotation and FP8 cache adaptation when required."""
    from flash_attn_interface import flash_attn_with_kvcache

    q, _, _, _, _, _, _, _, _, table = inputs
    page = workload.page_size
    shape = (-1, page, workload.heads_kv, workload.dim)
    positions = torch.cat(
        [
            torch.arange(old, old + length, device=q.device)
            for old, length in zip(workload.cache_lens, workload.q_lens, strict=True)
        ]
    ).int()
    requests = torch.repeat_interleave(
        torch.arange(workload.batch, device=q.device),
        torch.tensor(workload.q_lens, device=q.device),
    )
    rows = table[requests, positions.long() // page].long() * page + positions % page
    rotate = flashinfer_op("rope.apply_rope_with_cos_sin_cache")
    rotary_cache = None
    if workload.fuse_rope:
        rotary_dim = workload.rotary_dim or workload.dim
        half = rotary_dim // 2
        frequency = workload.rope_base ** (-torch.arange(half, device=q.device).float() / half)
        angles = (
            torch.arange(
                max(a + b for a, b in zip(workload.cache_lens, workload.q_lens, strict=True)),
                device=q.device,
            )[:, None]
            * frequency
        )
        rotary_cache = torch.cat((angles.cos().to(q.dtype), angles.sin().to(q.dtype)), -1).float()

    def run(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, cu_q, seqlens, table):
        if rotary_cache is not None:
            q_rot, k_rot = rotate(
                positions, q.flatten(1), k_new.flatten(1), workload.dim, rotary_cache, True
            )
            q, k_new = q_rot.reshape_as(q), k_rot.reshape_as(k_new)
        quantized = k_pages.dtype == torch.float8_e4m3fn
        if quantized:
            keys = (k_pages.float() * k_scale[0]).to(q.dtype)
            values = (v_pages.float() * v_scale[0]).to(q.dtype)
        else:
            keys, values = k_pages, v_pages
        out = flash_attn_with_kvcache(
            q,
            keys.view(shape),
            values.view(shape),
            k=k_new,
            v=v_new,
            cache_seqlens=seqlens,
            page_table=table,
            cu_seqlens_q=cu_q,
            cu_seqlens_k_new=cu_q,
            max_seqlen_q=workload.max_seqlen_q,
            causal=workload.is_causal,
            softmax_scale=workload.sm_scale,
            softcap=float(workload.softcap or 0.0),
        )
        if quantized:
            # Persist only newly appended tokens in the caller's quantized cache.
            k_pages[rows] = (k_new.float() / k_scale[0]).to(k_pages.dtype)
            v_pages[rows] = (v_new.float() / v_scale[0]).to(v_pages.dtype)
        return out[0] if isinstance(out, tuple) else out

    return run


@pytest.mark.parametrize(
    "case", bench.cases(GQAPrefillPagedWithKVCacheFwdOp), ids=lambda case: case.id
)
def test_gqa_prefill_paged_with_kv_cache_fwd_bench(case) -> None:
    inputs = case.inputs
    op = GQAPrefillPagedWithKVCacheFwdOp(**case.arguments)
    # Every implementation writes k_new and v_new into k_pages and v_pages.
    implementations = {"torch-ref": private_inputs(case.reference, inputs, 3, 4)}
    fa3_fn = _fa3_gqa_prefill_paged(case.workload, inputs)
    if fa3_fn is not None:
        implementations["fa3"] = private_inputs(
            lambda *args: paged_prefill_result(fa3_fn, *args), inputs, 3, 4
        )
    bench.Runner(op, case).compare(implementations)


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


def _fa3_gqa_paged_rope(workload, inputs):
    """Gather read-only pages, rotate with FlashInfer, then run FA3 packed attention.

    FA3's cache entry rotates newly appended keys, not historical read-only keys. This
    adapter materializes each request's keys, so even a physical page shared at different
    logical positions is rotated correctly. Gathering and rotation are inside timing.
    """
    from flash_attn_interface import flash_attn_varlen_func
    from flashinfer.rope import apply_rope_with_cos_sin_cache

    q, _, _, table = inputs[:4]
    device, dim = q.device, workload.dim
    key_positions = torch.cat([torch.arange(n, device=device) for n in workload.cache_lens])
    requests = torch.repeat_interleave(
        torch.arange(workload.batch, device=device),
        torch.tensor(workload.cache_lens, device=device),
    )
    key_pages = table[requests, key_positions // workload.page_size].long()
    key_offsets = key_positions % workload.page_size
    query_positions = torch.cat(
        [
            torch.arange(kv - qn, kv, device=device)
            for qn, kv in zip(workload.q_lens, workload.cache_lens, strict=True)
        ]
    ).int()
    key_positions = key_positions.int()
    cu_k = torch.tensor([0, *accumulate(workload.cache_lens)], device=device, dtype=torch.int32)
    scratch_q = torch.empty(key_positions.numel(), dim, device=device, dtype=q.dtype)
    scratch_k = torch.empty(q.shape[0], dim, device=device, dtype=q.dtype)

    def run(q, k_pages, v_pages, _table, _lengths, cu_q, _qs, _ks, _vs, cos, sin):
        k = k_pages[key_pages, key_offsets]
        v = v_pages[key_pages, key_offsets]
        rotary_cache = torch.cat((cos, sin), -1).float()
        q_rot, _ = apply_rope_with_cos_sin_cache(
            query_positions,
            q.flatten(1),
            scratch_k,
            dim,
            rotary_cache,
            workload.rope_layout == "neox",
        )
        _, k_rot = apply_rope_with_cos_sin_cache(
            key_positions,
            scratch_q,
            k.flatten(1),
            dim,
            rotary_cache,
            workload.rope_layout == "neox",
        )
        out = flash_attn_varlen_func(
            q_rot.view_as(q),
            k_rot.view_as(k),
            v,
            cu_q,
            cu_k,
            max(workload.q_lens),
            max(workload.cache_lens),
            causal=workload.is_causal,
            softmax_scale=workload.sm_scale,
            window_size=(workload.window_size_left, workload.window_size_right),
            softcap=float(workload.softcap or 0.0),
        )
        return out[0] if isinstance(out, tuple) else out

    return run


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
    it cannot serve the row: tensor-core decode takes one query token per request
    and no right window. Tensor cores also support GQA groups larger than eight."""
    if set(workload.q_lens) != {1}:
        return None
    if workload.window_size_right >= 0:
        return None
    q, k_pages, v_pages, page_table, cache_seqlens = inputs[:5]
    page_size = workload.page_size
    pages_per_request = ((cache_seqlens + page_size - 1) // page_size).tolist()
    indptr = torch.tensor([0, *accumulate(pages_per_request)], dtype=torch.int32, device=q.device)
    indices = torch.cat([page_table[b, :n] for b, n in enumerate(pages_per_request)])
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer_op("decode.BatchDecodeWithPagedKVCacheWrapper")(
        workspace, kv_layout="NHD", use_tensor_cores=True
    )
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


@pytest.mark.parametrize("case", bench.cases(GQAPagedFwdOp), ids=lambda case: case.id)
def test_gqa_paged_fwd_bench(case) -> None:
    workload = case.workload
    inputs = case.inputs
    op = GQAPagedFwdOp(**case.arguments)
    implementations = {"torch-ref": case.reference}
    if workload.pos_encoding_mode == "rope":
        implementations["fa3"] = _fa3_gqa_paged_rope(workload, inputs)
        bench.Runner(op, case).compare(implementations)
        return
    fa3_fn = _fa3_gqa_paged(workload)
    if fa3_fn is not None:
        implementations["fa3"] = fa3_fn
    flashinfer_fn = _flashinfer_gqa_paged(workload, inputs)
    if flashinfer_fn is not None:
        implementations[FLASHINFER_TAG] = flashinfer_fn
    bench.Runner(op, case).compare(implementations)
