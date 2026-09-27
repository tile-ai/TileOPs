"""Benchmark the TileOPs grouped-query attention ops, one case per manifest call, against FA3, FlashInfer and torch."""

import math

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

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
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.gqa import (
    GQAPrefillPagedWithKVCacheFwdCall,
    GroupedQueryAttentionBwdCall,
    GroupedQueryAttentionDecodePagedCall,
    GroupedQueryAttentionDenseDecodeCall,
    GroupedQueryAttentionDensePrefillCall,
    GroupedQueryAttentionVarlenCall,
)


def _fa3_gqa_bwd(workload: GroupedQueryAttentionBwdCall):
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
        raw = flash_attn_func(q, k, v, causal=workload.is_causal)
        outputs = raw if isinstance(raw, tuple) else (raw,)
        return backward_of(outputs[0])(grad_output, *(None,) * (len(outputs) - 1))

    return baseline_fn


def _torch_gqa_bwd(workload):
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
            is_causal=workload.is_causal,
            enable_gqa=True,
        )
        # Transposing grad_output into SDPA's layout is a view, so the baseline
        # measures SDPA's backward alone.
        return backward_of(out)(grad_output.transpose(1, 2))

    return fn


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionBwdOp))
def test_gqa_bwd_bench(call) -> None:
    """Backward is timed in training, so the kernels tune."""
    workload = GroupedQueryAttentionBwdCall(call)
    inputs = workload.gen_inputs()

    op = GroupedQueryAttentionBwdOp(**workload.arguments(), tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_bwd(workload)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_gqa_bwd(workload)

    bm.compare(functors, *inputs)
    # No FlashInfer baseline for bwd (FlashInfer has no backward API)


def _bench_packed(op_cls, call, cu_kv: str) -> None:
    """Time a packed GQA op against its reference and FA3 over the same layout."""
    workload = GroupedQueryAttentionVarlenCall(call, cu_kv)
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    tolerance = reference_tolerance(workload.dtype)
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
    functors = {"tileops": op, "torch-ref": workload.ref_program}
    fa3_fn = _fa3_gqa_varlen(workload, workload.wl, workload.wr)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, workload.ref_program, *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionPrefillVarlenFwdOp))
def test_gqa_prefill_varlen_fwd_bench(call) -> None:
    _bench_packed(GroupedQueryAttentionPrefillVarlenFwdOp, call, "cu_seqlens_kv")


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionSlidingWindowVarlenFwdOp))
def test_gqa_sliding_window_varlen_fwd_bench(call) -> None:
    _bench_packed(GroupedQueryAttentionSlidingWindowVarlenFwdOp, call, "cu_seqlens_k")


def _fa3_gqa_dense_decode(workload: GroupedQueryAttentionDenseDecodeCall):
    """Return the contiguous FA3 decode baseline where its defaults match."""
    if workload.sm_scale != workload.dim**-0.5 or workload.softcap != 0.0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    cache_seqlens = torch.full(
        (workload.batch,), workload.seq_len_kv, dtype=torch.int32, device="cuda"
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


def _fa3_gqa_varlen(
    workload: GroupedQueryAttentionVarlenCall,
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
            workload.max_seqlen_q,
            workload.max_seqlen_kv,
            causal=workload.is_causal,
            window_size=(window_size_left, window_size_right),
        )
        return out[0] if isinstance(out, tuple) else out

    return _run


def _flashinfer_gqa_varlen(
    workload: GroupedQueryAttentionVarlenCall,
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
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads_kv,
        head_dim_qk=workload.dim,
        causal=workload.is_causal,
        window_left=window_size_left,
        q_data_type=q.dtype,
    )

    def _run(q, k, v, _cu_seqlens_q, _cu_seqlens_kv):
        return wrapper.run(q, k, v)

    return _run


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionVarlenFwdOp))
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
    fa3_fn = _fa3_gqa_varlen(workload, workload.wl, workload.wr)
    if fa3_fn is not None:
        assert_matches_reference(fa3_fn, functors["torch-ref"], *inputs, **tolerance)
        functors["fa3"] = fa3_fn
    flashinfer_fn = _flashinfer_gqa_varlen(workload, workload.wl, workload.wr, *inputs)
    if flashinfer_fn is not None:
        assert_matches_reference(flashinfer_fn, functors["torch-ref"], *inputs, **tolerance)
        functors[FLASHINFER_TAG] = flashinfer_fn
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
    fa3_fn = _fa3_gqa_prefill_paged(workload, cache_dtype, workload.fuse_rope, workload.softcap)
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

    assert_matches_reference(op, fa3_fn, *inputs, **reference_tolerance(workload.dtype))
    bm.compare({"tileops": op, "fa3": fa3_fn}, *inputs)


class GroupedQueryAttentionDecodePagedTestBaseline(GroupedQueryAttentionDecodePagedCall):
    """Times SDPA on the reassembled pages, not an explicit softmax.

    ``sdpa_kernel(MATH)`` replaces the test reference's explicit
    matmul/softcap/softmax chain, so the ratio is against torch's own attention.
    """

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then GQA (expand to heads) + SDPA."""
        batch, _, dim = q.shape
        seqlen_kv, _, _ = k.shape
        kv_group_num = self.heads // self.heads_kv
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b : i_b + 1, :, :]
            k_logical = torch.zeros(seqlen_kv, self.heads_kv, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, self.heads_kv, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(real_seqlen_kv[i_b].item() / self.page_size)
            for i_paged in range(num_pages):
                start_pos = block_table[i_b, i_paged].item() * self.page_size
                end_pos = min(start_pos + self.page_size, seqlen_kv)
                page_len = end_pos - start_pos
                k_logical[i_paged * self.page_size : i_paged * self.page_size + page_len, :, :] = k[
                    start_pos:end_pos, :, :
                ]
                v_logical[i_paged * self.page_size : i_paged * self.page_size + page_len, :, :] = v[
                    start_pos:end_pos, :, :
                ]
            k_logical = k_logical[: real_seqlen_kv[i_b].item(), :, :]
            v_logical = v_logical[: real_seqlen_kv[i_b].item(), :, :]
            group_id = torch.arange(self.heads, dtype=torch.long, device=q.device) // kv_group_num
            k_bhsd = k_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
            v_bhsd = v_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
            q_bhsd = q_b.unsqueeze(2)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.squeeze(2)
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)


def _fa3_gqa_decode_paged(workload, k, v):
    """Set up FA3 paged decode. Returns callable or None.

    FA3 requires page_block_size to be a multiple of 256.
    """
    if workload.page_size % 256 != 0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    num_pages = k.shape[0] // workload.page_size
    k_paged = k.view(num_pages, workload.page_size, workload.heads_kv, workload.dim)
    v_paged = v.view(num_pages, workload.page_size, workload.heads_kv, workload.dim)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim) — add seq dim for flash_attn
        out = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_paged,
            v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            page_table=block_table.int(),
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.squeeze(1)

    return baseline_fn


def _flashinfer_gqa_decode_paged(workload, q, k, v, real_seqlen_kv, block_table):
    """Set up FlashInfer paged decode wrapper. Returns callable or None.

    FlashInfer decode kernel supports group_size (Q/KV head ratio) up to 8.
    """
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    except ImportError:
        return None

    if workload.heads // workload.heads_kv > 8:
        return None  # FlashInfer decode kernel does not support group_size > 8

    batch = q.shape[0]
    num_pages = k.shape[0] // workload.page_size
    k_paged = k.view(num_pages, workload.page_size, workload.heads_kv, workload.dim)
    v_paged = v.view(num_pages, workload.page_size, workload.heads_kv, workload.dim)
    kv_data = (k_paged, v_paged)

    pages_per_batch = (real_seqlen_kv.int() + workload.page_size - 1) // workload.page_size
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    indptr[1:] = torch.cumsum(pages_per_batch, dim=0)

    indices_list = []
    for b in range(batch):
        n = pages_per_batch[b].item()
        indices_list.append(block_table[b, :n])
    indices = torch.cat(indices_list)

    last_page_len = (real_seqlen_kv.int() - 1) % workload.page_size + 1

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads_kv,
        head_dim=workload.dim,
        page_size=workload.page_size,
        q_data_type=workload.dtype,
    )

    def run_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim)
        return wrapper.run(q, kv_data)

    return run_fn


@pytest.mark.parametrize("call", manifest_calls(GroupedQueryAttentionDecodePagedWithKVCacheFwdOp))
def test_gqa_decode_paged_bench(call) -> None:
    workload = GroupedQueryAttentionDecodePagedTestBaseline(call)
    inputs = workload.gen_inputs()
    q, k, v, real_seqlen_kv, block_table = inputs

    op = GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    fa3_fn = _fa3_gqa_decode_paged(workload, k, v)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    fi_fn = _flashinfer_gqa_decode_paged(workload, *inputs)
    if fi_fn is not None:
        functors[FLASHINFER_TAG] = fi_fn

    if fa3_fn is None and fi_fn is None:
        functors["torch-ref"] = workload.ref_program

    bm.compare(functors, *inputs)
