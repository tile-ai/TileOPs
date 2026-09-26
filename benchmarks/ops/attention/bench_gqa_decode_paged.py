"""Benchmark the decode subset of the unified paged GQA interface."""

import pytest
import torch

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, then_dtype, workload_params
from benchmarks.ops.attention.workload_args import gqa_decode_paged_args
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionPagedFwdOp
from workloads.attention.gqa import GroupedQueryAttentionPagedDecodeWorkload


def _checked_baseline(name, fn, reference, inputs, dtype):
    """Return a runnable baseline only after it proves semantic parity."""
    if fn is None:
        return None
    try:
        assert_matches_reference(fn, reference, *inputs, **reference_tolerance(dtype))
    except Exception as exc:
        print(f"  [skip] {name}: {str(exc).splitlines()[0]}")
        return None
    return fn


def _fa3_gqa_decode_paged(test):
    """FA3 over the same pages and score semantics, when this build supports it."""
    if test.page_size % 256 != 0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None

    def run(
        q,
        k_pages,
        v_pages,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale,
        k_scale,
        v_scale,
        rope_cos,
        rope_sin,
    ):
        del cu_seqlens_q, q_scale, k_scale, v_scale, rope_cos, rope_sin
        out = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_pages,
            v_pages,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            softmax_scale=test.sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=test.softcap,
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.squeeze(1)

    return run


def _flashinfer_gqa_decode_paged(test, inputs):
    """FlashInfer decode planned with the same scale, cap, dtype and page mapping."""
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    except ImportError:
        return None
    if test.heads // test.heads_kv > 8:
        return None

    q, k_pages, v_pages, page_table, cache_seqlens, *_ = inputs
    try:
        pages_per_request = (cache_seqlens + test.page_size - 1) // test.page_size
        indptr = torch.zeros(test.batch + 1, dtype=torch.int32, device=q.device)
        indptr[1:] = torch.cumsum(pages_per_request, dim=0)
        indices = torch.cat(
            [page_table[b, : int(pages_per_request[b].item())] for b in range(test.batch)]
        )
        last_page_len = (cache_seqlens - 1) % test.page_size + 1
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=test.heads,
            num_kv_heads=test.heads_kv,
            head_dim=test.dim,
            page_size=test.page_size,
            q_data_type=test.dtype,
            kv_data_type=test.dtype,
            o_data_type=test.dtype,
            sm_scale=test.sm_scale,
            logits_soft_cap=test.softcap,
            window_left=-1,
        )
    except Exception as exc:
        print(f"  [skip] flashinfer setup: {str(exc).splitlines()[0]}")
        return None

    def run(q, k_pages, v_pages, *_metadata):
        return wrapper.run(q, (k_pages, v_pages))

    return run


_GQA_PAGED_BENCH_PARAMS = workload_params(
    load_workloads(GroupedQueryAttentionPagedFwdOp),
    then_dtype(gqa_decode_paged_args),
)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seqlen_kv, dim, page_size, sm_scale, softcap, dtype",
    _GQA_PAGED_BENCH_PARAMS,
)
def test_gqa_decode_paged_bench(
    batch: int,
    heads: int,
    heads_kv: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    sm_scale: float | None,
    softcap: float | None,
    dtype: torch.dtype,
) -> None:
    test = GroupedQueryAttentionPagedDecodeWorkload(
        batch,
        heads,
        heads_kv,
        seqlen_kv,
        dim,
        page_size,
        dtype,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionPagedFwdOp(
        sm_scale=sm_scale,
        softcap=softcap,
    )
    bm = ManifestBenchmark(op, test)

    assert_matches_reference(op, test.ref_program, *inputs, **reference_tolerance(dtype))
    functors = {"tileops": op, "torch-ref": test.ref_program}
    fa3 = _checked_baseline("fa3", _fa3_gqa_decode_paged(test), test.ref_program, inputs, dtype)
    if fa3 is not None:
        functors["fa3"] = fa3
    flashinfer = _checked_baseline(
        "flashinfer",
        _flashinfer_gqa_decode_paged(test, inputs),
        test.ref_program,
        inputs,
        dtype,
    )
    if flashinfer is not None:
        functors["flashinfer"] = flashinfer
    bm.compare(functors, *inputs)
