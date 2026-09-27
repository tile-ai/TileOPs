"""Benchmarks for multi-head attention backward and paged decode, one case per manifest call, against FA3, FlashInfer and torch."""

import pytest
import torch
from torch.nn import functional as F

from benchmarks.baselines import FLASHINFER_TAG
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from workloads.mha import MhaBwdCall, MhaDecodePagedCall


def _fa3_mha_bwd(workload: MhaBwdCall):
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


def _torch_mha_bwd(workload):
    """Torch SDPA backward baseline (includes forward recompute)."""

    @torch.enable_grad()
    def fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=workload.is_causal
        )
        # Transposing grad_output into SDPA's layout is a view, so the baseline
        # measures SDPA's backward alone.
        return backward_of(out)(grad_output.transpose(1, 2))

    return fn


@pytest.mark.parametrize("call", manifest_calls(MultiHeadAttentionBwdOp))
def test_mha_bwd_bench(call) -> None:
    """Backward is timed in training, so the kernels tune."""
    workload = MhaBwdCall(call)
    inputs = workload.gen_inputs()

    op = MultiHeadAttentionBwdOp(**workload.arguments(), tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    fa3_fn = _fa3_mha_bwd(workload)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_mha_bwd(workload)

    bm.compare(functors, *inputs)


def _fa3_mha_decode_paged(workload, k, v):
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
    k_paged = k.view(num_pages, workload.page_size, workload.heads, workload.dim)
    v_paged = v.view(num_pages, workload.page_size, workload.heads, workload.dim)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        out = flash_attn_with_kvcache(
            q, k_paged, v_paged, cache_seqlens=real_seqlen_kv.int(), page_table=block_table.int()
        )
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_mha_decode_paged(workload, q, k, v, real_seqlen_kv, block_table):
    """Set up FlashInfer paged decode wrapper. Returns callable or None."""
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    except ImportError:
        return None

    batch = q.shape[0]
    num_pages = k.shape[0] // workload.page_size
    k_paged = k.view(num_pages, workload.page_size, workload.heads, workload.dim)
    v_paged = v.view(num_pages, workload.page_size, workload.heads, workload.dim)
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
        num_kv_heads=workload.heads,
        head_dim=workload.dim,
        page_size=workload.page_size,
        q_data_type=workload.dtype,
    )

    def run_fn(q, k, v, real_seqlen_kv, block_table):
        return wrapper.run(q.squeeze(1), kv_data).unsqueeze(1)

    return run_fn


@pytest.mark.parametrize("call", manifest_calls(MultiHeadAttentionDecodePagedWithKVCacheFwdOp))
def test_mha_decode_paged_bench(call) -> None:
    workload = MhaDecodePagedCall(call)
    inputs = workload.gen_inputs()
    q, k, v, real_seqlen_kv, block_table = inputs

    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(**workload.arguments(), tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    fa3_fn = _fa3_mha_decode_paged(workload, k, v)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    fi_fn = _flashinfer_mha_decode_paged(workload, *inputs)
    if fi_fn is not None:
        functors[FLASHINFER_TAG] = fi_fn

    if fa3_fn is None and fi_fn is None:
        functors["torch-ref"] = workload.ref_program

    bm.compare(functors, *inputs)
