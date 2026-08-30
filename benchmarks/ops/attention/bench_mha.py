import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    backward_of,
    then_dtype,
    workload_params,
)
from benchmarks.ops.attention.workload_args import mha_qkv_args
from tileops.manifest import load_workloads
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from workloads.attention.mha import (
    MhaBwdWorkload,
    MhaFwdWorkload,
)


def _fa3_mha_fwd(test: MhaFwdWorkload):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(q, k, v, causal=test.is_causal)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _fa3_mha_bwd(test: MhaBwdWorkload):
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


def _flashinfer_mha_fwd(test: MhaFwdWorkload, q, k, v):
    """Set up FlashInfer batched prefill wrapper. Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
    except ImportError:
        return None

    B, S, H, D = q.shape
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * S

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens,
        kv_indptr=cu_seqlens,
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=D,
        causal=test.is_causal,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D),
            k.reshape(-1, H, D),
            v.reshape(-1, H, D),
        ).reshape(B, S, H, D)

    return run_fn


def _torch_mha_fwd(test):
    """Torch SDPA forward baseline."""

    def fn(q, k, v):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=test.is_causal
        )
        return out.transpose(1, 2)

    return fn


def _torch_mha_bwd(test):
    """Torch SDPA backward baseline (includes forward recompute)."""

    @torch.enable_grad()
    def fn(q, k, v, o, grad_output, lse):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=test.is_causal
        )
        # Transposing grad_output into SDPA's layout is a view, so the baseline
        # measures SDPA's backward alone.
        return backward_of(out)(grad_output.transpose(1, 2))

    return fn


_MHA_FWD_BENCH_PARAMS = workload_params(
    load_workloads(MultiHeadAttentionFwdOp), then_dtype(mha_qkv_args, tune=True)
)


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", _MHA_FWD_BENCH_PARAMS)
def test_mha_fwd_bench(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype, tune: bool
) -> None:
    test = MhaFwdWorkload(batch, heads, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, tune=tune)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_mha_fwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    fi_fn = _flashinfer_mha_fwd(test, *inputs)
    if fi_fn is not None:
        functors["flashinfer"] = fi_fn

    if fa3_fn is None and fi_fn is None:
        functors["torch-sdpa"] = _torch_mha_fwd(test)

    bm.compare(functors, *inputs)


_MHA_BWD_BENCH_PARAMS = workload_params(
    load_workloads(MultiHeadAttentionBwdOp), then_dtype(mha_qkv_args, tune=True)
)


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", _MHA_BWD_BENCH_PARAMS)
def test_mha_bwd_bench(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype, tune: bool
) -> None:
    test = MhaBwdWorkload(batch, heads, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, tune=tune)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_mha_bwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_mha_bwd(test)

    bm.compare(functors, *inputs)
