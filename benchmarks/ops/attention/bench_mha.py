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
from tileops.ops import MultiHeadAttentionBwdOp
from workloads.attention.mha import MhaBwdWorkload


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
