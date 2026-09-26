import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops import MultiHeadAttentionBwdOp
from workloads.attention.mha import MhaBwdCall


def _fa3_mha_bwd(test: MhaBwdCall):
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


@pytest.mark.parametrize("call", manifest_calls(MultiHeadAttentionBwdOp))
def test_mha_bwd_bench(call) -> None:
    """Backward is timed in training, so the kernels tune."""
    test = MhaBwdCall(call)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionBwdOp(**test.arguments(), tune=True)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    fa3_fn = _fa3_mha_bwd(test)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn
    else:
        functors["torch-sdpa"] = _torch_mha_bwd(test)

    bm.compare(functors, *inputs)
