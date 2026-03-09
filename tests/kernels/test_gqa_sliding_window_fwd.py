"""Tests for GqaSlidingWindowFwdKernel against FA3 ground truth."""
import pytest
import torch

flash_attn = pytest.importorskip("flash_attn",
                                  reason="flash_attn (FA3) not installed")

from tileops.kernels.deepseek_nsa import GqaSlidingWindowFwdKernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")


def fa3_ref(q, k, v, causal, window_left, window_right):
    """FA3 reference: q/k/v shape [B, S, H, D] or [B, S, H_kv, D]."""
    window_size = (window_left, window_right)
    o = flash_attn.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    return o


CONFIGS = [
    # (batch, seq, heads, heads_kv, dim, is_causal, wl, wr)
    (2, 512,  8,  2, 64, True,  -1,  -1),   # causal full
    (2, 512,  8,  2, 64, True,  128, -1),   # causal + left window
    (2, 512,  8,  2, 64, False, -1,  -1),   # bidirectional full
    (2, 512,  8,  2, 64, False, 64,  64),   # bidirectional window
    (2, 128,  8,  1, 128, True,  1,  -1),   # tiny left window (edge case)
]


@pytest.mark.parametrize("batch,seq,heads,heads_kv,dim,causal,wl,wr", CONFIGS)
def test_gqa_sliding_window_fwd_simple(batch, seq, heads, heads_kv, dim, causal, wl, wr):
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(batch, seq, heads,    dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq, heads_kv, dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq, heads_kv, dim, dtype=dtype, device=device) * 0.1

    kernel = GqaSlidingWindowFwdKernel(
        batch=batch, heads=heads, heads_kv=heads_kv, seq_len=seq, dim=dim,
        is_causal=causal, window_size_left=wl, window_size_right=wr,
        dtype=dtype)

    o_ref = fa3_ref(q, k, v, causal, wl, wr)
    o_out, _ = kernel.forward(q, k, v)

    torch.testing.assert_close(o_out, o_ref, atol=1e-2, rtol=1e-2)
