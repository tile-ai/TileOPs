"""One op instance, two element types.

Dtype used to be fixed at construction, so an instance served exactly one
element type and per-instance state could safely hold anything derived from
it. Now the tensors decide, and the family caches are hand-written per
family — this covers the invariant they all have to satisfy.

The rest of the suite parametrizes dtype but builds a fresh op per case, so
it never exercises a second dtype through an instance that already has an
entry.
"""

import pytest
import torch

from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.reduction.reduce import SumFwdOp

_DTYPES = (torch.float16, torch.bfloat16)


def _assert_two_entries(op, cache_name="_kernel_cache"):
    cache = getattr(op, cache_name)
    assert len(cache) == 2, f"expected one entry per dtype, got {list(cache)}"
    kernels = list(cache.values())
    assert kernels[0] is not kernels[1], "both dtypes reused one kernel"


@pytest.mark.smoke
def test_reduction_serves_two_dtypes_from_one_instance():
    op = SumFwdOp(dim=-1)
    for dtype in _DTYPES:
        x = torch.randn(8, 128, dtype=dtype, device="cuda")
        y = op(x)
        assert y.dtype == dtype
        torch.testing.assert_close(y, x.sum(-1), atol=2e-2, rtol=2e-2)
    _assert_two_entries(op)


@pytest.mark.smoke
def test_rms_norm_serves_two_dtypes_from_one_instance():
    n = 256
    op = RMSNormFwdOp(normalized_shape=(n,))
    for dtype in _DTYPES:
        x = torch.randn(16, n, dtype=dtype, device="cuda")
        w = torch.randn(n, dtype=dtype, device="cuda")
        y = op(x, w)
        assert y.dtype == dtype
    _assert_two_entries(op)


@pytest.mark.smoke
def test_layer_norm_keys_on_both_shape_and_dtype():
    """A second dtype at the same shape must not reuse the first entry."""
    n = 256
    op = LayerNormFwdOp(normalized_shape=(n,))
    for dtype in _DTYPES:
        x = torch.randn(16, n, dtype=dtype, device="cuda")
        w = torch.randn(n, dtype=dtype, device="cuda")
        b = torch.randn(n, dtype=dtype, device="cuda")
        assert op(x, w, b).dtype == dtype
    assert set(op._kernel_cache) == {(16, dt) for dt in _DTYPES}


@pytest.mark.smoke
def test_roofline_reports_the_most_recent_forward():
    """`self.dtype` is most-recent-forward, so bytes follow the last call."""
    op = SumFwdOp(dim=-1)
    op(torch.randn(8, 128, dtype=torch.float32, device="cuda"))
    _, bytes_fp32 = op.eval_roofline()
    op(torch.randn(8, 128, dtype=torch.float16, device="cuda"))
    _, bytes_fp16 = op.eval_roofline()
    assert bytes_fp16 < bytes_fp32


@pytest.mark.smoke
def test_attention_decode_reselects_the_kernel_per_dtype():
    """The decode ops pick a kernel *slot* from the element type.

    float16 at batch=1 takes the warp-specialized kernel and bfloat16 falls
    back, so one instance must hold two different kernel classes — the case
    that used to need two ops with two constructor dtypes.
    """
    from tileops.ops.attention.gqa import (
        GroupedQueryAttentionDecodeWithKVCacheFwdOp,
    )

    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(1, 32, 4, 8192, 128)
    fp16 = op._get_kernel(torch.float16)
    bf16 = op._get_kernel(torch.bfloat16)
    assert fp16.__class__.__name__ == "GQADecodeBs1Kernel"
    assert bf16.__class__.__name__ == "GQADecodeKernel"
    _assert_two_entries(op)


@pytest.mark.smoke
def test_attention_square_prefill_reselects_the_kernel_per_dtype():
    """The BSHD square wrapper resolves its prefill kernel from the tensors.

    Causal dim-128 takes the warp-specialized dense slot; the element type used
    to reach that slot through a constructor dtype read by default_kernel_map.
    """
    from tileops.ops.attention.gqa import GroupedQueryAttentionFwdOp

    batch, heads, heads_kv, seq_len, dim = 1, 8, 2, 256, 128
    op = GroupedQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, is_causal=True)
    for dtype in _DTYPES:
        q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seq_len, heads_kv, dim, dtype=dtype, device="cuda")
        v = torch.randn_like(k)
        assert op(q, k, v).dtype == dtype
        kernel = op._get_kernel(dtype)
        assert kernel.__class__.__name__ == "GQAPrefillFwdWsPersistentCausalKernel"
        assert kernel.dtype == dtype
    _assert_two_entries(op)


@pytest.mark.smoke
def test_attention_mha_serves_two_dtypes_from_one_instance():
    """MHA delegates to the GQA wrapper and shares its per-dtype kernel cache."""
    from tileops.ops.attention.mha import MultiHeadAttentionFwdOp

    batch, heads, seq_len, dim = 1, 8, 256, 64
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, is_causal=False)
    for dtype in _DTYPES:
        q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        assert op(q, k, v).dtype == dtype
        assert op._get_kernel(dtype).__class__.__name__ == "GQAPrefillFwdKernel"
    _assert_two_entries(op)


@pytest.mark.smoke
def test_moe_unpermute_serves_two_dtypes_from_one_instance():
    from tileops.ops.moe.routed_expert.unpermute import MoeUnpermuteFwdOp

    total_tokens, top_k, hidden = 16, 2, 128
    numel = total_tokens * top_k
    op = MoeUnpermuteFwdOp(total_tokens, top_k, hidden, padded_batch_sum=numel)
    fwd_idx = torch.arange(numel, device="cuda", dtype=torch.int32)
    for dtype in _DTYPES:
        mm2_pad = torch.randn(numel, hidden, dtype=dtype, device="cuda")
        weights = torch.rand(total_tokens, top_k, dtype=torch.float32, device="cuda")
        assert op(mm2_pad, fwd_idx, weights).dtype == dtype
    _assert_two_entries(op)


@pytest.mark.smoke
def test_cb_producer_serves_two_dtypes_from_one_instance():
    from tileops.ops.cb_producer import CBProducerOp

    batch, chunks, groups, chunk_len, d_state = 1, 2, 1, 64, 64
    op = CBProducerOp(batch, chunks, groups, chunk_len, d_state)
    s = chunks * chunk_len
    for dtype in _DTYPES:
        c = torch.randn(batch, s, groups, d_state, dtype=dtype, device="cuda")
        b = torch.randn(batch, s, groups, d_state, dtype=dtype, device="cuda")
        assert op(c, b).dtype == dtype
    _assert_two_entries(op)


@pytest.mark.smoke
def test_mismatched_input_dtypes_are_rejected():
    """The anchor selects the kernel; the others must agree with it."""
    n = 256
    op = RMSNormFwdOp(normalized_shape=(n,))
    x = torch.randn(16, n, dtype=torch.float16, device="cuda")
    w = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        op(x, w)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
