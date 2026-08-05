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

    Causal dim-128 takes the warp-specialized dense slot.
    """
    from tileops.ops.attention.gqa import GroupedQueryAttentionFwdOp

    batch, heads, heads_kv, seq_len, dim = 1, 8, 2, 256, 128
    op = GroupedQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, is_causal=True)
    for dtype in _DTYPES:
        q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seq_len, heads_kv, dim, dtype=dtype, device="cuda")
        v = torch.randn_like(k)
        output = op(q, k, v)
        assert output.dtype == dtype
        kernel = op._get_kernel(dtype)
        assert kernel.__class__.__name__ == "GQAPrefillFwdWsPersistentCausalKernel"
        assert kernel.dtype == dtype
    _assert_two_entries(op)


@pytest.mark.smoke
def test_attention_mha_serves_two_dtypes_from_one_instance():
    """MHA owns no kernels; the per-dtype cache lives on the GQA delegate."""
    from tileops.ops.attention.mha import MultiHeadAttentionFwdOp

    batch, heads, seq_len, dim = 1, 8, 256, 64
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, is_causal=False)
    for dtype in _DTYPES:
        q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output = op(q, k, v)
        assert output.dtype == dtype
        assert op._get_kernel(dtype).__class__.__name__ == "GQAPrefillFwdKernel"
    _assert_two_entries(op._gqa_op)

    # MHA owns no cache, so autotune has to reach the delegate's kernels — one
    # per dtype — rather than relying on attribute traversal finding them here.
    tuned = []
    for kernel in op._gqa_op._kernel_cache.values():
        kernel.autotune = lambda *_args, _k=kernel: tuned.append(id(_k))
    op.autotune()
    assert len(tuned) == len(_DTYPES)


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


# Bool operands run on a different kernel than the same op's other dtypes: the
# storage is reinterpreted as uint8. That fact used to live in a `_bool_storage`
# attribute written while building a kernel, so a bool call followed by an
# integer one left it describing the wrong call. It now travels inside the
# cache entry, and these sequences would fail if it moved back out.


@pytest.mark.smoke
def test_bitwise_alternates_between_bool_and_integer_storage():
    """bool -> int32 -> bool on one instance, each answered by its own kernel."""
    from tileops.ops.elementwise import BitwiseAndFwdOp

    op = BitwiseAndFwdOp((64,), (64,))
    b = torch.tensor([True, False] * 32, device="cuda")
    i = torch.arange(64, device="cuda", dtype=torch.int32)

    torch.testing.assert_close(op(b, ~b), b & ~b)
    torch.testing.assert_close(op(i, i + 1), i & (i + 1))
    torch.testing.assert_close(op(b, b), b & b)  # back to bool after the int kernel

    assert set(op._entries) == {torch.bool, torch.int32}


@pytest.mark.smoke
def test_logical_and_output_stays_bool_across_input_storage():
    """A float input produces a bool output without disturbing the bool entry."""
    from tileops.ops.elementwise import LogicalAndFwdOp

    op = LogicalAndFwdOp((64,), (64,))
    b = torch.tensor([True, False] * 32, device="cuda")
    f = torch.tensor([0.0, 1.0] * 32, device="cuda")

    torch.testing.assert_close(op(b, ~b), torch.logical_and(b, ~b))
    torch.testing.assert_close(op(f, f), torch.logical_and(f, f))
    torch.testing.assert_close(op(b, b), torch.logical_and(b, b))

    assert set(op._entries) == {torch.bool, torch.float32}


@pytest.mark.smoke
def test_masked_fill_alternates_between_bool_and_float_input():
    """The scalar is re-validated per element type, the mask stays bool."""
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    op = MaskedFillScalarFwdOp(input=(64,), mask=(64,), value=1)
    mask = torch.tensor([True, False] * 32, device="cuda")
    b = torch.zeros(64, device="cuda", dtype=torch.bool)
    f = torch.zeros(64, device="cuda", dtype=torch.float32)

    torch.testing.assert_close(op(b, mask), b.masked_fill(mask, 1))
    torch.testing.assert_close(op(f, mask), f.masked_fill(mask, 1))
    torch.testing.assert_close(op(b, mask), b.masked_fill(mask, 1))

    assert set(op._entries) == {torch.bool, torch.float32}


@pytest.mark.smoke
def test_every_elementwise_op_records_its_dtype_after_a_forward():
    """No op may reach a result without recording the element type it used.

    Recording lives in the shared cache lookup, so an op that answers on its own
    path has to record for itself. Enumerating the family turns a forgotten one
    into a failure here rather than into roofline numbers describing an earlier
    call — which is how `RoundFwdOp` shipped a stale dtype.
    """
    import inspect

    import tileops.ops.elementwise as ew
    from tileops.ops.elementwise._base import _PerDtypeKernels

    checked, unrecorded = 0, []
    for name in dir(ew):
        cls = getattr(ew, name)
        if not (inspect.isclass(cls) and issubclass(cls, _PerDtypeKernels)):
            continue
        if name.startswith("_") or getattr(cls, "_op_name", None) is None:
            continue  # a template base, not a concrete op
        params = inspect.signature(cls.__init__).parameters
        # Only the single-tensor, single-shape ops can be driven generically.
        if set(params) - {"self", "kernel_map", "tune"} != {"N_total"}:
            continue
        op = cls(N_total=64)
        try:
            op(torch.randn(64, device="cuda", dtype=torch.float32))
        except Exception:
            continue  # dtype outside this op's union; another dtype covers it
        checked += 1
        if op.dtype is None:
            unrecorded.append(name)

    assert checked >= 20, f"only drove {checked} ops; the enumeration stopped working"
    assert not unrecorded, f"ops reached a result without recording a dtype: {unrecorded}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
