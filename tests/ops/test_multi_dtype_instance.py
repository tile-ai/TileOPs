"""One op instance, two element types.

The rest of the suite parametrizes dtype but builds a fresh op per case, so it
never drives a second dtype through an instance that already holds an entry.
This covers the invariant every family owes its L1 kernel slots.
"""

import pytest
import torch

from tileops.manifest import load_manifest
from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.reduction.reduce import SumFwdOp

_DTYPES = (torch.float16, torch.bfloat16)


def _assert_two_entries(op):
    """One kernel per dtype. ``iter_kernels`` dedups, so two means two."""
    kernels = list(op.iter_kernels())
    assert len(kernels) == 2, f"expected one kernel per dtype, got {len(kernels)}"


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
    assert set(op.built_kernels("layer_norm")) == {(16, dt) for dt in _DTYPES}


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
    back, so one instance must hold two different kernel classes.
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
    from tileops.ops.attention.gqa import GroupedQueryAttentionPrefillDenseFwdOp

    batch, heads, heads_kv, seq_len, dim = 1, 8, 2, 256, 128
    op = GroupedQueryAttentionPrefillDenseFwdOp(batch, heads, heads_kv, seq_len, dim, is_causal=True)
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
        assert op._get_kernel(dtype).__class__.__name__ == "GQAPrefillDenseFwdKernel"
    _assert_two_entries(op._gqa_op)

    # MHA builds no kernel of its own, so autotune has to reach the delegate's
    # kernels, one per dtype, through ``kernel_delegates``.
    tuned = []
    for kernel in list(op._gqa_op.iter_kernels()):
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
    from tileops.ops.cb_producer import CBProducerFwdOp

    batch, chunks, groups, chunk_len, d_state = 1, 2, 1, 64, 64
    op = CBProducerFwdOp(batch, chunks, groups, chunk_len, d_state)
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


# One instance alternating between bool and non-bool operands: the two are served
# by different kernels, so anything derived from one must not survive into the
# other.


@pytest.mark.smoke
def test_bitwise_alternates_between_bool_and_integer_storage():
    """bool -> int32 -> bool on one instance, each answered by its own kernel."""
    from tileops.ops.elementwise import BitwiseAndFwdOp

    op = BitwiseAndFwdOp()
    b = torch.tensor([True, False] * 32, device="cuda")
    i = torch.arange(64, device="cuda", dtype=torch.int32)

    torch.testing.assert_close(op(b, ~b), b & ~b)
    torch.testing.assert_close(op(i, i + 1), i & (i + 1))
    torch.testing.assert_close(op(b, b), b & b)  # back to bool after the int kernel

    built = tuple(op.built_kernels(op._op_name).values())
    assert len(built) == 2, "bool and int32 are two specializations"
    assert len({type(k) for k in built}) == 2, "and two different kernel classes"


@pytest.mark.smoke
def test_logical_and_output_stays_bool_across_input_storage():
    """A float input produces a bool output without disturbing the bool entry."""
    from tileops.ops.elementwise import LogicalAndFwdOp

    op = LogicalAndFwdOp()
    b = torch.tensor([True, False] * 32, device="cuda")
    f = torch.tensor([0.0, 1.0] * 32, device="cuda")

    torch.testing.assert_close(op(b, ~b), torch.logical_and(b, ~b))
    torch.testing.assert_close(op(f, f), torch.logical_and(f, f))
    torch.testing.assert_close(op(b, b), torch.logical_and(b, b))

    built = tuple(op.built_kernels(op._op_name).values())
    assert len(built) == 2, "bool and float32 are two specializations"
    assert len({type(k) for k in built}) == 2, "and two different kernel classes"


@pytest.mark.smoke
def test_masked_fill_alternates_between_bool_and_float_input():
    """The scalar is re-validated per element type, the mask stays bool."""
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    op = MaskedFillScalarFwdOp(value=1)
    mask = torch.tensor([True, False] * 32, device="cuda")
    b = torch.zeros(64, device="cuda", dtype=torch.bool)
    f = torch.zeros(64, device="cuda", dtype=torch.float32)

    torch.testing.assert_close(op(b, mask), b.masked_fill(mask, 1))
    torch.testing.assert_close(op(f, mask), f.masked_fill(mask, 1))
    torch.testing.assert_close(op(b, mask), b.masked_fill(mask, 1))

    assert len(op.built_kernels(op._op_name)) == 2, "bool and float32 are two specializations"


def _single_tensor_elementwise_ops():
    """Every concrete elementwise op a flat tensor alone can drive.

    Selected off the manifest — one declared input, no declared params — rather than
    off the constructor, which no longer says anything about arity.
    """
    import inspect

    import tileops.ops.elementwise as ew
    from tileops.ops.elementwise._base import UnaryOp

    manifest = load_manifest()
    found = {}
    for name in dir(ew):
        cls = getattr(ew, name)
        if not (inspect.isclass(cls) and issubclass(cls, UnaryOp)):
            continue
        if name.startswith("_") or getattr(cls, "_op_name", None) is None:
            continue  # a template base, not a concrete op
        signature = manifest.get(name, {}).get("signature", {})
        if len(signature.get("inputs", {})) == 1 and not signature.get("params"):
            found[name] = cls
    return found


_SINGLE_TENSOR_OPS = _single_tensor_elementwise_ops()


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(_SINGLE_TENSOR_OPS))
def test_single_tensor_op_records_its_dtype(name):
    """No op may reach a result without recording the element type it used.

    An op answering on its own path, such as an integer identity or predicate
    fallback, records for itself, so every declared dtype is driven, not just float32.
    """
    op = _SINGLE_TENSOR_OPS[name]()
    declared = load_manifest()[name]["signature"]["inputs"]
    (spec,) = declared.values()
    dtypes = [getattr(torch, d.strip()) for d in spec["dtype"].split("|")]
    dtypes = [d for d in dtypes if d not in (torch.float64, torch.complex64, torch.complex128)]
    assert dtypes, f"{name} declares no drivable input dtype"

    for dtype in dtypes:
        op = _SINGLE_TENSOR_OPS[name]()
        if dtype == torch.bool:
            x = torch.tensor([True, False] * 32, device="cuda")
        elif dtype.is_floating_point:
            x = torch.rand(64, device="cuda", dtype=dtype) + 0.5
        else:
            x = torch.arange(1, 65, device="cuda", dtype=dtype)
        op(x)  # an unexpected failure here is a real defect, not a skip
        assert op.dtype == dtype, f"{name} did not record {dtype}"


@pytest.mark.smoke
def test_enumeration_still_sees_the_family():
    """The sweep is worthless if the enumeration silently stops matching."""
    assert len(_SINGLE_TENSOR_OPS) >= 23, (
        f"only {len(_SINGLE_TENSOR_OPS)} single-tensor ops found; the filter broke"
    )
    for expected in (
        "AbsFwdOp",
        "ReciprocalFwdOp",
        "IsnanFwdOp",
        "LogicalNotFwdOp",
        "BitwiseNotFwdOp",
    ):
        assert expected in _SINGLE_TENSOR_OPS, f"{expected} dropped out of the sweep"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
