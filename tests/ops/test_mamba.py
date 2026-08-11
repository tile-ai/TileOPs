import pytest
import torch

from tests.test_base import TestBase, allclose_compare
from tileops.ops.cb_producer import CBProducerOp
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba2_fwd import Mamba2FwdOp
from tileops.ops.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.ssd_decode import SSDDecodeOp
from tileops.ops.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import (
    DaCumsumFwdFixture,
    DaCumsumFwdWorkload,
    SSDChunkScanFwdFixture,
    SSDChunkScanFwdWorkload,
    SSDChunkStateFwdFixture,
    SSDChunkStateFwdWorkload,
    SSDDecodeFixture,
    SSDDecodeWorkload,
    SSDStatePassingFwdFixture,
    SSDStatePassingFwdWorkload,
    da_cumsum_fwd_ref,
    ssd_chunk_state_fwd_ref,
)
from workloads.mamba2_e2e import (
    Mamba2PrimaryWorkload,
    mamba2_direct_ref,
    mamba2_fwd_ref,
)


def cb_producer_fwd_ref(
    C_mat: torch.Tensor,
    B_mat: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """PyTorch reference for cb_producer_fwd.

    Computes the causal C@B coupling matrix:
    cb[b,c,g,l,s] = C[b,c*Q+l,g,:] @ B[b,c*Q+s,g,:]^T for s <= l, else 0

    Returns:
        cb: (batch, num_chunks, n_groups, chunk_len, chunk_len) dtype
    """
    b, S, G, N = C_mat.shape
    Q = chunk_len
    C = num_chunks
    C_chunked = C_mat.reshape(b, C, Q, G, N)
    B_chunked = B_mat.reshape(b, C, Q, G, N)
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", C_chunked.float(), B_chunked.float())
    mask = torch.tril(torch.ones(Q, Q, device=C_mat.device, dtype=torch.bool))
    cb = cb * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return cb.to(dtype)


@pytest.mark.parametrize("batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune", [
    pytest.param(1, 2, 64, 1, 64, torch.float16,  False, marks=pytest.mark.smoke),
    pytest.param(1, 2, 64, 1, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
    pytest.param(1, 2, 64, 2, 64, torch.float16,  False, marks=pytest.mark.smoke),
    pytest.param(1, 2, 64, 1, 64, torch.float16,  True,  marks=pytest.mark.full),
    pytest.param(1, 2, 64, 1, 96, torch.float16,  False, marks=pytest.mark.full),
    pytest.param(1, 2, 128, 1, 64, torch.bfloat16, False, marks=pytest.mark.full),
    pytest.param(1, 2, 256, 1, 64, torch.float16,  False, marks=pytest.mark.full),
    pytest.param(2, 4, 64, 4, 128, torch.bfloat16, False, marks=pytest.mark.full),
])
def test_cb_producer_fwd(batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune):
    op = CBProducerOp(batch, num_chunks, n_groups, chunk_len, d_state, tune=tune)
    seq_len = num_chunks * chunk_len
    C_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device="cuda") * 0.1
    B_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device="cuda") * 0.1
    ref = cb_producer_fwd_ref(C_mat, B_mat, num_chunks, chunk_len, dtype)
    out = op(C_mat, B_mat)
    allclose_compare(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_cb_producer_fwd_noncontiguous():
    """CBProducerOp must handle non-contiguous inputs."""
    batch, num_chunks, chunk_len, n_groups, d_state = 1, 2, 64, 1, 64
    dtype = torch.float16
    seq_len = num_chunks * chunk_len
    C_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device="cuda")
    B_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device="cuda")
    C_mat = C_full[:, ::2, :, :]
    B_mat = B_full[:, ::2, :, :]
    assert not C_mat.is_contiguous()
    assert not B_mat.is_contiguous()
    ref = cb_producer_fwd_ref(C_mat.contiguous(), B_mat.contiguous(), num_chunks, chunk_len, dtype)
    out = CBProducerOp(batch, num_chunks, n_groups, chunk_len, d_state)(C_mat, B_mat)
    allclose_compare(out, ref, atol=1e-3, rtol=1e-3)


class DaCumsumFwdTest(DaCumsumFwdWorkload, TestBase):
    pass


@DaCumsumFwdFixture
def test_da_cumsum_fwd(batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype, tune):
    test = DaCumsumFwdTest(
        batch, num_chunks, chunk_len, n_heads,
        has_dt_bias=has_dt_bias, dt_softplus=dt_softplus, dtype=dtype,
    )
    op = DaCumsumFwdOp(
        chunk_len=chunk_len,
        has_dt_bias=has_dt_bias,
        dt_softplus=dt_softplus,
        dtype=dtype,
        tune=tune,
    )
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_da_cumsum_fwd_missing_bias_raises():
    """DaCumsumFwdKernel must raise when has_dt_bias=True but dt_bias is None."""
    from tileops.kernels.mamba import DaCumsumFwdKernel
    kernel = DaCumsumFwdKernel(
        batch=1, num_chunks=2, chunk_len=64, n_heads=4,
        seq_len=128, has_dt_bias=True,
    )
    dt = torch.randn(1, 128, 4, dtype=torch.float32, device="cuda")
    A = -torch.rand(4, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="dt_bias is required"):
        kernel(dt, A, dt_bias=None)


@pytest.mark.smoke
def test_da_cumsum_fwd_padded_head_tile():
    """Five heads against block_h=4 is the only shape reaching the masked tail."""
    batch, n_heads, chunk_len, num_chunks = 1, 5, 64, 2
    seq_len = chunk_len * num_chunks
    op = DaCumsumFwdOp(chunk_len=chunk_len, dtype=torch.float32)
    dt = torch.rand(batch, seq_len, n_heads, dtype=torch.float32, device="cuda")
    A = -torch.rand(n_heads, dtype=torch.float32, device="cuda")

    dt_out, dA_cumsum = op(dt, A)
    ref_dt, ref_cumsum = da_cumsum_fwd_ref(
        dt, A, num_chunks, chunk_len, dtype=torch.float32,
    )
    torch.testing.assert_close(dt_out, ref_dt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dA_cumsum, ref_cumsum, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
def test_da_cumsum_fwd_rejects_undeclared_output_dtype(dtype):
    """A dtype outside the manifest union must raise, not silently cast.

    The op is the gate, not the kernel: the manifest is backend-independent, so
    a kernel-only check would let a ``kernel_map`` override accept a dtype the
    spec forbids.
    """
    with pytest.raises(ValueError, match="dt_out dtype must be one of"):
        DaCumsumFwdOp(chunk_len=64, dtype=dtype)



class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    pass


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDChunkScanFwdOp(tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)


class SSDChunkStateFwdTest(SSDChunkStateFwdWorkload, TestBase):
    pass


@SSDChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx,
):
    test = SSDChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    op = SSDChunkStateFwdOp(has_seq_idx=has_seq_idx, tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3
    test.check(op, *inputs, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ssd_chunk_state_fwd_seq_end_negative():
    """Kernel must zero the whole chunk when the last token's seq_idx is -1."""
    batch, num_chunks, chunk_len = 1, 2, 64
    n_heads, d_head, d_state, n_groups = 4, 64, 32, 1
    dtype = torch.float16
    b, c, Q, h, p, n, g = batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups
    seq_len = c * Q

    x = torch.randn(b, seq_len, h, p, dtype=dtype, device="cuda") * 0.1
    Bmat = torch.randn(b, seq_len, g, n, dtype=dtype, device="cuda") * 0.1
    dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
    dt = torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda") * 0.1 + 0.01

    # First chunk ends with seq_idx == -1 (whole chunk should zero out).
    # Second chunk is a normal sequence (seq_idx == 1 throughout).
    seq_idx = torch.ones(b, seq_len, dtype=torch.int32, device="cuda")
    seq_idx[:, :Q] = -1

    op = SSDChunkStateFwdOp(has_seq_idx=True)
    out = op(x, Bmat, dt, dA_cumsum, seq_idx)
    ref = ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, g, seq_idx=seq_idx)

    from tests.test_base import allclose_compare
    atol = 1e-3
    rtol = 1e-3
    allclose_compare(out, ref, atol=atol, rtol=rtol)

    # Pin the semantic: chunk 0 (seq_idx == -1 throughout) must be exactly zero;
    # chunk 1 (seq_idx == 1 throughout) must have non-zero state.
    allclose_compare(out[:, 0], torch.zeros_like(out[:, 0]), atol=0.0, rtol=0.0)
    assert out[:, 1].abs().max().item() > 0


class SSDStatePassingFwdTest(SSDStatePassingFwdWorkload, TestBase):
    pass


@SSDStatePassingFwdFixture
def test_ssd_state_passing_fwd(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    op = SSDStatePassingFwdOp(tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3
    test.check(op, *inputs, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("config", [
    {"block_d": 64,  "threads": 32,  "vectorize": True},
    {"block_d": 128, "threads": 64,  "vectorize": True},
    {"block_d": 256, "threads": 128, "vectorize": True},
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ssd_state_passing_fwd_vectorize(config, dtype):
    """Exercises the vectorize=True code path (lo/hi split per thread)."""
    batch, num_chunks, n_heads, d_state = 2, 4, 8, 128
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    op = SSDStatePassingFwdOp(tune=False)
    inputs = test.gen_inputs()
    op(*inputs)
    op.kernel.config = config
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    test.check(op, *inputs, atol=atol, rtol=1e-3)


class SSDDecodeTest(SSDDecodeWorkload, TestBase):
    pass


@SSDDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDDecodeOp(tune=tune)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Run reference on a clone of state so the two runs start from the same point.
    state_ref = state.clone()
    y_ref = test.ref_program(A, dt, x, B_in, C_in, state_ref)

    # Run kernel; state is updated in-place.
    y_op = op(A, dt, x, B_in, C_in, state)

    atol = 1e-3
    rtol = 1e-3
    allclose_compare(y_op, y_ref, atol=atol, rtol=rtol)
    allclose_compare(state, state_ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mamba2_primary_matches_direct_recurrence_across_chunk_boundary(dtype):
    """The no-bias path returns post-update y and final state across a boundary."""
    torch.manual_seed(42)
    workload = Mamba2PrimaryWorkload(
        1, 512, 4, 64, 32, 2, dtype,
        chunk_size=256,
        dt_softplus=True,
    )
    inputs = workload.gen_inputs()
    op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    actual = op.forward(*inputs, return_final_states=True)
    expected = mamba2_direct_ref(*inputs)

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    assert actual[0].shape == (1, 512, 4, 64)
    assert actual[1].shape == (1, 4, 64, 32)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.float32
    allclose_compare(actual[0], expected[0], atol=atol, rtol=1e-3)
    allclose_compare(actual[1], expected[1], atol=atol, rtol=1e-3)
    assert next(iter(op._da_cumsum_ops.values())).has_dt_bias is False


@pytest.mark.smoke
@pytest.mark.parametrize("with_bias,with_initial", [(True, False), (False, True)])
def test_mamba2_optional_variants_regressions(with_bias, with_initial):
    """Primary specialization must preserve bias and initial-state variants."""
    torch.manual_seed(43)
    workload = Mamba2PrimaryWorkload(
        1, 512, 4, 64, 32, 1, torch.float16,
        chunk_size=256,
        dt_softplus=True,
    )
    inputs = workload.gen_inputs()
    bias = torch.randn(4, device="cuda", dtype=torch.float32) * 0.1 if with_bias else None
    initial = (
        torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float32) * 0.1
        if with_initial else None
    )
    op = Mamba2FwdOp(
        chunk_size=256,
        dt_softplus=True,
        has_initial_states=with_initial,
    )
    actual = op.forward(
        *inputs,
        dt_bias=bias,
        initial_states=initial,
        return_final_states=True,
    )
    expected = mamba2_fwd_ref(
        *inputs, bias, 256, True, initial_states=initial
    )
    allclose_compare(actual[0], expected[0], atol=1e-2, rtol=1e-3)
    allclose_compare(actual[1], expected[1], atol=1e-2, rtol=1e-3)
    assert next(iter(op._da_cumsum_ops.values())).has_dt_bias is with_bias


@pytest.mark.smoke
def test_mamba2_bias_dispatch_cache_is_call_sensitive():
    """One reusable op keeps distinct kernels for biased and unbiased calls."""
    torch.manual_seed(45)
    workload = Mamba2PrimaryWorkload(
        1, 256, 4, 64, 32, 1, torch.float16,
        chunk_size=256,
        dt_softplus=True,
    )
    inputs = workload.gen_inputs()
    bias = torch.randn(4, device="cuda", dtype=torch.float32) * 0.1
    op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)

    unbiased_y, unbiased_final = op.forward(*inputs)
    biased_y, biased_final = op.forward(*inputs, dt_bias=bias)
    unbiased_again_y, _ = op.forward(*inputs)

    assert unbiased_final is None
    assert biased_final is None
    assert set(op._da_cumsum_ops) == {
        (torch.float16, False),
        (torch.float16, True),
    }
    assert torch.equal(unbiased_y, unbiased_again_y)
    assert not torch.equal(unbiased_y, biased_y)


@pytest.mark.parametrize(
    "batch,seqlen,n_heads,d_head,d_state,n_groups,dtype",
    [
        pytest.param(
            1, 512, 4, 64, 32, 2, torch.float16,
            id="smoke-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            1, 512, 4, 64, 32, 2, torch.bfloat16,
            id="smoke-bf16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            1, 2048, 80, 64, 128, 1, torch.bfloat16,
            id="mamba2-2p7b-b1-s2k",
            marks=pytest.mark.full,
        ),
        pytest.param(
            1, 8192, 64, 64, 128, 1, torch.float16,
            id="mamba2-1p3b-b1-s8k",
            marks=pytest.mark.full,
        ),
    ],
)
def test_mamba2_primary_manifest_outputs(
    batch, seqlen, n_heads, d_head, d_state, n_groups, dtype
):
    """Both manifest outputs agree with independent PyTorch and official Mamba."""
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    torch.manual_seed(44)
    workload = Mamba2PrimaryWorkload(
        batch, seqlen, n_heads, d_head, d_state, n_groups, dtype,
        chunk_size=256,
        dt_softplus=True,
    )
    inputs = workload.gen_inputs()
    op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    actual = op.forward(*inputs, return_final_states=True)
    independent = workload.ref_program(*inputs)
    official_y, official_final = ssd_combined.mamba_chunk_scan_combined(
        *inputs,
        256,
        dt_bias=None,
        initial_states=None,
        dt_softplus=True,
        return_final_states=True,
    )
    official = official_y.float(), official_final.float()

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    for output, ref, baseline in zip(actual, independent, official, strict=True):
        assert output.dtype == torch.float32
        allclose_compare(output, ref, atol=atol, rtol=1e-3)
        allclose_compare(output, baseline, atol=atol, rtol=1e-3)

    assert actual[0].shape == (batch, seqlen, n_heads, d_head)
    assert actual[1].shape == (batch, n_heads, d_head, d_state)
    assert op._chunk_state_op.kernel.config == {
        "block_n": min(128, d_state),
        "block_p": 64,
        "block_l": 32,
        "threads": 128,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
