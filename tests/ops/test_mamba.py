import pytest
import torch

from tests.test_base import TestBase, allclose_compare
from tileops.backend import BUILTIN
from tileops.ops.mamba.cb_producer import CBProducerFwdOp
from tileops.ops.mamba.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_decode import SSDDecodeFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
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
    cb_producer_fwd_ref,
    da_cumsum_fwd_ref,
    ssd_chunk_state_fwd_ref,
)
from workloads.mamba2_e2e import mamba2_fwd_ref


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune",
    [
        pytest.param(1, 2, 64, 1, 64, torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(1, 2, 64, 1, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param(1, 2, 64, 2, 64, torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(1, 2, 64, 1, 64, torch.float16, True, marks=pytest.mark.full),
        pytest.param(1, 2, 64, 1, 96, torch.float16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 128, 1, 64, torch.bfloat16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 256, 1, 64, torch.float16, False, marks=pytest.mark.full),
        pytest.param(2, 4, 64, 4, 128, torch.bfloat16, False, marks=pytest.mark.full),
    ],
)
def test_cb_producer_fwd(batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune):
    op = CBProducerFwdOp(chunk_len, tune=tune)
    seq_len = num_chunks * chunk_len
    C_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device="cuda") * 0.1
    B_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device="cuda") * 0.1
    ref = cb_producer_fwd_ref(C_mat, B_mat, num_chunks, chunk_len, dtype)
    out = op(C_mat, B_mat)
    allclose_compare(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_cb_producer_fwd_noncontiguous():
    """CBProducerFwdOp must handle non-contiguous inputs."""
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
    out = CBProducerFwdOp(chunk_len)(C_mat, B_mat)
    allclose_compare(out, ref, atol=1e-3, rtol=1e-3)


class DaCumsumFwdTest(DaCumsumFwdWorkload, TestBase):
    pass


@DaCumsumFwdFixture
def test_da_cumsum_fwd(
    batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype, tune
):
    test = DaCumsumFwdTest(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        has_dt_bias=has_dt_bias,
        dt_softplus=dt_softplus,
        dtype=dtype,
    )
    op = DaCumsumFwdOp(
        chunk_len=chunk_len,
        dt_softplus=dt_softplus,
        out_dtype=dtype,
        tune=tune,
    )
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_da_cumsum_fwd_missing_bias_raises():
    """DaCumsumFwdKernel must raise when has_dt_bias=True but dt_bias is None."""
    from tileops.kernels.mamba import DaCumsumFwdKernel

    kernel = DaCumsumFwdKernel(
        batch=1,
        num_chunks=2,
        chunk_len=64,
        n_heads=4,
        seq_len=128,
        has_dt_bias=True,
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
    op = DaCumsumFwdOp(chunk_len=chunk_len, out_dtype=torch.float32)
    dt = torch.rand(batch, seq_len, n_heads, dtype=torch.float32, device="cuda")
    A = -torch.rand(n_heads, dtype=torch.float32, device="cuda")

    dt_out, dA_cumsum = op(dt, A)
    ref_dt, ref_cumsum = da_cumsum_fwd_ref(
        dt,
        A,
        num_chunks,
        chunk_len,
        dtype=torch.float32,
    )
    torch.testing.assert_close(dt_out, ref_dt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dA_cumsum, ref_cumsum, atol=1e-5, rtol=1e-5)


class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    pass


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune
):
    test = SSDChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
    )
    op = SSDChunkScanFwdOp(tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)


class SSDChunkStateFwdTest(SSDChunkStateFwdWorkload, TestBase):
    pass


@SSDChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch,
    num_chunks,
    chunk_len,
    n_heads,
    d_head,
    d_state,
    n_groups,
    dtype,
    tune,
    has_seq_idx,
):
    test = SSDChunkStateFwdTest(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        d_head,
        d_state,
        n_groups,
        dtype,
        has_seq_idx,
    )
    op = SSDChunkStateFwdOp(tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3
    test.check(op, *inputs, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ssd_chunk_state_fwd_seq_idx_semantics():
    """Exercise negative chunk ends and the optional unmasked path."""
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

    op = SSDChunkStateFwdOp()
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

    poison = torch.full((b, seq_len), -1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    del poison
    out = op(x, Bmat, dt, dA_cumsum)
    ref = ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, g)
    allclose_compare(out, ref, atol=atol, rtol=rtol)


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
@pytest.mark.parametrize(
    "config",
    [
        {"block_d": 64, "threads": 32, "vectorize": True},
        {"block_d": 128, "threads": 64, "vectorize": True},
        {"block_d": 256, "threads": 128, "vectorize": True},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ssd_state_passing_fwd_vectorize(config, dtype):
    """Exercises the vectorize=True code path (lo/hi split per thread)."""
    batch, num_chunks, n_heads, d_state = 2, 4, 8, 128
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    op = SSDStatePassingFwdOp(tune=False, target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = op.built_kernels("ssd_state_passing_fwd").values()
    kernel.config = config
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    test.check(op, *inputs, atol=atol, rtol=1e-3)


class SSDDecodeTest(SSDDecodeWorkload, TestBase):
    pass


@SSDDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDDecodeFwdOp(tune=tune)
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
@pytest.mark.parametrize(
    "batch,seqlen,n_heads,d_head,d_state,n_groups,chunk_size",
    [
        (1, 256, 4, 64, 32, 1, 256),
        (2, 512, 8, 64, 64, 2, 256),
        (1, 512, 4, 64, 128, 1, 256),  # d_state=16 not supported by SSDChunkScanFwdKernel
    ],
)
def test_mamba2_fwd_e2e(batch, seqlen, n_heads, d_head, d_state, n_groups, chunk_size, dtype):
    """Mamba2FwdOp output must match the pure-PyTorch reference within tolerance."""
    dev = "cuda"
    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, n_heads, d_head, dtype=dtype, device=dev) * 0.1
    dt_raw = torch.randn(batch, seqlen, n_heads, dtype=torch.float32, device=dev) * 0.5
    A = -torch.rand(n_heads, dtype=torch.float32, device=dev)
    B = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype, device=dev) * 0.1
    C = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype, device=dev) * 0.1
    dt_bias = torch.randn(n_heads, dtype=torch.float32, device=dev) * 0.1

    op = Mamba2FwdOp(chunk_size=chunk_size, dt_softplus=True)
    y_op, _ = op(x, dt_raw, A, B, C, dt_bias=dt_bias)
    y_ref, _ = mamba2_fwd_ref(x, dt_raw, A, B, C, dt_bias, chunk_size, dt_softplus=True)

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    allclose_compare(y_op.float(), y_ref.float(), atol=atol, rtol=1e-3)
