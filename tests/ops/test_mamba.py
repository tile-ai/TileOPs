import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops.ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.device import run_device
from workloads.mamba import (
    SSDChunkCumsumFwdFixture,
    SSDChunkCumsumFwdWorkload,
    SSDChunkScanFwdFixture,
    SSDChunkScanFwdWorkload,
    SSDChunkStateFwdFixture,
    SSDChunkStateFwdWorkload,
    SSDDecodeFixture,
    SSDDecodeWorkload,
    SSDStatePassingFwdFixture,
    SSDStatePassingFwdWorkload,
    coupling_verification,
    mamba2_fwd_ref,
    mamba2_verification,
    ssd_chunk_coupling_fwd_ref,
    ssd_chunk_state_fwd_ref,
    ssd_decode_result,
)
from workloads.numerics import compare_outputs


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune",
    [
        pytest.param(
            1,
            2,
            64,
            1,
            64,
            torch.float16,
            False,
            marks=[pytest.mark.smoke, pytest.mark.packaging(family="mamba")],
        ),
        pytest.param(1, 2, 64, 1, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param(1, 2, 64, 2, 64, torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(1, 2, 64, 1, 64, torch.float16, True, marks=pytest.mark.full),
        pytest.param(1, 2, 64, 1, 96, torch.float16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 128, 1, 64, torch.bfloat16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 256, 1, 64, torch.float16, False, marks=pytest.mark.full),
        pytest.param(2, 4, 64, 4, 128, torch.bfloat16, False, marks=pytest.mark.full),
    ],
)
def test_ssd_chunk_coupling_fwd(batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune):
    op = SSDChunkCouplingFwdOp(chunk_len)
    if tune:
        op.request_tune()
    seq_len = num_chunks * chunk_len
    C_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device=run_device()) * 0.1
    B_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device=run_device()) * 0.1
    ref = ssd_chunk_coupling_fwd_ref(C_mat, B_mat, num_chunks, chunk_len, dtype)
    out = op(C_mat, B_mat)
    compare_outputs(out, ref, coupling_verification())


@pytest.mark.smoke
def test_ssd_chunk_coupling_fwd_noncontiguous():
    """SSDChunkCouplingFwdOp must handle non-contiguous inputs."""
    batch, num_chunks, chunk_len, n_groups, d_state = 1, 2, 64, 1, 64
    dtype = torch.float16
    seq_len = num_chunks * chunk_len
    C_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device=run_device())
    B_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device=run_device())
    C_mat = C_full[:, ::2, :, :]
    B_mat = B_full[:, ::2, :, :]
    assert not C_mat.is_contiguous()
    assert not B_mat.is_contiguous()
    ref = ssd_chunk_coupling_fwd_ref(
        C_mat.contiguous(), B_mat.contiguous(), num_chunks, chunk_len, dtype
    )
    out = SSDChunkCouplingFwdOp(chunk_len)(C_mat, B_mat)
    compare_outputs(out, ref, coupling_verification())


class SSDChunkCumsumFwdTest(SSDChunkCumsumFwdWorkload, TestBase):
    pass


@SSDChunkCumsumFwdFixture
def test_ssd_chunk_cumsum_fwd(
    batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype, tune
):
    test = SSDChunkCumsumFwdTest(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        has_dt_bias=has_dt_bias,
        dt_softplus=dt_softplus,
        dtype=dtype,
    )
    op = SSDChunkCumsumFwdOp(
        chunk_len=chunk_len,
        dt_softplus=dt_softplus,
        out_dtype=dtype,
    )
    if tune:
        op.request_tune()
    inputs = test.gen_inputs()
    test.check(op, *inputs)


@pytest.mark.smoke
def test_ssd_chunk_cumsum_fwd_padded_head_tile():
    """Five heads against block_h=4 is the only shape reaching the masked tail."""
    batch, n_heads, chunk_len, num_chunks = 1, 5, 64, 2
    seq_len = chunk_len * num_chunks
    op = SSDChunkCumsumFwdOp(chunk_len=chunk_len, out_dtype=torch.float32)
    dt = torch.rand(batch, seq_len, n_heads, dtype=torch.float32, device=run_device())
    A = -torch.rand(n_heads, dtype=torch.float32, device=run_device())

    test = SSDChunkCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    test.check(op, dt, A, None)


class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    pass


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune
):
    test = SSDChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
    )
    op = SSDChunkScanFwdOp()
    if tune:
        op.request_tune()
    inputs = test.gen_inputs()
    test.check(op, *inputs)


class SSDChunkStateFwdTest(SSDChunkStateFwdWorkload, TestBase):
    pass


@SSDChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx
):
    test = SSDChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx
    )
    op = SSDChunkStateFwdOp()
    if tune:
        op.request_tune()
    inputs = test.gen_inputs()
    test.check(op, *inputs)


@pytest.mark.cuda_only
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

    workload = SSDChunkStateFwdWorkload(b, c, Q, h, p, n, g, dtype, True)
    compare_outputs(out, ref, workload.verification(x))

    # Pin the semantic: chunk 0 (seq_idx == -1 throughout) must be exactly zero;
    # chunk 1 (seq_idx == 1 throughout) must have non-zero state.
    assert torch.count_nonzero(out[:, 0]) == 0
    assert out[:, 1].abs().max().item() > 0

    poison = torch.full((b, seq_len), -1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    del poison
    out = op(x, Bmat, dt, dA_cumsum)
    ref = ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, g)
    compare_outputs(out, ref, workload.verification(x))


class SSDStatePassingFwdTest(SSDStatePassingFwdWorkload, TestBase):
    pass


@SSDStatePassingFwdFixture
def test_ssd_state_passing_fwd(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    op = SSDStatePassingFwdOp()
    if tune:
        op.request_tune()
    inputs = test.gen_inputs()
    test.check(op, *inputs)


class SSDDecodeTest(SSDDecodeWorkload, TestBase):
    pass


@SSDDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDRecurrentFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs(), runs=lambda *args: ssd_decode_result(op, *args))


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
    dev = run_device()
    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, n_heads, d_head, dtype=dtype, device=dev) * 0.1
    dt_raw = torch.randn(batch, seqlen, n_heads, dtype=torch.float32, device=dev) * 0.5
    A = -torch.rand(n_heads, dtype=torch.float32, device=dev)
    B = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype, device=dev) * 0.1
    C = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype, device=dev) * 0.1
    dt_bias = torch.randn(n_heads, dtype=torch.float32, device=dev) * 0.1

    op = Mamba2FwdOp(chunk_size=chunk_size, dt_softplus=True)
    got = op(x, dt_raw, A, B, C, dt_bias=dt_bias)
    expected = mamba2_fwd_ref(x, dt_raw, A, B, C, dt_bias, chunk_size, dt_softplus=True)

    compare_outputs(got, expected, mamba2_verification(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ssd_state_passing_fwd_vectorized_short_scan(dtype):
    """A short scan over a flattened 8192-wide state splits each thread's state in two."""
    test = SSDStatePassingFwdTest(2, 4, 8, 8192, dtype)
    test.check(SSDStatePassingFwdOp(), *test.gen_inputs())
