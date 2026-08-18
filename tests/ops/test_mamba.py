from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import TestBase, allclose_compare
from tileops.ops.cb_producer import CBProducerOp
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba2_fwd import Mamba2FwdOp
from tileops.ops.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.ssd_decode import SSDDecodeOp
from tileops.ops.ssd_state_passing import SSDStatePassingFwdOp
from tileops.perf import formulas
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


def mamba2_fwd_ref(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
    dt_softplus: bool,
) -> torch.Tensor:
    """Pure-PyTorch reference for the Mamba-2 State-Space Dual (SSD) forward pass.

    Computes the same result as mamba_chunk_scan_combined from mamba_ssm:
      out[l,p] = exp(dA[l]) * C[l] @ prev_state
               + sum_{s<=l} (C[l]@B[s]) * exp(dA[l]-dA[s]) * dt[s] * x[s,p]

    Inputs:
        x:           (B, S, H, P)     dtype
        dt:          (B, S, H)        float32
        A:           (H,)             float32  (log-space, <= 0)
        B:           (B, S, G, N)     dtype
        C:           (B, S, G, N)     dtype
        dt_bias:     (H,)             float32, optional
        chunk_size:  int
        dt_softplus: bool

    Returns:
        y: (B, S, H, P)  float32
    """
    b, S, h, p = x.shape
    n = B.shape[-1]
    g = B.shape[2]
    hpg = h // g
    Q = chunk_size
    num_chunks = S // Q

    # Step 1: DaCumsum
    dt_val = dt.float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias.float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    dt_val = torch.clamp(dt_val, min=0.0)
    dt_chunked = dt_val.reshape(b, num_chunks, Q, h).permute(0, 3, 1, 2)
    dA = dt_chunked * A.float().view(1, h, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)

    # Step 2: CB = C[l] @ B[s]^T per chunk, lower-triangular, group-owned.
    B_c = B.float().reshape(b, num_chunks, Q, g, n)
    C_c = C.float().reshape(b, num_chunks, Q, g, n)
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", C_c, B_c)
    mask = torch.ones(Q, Q, device=x.device, dtype=torch.bool).tril()
    cb = cb * mask.view(1, 1, 1, Q, Q)

    # Step 3: SSDChunkState
    decay = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    decay_c = decay.permute(0, 2, 3, 1)
    dt_c = dt_chunked.permute(0, 2, 3, 1)
    x_c = x.float().reshape(b, num_chunks, Q, h, p)
    B_heads = B_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]
    wx = x_c * (decay_c * dt_c).unsqueeze(-1)
    chunk_states = torch.einsum("bcqhp,bcqhn->bchpn", wx, B_heads)

    # Step 4: SSDStatePassing
    exp_dA_chunk = torch.exp(dA_cumsum[:, :, :, -1])
    s = torch.zeros(b, h, p, n, device=x.device, dtype=torch.float32)
    prev_states_list = []
    for ci in range(num_chunks):
        prev_states_list.append(s.unsqueeze(1))
        scale = exp_dA_chunk[:, :, ci].view(b, h, 1, 1)
        s = scale * s + chunk_states[:, ci]
    prev_states = torch.cat(prev_states_list, dim=1)

    # Step 5: SSDChunkScan
    dA_c = dA_cumsum.permute(0, 2, 3, 1)
    C_heads = C_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]

    y_hist = torch.einsum("bcqhn,bchpn->bcqhp", C_heads, prev_states.float())
    y_hist = y_hist * torch.exp(dA_c).unsqueeze(-1)

    dA_l = dA_cumsum.unsqueeze(-1)
    dA_s = dA_cumsum.unsqueeze(-2)
    decay_ls = torch.exp(dA_l - dA_s).masked_fill(
        ~mask.view(1, 1, 1, Q, Q), 0.0
    ).permute(0, 2, 1, 3, 4)
    cb_heads = cb[:, :, torch.arange(h, device=x.device) // hpg, :, :]
    lcb = cb_heads * decay_ls * dt_c.permute(0, 1, 3, 2).unsqueeze(-2)
    wx_t = x_c.permute(0, 1, 3, 2, 4)
    y_intra = torch.einsum("bchls,bchsp->bchlp", lcb, wx_t).permute(0, 1, 3, 2, 4)

    return (y_hist + y_intra).reshape(b, S, h, p)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,seqlen,n_heads,d_head,d_state,n_groups,chunk_size", [
    (1, 256,  4, 64, 32,  1, 256),
    (2, 512,  8, 64, 64,  2, 256),
    (1, 512,  4, 64, 128, 1, 256),   # d_state=16 not supported by SSDChunkScanFwdKernel
])
def test_mamba2_fwd_e2e(batch, seqlen, n_heads, d_head, d_state, n_groups, chunk_size, dtype):
    """Mamba2FwdOp output must match the pure-PyTorch reference within tolerance."""
    dev = "cuda"
    torch.manual_seed(42)
    x       = torch.randn(batch, seqlen, n_heads, d_head,   dtype=dtype,          device=dev) * 0.1
    dt_raw  = torch.randn(batch, seqlen, n_heads,           dtype=torch.float32,  device=dev) * 0.5
    A       = -torch.rand(n_heads,                          dtype=torch.float32,  device=dev)
    B       = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype,          device=dev) * 0.1
    C       = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype,          device=dev) * 0.1
    dt_bias = torch.randn(n_heads,                          dtype=torch.float32,  device=dev) * 0.1

    op = Mamba2FwdOp(
        chunk_size=chunk_size,
        dt_softplus=True,
        has_initial_states=False,
    )
    y_op, _ = op.forward(x, dt_raw, A, B, C, dt_bias=dt_bias)
    y_ref   = mamba2_fwd_ref(x, dt_raw, A, B, C, dt_bias, chunk_size, dt_softplus=True)

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    allclose_compare(y_op.float(), y_ref.float(), atol=atol, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])


# ----------------------------------------------------------------------
# Composite-vs-stage contract for the Mamba-2 / State-Space Dual (SSD) rooflines.
# ----------------------------------------------------------------------


# Small representative Mamba-2 geometry: S = NC * Q, G divides H.
B, NC, Q, H, P, N, G = 2, 4, 128, 4, 16, 32, 1
S = NC * Q
TOKENS = B * S * H


def _da_cumsum_op(dt_softplus: bool, has_dt_bias: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seq_len=S, n_heads=H, dt_softplus=dt_softplus,
        dt_bias_shape=(H,) if has_dt_bias else None,
        dtype=torch.float16)


def _chunk_state_op() -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16)


def _state_passing_op(d_state: int, has_initial_states: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, n_heads=H, d_state=d_state,
        initial_states_shape=(B, H, d_state) if has_initial_states else None,
        dtype=torch.float32)


def _mamba2_op(has_dt_bias: bool, has_initial_states: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seqlen=S, num_chunks=NC, chunk_size=Q, n_heads=H, d_head=P,
        d_state=N, n_groups=G, dtype=torch.float16, dt_softplus=True,
        dt_bias_shape=(H,) if has_dt_bias else None,
        initial_states_shape=(B, H, P, N) if has_initial_states else None)


# One public roofline function reads which optional inputs the call passed, so
# the four presence configurations exercise the same helper.
@pytest.mark.parametrize(("has_dt_bias", "has_initial_states"),
                         [(False, False), (True, False),
                          (False, True), (True, True)])
@pytest.mark.smoke
def test_mamba2_fwd_roofline_flops_equal_stage_sum(has_dt_bias: bool,
                                                   has_initial_states: bool):
    """Composite FLOPs must equal the sum of the five standalone stages."""
    composite_flops, _ = formulas.mamba2_fwd_roofline(
        _mamba2_op(has_dt_bias, has_initial_states))

    stage_flops = 0
    stage_flops += formulas.da_cumsum_fwd_roofline(
        _da_cumsum_op(dt_softplus=True, has_dt_bias=has_dt_bias))[0]
    stage_flops += formulas.cb_producer_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, n_groups=G, chunk_len=Q, d_state=N,
        dtype=torch.float16))[0]
    stage_flops += formulas.ssd_chunk_state_fwd_roofline(_chunk_state_op())[0]
    # State passing runs over the flattened d_head * d_state dimension.
    stage_flops += formulas.ssd_state_passing_fwd_roofline(
        _state_passing_op(P * N, has_initial_states))[0]
    stage_flops += formulas.ssd_chunk_scan_fwd_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16))[0]

    assert composite_flops == stage_flops
