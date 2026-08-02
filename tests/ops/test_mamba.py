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
from tileops.testing.mamba2_reference import (
    mamba2_fwd_ref,
    ssd_chunk_scan_fwd_ref,
)
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
)


def da_cumsum_fwd_ref(
    dt: torch.Tensor,
    A: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for da_cumsum_fwd.

    Applies the same bias / softplus / clamp pipeline as the kernel, then
    computes dt_out and the chunk-local inclusive prefix sum of dA = dt_out * A.

    Returns:
        dt_out:    (batch, n_heads, num_chunks, chunk_len) dtype — in target dtype
        dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32
    """
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks
    dt_val = dt.float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias.float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    dt_val = torch.clamp(dt_val, min=dt_min, max=dt_max)
    dt_chunked = dt_val.reshape(b, C, Q, h)           # (b, C, Q, h)
    dt_out = dt_chunked.permute(0, 3, 1, 2).contiguous().to(dtype)  # (b, h, C, Q) in target dtype
    dA = dt_chunked * A.float()                        # (b, C, Q, h)
    dA_cumsum = dA.cumsum(dim=2).permute(0, 3, 1, 2).contiguous()  # (b, h, C, Q)
    return dt_out, dA_cumsum


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
    op = CBProducerOp(batch, num_chunks, n_groups, chunk_len, d_state, dtype=dtype, tune=tune)
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
    out = CBProducerOp(batch, num_chunks, n_groups, chunk_len, d_state, dtype=dtype)(C_mat, B_mat)
    allclose_compare(out, ref, atol=1e-3, rtol=1e-3)


class DaCumsumFwdTest(DaCumsumFwdWorkload, TestBase):
    def ref_program(self, dt, A, dt_bias):
        return da_cumsum_fwd_ref(
            dt, A, self.num_chunks, self.chunk_len,
            dt_bias=dt_bias if self.has_dt_bias else None,
            dt_softplus=self.dt_softplus,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dtype=self.dtype,
        )


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




class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.n_groups)


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDChunkScanFwdOp(tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)


def ssd_chunk_state_fwd_ref(
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    n_groups: int,
    seq_idx=None,
) -> torch.Tensor:
    """PyTorch reference for ssd_chunk_state_fwd."""
    b, seq_len, h, p = x.shape
    _, _, c, Q = dt.shape
    n = Bmat.shape[-1]
    heads_per_group = h // n_groups

    x_chunked = x.float().reshape(b, c, Q, h, p)
    B_chunked = Bmat.float().reshape(b, c, Q, n_groups, n)
    B_heads = B_chunked[:, :, :, torch.arange(h) // heads_per_group, :]

    dA = dA_cumsum.float().permute(0, 2, 1, 3)
    dA_end = dA[:, :, :, -1:]
    decay = torch.exp(torch.clamp(dA_end - dA, max=0.0))

    dt_chunked = dt.float().permute(0, 2, 1, 3)
    weight = decay * dt_chunked

    if seq_idx is not None:
        seq_chunked = seq_idx.reshape(b, c, Q)
        seq_end = seq_chunked[..., -1:]
        same = ((seq_end >= 0) & (seq_chunked == seq_end)).unsqueeze(3)
        weight = weight * same.permute(0, 1, 3, 2)

    w = weight.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
    contrib = w * B_heads.unsqueeze(-1) * x_chunked.unsqueeze(-2)
    out = contrib.sum(dim=2)
    return out.permute(0, 1, 2, 4, 3)


class SSDChunkStateFwdTest(SSDChunkStateFwdWorkload, TestBase):
    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups, seq_idx=seq_idx)


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


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the inter-chunk recurrent scan.

    Matches mamba convention: out[:,c] = state *before* processing chunk c,
    so out[:,0] = initial_states and final_states = state after chunk C-1.
    """
    b, c, h, d = states.shape
    # out[:,0] = s_{-1} = initial_states (state before chunk 0)
    out = [initial_states.float().clone()]
    s = initial_states.float()

    for ci in range(c):
        scale = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1)
        u = states[:, ci, :, :].float()
        s = scale * s + u
        if ci < c - 1:
            out.append(s.clone())

    return torch.stack(out, dim=1), s


class SSDStatePassingFwdTest(SSDStatePassingFwdWorkload, TestBase):
    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)


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


def ssd_decode_ref(
    A: torch.Tensor,      # (H, P, N)     float32
    dt: torch.Tensor,     # (B, H, P)     float32
    x: torch.Tensor,      # (B, H, P)     any dtype
    B_in: torch.Tensor,   # (B, G, N)     any dtype
    C_in: torch.Tensor,   # (B, G, N)     any dtype
    state: torch.Tensor,  # (B, H, P, N)  float32  -- updated in-place
) -> torch.Tensor:
    """PyTorch reference for ssd_decode.

    Matches the official Mamba-2 selective_state_update interface:
      A:  (nheads, headdim, d_state)   — repeated from (nheads,) in mamba2.step()
      dt: (batch,  nheads,  headdim)   — repeated from (batch, nheads) in mamba2.step()

    Returns:
      y_out: (B, H, P)  float32

    Semantics:
      g                = h // (n_heads // n_groups)
      dA[b, h, p, n]   = exp(dt[b,h,p] * A[h,p,n])
      state[b,h,p,n]  <- dA[b,h,p,n] * state[b,h,p,n]
                         + dt[b,h,p] * B_in[b,g,n] * x[b,h,p]   (in-place)
      y_out[b, h, p]   = sum_n  state[b, h, p, n] * C_in[b, g, n]
    """
    B, H, P = dt.shape
    G = B_in.shape[1]
    heads_per_group = H // G

    # Expand B/C from groups to heads: (B, H, N)
    head_idx = torch.arange(H, device=B_in.device) // heads_per_group
    B_heads = B_in.float()[:, head_idx, :]   # (B, H, N)
    C_heads = C_in.float()[:, head_idx, :]   # (B, H, N)

    # dA[b, h, p, n] = exp(dt[b,h,p] * A[h,p,n])
    dA = torch.exp(
        dt.float()[:, :, :, None] * A.float()[None, :, :, :]
    )  # (B, H, P, N)

    # dBx[b, h, p, n] = dt[b,h,p] * B[b,h,n] * x[b,h,p]
    dBx = (
        dt.float()[:, :, :, None]
        * x.float()[:, :, :, None]
        * B_heads[:, :, None, :]
    )  # (B, H, P, N)

    # Update state in-place
    new_state = dA * state.float() + dBx
    state.copy_(new_state)

    # y_out[b, h, p] = sum_n state[b, h, p, n] * C[b, h, n]
    y_out = torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)
    return y_out


class SSDDecodeTest(SSDDecodeWorkload, TestBase):
    def ref_program(self, A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)


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
