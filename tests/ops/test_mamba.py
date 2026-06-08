import pytest
import torch
import torch.nn.functional as F

from tests.test_base import TestBase, allclose_compare
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba2_fwd import Mamba2FwdOp
from tileops.ops.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.ssd_decode import SSDDecodeOp
from tileops.ops.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import (
    DaCumsumFwdFixture,
    SSDChunkScanFwdFixture,
    SSDChunkStateFwdFixture,
    SSDDecodeFixture,
    SSDStatePassingFwdFixture,
)
from workloads.mamba import DaCumsumFwdTest as _DaCumsumFwdTestWorkload
from workloads.mamba import SSDChunkScanFwdTest as _SSDChunkScanFwdTestWorkload
from workloads.mamba import SSDChunkStateFwdTest as _SSDChunkStateFwdTestWorkload
from workloads.mamba import SSDDecodeTest as _SSDDecodeTestWorkload
from workloads.mamba import (
    SSDStatePassingFwdTest as _SSDStatePassingFwdTestWorkload,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for da_cumsum_fwd.

    Applies the same bias / softplus / clamp pipeline as the kernel, then
    computes dt_out and the chunk-local inclusive prefix sum of dA = dt_out * A.

    Returns:
        dt_out:    (batch, n_heads, num_chunks, chunk_len) float32
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
    dt_out = dt_chunked.permute(0, 3, 1, 2).contiguous()  # (b, h, C, Q)
    dA = dt_chunked * A.float()                        # (b, C, Q, h)
    dA_cumsum = dA.cumsum(dim=2).permute(0, 3, 1, 2).contiguous()  # (b, h, C, Q)
    return dt_out, dA_cumsum


class DaCumsumFwdTest(_DaCumsumFwdTestWorkload, TestBase):
    def ref_program(self, dt, A, dt_bias):
        return da_cumsum_fwd_ref(
            dt, A, self.num_chunks, self.chunk_len,
            dt_bias=dt_bias if self.has_dt_bias else None,
            dt_softplus=self.dt_softplus,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
        )


@DaCumsumFwdFixture
def test_da_cumsum_fwd(batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, tune):
    test = DaCumsumFwdTest(
        batch, num_chunks, chunk_len, n_heads,
        has_dt_bias=has_dt_bias, dt_softplus=dt_softplus,
    )
    op = DaCumsumFwdOp(
        batch, num_chunks, chunk_len, n_heads,
        seq_len=num_chunks * chunk_len,
        has_dt_bias=has_dt_bias,
        dt_softplus=dt_softplus,
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


def ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups):
    """Official-aligned PyTorch reference for chunk scan.

    Inputs (official layouts):
      x:           [B, S, H, P]        dtype
      cb:          [B, C, G, L, L]     dtype    group-owned
      dA_cumsum:   [B, H, C, L]        float32
      C:           [B, S, G, N]        dtype    group-owned
      prev_states: [B, C, H, P, N]     float32  P before N
      dt:          [B, H, C, L]        dtype

    Output: [B, S, H, P]  float32
    """
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C.shape[-1]
    g = n_groups
    heads_per_group = h // g

    x_chunked = x.float().reshape(b, c, L, h, p)             # [B, C, L, H, P]
    C_chunked = C.float().reshape(b, c, L, g, n)              # [B, C, L, G, N]
    # broadcast C from groups to heads: [B, C, L, H, N]
    C_heads = C_chunked[:, :, :, torch.arange(h, device=x.device) // heads_per_group, :]

    # dA_cumsum: [B, H, C, L] -> [B, C, L, H] for broadcast
    dA = dA_cumsum.float().permute(0, 2, 3, 1)  # [B, C, L, H]

    # --- History path: exp(dA_l) * C[l] @ prev_states[p, n] ---
    # prev_states: [B, C, H, P, N]
    # C_heads:     [B, C, L, H, N] -> einsum over n: [B, C, L, H, P]
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_heads, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)  # scale by exp(dA_l)

    # --- Intra-chunk path: sum_{s<=l} cb[l,s] * exp(dA_l - dA_s) * dt[s] * x[s] ---
    # cb: [B, C, G, L, L]; broadcast to heads [B, C, H, L, L]
    cb_chunked = cb.float()  # [B, C, G, L, L]
    cb_heads = cb_chunked[:, :, torch.arange(h, device=x.device) // heads_per_group, :, :]

    # decay[b,c,h,l,s] = exp(dA_cumsum[l] - dA_cumsum[s])
    dA_l = dA_cumsum.float().unsqueeze(-1)  # [B, H, C, L, 1]
    dA_s = dA_cumsum.float().unsqueeze(-2)  # [B, H, C, 1, L]
    decay = torch.exp(dA_l - dA_s)         # [B, H, C, L, L]

    # causal mask
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
    decay = decay.permute(0, 2, 1, 3, 4)   # [B, C, H, L, L]

    # dt: [B, H, C, L] -> [B, C, H, 1, L]
    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)  # [B, C, H, 1, L]

    # lcb[b,c,h,l,s] = cb[l,s] * decay[l,s] * dt[s]
    lcb = cb_heads * decay * dt_s  # [B, C, H, L, L]

    # y_diag[b,c,l,h,p] = sum_s lcb[b,c,h,l,s] * x[b,c,s,h,p]
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_chunked)

    # combine and reshape to [B, S, H, P]
    out = (y_off + y_diag).reshape(b, S, h, p)
    return out


class SSDChunkScanFwdTest(_SSDChunkScanFwdTestWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.n_groups)


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
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


class SSDChunkStateFwdTest(_SSDChunkStateFwdTestWorkload, TestBase):
    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups, seq_idx=seq_idx)


@SSDChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx,
):
    test = SSDChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    op = SSDChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
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

    op = SSDChunkStateFwdOp(b, c, Q, h, p, n, g, dtype, has_seq_idx=True)
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


class SSDStatePassingFwdTest(_SSDStatePassingFwdTestWorkload, TestBase):
    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)


@SSDStatePassingFwdFixture
def test_ssd_state_passing_fwd(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    op = SSDStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
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
    op = SSDStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=False)
    op.kernel.config = config
    inputs = test.gen_inputs()
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


class SSDDecodeTest(_SSDDecodeTestWorkload, TestBase):
    def ref_program(self, A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)


@SSDDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
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
    """Pure-PyTorch reference for the Mamba-2 SSD forward pass.

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
    dt_chunked = dt_val.reshape(b, num_chunks, Q, h).permute(0, 3, 1, 2)  # (B, H, C, Q)
    dA = dt_chunked * A.float().view(1, h, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)  # (B, H, C, Q)

    # Step 2: CB = C[l] @ B[s]^T per chunk, lower-triangular, group-owned
    B_c = B.float().reshape(b, num_chunks, Q, g, n)  # (B, C, Q, G, N)
    C_c = C.float().reshape(b, num_chunks, Q, g, n)  # (B, C, Q, G, N)
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", C_c, B_c)  # (B, C, G, Q, Q)
    mask = torch.ones(Q, Q, device=x.device, dtype=torch.bool).tril()
    cb = cb * mask.view(1, 1, 1, Q, Q)

    # Step 3: SSDChunkState
    decay = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)   # (B, H, C, Q)
    decay_c = decay.permute(0, 2, 3, 1)                       # (B, C, Q, H)
    dt_c = dt_chunked.permute(0, 2, 3, 1)                     # (B, C, Q, H)
    x_c = x.float().reshape(b, num_chunks, Q, h, p)           # (B, C, Q, H, P)
    B_heads = B_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]
    wx = x_c * (decay_c * dt_c).unsqueeze(-1)                 # (B, C, Q, H, P)
    chunk_states = torch.einsum("bcqhp,bcqhn->bchpn", wx, B_heads)  # (B, C, H, P, N)

    # Step 4: SSDStatePassing
    exp_dA_chunk = torch.exp(dA_cumsum[:, :, :, -1])  # (B, H, C)
    s = torch.zeros(b, h, p, n, device=x.device, dtype=torch.float32)
    prev_states_list = []
    for ci in range(num_chunks):
        prev_states_list.append(s.unsqueeze(1))
        scale = exp_dA_chunk[:, :, ci].view(b, h, 1, 1)
        s = scale * s + chunk_states[:, ci]
    prev_states = torch.cat(prev_states_list, dim=1).to(x.dtype)  # (B, C, H, P, N)

    # Step 5: SSDChunkScan
    dA_c = dA_cumsum.permute(0, 2, 3, 1)  # (B, C, Q, H)
    C_heads = C_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]

    # History: exp(dA[l]) * C[l] @ prev_state  -> (B, C, Q, H, P)
    y_hist = torch.einsum("bcqhn,bchpn->bcqhp", C_heads, prev_states.float())
    y_hist = y_hist * torch.exp(dA_c).unsqueeze(-1)

    # Intra: sum_s cb[l,s] * exp(dA[l]-dA[s]) * dt[s] * x[s]  -> (B, C, Q, H, P)
    dA_l = dA_cumsum.unsqueeze(-1)   # (B, H, C, Q, 1)
    dA_s = dA_cumsum.unsqueeze(-2)   # (B, H, C, 1, Q)
    decay_ls = torch.exp(dA_l - dA_s).masked_fill(
        ~mask.view(1, 1, 1, Q, Q), 0.0
    ).permute(0, 2, 1, 3, 4)  # (B, C, H, Q, Q)
    cb_heads = cb[:, :, torch.arange(h, device=x.device) // hpg, :, :]  # (B, C, H, Q, Q)
    lcb = cb_heads * decay_ls * dt_c.permute(0, 1, 3, 2).unsqueeze(-2)   # (B, C, H, Q, Q)
    wx_t = x_c.permute(0, 1, 3, 2, 4)   # (B, C, H, Q, P)
    # y_intra result is (B, C, H, Q, P); permute to (B, C, Q, H, P)
    y_intra = torch.einsum("bchls,bchsp->bchlp", lcb, wx_t).permute(0, 1, 3, 2, 4)

    return (y_hist + y_intra).reshape(b, S, h, p)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,seqlen,n_heads,d_head,d_state,n_groups,chunk_size", [
    (1, 256,  4, 64, 32, 1, 256),
    (2, 512,  8, 64, 64, 2, 256),
    (1, 512,  4, 32, 16, 1, 256),
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
        batch=batch,
        seqlen=seqlen,
        n_heads=n_heads,
        d_head=d_head,
        d_state=d_state,
        n_groups=n_groups,
        dtype=dtype,
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
