import pytest
import torch

from tests.test_base import TestBase, allclose_compare
from tileops.ops.da_cumsum import DaCumsumFwdOp
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
) -> torch.Tensor:
    """PyTorch reference for da_cumsum_fwd."""
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks
    dt_chunked = dt.float().reshape(b, C, Q, h)
    dA = dt_chunked * A.float()
    dA_cumsum = dA.cumsum(dim=2)
    return dA_cumsum.permute(0, 3, 1, 2).contiguous()


class DaCumsumFwdTest(_DaCumsumFwdTestWorkload, TestBase):
    def ref_program(self, dt, A):
        return da_cumsum_fwd_ref(dt, A, self.num_chunks, self.chunk_len)


@DaCumsumFwdFixture
def test_da_cumsum_fwd(batch, num_chunks, chunk_len, n_heads, tune):
    test = DaCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    op = DaCumsumFwdOp(
        batch, num_chunks, chunk_len, n_heads,
        seq_len=num_chunks * chunk_len,
        tune=tune,
    )
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-5, rtol=1e-5)


def ssd_chunk_scan_fwd_torch(x, cb, dA_cumsum, C, prev_states, dt):
    """Triton-aligned PyTorch reference for chunk scan."""
    b, c, L, h, p = x.shape

    y_off = torch.einsum("bclhn,bchnp->bclhp", C.float(), prev_states.float())
    a_l = dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1)
    y_off = y_off * torch.exp(a_l)

    a_lhs = dA_cumsum.unsqueeze(-1)
    a_rhs = dA_cumsum.unsqueeze(-2)
    decay = torch.exp(a_lhs - a_rhs)
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0)
    decay = decay.permute(0, 2, 1, 3, 4)
    dt_s = dt.float().permute(0, 1, 3, 2).unsqueeze(-2)
    lcb = cb.float() * decay * dt_s
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x.float())

    return y_off + y_diag


class SSDChunkScanFwdTest(_SSDChunkScanFwdTestWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_torch(x, cb, dA_cumsum, C, prev_states, dt)


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune):
    test = SSDChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)
    op = SSDChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
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
        same = (seq_chunked == seq_end).unsqueeze(3)
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


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the inter-chunk recurrent scan."""
    b, c, h, d = states.shape
    out = []
    s = initial_states.float()

    for ci in range(c):
        scale = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1)
        u = states[:, ci, :, :].float()
        s = scale * s + u
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


def ssd_decode_ref(
    A: torch.Tensor,      # (H,)          float32
    dt: torch.Tensor,     # (B, H)        float32
    x: torch.Tensor,      # (B, H, P)     any dtype
    B_in: torch.Tensor,   # (B, G, N)     any dtype
    C_in: torch.Tensor,   # (B, G, N)     any dtype
    state: torch.Tensor,  # (B, H, P, N)  float32  -- updated in-place
) -> torch.Tensor:
    """PyTorch reference for ssd_decode."""
    B, H = dt.shape
    G = B_in.shape[1]
    heads_per_group = H // G

    dA = torch.exp(dt.float() * A.float())

    head_idx = torch.arange(H, device=B_in.device) // heads_per_group
    B_heads = B_in.float()[:, head_idx, :]
    C_heads = C_in.float()[:, head_idx, :]

    dBx = (
        dt.float()[:, :, None, None]
        * x.float()[:, :, :, None]
        * B_heads[:, :, None, :]
    )

    new_state = dA[:, :, None, None] * state.float() + dBx
    state.copy_(new_state)

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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
