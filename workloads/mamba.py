from typing import Optional

import torch
import torch.nn.functional as F

from workloads.workload_base import FixtureBase, WorkloadBase


class DaCumsumFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype, tune",
                [
                    # feature: no bias, no softplus (baseline path)
                    pytest.param(
                        1, 2, 64, 4, False, False, torch.float16, False, marks=pytest.mark.smoke
                    ),
                    # feature: bias only (has_dt_bias branch, no softplus)
                    pytest.param(
                        1, 2, 64, 4, True, False, torch.bfloat16, False, marks=pytest.mark.smoke
                    ),
                    # feature: softplus only (no bias, dt_softplus branch)
                    pytest.param(
                        1, 2, 64, 4, False, True, torch.float16, False, marks=pytest.mark.smoke
                    ),
                    # feature: bias + softplus (full pipeline)
                    pytest.param(
                        1, 2, 64, 4, True, True, torch.bfloat16, False, marks=pytest.mark.full
                    ),
                    # shape: larger batch and chunk count
                    pytest.param(
                        2, 4, 64, 8, False, False, torch.float16, False, marks=pytest.mark.full
                    ),
                    # shape: larger chunk_len tile
                    pytest.param(
                        1, 2, 128, 4, False, False, torch.bfloat16, False, marks=pytest.mark.full
                    ),
                    # shape + feature: large shape with full pipeline
                    pytest.param(
                        2, 4, 128, 16, True, True, torch.float16, False, marks=pytest.mark.full
                    ),
                ],
            ),
        ]


class DaCumsumFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        has_dt_bias: bool = False,
        dt_softplus: bool = False,
        dtype: torch.dtype = torch.float32,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.has_dt_bias = has_dt_bias
        self.dt_softplus = dt_softplus
        self.dtype = dtype
        self.dt_min = dt_min
        self.dt_max = dt_max

    def gen_inputs(self):
        b, C, Q, h = self.batch, self.num_chunks, self.chunk_len, self.n_heads
        seq_len = C * Q
        # Raw dt values; softplus maps R -> R+, so randn covers both sides of the nonlinearity.
        # A <= 0 (negative decay)
        dt_raw = torch.randn(b, seq_len, h, dtype=torch.float32, device="cuda")
        A = -torch.rand(h, dtype=torch.float32, device="cuda")
        # Absent means None: the op builds the kernel without that branch, and a
        # zero tensor would instead build the one that reads it.
        dt_bias = (
            torch.randn(h, dtype=torch.float32, device="cuda") * 0.5 if self.has_dt_bias else None
        )
        return dt_raw, A, dt_bias

    def ref_program(self, dt, A, dt_bias):
        return da_cumsum_fwd_ref(
            dt,
            A,
            self.num_chunks,
            self.chunk_len,
            dt_bias=dt_bias if self.has_dt_bias else None,
            dt_softplus=self.dt_softplus,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dtype=self.dtype,
        )


class SSDChunkScanFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune",
                [
                    pytest.param(
                        1, 2, 64, 4, 64, 32, 1, torch.float16, False, marks=pytest.mark.smoke
                    ),
                    pytest.param(
                        1, 2, 128, 4, 128, 32, 1, torch.bfloat16, False, marks=pytest.mark.smoke
                    ),
                    pytest.param(
                        2, 4, 64, 8, 64, 64, 2, torch.float16, False, marks=pytest.mark.full
                    ),
                    pytest.param(
                        2, 2, 64, 4, 64, 32, 2, torch.bfloat16, False, marks=pytest.mark.full
                    ),
                ],
            ),
        ]


class SSDChunkScanFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, c, L, h, p, n, g = (
            self.batch,
            self.num_chunks,
            self.chunk_len,
            self.n_heads,
            self.d_head,
            self.d_state,
            self.n_groups,
        )
        S = c * L

        # Official layouts (aligned with _chunk_scan_fwd in mamba_ssm)
        x = torch.randn(b, S, h, p, dtype=self.dtype, device="cuda") * 0.1
        cb = torch.randn(b, c, g, L, L, dtype=self.dtype, device="cuda") * 0.1
        dA_cumsum = -torch.rand(b, h, c, L, dtype=torch.float32, device="cuda").cumsum(-1)
        C = torch.randn(b, S, g, n, dtype=self.dtype, device="cuda") * 0.1
        prev_states = torch.randn(b, c, h, p, n, dtype=torch.float32, device="cuda") * 0.1
        dt = torch.rand(b, h, c, L, dtype=self.dtype, device="cuda") * 0.1 + 0.01
        return x, cb, dA_cumsum, C, prev_states, dt

    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.n_groups)


class SSDChunkStateFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx",
                [
                    pytest.param(
                        1,
                        2,
                        64,
                        4,
                        64,
                        32,
                        1,
                        torch.float16,
                        False,
                        False,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        1,
                        2,
                        128,
                        4,
                        128,
                        32,
                        1,
                        torch.bfloat16,
                        False,
                        False,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        2,
                        4,
                        64,
                        8,
                        64,
                        64,
                        2,
                        torch.float16,
                        False,
                        False,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        2,
                        2,
                        64,
                        4,
                        64,
                        32,
                        2,
                        torch.bfloat16,
                        False,
                        False,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        2,
                        4,
                        64,
                        8,
                        64,
                        64,
                        2,
                        torch.float16,
                        False,
                        True,
                        marks=pytest.mark.full,
                    ),
                ],
            ),
        ]


class SSDChunkStateFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        has_seq_idx: bool = False,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.has_seq_idx = has_seq_idx

    def gen_inputs(self):
        b, c, Q, h, p, n, g = (
            self.batch,
            self.num_chunks,
            self.chunk_len,
            self.n_heads,
            self.d_head,
            self.d_state,
            self.n_groups,
        )
        seq_len = c * Q
        x = torch.randn(b, seq_len, h, p, dtype=self.dtype, device="cuda") * 0.1
        Bmat = torch.randn(b, seq_len, g, n, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: monotonically non-increasing (negative values, cumsum of negatives)
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
        dt = torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        seq_idx = None
        if self.has_seq_idx:
            # simulate two packed sequences per batch row, split at midpoint
            seq_idx = torch.zeros(b, seq_len, dtype=torch.int32, device="cuda")
            seq_idx[:, seq_len // 2 :] = 1
        return x, Bmat, dt, dA_cumsum, seq_idx

    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups, seq_idx=seq_idx)


class SSDDecodeFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, n_heads, d_head, d_state, n_groups, dtype, tune",
                [
                    pytest.param(
                        1,
                        4,
                        64,
                        16,
                        1,
                        torch.float16,
                        False,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        1,
                        4,
                        64,
                        16,
                        1,
                        torch.bfloat16,
                        False,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        2,
                        8,
                        64,
                        32,
                        2,
                        torch.float16,
                        False,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        2,
                        8,
                        128,
                        64,
                        4,
                        torch.bfloat16,
                        False,
                        marks=pytest.mark.full,
                    ),
                ],
            ),
        ]


class SSDDecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, h, p, n, g = (
            self.batch,
            self.n_heads,
            self.d_head,
            self.d_state,
            self.n_groups,
        )
        # A <= 0 (negative decay), dt > 0 (post-softplus)
        A = -torch.rand(h, p, n, dtype=torch.float32, device="cuda")
        dt = torch.rand(b, h, p, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        x = torch.randn(b, h, p, dtype=self.dtype, device="cuda") * 0.1
        B_in = torch.randn(b, g, n, dtype=self.dtype, device="cuda") * 0.1
        C_in = torch.randn(b, g, n, dtype=self.dtype, device="cuda") * 0.1
        state = torch.randn(b, h, p, n, dtype=torch.float32, device="cuda") * 0.1
        return A, dt, x, B_in, C_in, state

    def ref_program(self, A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)


class SSDStatePassingFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, num_chunks, n_heads, d_state, dtype, tune",
                [
                    pytest.param(1, 2, 4, 32, torch.float16, False, marks=pytest.mark.smoke),
                    pytest.param(1, 2, 4, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
                    pytest.param(2, 4, 8, 64, torch.float16, False, marks=pytest.mark.full),
                    pytest.param(2, 4, 8, 64, torch.bfloat16, False, marks=pytest.mark.full),
                ],
            ),
        ]


class SSDStatePassingFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_state: int,
        dtype: torch.dtype,
        has_initial_states: bool = True,
    ):
        self.has_initial_states = has_initial_states
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_state = d_state
        self.dtype = dtype

    def gen_inputs(self):
        b, c, h, d = self.batch, self.num_chunks, self.n_heads, self.d_state
        states = torch.randn(b, c, h, d, dtype=self.dtype, device="cuda") * 0.1
        dA_chunk_cumsum = -torch.rand(b, h, c, dtype=torch.float32, device="cuda").cumsum(-1)
        # Absent means None: the op then builds the kernel that starts from zero.
        initial_states = (
            torch.randn(b, h, d, dtype=torch.float32, device="cuda") * 0.1
            if self.has_initial_states
            else None
        )
        return states, dA_chunk_cumsum, initial_states

    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)


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
    dt_chunked = dt_val.reshape(b, C, Q, h)  # (b, C, Q, h)
    dt_out = dt_chunked.permute(0, 3, 1, 2).contiguous().to(dtype)  # (b, h, C, Q) in target dtype
    dA = dt_chunked * A.float()  # (b, C, Q, h)
    dA_cumsum = dA.cumsum(dim=2).permute(0, 3, 1, 2).contiguous()  # (b, h, C, Q)
    return dt_out, dA_cumsum


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

    x_chunked = x.float().reshape(b, c, L, h, p)  # [B, C, L, H, P]
    C_chunked = C.float().reshape(b, c, L, g, n)  # [B, C, L, G, N]
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
    decay = torch.exp(dA_l - dA_s)  # [B, H, C, L, L]

    # causal mask
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
    decay = decay.permute(0, 2, 1, 3, 4)  # [B, C, H, L, L]

    # dt: [B, H, C, L] -> [B, C, H, 1, L]
    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)  # [B, C, H, 1, L]

    # lcb[b,c,h,l,s] = cb[l,s] * decay[l,s] * dt[s]
    lcb = cb_heads * decay * dt_s  # [B, C, H, L, L]

    # y_diag[b,c,l,h,p] = sum_s lcb[b,c,h,l,s] * x[b,c,s,h,p]
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_chunked)

    # combine and reshape to [B, S, H, P]
    out = (y_off + y_diag).reshape(b, S, h, p)
    return out


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


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the inter-chunk recurrent scan.

    Matches mamba convention: out[:,c] = state *before* processing chunk c,
    so out[:,0] = initial_states and final_states = state after chunk C-1.
    """
    b, c, h, d = states.shape
    # out[:,0] = s_{-1} = initial_states, or zero when the call omits it.
    s = (
        torch.zeros(b, h, d, dtype=torch.float32, device=states.device)
        if initial_states is None
        else initial_states.float()
    )
    out = [s.clone()]

    for ci in range(c):
        scale = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1)
        u = states[:, ci, :, :].float()
        s = scale * s + u
        if ci < c - 1:
            out.append(s.clone())

    return torch.stack(out, dim=1), s


def ssd_decode_ref(
    A: torch.Tensor,  # (H, P, N)     float32
    dt: torch.Tensor,  # (B, H, P)     float32
    x: torch.Tensor,  # (B, H, P)     any dtype
    B_in: torch.Tensor,  # (B, G, N)     any dtype
    C_in: torch.Tensor,  # (B, G, N)     any dtype
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
    B_heads = B_in.float()[:, head_idx, :]  # (B, H, N)
    C_heads = C_in.float()[:, head_idx, :]  # (B, H, N)

    # dA[b, h, p, n] = exp(dt[b,h,p] * A[h,p,n])
    dA = torch.exp(dt.float()[:, :, :, None] * A.float()[None, :, :, :])  # (B, H, P, N)

    # dBx[b, h, p, n] = dt[b,h,p] * B[b,h,n] * x[b,h,p]
    dBx = (
        dt.float()[:, :, :, None] * x.float()[:, :, :, None] * B_heads[:, :, None, :]
    )  # (B, H, P, N)

    # Update state in-place
    new_state = dA * state.float() + dBx
    state.copy_(new_state)

    # y_out[b, h, p] = sum_n state[b, h, p, n] * C[b, h, n]
    y_out = torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)
    return y_out


def cb_producer_fwd_ref(
    C_mat: torch.Tensor,
    B_mat: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal C@B coupling matrix: ``cb[b,c,g,l,s] = C[b,cQ+l,g,:] @ B[b,cQ+s,g,:]`` for s <= l."""
    batch, _, groups, state = C_mat.shape
    c_chunked = C_mat.reshape(batch, num_chunks, chunk_len, groups, state)
    b_chunked = B_mat.reshape(batch, num_chunks, chunk_len, groups, state)
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", c_chunked.float(), b_chunked.float())
    mask = torch.tril(torch.ones(chunk_len, chunk_len, device=C_mat.device, dtype=torch.bool))
    return (cb * mask).to(dtype)


class CBProducerFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.dtype = dtype

    def gen_inputs(self):
        shape = (self.batch, self.num_chunks * self.chunk_len, self.n_groups, self.d_state)
        c_mat = torch.randn(shape, dtype=self.dtype, device="cuda") * 0.1
        b_mat = torch.randn(shape, dtype=self.dtype, device="cuda") * 0.1
        return c_mat, b_mat

    def ref_program(self, C_mat, B_mat):
        return cb_producer_fwd_ref(C_mat, B_mat, self.num_chunks, self.chunk_len, self.dtype)
