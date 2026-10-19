import math
from typing import Optional

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import CallWorkload, FixtureBase, WorkloadBase


class SSDChunkCumsumFwdFixture(FixtureBase):
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


class SSDChunkCumsumFwdWorkload(WorkloadBase):
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
        dt_raw = torch.randn(b, seq_len, h, dtype=torch.float32, device=run_device())
        A = -torch.rand(h, dtype=torch.float32, device=run_device())
        # Absent means None: the op builds the kernel without that branch, and a
        # zero tensor would instead build the one that reads it.
        dt_bias = (
            torch.randn(h, dtype=torch.float32, device=run_device()) * 0.5
            if self.has_dt_bias
            else None
        )
        return dt_raw, A, dt_bias

    def ref_program(self, dt, A, dt_bias):
        return ssd_chunk_cumsum_fwd_ref(
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
        x = torch.randn(b, S, h, p, dtype=self.dtype, device=run_device()) * 0.1
        cb = torch.randn(b, c, g, L, L, dtype=self.dtype, device=run_device()) * 0.1
        dA_cumsum = -torch.rand(b, h, c, L, dtype=torch.float32, device=run_device()).cumsum(-1)
        C = torch.randn(b, S, g, n, dtype=self.dtype, device=run_device()) * 0.1
        prev_states = torch.randn(b, c, h, p, n, dtype=torch.float32, device=run_device()) * 0.1
        dt = torch.rand(b, h, c, L, dtype=self.dtype, device=run_device()) * 0.1 + 0.01
        return x, cb, dA_cumsum, C, prev_states, dt

    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.n_groups)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        atol = 1e-3 if inputs[0].dtype == torch.float16 else 2e-3
        return Exact(atol=atol, rtol=1e-5)


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
                    # A d_state that is not a whole number of 16-column GEMM steps.
                    pytest.param(
                        1,
                        2,
                        64,
                        4,
                        64,
                        24,
                        1,
                        torch.float16,
                        False,
                        False,
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
        x = torch.randn(b, seq_len, h, p, dtype=self.dtype, device=run_device()) * 0.1
        Bmat = torch.randn(b, seq_len, g, n, dtype=self.dtype, device=run_device()) * 0.1
        # dA_cumsum: monotonically non-increasing (negative values, cumsum of negatives)
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device=run_device()).cumsum(-1)
        dt = torch.rand(b, h, c, Q, dtype=torch.float32, device=run_device()) * 0.1 + 0.01
        seq_idx = None
        if self.has_seq_idx:
            # simulate two packed sequences per batch row, split at midpoint
            seq_idx = torch.zeros(b, seq_len, dtype=torch.int32, device=run_device())
            seq_idx[:, seq_len // 2 :] = 1
        return x, Bmat, dt, dA_cumsum, seq_idx

    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups, seq_idx=seq_idx)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        atol = 1e-3 if inputs[0].dtype == torch.float16 else 1.6e-2
        return Exact(atol=atol, rtol=1e-3)


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
        A = -torch.rand(h, p, n, dtype=torch.float32, device=run_device())
        dt = torch.rand(b, h, p, dtype=torch.float32, device=run_device()) * 0.1 + 0.01
        x = torch.randn(b, h, p, dtype=self.dtype, device=run_device()) * 0.1
        B_in = torch.randn(b, g, n, dtype=self.dtype, device=run_device()) * 0.1
        C_in = torch.randn(b, g, n, dtype=self.dtype, device=run_device()) * 0.1
        state = torch.randn(b, h, p, n, dtype=torch.float32, device=run_device()) * 0.1
        return A, dt, x, B_in, C_in, state

    def ref_program(self, A, dt, x, B_in, C_in, state):
        output = ssd_decode_ref(A, dt, x, B_in, C_in, state)
        return output, state

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=1e-3, rtol=1e-3)


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
        states = torch.randn(b, c, h, d, dtype=self.dtype, device=run_device()) * 0.1
        dA_chunk_cumsum = -torch.rand(b, h, c, dtype=torch.float32, device=run_device()).cumsum(-1)
        # Absent means None: the op then builds the kernel that starts from zero.
        initial_states = (
            torch.randn(b, h, d, dtype=torch.float32, device=run_device()) * 0.1
            if self.has_initial_states
            else None
        )
        return states, dA_chunk_cumsum, initial_states

    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        atol = 1e-3 if inputs[0].dtype == torch.float16 else 1.6e-2
        return Exact(atol=atol, rtol=1e-3)


def ssd_chunk_cumsum_fwd_ref(
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
    """PyTorch reference for ssd_chunk_cumsum_fwd.

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


def ssd_chunk_coupling_fwd_ref(
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


# Manifest calls: shapes, dtypes and presence come from a workload row; these
# classes condition the values the row does not determine and carry the reference.


def _decay_cumsum(like: torch.Tensor) -> torch.Tensor:
    """A non-increasing float32 cumulative sum of negative decays, shaped like *like*."""
    return -torch.rand(like.shape, dtype=torch.float32, device=like.device).cumsum(-1)


def _step_sizes(like: torch.Tensor) -> torch.Tensor:
    """Positive post-softplus step sizes in ``[0.01, 0.11)``, with *like*'s shape and dtype."""
    return torch.rand(like.shape, device=like.device).to(like.dtype) * 0.1 + 0.01


class SSDChunkCumsumFwdCall(CallWorkload, SSDChunkCumsumFwdWorkload):
    """A manifest call of SSDChunkCumsumFwdOp with a negative decay ``A``."""

    def gen_inputs(self):
        dt, A, dt_bias = super().gen_inputs()
        return dt, -A.abs(), None if dt_bias is None else dt_bias * 0.5

    def ref_program(self, dt, A, dt_bias):
        indices = self.call.indices
        return ssd_chunk_cumsum_fwd_ref(
            dt,
            A,
            indices["NC"],
            indices["chunk_len"],
            dt_bias=dt_bias,
            dt_softplus=indices["dt_softplus"],
            dt_min=indices["dt_min"],
            dt_max=indices["dt_max"],
            dtype=getattr(torch, indices["out_dtype"]),
        )


class SSDChunkCouplingFwdCall(CallWorkload):
    """A manifest call of SSDChunkCouplingFwdOp."""

    def gen_inputs(self):
        return tuple(t * 0.1 for t in super().gen_inputs())

    def ref_program(self, C_mat, B_mat):
        indices = self.call.indices
        return ssd_chunk_coupling_fwd_ref(
            C_mat, B_mat, indices["NC"], indices["chunk_len"], C_mat.dtype
        )

    def verification(self, *inputs):
        return coupling_verification()


class SSDChunkStateFwdCall(CallWorkload, SSDChunkStateFwdWorkload):
    """A manifest call of SSDChunkStateFwdOp; a passed ``seq_idx`` packs two sequences."""

    def gen_inputs(self):
        x, Bmat, dt, dA_cumsum, seq_idx = super().gen_inputs()
        if seq_idx is not None:
            seq_idx = torch.zeros_like(seq_idx)
            seq_idx[:, seq_idx.shape[1] // 2 :] = 1
        return x * 0.1, Bmat * 0.1, _step_sizes(dt), _decay_cumsum(dA_cumsum), seq_idx

    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(
            x, Bmat, dt, dA_cumsum, self.call.indices["G"], seq_idx=seq_idx
        )


class SSDStatePassingFwdCall(CallWorkload, SSDStatePassingFwdWorkload):
    """A manifest call of SSDStatePassingFwdOp."""

    def gen_inputs(self):
        states, dA_chunk_cumsum, initial_states = super().gen_inputs()
        return (
            states * 0.1,
            _decay_cumsum(dA_chunk_cumsum),
            None if initial_states is None else initial_states * 0.1,
        )

    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)


class SSDChunkScanFwdCall(CallWorkload, SSDChunkScanFwdWorkload):
    """A manifest call of SSDChunkScanFwdOp."""

    def gen_inputs(self):
        x, cb, dA_cumsum, C, prev_states, dt = super().gen_inputs()
        return (
            x * 0.1,
            cb * 0.1,
            _decay_cumsum(dA_cumsum),
            C * 0.1,
            prev_states * 0.1,
            _step_sizes(dt),
        )

    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.call.indices["G"])


class SSDDecodeFwdCall(CallWorkload, SSDDecodeWorkload):
    """Mamba-2 decode: each head shares its decay rate and time step across channels.

    Dense copies retain the op's input layout; the generic SSDDecodeWorkload also
    covers independent channel/state rates.
    """

    def gen_inputs(self):
        A, dt, x, B_in, C_in, state = super().gen_inputs()
        A = -A[:, :1, :1].abs().expand_as(A).contiguous()
        dt = _step_sizes(dt[:, :, :1]).expand_as(dt).contiguous()
        return A, dt, x * 0.1, B_in * 0.1, C_in * 0.1, state * 0.1

    def ref_program(self, A, dt, x, B_in, C_in, state):
        output = ssd_decode_ref(A, dt, x, B_in, C_in, state)
        return output, state


# The end-to-end Mamba-2 SSD forward pass: its reference and its manifest calls.
def mamba2_fwd_ref(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
    dt_softplus: bool,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the Mamba-2 State-Space Dual (SSD) forward pass.

    Computes the same result as mamba_chunk_scan_combined from mamba_ssm:
      out[l,p] = exp(dA[l]) * C[l] @ prev_state
               + sum_{s<=l} (C[l]@B[s]) * exp(dA[l]-dA[s]) * dt[s] * x[s,p]

    Inputs:
        x:              (B, S, H, P)     dtype
        dt:             (B, S, H)        float32
        A:              (H,)             float32  (log-space, <= 0)
        B:              (B, S, G, N)     dtype
        C:              (B, S, G, N)     dtype
        dt_bias:        (H,)             float32, optional
        chunk_size:     int
        dt_softplus:    bool
        initial_states: (B, H, P, N)     float32, optional

    Returns:
        y:            (B, S, H, P)  float32
        final_states: (B, H, P, N)  float32
    """
    b, S, h, p = x.shape
    n = B.shape[-1]
    g = B.shape[2]
    hpg = h // g
    Q = chunk_size
    num_chunks = S // Q

    # Step 1: SSDChunkCumsum
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
    s = (
        torch.zeros(b, h, p, n, device=x.device, dtype=torch.float32)
        if initial_states is None
        else initial_states.float().clone()
    )
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
    decay_ls = (
        torch.exp(dA_l - dA_s).masked_fill(~mask.view(1, 1, 1, Q, Q), 0.0).permute(0, 2, 1, 3, 4)
    )
    cb_heads = cb[:, :, torch.arange(h, device=x.device) // hpg, :, :]
    lcb = cb_heads * decay_ls * dt_c.permute(0, 1, 3, 2).unsqueeze(-2)
    wx_t = x_c.permute(0, 1, 3, 2, 4)
    y_intra = torch.einsum("bchls,bchsp->bchlp", lcb, wx_t).permute(0, 1, 3, 2, 4)

    return (y_hist + y_intra).reshape(b, S, h, p), s


class Mamba2FwdCall(CallWorkload):
    """A manifest call of Mamba2FwdOp with a negative decay ``A``."""

    def gen_inputs(self):
        x, dt, A, B, C, dt_bias, initial_states = super().gen_inputs()
        return (
            x * 0.1,
            dt * 0.5,
            -A.abs(),
            B * 0.1,
            C * 0.1,
            None if dt_bias is None else dt_bias * 0.1,
            None if initial_states is None else initial_states * 0.1,
        )

    def ref_program(self, x, dt, A, B, C, dt_bias, initial_states):
        indices = self.call.indices
        return mamba2_fwd_ref(
            x, dt, A, B, C, dt_bias, indices["chunk_size"], indices["dt_softplus"], initial_states
        )

    def verification(self, *inputs):
        return mamba2_verification(inputs[0].dtype)


def coupling_verification():
    from workloads.numerics import Exact

    return Exact(atol=1e-3, rtol=1e-3)


def mamba2_verification(dtype):
    from workloads.numerics import Exact

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    return Exact(atol=atol, rtol=1e-3)


def ssd_decode_result(subject, *inputs):
    """Observe both the returned projection and the updated recurrent state."""
    output = subject(*inputs)
    return output, inputs[-1]


# The depthwise causal convolution ahead of the SSD scan: its references and manifest calls.
def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    silu: bool = True,
) -> torch.Tensor:
    """Float32 reference of CausalConv1dFwdOp; overwrites *conv_state* as the op does.

    A sequence on tokens ``[a, e)`` is preceded by its state where ``has_initial_state`` is
    set and by zeros otherwise; its state becomes its last ``W - 1`` tokens of that history
    and ``x[a:e]``. An empty sequence keeps its state, as vLLM's causal_conv1d_fn leaves it.
    """
    width = weight.shape[1]
    w = weight.float().t()  # (W, D)
    out = torch.empty_like(x)
    starts = cu_seqlens.tolist()
    flags = [False] * (len(starts) - 1) if has_initial_state is None else has_initial_state.tolist()
    for s in range(len(starts) - 1):
        a, e = starts[s], starts[s + 1]
        hist = conv_state[s].float().t() if flags[s] else x.new_zeros(width - 1, x.shape[1]).float()
        xp = torch.cat([hist, x[a:e].float()])
        acc = sum(w[k] * xp[k : k + e - a] for k in range(width))
        if bias is not None:
            acc = acc + bias.float()
        out[a:e] = (F.silu(acc) if silu else acc).to(x.dtype)
        if e > a:
            conv_state[s] = xp[e - a :].t().to(conv_state.dtype)
    return out


def causal_conv1d_update_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu: bool = True,
) -> torch.Tensor:
    """Float32 reference of CausalConv1dUpdateFwdOp; advances *conv_state* as the op does."""
    xp = torch.cat([conv_state.float(), x.float()[:, :, None]], dim=-1)  # (B, D, W)
    acc = (xp * weight.float()).sum(-1)
    if bias is not None:
        acc = acc + bias.float()
    conv_state.copy_(xp[:, :, 1:].to(conv_state.dtype))
    return (F.silu(acc) if silu else acc).to(x.dtype)


def conv_state_result(subject, *inputs):
    """Observe the returned output and the convolution state, the third input of both ops."""
    output = subject(*inputs)
    return output, inputs[2]


class CausalConv1dFwdCall(CallWorkload):
    """A manifest call of CausalConv1dFwdOp: filters in (-0.5, 0.5), every other sequence
    carrying a state."""

    def gen_inputs(self):
        x, weight, conv_state, cu_seqlens, bias, has_initial_state = super().gen_inputs()
        if has_initial_state is not None:
            has_initial_state = torch.arange(has_initial_state.numel(), device=x.device) % 2 == 0
        return x, weight * 0.25, conv_state, cu_seqlens, bias, has_initial_state

    def ref_program(self, x, weight, conv_state, cu_seqlens, bias, has_initial_state):
        silu = self.call.arguments({}).get("activation", "silu") is not None
        output = causal_conv1d_ref(x, weight, conv_state, cu_seqlens, bias, has_initial_state, silu)
        return output, conv_state


class CausalConv1dUpdateFwdCall(CallWorkload):
    """A manifest call of CausalConv1dUpdateFwdOp: filters in (-0.5, 0.5)."""

    def gen_inputs(self):
        x, weight, conv_state, bias = super().gen_inputs()
        return x, weight * 0.25, conv_state, bias

    def ref_program(self, x, weight, conv_state, bias):
        silu = self.call.arguments({}).get("activation", "silu") is not None
        return causal_conv1d_update_ref(x, weight, conv_state, bias, silu), conv_state


def selective_scan_ref(
    u,
    delta,
    A,
    B_in,
    C_in,
    D_in=None,
    z=None,
    delta_bias=None,
    seq_idx=None,
    initial_states=None,
    delta_softplus=True,
):
    """Mamba-1 S6 selective scan in float32, one step at a time: ``(y, final_states)``.

    The recurrence of ``mamba_ssm``'s ``selective_scan_ref``, with ``seq_idx`` restarting the
    state at every step whose id differs from the step before, and ``initial_states`` the
    state before the first step.
    """
    batch, dim, seqlen = u.shape
    groups = B_in.shape[1]
    uf = u.float()
    dt = delta.float() + (delta_bias.float()[:, None] if delta_bias is not None else 0)
    if delta_softplus:
        dt = F.softplus(dt)
    rep = dim // groups
    Bf = B_in.float().repeat_interleave(rep, 1)
    Cf = C_in.float().repeat_interleave(rep, 1)
    h = (
        initial_states.float().clone()
        if initial_states is not None
        else torch.zeros(batch, dim, A.shape[1], device=u.device)
    )
    ys = []
    for t in range(seqlen):
        if seq_idx is not None and t > 0:
            restart = seq_idx[:, t] != seq_idx[:, t - 1]
            h = torch.where(restart[:, None, None], 0.0, h)
        h = (
            torch.exp(dt[:, :, t, None] * A) * h
            + (dt[:, :, t] * uf[:, :, t])[..., None] * Bf[..., t]
        )
        ys.append((h * Cf[..., t]).sum(-1))
    y = torch.stack(ys, -1)
    if D_in is not None:
        y = y + uf * D_in.float()[:, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(u.dtype), h


class SelectiveScanFwdCall(CallWorkload):
    """A manifest call of SelectiveScanFwdOp, at the magnitudes of a Mamba-1 block.

    ``dt`` is log-uniform in [1e-3, 1e-1] after the softplus, as Mamba-1 initializes it, with
    ``delta_bias`` its inverse softplus where the call passes one. ``A`` holds the S4D-real
    rates ``-(1 .. N)`` scaled per channel and ``D_in`` sits near one. A passed ``seq_idx``
    packs four sequences of uneven length.
    """

    def gen_inputs(self):
        u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states = super().gen_inputs()
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]
        gen = self.rng(device=u.device)
        lo, hi = math.log(1e-3), math.log(1e-1)
        dt = torch.exp(torch.rand(dim, generator=gen, device=u.device) * (hi - lo) + lo)
        inverse = dt + torch.log(-torch.expm1(-dt))
        raw = 0.1 * torch.randn(batch, dim, seqlen, generator=gen, device=u.device)
        if self.call.arguments({}).get("delta_softplus", True):
            delta = (raw if delta_bias is not None else raw + inverse[:, None]).to(delta.dtype)
            if delta_bias is not None:
                delta_bias = inverse
        else:
            # Without the softplus, delta is the step itself: positive, around dt.
            delta = (dt[:, None] * torch.exp(raw)).to(delta.dtype)
            if delta_bias is not None:
                delta_bias = torch.zeros_like(dt)
        scale = 0.5 + torch.rand(dim, 1, generator=gen, device=u.device)
        A = -torch.arange(1, dstate + 1, dtype=torch.float32, device=u.device)[None, :] * scale
        if D_in is not None:
            D_in = 1.0 + 0.1 * torch.randn(dim, generator=gen, device=u.device)
        if seq_idx is not None:
            # Four sequences, cut at 25%, 62% and 76% of the batch row.
            cuts = [0, seqlen // 4, seqlen * 31 // 50, seqlen * 19 // 25, seqlen]
            ids = torch.cat(
                [
                    torch.full((b - a,), i, dtype=torch.int32)
                    for i, (a, b) in enumerate(zip(cuts, cuts[1:], strict=False))
                ]
            )
            seq_idx = ids.to(u.device)[None].expand(batch, seqlen).contiguous()
        if initial_states is not None:
            initial_states = 0.5 * torch.randn(batch, dim, dstate, generator=gen, device=u.device)
        return u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states

    def ref_program(self, u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states):
        softplus = self.call.arguments({}).get("delta_softplus", True)
        return selective_scan_ref(
            u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states, softplus
        )
