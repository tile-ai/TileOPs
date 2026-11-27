import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.workload_base import CallWorkload, WorkloadBase


@dataclass(eq=False)
class SSDChunkCumsumFwdWorkload(WorkloadBase):
    batch: int
    num_chunks: int
    chunk_len: int
    n_heads: int
    has_dt_bias: bool = False
    dt_softplus: bool = False
    dtype: torch.dtype = torch.float32
    dt_min: float = 0.0
    dt_max: float = float("inf")

    def gen_inputs(self):
        b, C, Q, h = self.batch, self.num_chunks, self.chunk_len, self.n_heads
        seq_len = C * Q
        # Raw dt values on both sides of the softplus; a non-positive decay A.
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


@dataclass(eq=False)
class SSDChunkScanFwdWorkload(WorkloadBase):
    batch: int
    num_chunks: int
    chunk_len: int
    n_heads: int
    d_head: int
    d_state: int
    n_groups: int
    dtype: torch.dtype

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
        atol = 1e-3 if inputs[0].dtype == torch.float16 else 2e-3
        return Exact(atol=atol, rtol=1e-5)


@dataclass(eq=False)
class SSDChunkStateFwdWorkload(WorkloadBase):
    batch: int
    num_chunks: int
    chunk_len: int
    n_heads: int
    d_head: int
    d_state: int
    n_groups: int
    dtype: torch.dtype
    has_seq_idx: bool = False

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
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device=run_device()).cumsum(-1)
        dt = torch.rand(b, h, c, Q, dtype=torch.float32, device=run_device()) * 0.1 + 0.01
        seq_idx = None
        if self.has_seq_idx:
            # Two packed sequences per batch row, split at the midpoint.
            seq_idx = torch.zeros(b, seq_len, dtype=torch.int32, device=run_device())
            seq_idx[:, seq_len // 2 :] = 1
        return x, Bmat, dt, dA_cumsum, seq_idx

    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups, seq_idx=seq_idx)

    def verification(self, *inputs):
        atol = 1e-3 if inputs[0].dtype == torch.float16 else 1.6e-2
        return Exact(atol=atol, rtol=1e-3)


@dataclass(eq=False)
class SSDDecodeWorkload(WorkloadBase):
    batch: int
    n_heads: int
    d_head: int
    d_state: int
    n_groups: int
    dtype: torch.dtype

    def gen_inputs(self):
        b, h, p, n, g = (
            self.batch,
            self.n_heads,
            self.d_head,
            self.d_state,
            self.n_groups,
        )
        # A non-positive decay and a positive, post-softplus step.
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
        return Exact(atol=1e-3, rtol=1e-3)


@dataclass(eq=False)
class SSDStatePassingFwdWorkload(WorkloadBase):
    batch: int
    num_chunks: int
    n_heads: int
    d_state: int
    dtype: torch.dtype
    has_initial_states: bool = True

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
        atol = 1e-3 if inputs[0].dtype == torch.float16 else 1.6e-2
        return Exact(atol=atol, rtol=1e-3)


class SSDStatePassingUnitWorkload(SSDStatePassingFwdWorkload):
    """Unit increments with optional periodic resets have an exact FP32 scan."""

    def __init__(self, *args, reset_period=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_period = reset_period

    def gen_inputs(self):
        b, c, h, d = self.batch, self.num_chunks, self.n_heads, self.d_state
        device = run_device()
        states = torch.ones((b, c, h, d), dtype=self.dtype, device=device)
        decay = torch.zeros((b, h, c), dtype=torch.float32, device=device)
        if self.reset_period:
            decay[:, :, :: self.reset_period] = -float("inf")
        initial = (
            torch.ones((b, h, d), dtype=torch.float32, device=device)
            if self.has_initial_states
            else None
        )
        return states, decay, initial

    def ref_program(self, states, decay, initial):
        b, c, h, d = states.shape
        position = torch.arange(c + 1, device=states.device)
        start = int(initial is not None)
        if self.reset_period:
            values = torch.where(position == 0, start, (position - 1) % self.reset_period + 1)
        else:
            values = position + start
        values = values.to(torch.float32)
        previous = values[:-1].view(1, c, 1, 1).expand(b, c, h, d).contiguous()
        final = values[-1].expand(b, h, d).contiguous()
        return previous, final

    def verification(self, *inputs):
        return Exact(atol=0, rtol=0)


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
    """``dt`` plus ``dt_bias``, softplus-ed and clamped, and the chunk-local inclusive prefix
    sum of ``dA = dt_out * A``.

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
    dt_chunked = dt_val.reshape(b, C, Q, h)
    dt_out = dt_chunked.permute(0, 3, 1, 2).contiguous().to(dtype)
    dA_cumsum = (dt_chunked * A.float()).cumsum(dim=2).permute(0, 3, 1, 2).contiguous()
    return dt_out, dA_cumsum


def ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups):
    """Float32 chunk scan in mamba_ssm's ``_chunk_scan_fwd`` layouts.

    ``out[l] = exp(dA[l]) * C[l] @ prev_state + sum_{s<=l} cb[l, s] * exp(dA[l] - dA[s])
    * dt[s] * x[s]`` within each chunk, ``cb`` and ``C`` read through each head's group.

    Inputs:
      x:           [B, S, H, P]
      cb:          [B, C, G, L, L]
      dA_cumsum:   [B, H, C, L]     float32
      C:           [B, S, G, N]
      prev_states: [B, C, H, P, N]  float32
      dt:          [B, H, C, L]

    Output: [B, S, H, P] float32
    """
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C.shape[-1]
    group_of_head = torch.arange(h, device=x.device) // (h // n_groups)
    x_chunked = x.float().reshape(b, c, L, h, p)
    C_heads = C.float().reshape(b, c, L, n_groups, n)[:, :, :, group_of_head, :]
    dA = dA_cumsum.float().permute(0, 2, 3, 1)
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_heads, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)
    cb_heads = cb.float()[:, :, group_of_head, :, :]
    decay = torch.exp(dA_cumsum.float().unsqueeze(-1) - dA_cumsum.float().unsqueeze(-2))
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0.0).permute(0, 2, 1, 3, 4)
    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)
    y_diag = torch.einsum("bchls,bcshp->bclhp", cb_heads * decay * dt_s, x_chunked)
    return (y_off + y_diag).reshape(b, S, h, p)


def ssd_chunk_state_fwd_ref(
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    n_groups: int,
    seq_idx=None,
) -> torch.Tensor:
    """Float32 state each chunk leaves from a zero start, ``[B, C, H, P, N]``.

    ``sum_l exp(dA[end] - dA[l]) * dt[l] * x[l] B[l]^T``, ``B`` read through each head's
    group. With ``seq_idx``, only the tokens of the sequence the chunk ends in, when that id
    is not negative, contribute.
    """
    b, seq_len, h, p = x.shape
    _, _, c, Q = dt.shape
    n = Bmat.shape[-1]
    x_chunked = x.float().reshape(b, c, Q, h, p)
    group_of_head = torch.arange(h, device=x.device) // (h // n_groups)
    B_heads = Bmat.float().reshape(b, c, Q, n_groups, n)[:, :, :, group_of_head, :]
    dA = dA_cumsum.float().permute(0, 2, 1, 3)
    decay = torch.exp(torch.clamp(dA[:, :, :, -1:] - dA, max=0.0))
    weight = decay * dt.float().permute(0, 2, 1, 3)
    if seq_idx is not None:
        seq_chunked = seq_idx.reshape(b, c, Q)
        seq_end = seq_chunked[..., -1:]
        same = (seq_end >= 0) & (seq_chunked == seq_end)
        weight = weight * same.unsqueeze(2)
    wx = x_chunked * weight.permute(0, 1, 3, 2).unsqueeze(-1)
    return torch.einsum("bcqhp,bcqhn->bchpn", wx, B_heads)


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 inter-chunk scan over ``states [B, C, H, ...]``: ``(prev_states, final)``.

    Matches mamba convention: ``prev_states[:, c]`` is the state before chunk ``c``, so
    ``prev_states[:, 0]`` is ``initial_states`` (zero when omitted) and ``final`` the state
    after the last chunk.
    """
    b, c, h = states.shape[:3]
    s = (
        torch.zeros(states[:, 0].shape, dtype=torch.float32, device=states.device)
        if initial_states is None
        else initial_states.float()
    )
    out = [s.clone()]
    for ci in range(c):
        decay = torch.exp(dA_chunk_cumsum[:, :, ci]).view(b, h, *(1,) * (s.dim() - 2))
        s = decay * s + states[:, ci].float()
        if ci < c - 1:
            out.append(s.clone())
    return torch.stack(out, dim=1), s


def ssd_decode_ref(
    A: torch.Tensor,
    dt: torch.Tensor,
    x: torch.Tensor,
    B_in: torch.Tensor,
    C_in: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """One Mamba-2 ``selective_state_update`` step in float32; updates *state* in place.

    ``A [H, P, N]`` and ``dt [B, H, P]`` are float32, as ``mamba2.step()`` repeats them from
    ``[H]`` and ``[B, H]``; ``x`` is ``[B, H, P]``, ``B_in`` and ``C_in`` ``[B, G, N]`` and
    ``state`` ``[B, H, P, N]``. Head ``h`` reads group ``h // (H // G)``::

        state <- exp(dt * A) * state + dt * x * B
        y      = state @ C

    Returns ``y [B, H, P]`` in float32.
    """
    heads = dt.shape[1]
    group_of_head = torch.arange(heads, device=B_in.device) // (heads // B_in.shape[1])
    B_heads = B_in.float()[:, group_of_head, :]
    C_heads = C_in.float()[:, group_of_head, :]
    dA = torch.exp(dt.float()[:, :, :, None] * A.float()[None, :, :, :])
    dBx = dt.float()[:, :, :, None] * x.float()[:, :, :, None] * B_heads[:, :, None, :]
    state.copy_(dA * state.float() + dBx)
    return torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)


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
            indices["chunk_size"],
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

    def ref_program(self, C, B):
        indices = self.call.indices
        return ssd_chunk_coupling_fwd_ref(C, B, indices["NC"], indices["chunk_size"], C.dtype)

    def verification(self, *inputs):
        return coupling_verification()


class SSDChunkStateFwdCall(CallWorkload, SSDChunkStateFwdWorkload):
    """A manifest call of SSDChunkStateFwdOp; a passed ``seq_idx`` packs two sequences."""

    def gen_inputs(self):
        x, B, dt, dA_cumsum, seq_idx = super().gen_inputs()
        if seq_idx is not None:
            seq_idx = torch.zeros_like(seq_idx)
            seq_idx[:, seq_idx.shape[1] // 2 :] = 1
        return x * 0.1, B * 0.1, _step_sizes(dt), _decay_cumsum(dA_cumsum), seq_idx

    def ref_program(self, x, B, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, B, dt, dA_cumsum, self.call.indices["G"], seq_idx=seq_idx)


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
        A, dt, x, B, C, state = super().gen_inputs()
        A = -A[:, :1, :1].abs().expand_as(A).contiguous()
        dt = _step_sizes(dt[:, :, :1]).expand_as(dt).contiguous()
        return A, dt, x * 0.1, B * 0.1, C * 0.1, state * 0.1

    def ref_program(self, A, dt, x, B, C, state):
        output = ssd_decode_ref(A, dt, x, B, C, state)
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
    """Float32 Mamba-2 State-Space Dual (SSD) forward, ``mamba_chunk_scan_combined``'s result.

    The five chunked steps in order: cumulative decay, the ``C @ B`` coupling, each chunk's
    state, the inter-chunk scan, and the chunk scan. ``x`` is ``[B, S, H, P]``, ``dt``
    ``[B, S, H]``, ``A [H]`` (log-space, <= 0), ``B`` and ``C`` ``[B, S, G, N]``, ``dt_bias``
    ``[H]`` and ``initial_states`` ``[B, H, P, N]``. Returns ``y [B, S, H, P]`` and the final
    states ``[B, H, P, N]``.
    """
    num_chunks, groups = x.shape[1] // chunk_size, B.shape[2]
    dt, dA_cumsum = ssd_chunk_cumsum_fwd_ref(
        dt, A, num_chunks, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus
    )
    cb = ssd_chunk_coupling_fwd_ref(C, B, num_chunks, chunk_size, torch.float32)
    chunk_states = ssd_chunk_state_fwd_ref(x, B, dt, dA_cumsum, groups)
    prev_states, final_states = ssd_state_passing_fwd_ref(
        chunk_states, dA_cumsum[..., -1], initial_states
    )
    return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, groups), final_states


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
    return Exact(atol=1e-3, rtol=1e-3)


def mamba2_verification(dtype):
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    return Exact(atol=atol, rtol=1e-3)


def ssd_decode_result(subject, *inputs):
    """Observe both the returned projection and the updated recurrent state."""
    output = subject(*inputs)
    return output, inputs[-1]


# The depthwise causal convolution ahead of the SSD scan: its references and manifest calls.
def _dilated_taps(history, x, weight, bias, dilation, silu):
    """Each token of ``x [q, D]`` convolved in float32 with the ``(W - 1) * dilation`` tokens of
    ``history [D, ...]`` before it, taps ``dilation`` apart, and that history followed by ``x``."""
    width = weight.shape[1]
    xp = torch.cat([history.float().t(), x.float()])
    w = weight.float()
    acc = sum(w[:, j] * xp[j * dilation : j * dilation + x.shape[0]] for j in range(width))
    if bias is not None:
        acc = acc + bias.float()
    return (F.silu(acc) if silu else acc), xp


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    silu: bool = True,
    cache_indices: torch.Tensor | None = None,
    dilation: int = 1,
) -> torch.Tensor:
    """Float32 reference of CausalConv1dFwdOp (vLLM causal_conv1d_fn; dilation as transformers
    Qwen4ExpTextPLELayer's F.conv1d); overwrites *conv_state* as the op does.

    A sequence on tokens ``[a, e)`` is preceded by its state, ``conv_state[cache_indices[s]]``
    or ``conv_state[s]``, where ``has_initial_state`` is set and by zeros otherwise; the first
    ``(W - 1) * dilation`` columns of its state become its last tokens of that history and
    ``x[a:e]``. An empty sequence keeps its state. A slot of -1 is never written; its
    sequence's outputs are unspecified and left zero.
    """
    span = (weight.shape[1] - 1) * dilation
    if conv_state.shape[2] < span or weight.shape[0] != x.shape[1]:
        raise ValueError("the state holds the dilated window of x's channels")
    out = torch.empty_like(x)
    bounds = cu_seqlens.tolist()
    for s, (a, e) in enumerate(zip(bounds, bounds[1:], strict=False)):
        slot = int(cache_indices[s]) if cache_indices is not None else s
        if slot < 0:
            out[a:e] = 0
            continue
        history = conv_state[slot, :, :span]
        zeros = torch.zeros_like(history)
        history = (
            zeros
            if has_initial_state is None
            else torch.where(has_initial_state[s], history, zeros)
        )
        acc, xp = _dilated_taps(history, x[a:e], weight, bias, dilation, silu)
        out[a:e] = acc.to(x.dtype)
        if e > a:
            conv_state[slot, :, :span] = xp[e - a :].t().to(conv_state.dtype)
    return out


def causal_conv1d_update_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: torch.Tensor | None = None,
    silu: bool = True,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    dilation: int = 1,
) -> torch.Tensor:
    """Float32 reference of CausalConv1dUpdateFwdOp (vLLM causal_conv1d_update with
    num_accepted_tokens); advances *conv_state* as the op does. A slot of -1 is never
    written; its sequence's outputs are unspecified and left zero."""
    span = (weight.shape[1] - 1) * dilation
    if weight.shape[0] != x.shape[1]:
        raise ValueError("weight must have x's channels")
    if query_start_loc is not None:
        bounds = query_start_loc.tolist()
    else:
        bounds = list(range(x.shape[0] + 1))
    out = torch.empty_like(x)
    for s, (a, e) in enumerate(zip(bounds, bounds[1:], strict=False)):
        slot = int(conv_state_indices[s]) if conv_state_indices is not None else s
        if slot < 0:
            out[a:e] = 0
            continue
        first = int(num_accepted_tokens[s]) - 1 if num_accepted_tokens is not None else 0
        history = conv_state[slot, :, first : first + span]
        acc, xp = _dilated_taps(history, x[a:e], weight, bias, dilation, silu)
        out[a:e] = acc.to(x.dtype)
        kept = span + (e - a - 1 if num_accepted_tokens is not None else 0)
        if e > a:
            conv_state[slot, :, :kept] = xp[xp.shape[0] - kept :].t().to(conv_state.dtype)
    return out


def conv_state_result(subject, *inputs, **keywords):
    """Observe the returned output and the convolution state, the third input of both ops."""
    output = subject(*inputs, **keywords)
    return output, inputs[2]


class CausalConv1dFwdCall(CallWorkload):
    """A manifest call of CausalConv1dFwdOp: filters in (-0.5, 0.5), every other sequence
    carrying a state."""

    def gen_inputs(self):
        x, weight, conv_state, cu_seqlens, bias, has_initial_state, cache_indices = (
            super().gen_inputs()
        )
        if has_initial_state is not None:
            has_initial_state = torch.arange(has_initial_state.numel(), device=x.device) % 2 == 0
        return x, weight * 0.25, conv_state, cu_seqlens, bias, has_initial_state, cache_indices

    def ref_program(
        self, x, weight, conv_state, cu_seqlens, bias, has_initial_state, cache_indices=None
    ):
        arguments = self.call.arguments()
        silu = arguments.get("activation", "silu") is not None
        output = causal_conv1d_ref(
            x, weight, conv_state, cu_seqlens, bias, has_initial_state, silu, cache_indices,
            arguments.get("dilation", 1),
        )  # fmt: skip
        return output, conv_state


class CausalConv1dUpdateFwdCall(CallWorkload):
    """A manifest call of CausalConv1dUpdateFwdOp: filters in (-0.5, 0.5)."""

    def gen_inputs(self):
        x, weight, conv_state, bias, *serving = super().gen_inputs()
        return x, weight * 0.25, conv_state, bias, *serving

    def ref_program(
        self,
        x,
        weight,
        conv_state,
        bias,
        conv_state_indices=None,
        num_accepted_tokens=None,
        query_start_loc=None,
    ):
        arguments = self.call.arguments()
        silu = arguments.get("activation", "silu") is not None
        output = causal_conv1d_update_ref(
            x, weight, conv_state, bias, silu, conv_state_indices, num_accepted_tokens,
            query_start_loc, arguments.get("dilation", 1),
        )  # fmt: skip
        return output, conv_state


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
    rates ``-(1 .. N)`` scaled per channel and ``D`` sits near one. A passed ``seq_idx``
    packs four sequences of uneven length.
    """

    def gen_inputs(self):
        u, delta, A, B, C, D, z, delta_bias, seq_idx, initial_states = super().gen_inputs()
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]
        gen = self.rng(device=u.device)
        lo, hi = math.log(1e-3), math.log(1e-1)
        dt = torch.exp(torch.rand(dim, generator=gen, device=u.device) * (hi - lo) + lo)
        inverse = dt + torch.log(-torch.expm1(-dt))
        raw = 0.1 * torch.randn(batch, dim, seqlen, generator=gen, device=u.device)
        if self.call.arguments().get("delta_softplus", True):
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
        if D is not None:
            D = 1.0 + 0.1 * torch.randn(dim, generator=gen, device=u.device)
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
        return u, delta, A, B, C, D, z, delta_bias, seq_idx, initial_states

    def ref_program(self, u, delta, A, B, C, D, z, delta_bias, seq_idx, initial_states):
        softplus = self.call.arguments().get("delta_softplus", True)
        return selective_scan_ref(
            u, delta, A, B, C, D, z, delta_bias, seq_idx, initial_states, softplus
        )
