"""Workload definitions for the linear_attention op family."""

import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


class DeltaNetFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        q = torch.randn(B, H, S, DK, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(B, H, S, DK, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(B, H, S, DV, device=run_device(), dtype=self.dtype) * 0.1
        beta = torch.rand(B, H, S, device=run_device(), dtype=self.dtype) * 0.5
        return q, k, v, beta

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        B, H, S, DK = k.shape
        _, _, _, DV = v.shape
        Aw, Au = prepare_wy_repr_deltanet_torch(k, beta, self.chunk_size)
        w, u = compute_w_u_torch(Aw, Au, k, v, beta, self.chunk_size)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        _S, o = kernel2_deltanet_torch(q, k, w, u, S_0, self.chunk_size)
        return o.to(self.dtype)


class DeltaNetDecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device=run_device(), dtype=self.dtype) * 0.1
        beta = torch.rand(B, H, device=run_device(), dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device=run_device(), dtype=self.dtype) * 0.1
        return q, k, v, beta, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = deltanet_decode_torch(q, k, v, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)


class GLADecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        scale: float = -1.0,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.scale = scale

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device=run_device(), dtype=self.dtype) * 0.1
        gk = -torch.rand(B, H, DK, device=run_device(), dtype=self.dtype)
        state = torch.randn(B, H, DK, DV, device=run_device(), dtype=self.dtype) * 0.1
        return q, k, v, gk, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gla_decode_torch(q, k, v, gk, state, self.scale)
        return o.to(self.dtype), new_state.to(self.dtype)


class DeltaNetInferenceWorkload(WorkloadBase):
    """BTHD ungated DeltaNet prefill or single-token decode with recurrent state.

    ``sequence_lengths`` packs the rows into one ``B = 1`` token axis and makes the call
    carry ``cu_seqlens``.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        sequence_lengths: tuple[int, ...] | None = None,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.sequence_lengths = sequence_lengths

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        rows = (
            (self.batch, self.seq_len)
            if self.sequence_lengths is None
            else (1, sum(self.sequence_lengths))
        )
        shape = (*rows, self.heads, self.dim)
        q = torch.randn(shape, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(shape, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(shape, device=run_device(), dtype=self.dtype) * 0.1
        beta = torch.rand(shape[:3], device=run_device(), dtype=self.dtype) * 0.5
        sequences = self.batch if self.sequence_lengths is None else len(self.sequence_lengths)
        initial_state = (
            torch.randn(
                sequences, self.heads, self.dim, self.dim, device=run_device(), dtype=torch.float32
            )
            * 0.01
        )
        if self.sequence_lengths is None:
            return q, k, v, beta, initial_state
        offsets = torch.zeros(sequences + 1, dtype=torch.int64)
        offsets[1:] = torch.tensor(self.sequence_lengths, dtype=torch.int64).cumsum(0)
        return q, k, v, beta, initial_state, offsets.to(run_device()), offsets

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

        del cu_seqlens_cpu
        fla_kernel = fused_recurrent_delta_rule if self.seq_len == 1 else chunk_delta_rule
        return fla_kernel(
            q,
            k,
            v,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )


class GatedDeltaNetFwdWorkload(WorkloadBase):
    """BTHD Gated DeltaNet inference prefill or decode, equal-length or packed.

    ``sequence_lengths`` packs the rows into one ``B = 1`` token axis and makes the call
    carry ``cu_seqlens``; ``value_heads`` gives the recurrence more heads than the key
    carries, with key head ``h // (value_heads // heads)`` serving value head ``h``.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        scale: float | None = None,
        has_initial_state: bool = False,
        value_heads: int | None = None,
        sequence_lengths: tuple[int, ...] | None = None,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.scale = scale
        self.has_initial_state = has_initial_state
        self.value_heads = heads if value_heads is None else value_heads
        self.sequence_lengths = sequence_lengths

    @property
    def _spans(self) -> tuple[tuple[int, int], ...]:
        """The ``(start, end)`` token span of every sequence in the call."""
        lengths = (
            (self.seq_len,) * self.batch if self.sequence_lengths is None else self.sequence_lengths
        )
        spans, start = [], 0
        for length in lengths:
            spans.append((start, start + length))
            start += length
        return tuple(spans)

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        packed = self.sequence_lengths is not None
        rows = (1, sum(self.sequence_lengths)) if packed else (self.batch, self.seq_len)
        q = torch.randn((*rows, self.heads, self.dim), device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn((*rows, self.heads, self.dim), device=run_device(), dtype=self.dtype) * 0.1
        shape = (*rows, self.value_heads, self.dim)
        v = torch.randn(shape, device=run_device(), dtype=self.dtype) * 0.1
        g = -torch.rand(shape[:3], device=run_device(), dtype=self.dtype)
        beta = torch.rand(shape[:3], device=run_device(), dtype=self.dtype) * 0.5
        initial_state = (
            torch.randn(
                len(self._spans),
                self.value_heads,
                self.dim,
                self.dim,
                device=run_device(),
                dtype=torch.float32,
            )
            * 0.01
            if self.has_initial_state
            else None
        )
        if not packed:
            return (
                (q, k, v, g, beta) if initial_state is None else (q, k, v, g, beta, initial_state)
            )
        offsets = torch.tensor(
            [0, *(end for _, end in self._spans)], dtype=torch.int64, device=run_device()
        )
        return q, k, v, g, beta, initial_state, offsets, offsets.cpu()

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cu_seqlens, cu_seqlens_cpu
        scale = self.dim**-0.5 if self.scale is None else self.scale
        group = self.value_heads // self.heads
        spans = self._spans
        states, output = [], torch.empty_like(v)
        for sequence, (first, last) in enumerate(spans):
            state = (
                torch.zeros(
                    self.value_heads, self.dim, self.dim, dtype=torch.float32, device=q.device
                )
                if initial_state is None
                else initial_state[sequence].float()
            )
            for token in range(last - first):
                index = (
                    (0, first + token) if self.sequence_lengths is not None else (sequence, token)
                )
                # Value head h reads the key head its group shares.
                q_t = q[index].float().repeat_interleave(group, dim=0) * scale
                k_t = k[index].float().repeat_interleave(group, dim=0)
                v_t = v[index].float()
                decay = g[index].float().exp()
                beta_t = beta[index].float()
                old_value = torch.einsum("hkv,hk->hv", state, k_t)
                value = beta_t.unsqueeze(-1) * (v_t - decay.unsqueeze(-1) * old_value)
                state = decay[:, None, None] * state + k_t.unsqueeze(-1) * value.unsqueeze(-2)
                output[index] = torch.einsum("hk,hkv->hv", q_t, state).to(q.dtype)
            states.append(state)
        return output, torch.stack(states)


class GLAChunkwiseWorkload(WorkloadBase):
    def __init__(
        self,
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        has_initial_state: bool = False,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.has_initial_state = has_initial_state

    def gen_inputs(self):
        B, T, H, K, V = self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, T, H, K, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(B, T, H, K, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(B, T, H, V, device=run_device(), dtype=self.dtype) * 0.1
        g = -torch.rand(B, T, H, K, device=run_device(), dtype=self.dtype)
        # Absent means None: the recurrence then starts from zeros. Present, it is
        # fp32, the dtype the recurrence carries the state in.
        initial_state = (
            torch.randn(B, H, K, V, device=run_device(), dtype=torch.float32) * 0.1
            if self.has_initial_state
            else None
        )
        return q, k, v, g, initial_state


class GLAInferenceWorkload(GLAChunkwiseWorkload):
    """Inference GLA prefill with a caller-owned optional recurrent state."""

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        has_initial_state: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__(batch, seq_len, heads, dim_k, dim_v, 64, dtype, has_initial_state)
        self.scale = scale

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.gla import chunk_gla, fused_recurrent_gla

        reference = fused_recurrent_gla if q.shape[1] == 1 else chunk_gla
        return reference(
            q,
            k,
            v,
            g,
            scale=self.scale if self.scale is not None else self.dim_k**-0.5,
            initial_state=initial_state,
            output_final_state=True,
        )


def compute_w_u_torch(Aw, Au, k, v, beta, chunk_size):
    B, H, S, DK = k.shape
    _, _, _, DV = v.shape
    BC = chunk_size
    num_chunks = S // BC
    k_beta = k.float() * beta.unsqueeze(-1)
    v_beta = v.float() * beta.unsqueeze(-1)
    Aw_ = Aw.reshape(B, H, num_chunks, BC, BC)
    Au_ = Au.reshape(B, H, num_chunks, BC, BC)
    k_beta_ = k_beta.reshape(B, H, num_chunks, BC, DK)
    v_beta_ = v_beta.reshape(B, H, num_chunks, BC, DV)
    w = torch.einsum("bhcij,bhcjd->bhcid", Aw_, k_beta_).reshape(B, H, S, DK)
    u = torch.einsum("bhcij,bhcjd->bhcid", Au_, v_beta_).reshape(B, H, S, DV)
    return w, u


def kernel2_deltanet_torch(q, k, w, u, S_0, chunk_size):
    """DeltaNet kernel2 reference (ungated)."""
    B, H, S_len, DK = q.shape
    _, _, _, DV = u.shape
    BC = chunk_size
    num_chunks = S_len // BC
    q, k, w, u = q.float(), k.float(), w.float(), u.float()
    h = S_0.float().clone()

    o = torch.zeros(B, H, S_len, DV, dtype=torch.float32, device=q.device)
    for c in range(num_chunks):
        i0, i1 = c * BC, (c + 1) * BC
        q_c = q[:, :, i0:i1, :]
        k_c = k[:, :, i0:i1, :]
        w_c = w[:, :, i0:i1, :]
        u_c = u[:, :, i0:i1, :]
        v_new_c = u_c - w_c @ h
        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.bool), diagonal=0)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)
        o_c = o_part + torch.einsum("bhnm,bhmv->bhnv", attn, v_new_c)
        o[:, :, i0:i1, :] = o_c
        h = h + torch.einsum("bhnk,bhnv->bhkv", k_c, v_new_c)
    return h, o


def prepare_wy_repr_deltanet_torch(k, beta, chunk_size):
    B, H, S, DK = k.shape
    assert S % chunk_size == 0
    BC = chunk_size
    Aw = torch.empty(B, H, S, BC, dtype=torch.float32, device=k.device)
    Au = torch.empty(B, H, S, BC, dtype=torch.float32, device=k.device)

    for b in range(B):
        for h in range(H):
            for c in range(S // BC):
                i0, i1 = c * BC, (c + 1) * BC
                kc = k[b, h, i0:i1, :].float()
                bc = beta[b, h, i0:i1].float()
                Gram = kc @ kc.T
                M = bc.unsqueeze(-1) * Gram
                A = torch.eye(BC, device=k.device) + torch.tril(M, diagonal=-1)
                A_inv = torch.linalg.inv(A)
                Aw[b, h, i0:i1, :] = A_inv
                Au[b, h, i0:i1, :] = A_inv

    return Aw, Au


def deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step delta rule (ungated)."""
    q, k, v = q.float(), k.float(), v.float()
    beta = beta.float()
    state = state.float()

    old_val = torch.einsum("bhkv,bhk->bhv", state, k)
    beta_unsq = beta.unsqueeze(-1)
    v_new = beta_unsq * (v - old_val)

    o_inter = torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    new_state = state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


def gla_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step GLA recurrence."""
    DK = q.shape[-1]
    if scale <= 0:
        scale = DK**-0.5

    q, k, v = q.float(), k.float(), v.float()
    gk = gk.float()
    state = state.float()

    alpha = torch.exp(gk)
    new_state = alpha.unsqueeze(-1) * state + k.unsqueeze(-1) * v.unsqueeze(-2)
    o = scale * torch.einsum("bhk,bhkv->bhv", q, new_state)

    return o, new_state


# Manifest calls: shapes, dtypes and presence come from a workload row; these
# classes condition the values the row does not determine and carry the reference.


def _step_sizes(like: torch.Tensor) -> torch.Tensor:
    """Delta-rule step sizes in ``[0, 0.5)``, with *like*'s shape and dtype."""
    return torch.rand(like.shape, device=like.device).to(like.dtype) * 0.5


def _log_gates(like: torch.Tensor) -> torch.Tensor:
    """Log-space forget gates in ``(-1, 0]``, with *like*'s shape and dtype."""
    return -torch.rand(like.shape, device=like.device).to(like.dtype)


def _small(t: torch.Tensor | None, scale: float = 0.1) -> torch.Tensor | None:
    return None if t is None else t * scale


class DeltaNetDecodeCall(CallWorkload):
    """A manifest call of DeltaNetDecodeFwdOp."""

    def gen_inputs(self):
        q, k, v, beta, state = super().gen_inputs()
        return _small(q), _small(k), _small(v), _step_sizes(beta), _small(state)

    def ref_program(self, q, k, v, beta, state):
        o, new_state = deltanet_decode_torch(q, k, v, beta, state)
        return o.to(q.dtype), new_state.to(q.dtype)


class GLADecodeCall(CallWorkload):
    """A manifest call of GLADecodeFwdOp."""

    def gen_inputs(self):
        q, k, v, gk, state = super().gen_inputs()
        return _small(q), _small(k), _small(v), _log_gates(gk), _small(state)

    def ref_program(self, q, k, v, gk, state):
        o, new_state = gla_decode_torch(q, k, v, gk, state, self.call.ix["scale"])
        return o.to(q.dtype), new_state.to(q.dtype)


class DeltaNetChunkwiseCall(CallWorkload):
    """A manifest call of DeltaNetFwdOp, DeltaNetAutogradFwdOp or DeltaNetBwdOp.

    The backward's saved buffers come back random; a caller that needs the forward's
    values runs the forward on ``q, k, v, beta``.
    """

    def gen_inputs(self):
        tensors = dict(zip(self.call.signature.inputs, super().gen_inputs(), strict=True))
        return tuple(_step_sizes(t) if name == "beta" else _small(t) for name, t in tensors.items())


class GLAChunkwiseCall(CallWorkload):
    """A manifest call of GLAFwdOp or GLABwdOp."""

    def gen_inputs(self):
        tensors = dict(zip(self.call.signature.inputs, super().gen_inputs(), strict=True))
        return tuple(_log_gates(t) if name == "g" else _small(t) for name, t in tensors.items())


class DeltaNetInferenceCall(CallWorkload):
    """A manifest call of DeltaNetInferenceFwdOp.

    FLA's ``chunk_delta_rule`` is the reference for a prefill call and its
    ``fused_recurrent_delta_rule`` for a single-token one, which is the kernel FLA
    supplies for decode.
    """

    def gen_inputs(self):
        q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu = super().gen_inputs()
        return (
            _small(q),
            _small(k),
            _small(v),
            _step_sizes(beta),
            _small(initial_state, 0.01),
            cu_seqlens,
            cu_seqlens_cpu,
        )

    def ref_program(self, q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu):
        from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

        fla_kernel = fused_recurrent_delta_rule if q.shape[1] == 1 else chunk_delta_rule
        return fla_kernel(
            q,
            k,
            v,
            beta,
            scale=self.call.ix["scale"],
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=self.call.ix["use_qk_l2norm_in_kernel"],
        )


class GatedDeltaNetFwdCall(CallWorkload):
    """A manifest call of GatedDeltaNetFwdOp.

    FLA's ``chunk_gated_delta_rule`` is the reference for a prefill call and its
    ``fused_recurrent_gated_delta_rule`` for a single-token one, which is the kernel FLA
    supplies for decode.
    """

    def gen_inputs(self):
        q, k, v, g, beta, initial_state, *rest = super().gen_inputs()
        return (
            _small(q),
            _small(k),
            _small(v),
            _log_gates(g),
            _step_sizes(beta),
            _small(initial_state, 0.01),
            *rest,
        )

    def ref_program(self, q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, *rest):
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        del rest
        scale = self.call.ix["scale"]
        arguments = dict(
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=self.call.ix["use_qk_l2norm_in_kernel"],
        )
        if q.shape[1] == 1:
            return fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, **arguments)
        # chunk_gated_delta_rule builds its chunk index on the host, and the host copy of
        # the offsets is what spares it a device-to-host synchronization for them.
        return chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, cu_seqlens_cpu=cu_seqlens_cpu, **arguments
        )


class GLAInferenceCall(CallWorkload):
    """A manifest call of GLAInferenceFwdOp; FLA's chunk_gla is the reference."""

    def gen_inputs(self):
        q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu = super().gen_inputs()
        return (
            _small(q),
            _small(k),
            _small(v),
            _log_gates(g),
            _small(initial_state),
            cu_seqlens,
            cu_seqlens_cpu,
        )

    def ref_program(self, q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu):
        from fla.ops.gla import chunk_gla, fused_recurrent_gla

        scale = self.call.ix["scale"]
        arguments = dict(
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        if q.shape[1] == 1:
            return fused_recurrent_gla(q, k, v, g, **arguments)
        # chunk_gla builds its chunk index on the host, and the host copy of the offsets
        # is what spares it a device-to-host synchronization for them.
        return chunk_gla(q, k, v, g, cu_seqlens_cpu=cu_seqlens_cpu, **arguments)
