import torch

from workloads.device import run_device
from workloads.linear_attention.input_values import _small, _step_sizes
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "DeltaNetChunkwiseCall",
    "DeltaNetDecodeCall",
    "DeltaNetDecodeWorkload",
    "DeltaNetFwdWorkload",
    "DeltaNetInferenceCall",
    "DeltaNetInferenceWorkload",
    "compute_w_u_torch",
    "deltanet_autograd_bwd_torch",
    "deltanet_decode_torch",
    "deltanet_differentiable_fwd_torch",
    "kernel2_deltanet_torch",
    "prepare_wy_repr_deltanet_torch",
]


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


class DeltaNetInferenceWorkload(WorkloadBase):
    """BTHD ungated DeltaNet prefill or single-token decode with recurrent state.

    ``sequence_lengths`` packs the rows into one ``B = 1`` token axis and makes the call
    carry ``cu_seqlens``; ``l2norm`` hands Q and K over unnormalized.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        sequence_lengths: tuple[int, ...] | None = None,
        l2norm: bool = False,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.sequence_lengths = sequence_lengths
        self.l2norm = l2norm

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
            use_qk_l2norm_in_kernel=self.l2norm,
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


def deltanet_differentiable_fwd_torch(q, k, v, beta, chunk_size):
    """Fully differentiable chunked forward matching DeltaNet (ungated)."""
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    BC = chunk_size
    NC = S // BC
    h = q.new_zeros(B, H, DK, DV)
    o_chunks = []
    eye = torch.eye(BC, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, :, sl, :].float()
        kc = k[:, :, sl, :].float()
        vc = v[:, :, sl, :].float()
        bc = beta[:, :, sl].float()
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        M = bc.unsqueeze(-1) * Gram
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        v_new = uc - wc @ h
        o_part = qc @ h
        attn = (qc @ kc.transpose(-2, -1)) * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        h = h + kc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def deltanet_autograd_bwd_torch(do, q, k, v, beta, chunk_size):
    """Compute backward gradients via autograd on the differentiable forward."""
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    beta_ = beta.float().detach().requires_grad_(True)

    o = deltanet_differentiable_fwd_torch(q_, k_, v_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dbeta = torch.autograd.grad(loss, [q_, k_, v_, beta_])
    return dq, dk, dv, dbeta


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


class DeltaNetDecodeCall(CallWorkload):
    """A manifest call of DeltaNetRecurrentFwdOp."""

    def gen_inputs(self):
        q, k, v, beta, state = super().gen_inputs()
        return _small(q), _small(k), _small(v), _step_sizes(beta), _small(state)

    def ref_program(self, q, k, v, beta, state):
        o, new_state = deltanet_decode_torch(q, k, v, beta, state)
        return o.to(q.dtype), new_state.to(q.dtype)


class DeltaNetChunkwiseCall(CallWorkload):
    """A manifest call of DeltaNetChunkFwdOp or DeltaNetChunkBwdOp.

    The backward's saved buffers come back random; a caller that needs the forward's
    values runs the forward on ``q, k, v, beta``.
    """

    def gen_inputs(self):
        tensors = dict(zip(self.call.signature.inputs, super().gen_inputs(), strict=True))
        return tuple(_step_sizes(t) if name == "beta" else _small(t) for name, t in tensors.items())


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
