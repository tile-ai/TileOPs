"""Workload definitions for the linear_attention op family."""

import torch

from workloads.workload_base import WorkloadBase


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
        q = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, S, DV, device="cuda", dtype=self.dtype) * 0.1
        beta = torch.rand(B, H, S, device="cuda", dtype=self.dtype) * 0.5
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
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        beta = torch.rand(B, H, device="cuda", dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
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
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        gk = -torch.rand(B, H, DK, device="cuda", dtype=self.dtype)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
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


class GatedDeltaNetFwdWorkload(WorkloadBase):
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
        q = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, S, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, S, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, S, device="cuda", dtype=self.dtype) * 0.5
        return q, k, v, g, beta

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        B, H, S, DK = k.shape
        _, _, _, DV = v.shape
        BC = self.chunk_size
        g_cum = g.float().reshape(B, H, S // BC, BC).cumsum(-1).reshape(B, H, S).to(g.dtype)
        Aw, Au = prepare_wy_repr_gated_torch(k, g_cum, beta, self.chunk_size)
        w, u = compute_w_u_torch(Aw, Au, k, v, beta, self.chunk_size)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        _S, o = kernel2_gated_deltanet_torch(q, k, g_cum, w, u, S_0, self.chunk_size)
        return o.to(self.dtype)


class GatedDeltaNetPrefillFwdWorkload(GatedDeltaNetFwdWorkload):
    """Inference prefill workload for Gated DeltaNet."""

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int,
        dtype: torch.dtype,
        layout: str = "bhtd",
    ) -> None:
        super().__init__(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
        self.layout = self._normalize_layout(layout)
        if self.layout == "bthd":
            self.shape = (batch, seq_len, heads, dim_k)
        else:
            self.shape = (batch, heads, seq_len, dim_k)

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        layout = layout.lower()
        if layout in ("bhtd", "bthd"):
            return layout
        raise ValueError(f"Unsupported layout: {layout}")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.layout != "bthd":
            return super().gen_inputs()

        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        q = torch.randn(B, S, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, S, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, S, H, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, S, H, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, S, H, device="cuda", dtype=self.dtype) * 0.5
        return q, k, v, g, beta

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The chunked form below indexes bhtd; gen_inputs also builds bthd.
        if self.layout == "bthd":
            q, k, v = (t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v))
            g, beta = (t.permute(0, 2, 1).contiguous() for t in (g, beta))
        B, H, S, DK = k.shape
        _, _, _, DV = v.shape
        BC = self.chunk_size
        g_cum = g.float().reshape(B, H, S // BC, BC).cumsum(-1).reshape(B, H, S).to(g.dtype)
        Aw, Au = prepare_wy_repr_gated_torch(k, g_cum, beta, BC)
        w, u = compute_w_u_torch(Aw, Au, k, v, beta, BC)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        final_state, o = kernel2_gated_deltanet_torch(q, k, g_cum, w, u, S_0, BC)
        if self.layout == "bthd":
            o = o.permute(0, 2, 1, 3).contiguous()
        return o.to(self.dtype), final_state.to(self.dtype)


class GatedDeltaNetDecodeWorkload(WorkloadBase):
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
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, device="cuda", dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, g, beta, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gated_deltanet_decode_torch(q, k, v, g, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)


class GLAChunkwiseWorkload(WorkloadBase):
    def __init__(self, batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

    def gen_inputs(self):
        B, T, H, K, V = self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, T, H, K, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=self.dtype)
        return q, k, v, g


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


def kernel2_gated_deltanet_torch(q, k, g, w, u, S_0, chunk_size):
    B, H, S_len, DK = q.shape
    _, _, _, DV = u.shape
    BC = chunk_size
    num_chunks = S_len // BC
    q, k, g, w, u = q.float(), k.float(), g.float(), w.float(), u.float()
    h = S_0.float().clone()

    o = torch.zeros(B, H, S_len, DV, dtype=torch.float32, device=q.device)
    for c in range(num_chunks):
        i0, i1 = c * BC, (c + 1) * BC
        q_c = q[:, :, i0:i1, :]
        k_c = k[:, :, i0:i1, :]
        g_c = g[:, :, i0:i1]
        w_c = w[:, :, i0:i1, :]
        u_c = u[:, :, i0:i1, :]

        g_last_val = g_c[:, :, -1:]
        v_new_c = u_c - (w_c * torch.exp(g_c + g_last_val).unsqueeze(-1)) @ h

        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h)
        o_part = o_part * torch.exp(g_c).unsqueeze(-1)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        Gamma_causal = torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
        mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.bool), diagonal=0)
        attn = (attn * Gamma_causal).masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)
        o_c = o_part + torch.einsum("bhnm,bhmv->bhnv", attn, v_new_c)
        o[:, :, i0:i1, :] = o_c

        g_last = g_c[:, :, -1:]
        k_scaled = k_c * torch.exp(g_last - g_c).unsqueeze(-1)
        h = h * torch.exp(g_last).view(B, H, 1, 1)
        h = h + torch.einsum("bhnk,bhnv->bhkv", k_scaled, v_new_c)
    return h, o


def prepare_wy_repr_gated_torch(k, g_cum, beta, chunk_size):
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
                gc = g_cum[b, h, i0:i1].float()
                bc = beta[b, h, i0:i1].float()
                Gram = kc @ kc.T
                Gamma = torch.exp(gc.unsqueeze(1) - gc.unsqueeze(0))
                M = bc.unsqueeze(-1) * (Gamma * Gram)
                A_g = torch.eye(BC, device=k.device) + torch.tril(M, diagonal=-1)
                A_g_inv = torch.linalg.inv(A_g)
                Aw[b, h, i0:i1, :] = A_g_inv
                Au[b, h, i0:i1, :] = A_g_inv

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


def gated_deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step gated delta rule."""
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()
    state = state.float()

    alpha = torch.exp(g)
    old_val = torch.einsum("bhkv,bhk->bhv", state, k)

    beta_unsq = beta.unsqueeze(-1)
    alpha_unsq = alpha.unsqueeze(-1)
    v_new = beta_unsq * v - alpha_unsq * beta_unsq * old_val

    o_inter = alpha_unsq * torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    new_state = alpha_unsq.unsqueeze(-1) * state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

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
