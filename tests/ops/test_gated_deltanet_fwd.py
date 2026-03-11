from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetFwdOp

# =============================================================================
# Torch reference implementations (test-only)
# =============================================================================

def _prepare_wy_repr_torch_ref(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
                gc = g[b, h, i0:i1].float()
                bc = beta[b, h, i0:i1].float()
                k_beta = kc * bc.unsqueeze(-1)
                Gram = k_beta @ k_beta.T

                L = torch.tril(Gram, diagonal=-1).neg()
                L.diagonal(0).fill_(1.0)
                L_inv = torch.linalg.inv(L)
                Aw[b, h, i0:i1, :] = L_inv

                Gram_g = Gram * torch.exp(gc.unsqueeze(1) - gc.unsqueeze(0))
                L_u = torch.tril(Gram_g, diagonal=-1).neg()
                L_u.diagonal(0).fill_(1.0)
                L_u_inv = torch.linalg.inv(L_u)
                Au[b, h, i0:i1, :] = L_u_inv

    return Aw, Au


def _compute_w_u_torch_ref(
    Aw: torch.Tensor,
    Au: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _kernel2_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    S_0: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

        v_new_c = u_c - (w_c * torch.exp(g_c).unsqueeze(-1)) @ h

        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h)
        o_part = o_part * torch.exp(g_c).unsqueeze(-1)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.bool), diagonal=0)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)
        o_c = o_part + torch.einsum("bhnm,bhmv->bhnv", attn, v_new_c)
        o[:, :, i0:i1, :] = o_c

        g_last = g_c[:, :, -1:]
        k_scaled = k_c * torch.exp(g_last - g_c).unsqueeze(-1)
        h = h * torch.exp(g_last).view(B, H, 1, 1)
        h = h + torch.einsum("bhnk,bhnv->bhkv", k_scaled, v_new_c)
    return h, o


# =============================================================================
# Forward correctness tests
# =============================================================================

def _get_tolerances(dtype: torch.dtype) -> dict:
    # Tolerances are looser than DEVELOPMENT.md defaults (fp16: 1e-3, bf16: 1.6e-2)
    # because Gated DeltaNet uses sequential chunk recurrence: each chunk's hidden
    # state h depends on all prior chunks, so fp32 rounding errors accumulate across
    # the chunk chain. With seq_len=128 and chunk_size=32 that is 4 serial steps of
    # matmul + exp + state update, which amplifies per-element error well beyond
    # single-kernel tolerances.
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    else:  # bfloat16
        return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class GatedDeltaNetFwdTest(TestBase):

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

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
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
        Aw, Au = _prepare_wy_repr_torch_ref(k, g, beta, self.chunk_size)
        w, u = _compute_w_u_torch_ref(Aw, Au, k, v, beta, self.chunk_size)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        _S, o = _kernel2_torch_ref(q, k, g, w, u, S_0, self.chunk_size)
        return o.to(self.dtype)


@GatedDeltaNetFwdFixture
def test_gated_deltanet_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    tols = _get_tolerances(dtype)
    inputs = test.gen_inputs()
    ref_o = test.ref_program(*inputs)
    op_o, _S, _Aw, _Au = op(*inputs)
    torch.testing.assert_close(op_o, ref_o, **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
