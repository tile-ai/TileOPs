from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import DeltaNetFwdOp

# =============================================================================
# Torch reference implementations (test-only)
# =============================================================================

def _prepare_wy_repr_torch_ref(
    k: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paper: A = I + strictLower(diag(beta) * KK^T), no Gamma matrix."""
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
                # diag(beta) * KK^T (no Gamma)
                Gram = kc @ kc.T
                M = bc.unsqueeze(-1) * Gram

                # A = I + strictLower(M), invert via direct linalg
                A = torch.eye(BC, device=k.device) + torch.tril(M, diagonal=-1)
                A_inv = torch.linalg.inv(A)
                Aw[b, h, i0:i1, :] = A_inv
                Au[b, h, i0:i1, :] = A_inv  # same matrix

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
    w: torch.Tensor,
    u: torch.Tensor,
    S_0: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeltaNet kernel2 reference (ungated): no exp(g) scaling, no state decay."""
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

        # v_new = u - w @ h (no exp scaling)
        v_new_c = u_c - w_c @ h

        # o = q @ h + causal_attn @ v_new (no exp scaling, no Gamma)
        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.bool), diagonal=0)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)
        o_c = o_part + torch.einsum("bhnm,bhmv->bhnv", attn, v_new_c)
        o[:, :, i0:i1, :] = o_c

        # h_new = h + k^T @ v_new (no decay)
        h = h + torch.einsum("bhnk,bhnv->bhkv", k_c, v_new_c)
    return h, o


# =============================================================================
# Forward correctness tests
# =============================================================================

def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.float16:
        return {"atol": 2e-2, "rtol": 2e-2}
    else:  # bfloat16
        return {"atol": 5e-2, "rtol": 5e-2}


class DeltaNetFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


class DeltaNetFwdTest(TestBase):

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
        Aw, Au = _prepare_wy_repr_torch_ref(k, beta, self.chunk_size)
        w, u = _compute_w_u_torch_ref(Aw, Au, k, v, beta, self.chunk_size)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        _S, o = _kernel2_torch_ref(q, k, w, u, S_0, self.chunk_size)
        return o.to(self.dtype)


@DeltaNetFwdFixture
def test_deltanet_fwd(
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
    test = DeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = DeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    tols = _get_tolerances(dtype)
    inputs = test.gen_inputs()
    ref_o = test.ref_program(*inputs)
    op_o, _S, _Aw, _Au, _w, _u = op(*inputs)
    torch.testing.assert_close(op_o, ref_o, **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
