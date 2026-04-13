"""Benchmark: TileOPs DeltaNet (ungated) chunkwise vs FLA chunk_delta_rule.

Compares forward and backward latency across sequence lengths and dtypes.

When FLA is not installed, benchmarks still run using a pure-torch reference
implementation as baseline (tagged "torch"), so the nightly CI is never
blocked by a missing optional dependency.

Layout convention:
    TileOPs uses BHSD: q/k [B, H, S, DK], v [B, H, S, DV], beta [B, H, S].
    FLA uses BTHK:     q/k [B, T, H, K],  v [B, T, H, V],  beta [B, T, H].
    Tensors are permuted before calling FLA to ensure both implementations
    compute the same function.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops import DeltaNetBwdOp, DeltaNetFwdOp, DeltaNetOp
from workloads.deltanet import DeltaNetFwdTest
from workloads.workload_base import FixtureBase


def _differentiable_fwd(q, k, v, beta, chunk_size):
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

    o = _differentiable_fwd(q_, k_, v_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dbeta = torch.autograd.grad(loss, [q_, k_, v_, beta_])
    return dq, dk, dv, dbeta


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


class _DeltaNetFwdTestBaseline(DeltaNetFwdTest):
    """Adds baseline ref_program for benchmark profiling."""

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

try:
    from fla.ops.delta_rule import chunk_delta_rule
except ImportError:
    chunk_delta_rule = None


def _to_fla_layout(q, k, v, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


# =============================================================================
# Forward benchmark
# =============================================================================

class DeltaNetFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 1) * elem


class DeltaNetVsFlaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 4096, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4096, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 4096, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 2048, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 32768, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.nightly),
            pytest.param(2, 2048, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 4096, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 32768, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.nightly),
        ]),
    ]


@DeltaNetVsFlaFwdFixture
def test_deltanet_vs_fla_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = _DeltaNetFwdTestBaseline(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = DeltaNetFwdBenchmark(test)
    inputs = test.gen_inputs()  # q, k, v, beta (BHSD)

    # --- TileOPs (BHSD) ---
    op = DeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if chunk_delta_rule is not None:
        # --- FLA (BTHK) ---
        q, k, v, beta = inputs
        scale = dim_k ** -0.5
        q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(q, k, v, beta)

        def fla_fwd():
            return chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)

        result_fla = bm.profile(fla_fwd)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch reference baseline ---
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# =============================================================================
# Backward benchmark
# =============================================================================

class DeltaNetBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 3) * elem


class DeltaNetVsFlaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 4096, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4096, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 4096, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 2048, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 2048, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 4096, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@DeltaNetVsFlaBwdFixture
def test_deltanet_vs_fla_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = _DeltaNetFwdTestBaseline(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = DeltaNetBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get S, Aw, Au, w, u; then profile bwd only ---
    fwd_op = DeltaNetFwdOp(B, H, S, DK, DV, BC, dtype)
    _o, S_fwd, Aw, Au, w_fwd, u_fwd = fwd_op.forward(q, k, v, beta)

    bwd_op = DeltaNetBwdOp(B, H, S, DK, DV, BC, dtype, tune=tune)
    result = bm.profile(bwd_op.forward, do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd)
    BenchmarkReport.record(bwd_op, locals(), result, tag="tileops")

    if chunk_delta_rule is not None:
        # --- FLA: bwd only via autograd (BTHK layout) ---
        scale = DK ** -0.5
        q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(q, k, v, beta)
        do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

        q_fla = q_fla.detach().requires_grad_(True)
        k_fla = k_fla.detach().requires_grad_(True)
        v_fla = v_fla.detach().requires_grad_(True)
        beta_fla = beta_fla.detach().requires_grad_(True)

        # Run fwd once to build computation graph, then time only backward
        o_fla, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)

        def fla_bwd():
            q_fla.grad = k_fla.grad = v_fla.grad = beta_fla.grad = None
            o_fla.backward(do_fla, retain_graph=True)
            return q_fla.grad, k_fla.grad, v_fla.grad

        result_fla = bm.profile(fla_bwd)
        BenchmarkReport.record(bwd_op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_bwd():
            return deltanet_autograd_bwd_torch(do, q, k, v, beta, BC)
        result_bl = bm.profile(torch_bwd)
        BenchmarkReport.record(bwd_op, locals(), result_bl, tag="torch")


# =============================================================================
# Combined fwd+bwd benchmark (fair comparison: both measure fwd+bwd total)
# =============================================================================

class DeltaNetFwdBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 6.0 * B * H * S * DK * DV  # fwd (2x) + bwd (4x)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (6 * DK + 5 * DV + 4) * elem


class DeltaNetVsFlaFwdBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 4096, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4096, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 4096, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 2048, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 2048, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 4096, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@DeltaNetVsFlaFwdBwdFixture
def test_deltanet_vs_fla_fwdbwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = _DeltaNetFwdTestBaseline(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = DeltaNetFwdBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size

    # --- TileOPs: combined fwd+bwd via DeltaNetOp ---
    op = DeltaNetOp(B, H, S, DK, DV, BC, dtype, tune=tune)

    q = (torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    k = (torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    v = (torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    beta = (torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5).detach().requires_grad_(True)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    def tileops_fwdbwd():
        q.grad = k.grad = v.grad = beta.grad = None
        o = op(q, k, v, beta)
        o.backward(do)
        return q.grad, k.grad, v.grad

    result = bm.profile_autograd(tileops_fwdbwd)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if chunk_delta_rule is not None:
        # --- FLA: fwd+bwd via autograd ---
        scale = DK ** -0.5
        q_fla = (q.data.permute(0, 2, 1, 3).contiguous()).detach().requires_grad_(True)
        k_fla = (k.data.permute(0, 2, 1, 3).contiguous()).detach().requires_grad_(True)
        v_fla = (v.data.permute(0, 2, 1, 3).contiguous()).detach().requires_grad_(True)
        beta_fla = (beta.data.permute(0, 2, 1).contiguous()).detach().requires_grad_(True)
        do_fla = do.permute(0, 2, 1, 3).contiguous()

        def fla_fwdbwd():
            q_fla.grad = k_fla.grad = v_fla.grad = beta_fla.grad = None
            o, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)
            o.backward(do_fla)
            return q_fla.grad, k_fla.grad, v_fla.grad

        result_fla = bm.profile_autograd(fla_fwdbwd)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_fwdbwd():
            return deltanet_autograd_bwd_torch(do, q.data, k.data, v.data, beta.data, BC)
        result_bl = bm.profile_autograd(torch_fwdbwd)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
