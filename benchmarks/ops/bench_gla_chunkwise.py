"""Benchmark: TileOPs GLA vs FLA chunk_gla.

Compares forward and backward latency across sequence lengths and dtypes.

When FLA is not installed, benchmarks still run using a pure-torch reference
implementation as baseline (tagged "baseline"), so the nightly CI is never
blocked by a missing optional dependency.

Layout convention:
    Both TileOPs and FLA use BTHD: q/k [B, T, H, K], v [B, T, H, V], g [B, T, H, K].
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops import GLABwdOp, GLAFwdOp
from workloads.linear_attention import GLAChunkwiseWorkload
from workloads.workload_base import FixtureBase


def gla_fwd_chunked_torch(q, k, v, g, chunk_size, scale=None):
    """Fully differentiable chunked GLA forward in float32."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BC = chunk_size
    NC = T // BC

    if scale is None:
        scale = K ** -0.5

    q = q.float() * scale
    k = k.float()
    v = v.float()
    g = g.float()

    g_cum = g.reshape(B, NC, BC, H, K).cumsum(dim=2).reshape(B, T, H, K)

    h = q.new_zeros(B, H, K, V)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))

    o_chunks = []
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, sl, :, :]
        kc = k[:, sl, :, :]
        vc = v[:, sl, :, :]
        gc = g_cum[:, sl, :, :]
        g_last = gc[:, -1:, :, :]

        q_gated = qc * torch.exp(gc)
        o_inter = torch.einsum("bthk,bhkv->bthv", q_gated, h)

        k_ungated = kc * torch.exp(-gc)
        A = torch.einsum("bihk,bjhk->bhij", q_gated, k_ungated)
        A = A * mask.unsqueeze(0).unsqueeze(0)
        o_intra = torch.einsum("bhij,bjhv->bihv", A, vc)

        o_chunks.append(o_inter + o_intra)

        k_adj = kc * torch.exp(g_last - gc)
        h = h * torch.exp(g_last).permute(0, 2, 3, 1).squeeze(-1).unsqueeze(-1)
        h = h + torch.einsum("bthk,bthv->bhkv", k_adj, vc)

    return torch.cat(o_chunks, dim=1)


def gla_autograd_bwd_torch(do, q, k, v, g, chunk_size, scale=-1.0):
    """Compute GLA backward gradients via autograd on the differentiable forward."""
    sc = (q.shape[-1] ** -0.5) if scale <= 0 else scale

    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)

    o = gla_fwd_chunked_torch(q_, k_, v_, g_, chunk_size, scale=sc)
    loss = (o * do.float()).sum()
    dq, dk, dv, dg = torch.autograd.grad(loss, [q_, k_, v_, g_])
    return dq, dk, dv, dg

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None


# Test helper (shared between fwd and bwd benchmarks)


# Forward benchmark

class GLAFwdBenchmark(BenchmarkBase[GLAChunkwiseWorkload]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        return 2.0 * B * H * T * K * V

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * T * H * (2 * K + 2 * V) * elem


class GLAFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


@GLAFwdFixture
def test_gla_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GLAChunkwiseWorkload(batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype)
    bm = GLAFwdBenchmark(test)
    inputs = test.gen_inputs()

    # --- TileOPs ---
    scale = dim_k ** -0.5
    op = GLAFwdOp(chunk_size=chunk_size, scale=scale, tune=tune)
    functors = {"tileops": op.forward}

    if chunk_gla is not None:
        # --- FLA ---
        q, k, v, g = inputs

        def fla_fwd():
            return chunk_gla(q, k, v, g, scale=scale)

        functors["fla"] = (fla_fwd, ())
    else:
        # --- Torch reference baseline ---
        functors["torch"] = lambda q, k, v, g: gla_fwd_chunked_torch(q, k, v, g, chunk_size)

    bm.compare(functors, *inputs, record_as=op, params=locals())


# Backward benchmark

class GLABwdBenchmark(BenchmarkBase[GLAChunkwiseWorkload]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        return 4.0 * B * H * T * K * V

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * T * H * (4 * K + 3 * V) * elem


class GLABwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


@pytest.mark.xfail(
    reason="TileLang emits a WGMMA descriptor for a B operand whose layout the "
           "assert rejects: 'Not a canonical GMMA_MN layout'. Fails on main too.",
    strict=False,
)
@GLABwdFixture
def test_gla_bwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GLAChunkwiseWorkload(batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype)
    bm = GLABwdBenchmark(test)

    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get h, then profile bwd only ---
    fwd_op = GLAFwdOp(chunk_size=BC, scale=scale)
    fwd_op.forward(q, k, v, g)
    h = fwd_op.kernel._h_out
    dht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    bwd_op = GLABwdOp(chunk_size=BC, scale=scale, tune=tune)
    functors = {"tileops": (bwd_op.forward, (q, k, v, g, h, do, dht))}

    if chunk_gla is not None:
        # --- FLA: bwd via autograd ---
        # NOTE: FLA's backward recomputes h internally (not saved from fwd),
        # so this measures bwd + h recomputation, not pure bwd.
        q_fla = q.float().detach().requires_grad_(True)
        k_fla = k.float().detach().requires_grad_(True)
        v_fla = v.float().detach().requires_grad_(True)
        g_fla = g.float().detach().requires_grad_(True)
        do_fla = do.float()

        o_fla, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=scale)

        def fla_bwd():
            q_fla.grad = k_fla.grad = v_fla.grad = g_fla.grad = None
            o_fla.backward(do_fla, retain_graph=True)
            return q_fla.grad, k_fla.grad, v_fla.grad

        functors["fla"] = (fla_bwd, ())
    else:
        # --- Torch autograd reference baseline ---
        def torch_bwd():
            return gla_autograd_bwd_torch(do, q, k, v, g, BC, scale=scale)

        functors["torch"] = (torch_bwd, ())

    # Only torch_bwd builds its graph inside the timed call.
    bm.compare(functors, record_as=bwd_op, params=locals(), needs_grad=("torch",))


# Combined fwd+bwd benchmark
