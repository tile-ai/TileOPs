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
from workloads.workload_base import FixtureBase, WorkloadBase


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


# =============================================================================
# Test helper (shared between fwd and bwd benchmarks)
# =============================================================================

class GLATest(WorkloadBase):

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

    def ref_program(self, q, k, v, g):
        return gla_fwd_chunked_torch(q, k, v, g, self.chunk_size)


# =============================================================================
# Forward benchmark
# =============================================================================

class GLAFwdBenchmark(BenchmarkBase[GLATest]):

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
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
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
    test = GLATest(batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype)
    bm = GLAFwdBenchmark(test)
    inputs = test.gen_inputs()

    # --- TileOPs ---
    scale = dim_k ** -0.5
    op = GLAFwdOp(batch, seq_len, heads, dim_k, dim_v, chunk_size,
                   scale=scale, dtype=dtype, tune=tune)
    result = bm.profile(op.forward, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if chunk_gla is not None:
        # --- FLA ---
        q, k, v, g = inputs

        def fla_fwd():
            return chunk_gla(q, k, v, g, scale=scale)

        result_fla = bm.profile(fla_fwd)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch reference baseline ---
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# =============================================================================
# Backward benchmark
# =============================================================================

class GLABwdBenchmark(BenchmarkBase[GLATest]):

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
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


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
    test = GLATest(batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype)
    bm = GLABwdBenchmark(test)

    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get h, then profile bwd only ---
    fwd_op = GLAFwdOp(B, T, H, K, V, BC, scale=scale, dtype=dtype)
    fwd_op.forward(q, k, v, g)
    h = fwd_op.kernel._h_out
    dht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    bwd_op = GLABwdOp(B, T, H, K, V, BC, scale=scale, dtype=dtype, tune=tune)
    result = bm.profile(bwd_op.forward, q, k, v, g, h, do, dht)
    BenchmarkReport.record(bwd_op, locals(), result, tag="tileops")

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

        result_fla = bm.profile(fla_bwd)
        BenchmarkReport.record(bwd_op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_bwd():
            return gla_autograd_bwd_torch(do, q, k, v, g, BC, scale=scale)
        result_bl = bm.profile(torch_bwd)
        BenchmarkReport.record(bwd_op, locals(), result_bl, tag="torch")


# =============================================================================
# Combined fwd+bwd benchmark
# =============================================================================

class GLAFwdBwdBenchmark(BenchmarkBase[GLATest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        return 6.0 * B * H * T * K * V

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * T * H * (6 * K + 5 * V) * elem


class GLAFwdBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


@GLAFwdBwdFixture
def test_gla_fwdbwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GLATest(batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype)
    bm = GLAFwdBwdBenchmark(test)

    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd + bwd ---
    fwd_op = GLAFwdOp(B, T, H, K, V, BC, scale=scale, dtype=dtype)
    bwd_op = GLABwdOp(B, T, H, K, V, BC, scale=scale, dtype=dtype, tune=tune)

    def tileops_fwdbwd():
        fwd_op.forward(q, k, v, g)
        h = fwd_op.kernel._h_out
        dht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
        return bwd_op.forward(q, k, v, g, h, do, dht)

    result = bm.profile_autograd(tileops_fwdbwd)
    BenchmarkReport.record(fwd_op, locals(), result, tag="tileops")

    if chunk_gla is not None:
        # --- FLA: fwd+bwd via autograd ---
        q_fla = q.float().detach().requires_grad_(True)
        k_fla = k.float().detach().requires_grad_(True)
        v_fla = v.float().detach().requires_grad_(True)
        g_fla = g.float().detach().requires_grad_(True)
        do_fla = do.float()

        def fla_fwdbwd():
            q_fla.grad = k_fla.grad = v_fla.grad = g_fla.grad = None
            o, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=scale)
            o.backward(do_fla)
            return q_fla.grad, k_fla.grad, v_fla.grad

        result_fla = bm.profile_autograd(fla_fwdbwd)
        BenchmarkReport.record(fwd_op, locals(), result_fla, tag="fla")
    else:
        def ref_autograd_fwdbwd():
            return gla_autograd_bwd_torch(do, q, k, v, g, BC, scale=scale)
        result_bl = bm.profile_autograd(ref_autograd_fwdbwd)
        BenchmarkReport.record(fwd_op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
