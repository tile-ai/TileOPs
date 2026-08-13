"""Benchmark: TileOPs GLA vs FLA chunk_gla.

Compares forward and backward latency across sequence lengths and dtypes.

FLA is required, not optional: this file exists to compare against chunk_gla, and a
torch reference is not a comparison worth recording.

Layout convention:
    Both TileOPs and FLA use BTHD: q/k [B, T, H, K], v [B, T, H, V], g [B, T, H, K].
"""

from typing import Optional

import pytest
import torch
from fla.ops.gla import chunk_gla

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, backward_of
from tileops.ops import GLABwdOp, GLAFwdOp
from workloads.linear_attention import GLAChunkwiseWorkload
from workloads.workload_base import FixtureBase

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

    # --- FLA ---
    q, k, v, g = inputs

    def fla_fwd():
        return chunk_gla(q, k, v, g, scale=scale)

    functors["fla"] = (fla_fwd, ())

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

    # --- FLA: the backward node, called directly ---
    # NOTE: FLA's backward recomputes h internally (not saved from fwd),
    # so this measures bwd + h recomputation, not pure bwd.
    q_fla = q.float().detach().requires_grad_(True)
    k_fla = k.float().detach().requires_grad_(True)
    v_fla = v.float().detach().requires_grad_(True)
    g_fla = g.float().detach().requires_grad_(True)
    do_fla = do.float()

    o_fla, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=scale)
    # One grad per forward output: fla returns (o, final_state).
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        return fla_backward(do_fla, None)

    functors["fla"] = (fla_bwd, ())

    bm.compare(functors, record_as=bwd_op, params=locals())


# Combined fwd+bwd benchmark
