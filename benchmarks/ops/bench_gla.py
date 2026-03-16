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
from tilelang.profiler import do_bench as _do_bench

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gla_bwd import _gla_autograd_bwd_ref, _gla_fwd_torch_ref
from tests.test_base import FixtureBase, TestBase
from tileops.ops import GLABwdOp, GLAFwdOp

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None


def _profile_manual(fn, bm, warmup=100, rep=100):
    """Profile a function using do_bench with cupti/event fallback."""
    latency = _do_bench(fn, warmup=warmup, rep=rep, backend='cupti')
    if latency <= 0:
        latency = _do_bench(fn, warmup=warmup, rep=rep, backend='event')
    result = {"latency_ms": latency}
    flops = bm.calculate_flops()
    if flops is not None:
        result["tflops"] = flops / latency * 1e-9
    memory = bm.calculate_memory()
    if memory is not None:
        result["bandwidth_tbs"] = memory / latency * 1e-9
    return result


# =============================================================================
# Test helper (shared between fwd and bwd benchmarks)
# =============================================================================

class GLATest(TestBase):

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
        return _gla_fwd_torch_ref(q, k, v, g, self.chunk_size)


# =============================================================================
# Forward benchmark
# =============================================================================

class GLAFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        return 2.0 * B * H * T * K * V

    def calculate_memory(self) -> Optional[float]:
        t = self.test
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
    BenchmarkReport.record("gla_fwd", locals(), result, tag="tileops")

    if chunk_gla is not None:
        # --- FLA ---
        q, k, v, g = inputs

        def fla_fwd():
            return chunk_gla(q, k, v, g, scale=scale)

        result_fla = bm.profile(fla_fwd)
        BenchmarkReport.record("gla_fwd", locals(), result_fla, tag="fla")
    else:
        # --- Torch reference baseline ---
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record("gla_fwd", locals(), result_bl, tag="baseline")


# =============================================================================
# Backward benchmark
# =============================================================================

class GLABwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        return 4.0 * B * H * T * K * V

    def calculate_memory(self) -> Optional[float]:
        t = self.test
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
    BenchmarkReport.record("gla_bwd", locals(), result, tag="tileops")

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

        result_fla = _profile_manual(fla_bwd, bm)
        BenchmarkReport.record("gla_bwd", locals(), result_fla, tag="fla_bwd_with_recompute")
    else:
        # --- Torch autograd reference baseline ---
        def torch_bwd():
            return _gla_autograd_bwd_ref(do, q, k, v, g, BC, scale=scale)
        result_bl = _profile_manual(torch_bwd, bm)
        BenchmarkReport.record("gla_bwd", locals(), result_bl, tag="baseline")


# =============================================================================
# Combined fwd+bwd benchmark
# =============================================================================

class GLAFwdBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, T, H, K, V = t.batch, t.seq_len, t.heads, t.dim_k, t.dim_v
        return 6.0 * B * H * T * K * V

    def calculate_memory(self) -> Optional[float]:
        t = self.test
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

    result = _profile_manual(tileops_fwdbwd, bm, warmup=50)
    BenchmarkReport.record("gla_fwdbwd", locals(), result, tag="tileops")

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

        result_fla = _profile_manual(fla_fwdbwd, bm, warmup=50)
        BenchmarkReport.record("gla_fwdbwd", locals(), result_fla, tag="fla")
    else:
        def torch_fwdbwd():
            return _gla_autograd_bwd_ref(do, q, k, v, g, BC, scale=scale)
        result_bl = _profile_manual(torch_fwdbwd, bm, warmup=50)
        BenchmarkReport.record("gla_fwdbwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
