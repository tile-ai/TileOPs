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
from tilelang.profiler import do_bench as _do_bench

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_deltanet_chunkwise_bwd import _autograd_bwd_ref
from tests.ops.test_deltanet_chunkwise_fwd import DeltaNetFwdTest
from tests.test_base import FixtureBase
from tileops.ops import DeltaNetBwdOp, DeltaNetFwdOp, DeltaNetOp

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


def _profile_manual(fn, bm, warmup=100, rep=100):
    """Profile a function using do_bench with cupti/event fallback, returning a result dict."""
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
# Forward benchmark
# =============================================================================

class DeltaNetFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 1) * elem


class DeltaNetVsFlaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 32768, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 32768, 4, 64, 64, 64, torch.bfloat16, False),
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
    test = DeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
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
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 3) * elem


class DeltaNetVsFlaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
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
    test = DeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
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

        result_fla = _profile_manual(fla_bwd, bm)
        BenchmarkReport.record(bwd_op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_bwd():
            return _autograd_bwd_ref(do, q, k, v, beta, BC)
        result_bl = _profile_manual(torch_bwd, bm)
        BenchmarkReport.record(bwd_op, locals(), result_bl, tag="torch")


# =============================================================================
# Combined fwd+bwd benchmark (fair comparison: both measure fwd+bwd total)
# =============================================================================

class DeltaNetFwdBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 6.0 * B * H * S * DK * DV  # fwd (2x) + bwd (4x)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (6 * DK + 5 * DV + 4) * elem


class DeltaNetVsFlaFwdBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
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
    test = DeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
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
        o.backward(do, retain_graph=True)
        return q.grad, k.grad, v.grad

    result = _profile_manual(tileops_fwdbwd, bm, warmup=50)
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

        result_fla = _profile_manual(fla_fwdbwd, bm, warmup=50)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_fwdbwd():
            return _autograd_bwd_ref(do, q.data, k.data, v.data, beta.data, BC)
        result_bl = _profile_manual(torch_fwdbwd, bm, warmup=50)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
