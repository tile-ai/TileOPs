"""Benchmark: TileOPs Gated DeltaNet vs FLA chunk_gated_delta_rule.

Compares forward and backward latency across sequence lengths and dtypes.

Layout convention:
    TileOPs uses BHSD: q/k [B, H, S, DK], v [B, H, S, DV], g/beta [B, H, S].
    FLA uses BTHK:     q/k [B, T, H, K],  v [B, T, H, V],  g/beta [B, T, H].
    Tensors are permuted before calling FLA to ensure both implementations
    compute the same function.
"""

from typing import Optional

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_fwd import GatedDeltaNetFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp

from .bench_gated_deltanet_fla_validation import _profile_manual, _to_fla_layout

# =============================================================================
# Forward benchmark
# =============================================================================

class GatedDeltaNetFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 2) * elem


class GatedDeltaNetVsFlaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 1024, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 32768, 4, 64, 64, 64, torch.float16, False),
            (2, 1024, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 32768, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


@GatedDeltaNetVsFlaFwdFixture
def test_gated_deltanet_vs_fla_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdBenchmark(test)
    inputs = test.gen_inputs()  # q, k, v, g, beta  (BHSD)

    # --- TileOPs (BHSD) ---
    op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gated_deltanet_fwd", locals(), result, tag="tileops")

    # --- FLA (BTHK) ---
    q, k, v, g, beta = inputs
    scale = dim_k ** -0.5
    q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)

    def fla_fwd():
        return chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)

    result_fla = bm.profile(fla_fwd)
    BenchmarkReport.record("gated_deltanet_fwd", locals(), result_fla, tag="fla")


# =============================================================================
# Backward benchmark
# =============================================================================

class GatedDeltaNetBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 4) * elem


class GatedDeltaNetVsFlaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 1024, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 1024, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


@GatedDeltaNetVsFlaBwdFixture
def test_gated_deltanet_vs_fla_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get S, then profile bwd only ---
    fwd_op = GatedDeltaNetFwdOp(B, H, S, DK, DV, BC, dtype)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)

    bwd_op = GatedDeltaNetBwdOp(B, H, S, DK, DV, BC, dtype, tune=tune)
    result = bm.profile(bwd_op.forward, do, q, k, v, g, beta, S_fwd)
    BenchmarkReport.record("gated_deltanet_bwd", locals(), result, tag="tileops")

    # --- FLA: fwd+bwd via autograd (BTHK layout) ---
    scale = DK ** -0.5
    q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)
    do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

    q_fla = q_fla.detach().requires_grad_(True)
    k_fla = k_fla.detach().requires_grad_(True)
    v_fla = v_fla.detach().requires_grad_(True)
    g_fla = g_fla.detach().requires_grad_(True)
    beta_fla = beta_fla.detach().requires_grad_(True)

    def fla_bwd():
        q_fla.grad = k_fla.grad = v_fla.grad = g_fla.grad = beta_fla.grad = None
        o, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
        o.backward(do_fla, retain_graph=True)
        return q_fla.grad, k_fla.grad, v_fla.grad

    result_fla = _profile_manual(fla_bwd, bm)
    BenchmarkReport.record("gated_deltanet_bwd", locals(), result_fla, tag="fla")


# =============================================================================
# Combined fwd+bwd benchmark (fair comparison: both measure fwd+bwd total)
# =============================================================================

class GatedDeltaNetFwdBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 6.0 * B * H * S * DK * DV  # fwd (2x) + bwd (4x)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (6 * DK + 5 * DV + 6) * elem


class GatedDeltaNetVsFlaFwdBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 1024, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 1024, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 8192, 4, 64, 64, 64, torch.bfloat16, False),
            (2, 16384, 4, 64, 64, 64, torch.bfloat16, False),
        ]),
    ]


@GatedDeltaNetVsFlaFwdBwdFixture
def test_gated_deltanet_vs_fla_fwdbwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size

    # --- TileOPs: combined fwd+bwd via GatedDeltaNetOp ---
    op = GatedDeltaNetOp(B, H, S, DK, DV, BC, dtype, tune=tune)

    q = (torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    k = (torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    v = (torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    g = (-torch.rand(B, H, S, device="cuda", dtype=dtype)).detach().requires_grad_(True)
    beta = (torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5).detach().requires_grad_(True)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    def tileops_fwdbwd():
        q.grad = k.grad = v.grad = g.grad = beta.grad = None
        o = op(q, k, v, g, beta)
        o.backward(do, retain_graph=True)
        return q.grad, k.grad, v.grad

    result = _profile_manual(tileops_fwdbwd, bm, warmup=50)
    BenchmarkReport.record("gated_deltanet_fwdbwd", locals(), result, tag="tileops")

    # --- FLA: fwd+bwd via autograd ---
    scale = DK ** -0.5
    q_fla = (q.data.permute(0, 2, 1, 3).contiguous()).detach().requires_grad_(True)
    k_fla = (k.data.permute(0, 2, 1, 3).contiguous()).detach().requires_grad_(True)
    v_fla = (v.data.permute(0, 2, 1, 3).contiguous()).detach().requires_grad_(True)
    g_fla = (g.data.permute(0, 2, 1).contiguous()).detach().requires_grad_(True)
    beta_fla = (beta.data.permute(0, 2, 1).contiguous()).detach().requires_grad_(True)
    do_fla = do.permute(0, 2, 1, 3).contiguous()

    def fla_fwdbwd():
        q_fla.grad = k_fla.grad = v_fla.grad = g_fla.grad = beta_fla.grad = None
        o, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
        o.backward(do_fla, retain_graph=True)
        return q_fla.grad, k_fla.grad, v_fla.grad

    result_fla = _profile_manual(fla_fwdbwd, bm, warmup=50)
    BenchmarkReport.record("gated_deltanet_fwdbwd", locals(), result_fla, tag="fla")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
