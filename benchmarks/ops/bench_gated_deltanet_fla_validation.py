"""Benchmark: TileOPs Gated DeltaNet numerical validation + performance vs FLA.

Cross-validates forward and backward outputs against FLA's chunk_gated_delta_rule
using cosine similarity, and measures latency for fwd, bwd, and fwd+bwd.

Layout convention:
    TileOPs uses BHSD: q/k [B, H, S, DK], v [B, H, S, DV], g/beta [B, H, S].
    FLA uses BTHK:     q/k [B, T, H, K],  v [B, T, H, V],  g/beta [B, T, H].
    Tensors are permuted before calling FLA to ensure both implementations
    compute the same function.

Related issue: https://github.com/tile-ai/TileOPs/issues/472
"""

from typing import Optional

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from tilelang.profiler import do_bench as _do_bench

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_fwd import GatedDeltaNetFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp


def _to_fla_layout(q, k, v, g, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


def _cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


# =============================================================================
# Benchmark classes
# =============================================================================

class GatedDeltaNetFwdValidationBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 2) * elem


class GatedDeltaNetBwdValidationBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 4) * elem


class GatedDeltaNetFwdBwdValidationBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 6.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (6 * DK + 5 * DV + 6) * elem


# =============================================================================
# Fixtures
# =============================================================================

class GatedDeltaNetFlaValidationFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype", [
            (2, 256, 4, 64, 64, 64, torch.float16),
            (2, 1024, 4, 64, 64, 64, torch.float16),
            (2, 4096, 4, 64, 64, 64, torch.float16),
            (2, 8192, 4, 64, 64, 64, torch.float16),
            (2, 16384, 4, 64, 64, 64, torch.float16),
        ]),
    ]


# =============================================================================
# Forward validation + benchmark
# =============================================================================

@GatedDeltaNetFlaValidationFixture
def test_gated_deltanet_fla_validation_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdValidationBenchmark(test)

    q = (torch.randn(batch, heads, seq_len, dim_k, device="cuda", dtype=dtype) * 0.1)
    k = (torch.randn(batch, heads, seq_len, dim_k, device="cuda", dtype=dtype) * 0.1)
    v = (torch.randn(batch, heads, seq_len, dim_v, device="cuda", dtype=dtype) * 0.1)
    g = -torch.rand(batch, heads, seq_len, device="cuda", dtype=dtype)
    beta = torch.rand(batch, heads, seq_len, device="cuda", dtype=dtype) * 0.5

    scale = dim_k ** -0.5

    # FLA reference
    q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)
    o_fla, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
    o_fla_bhsd = o_fla.permute(0, 2, 1, 3).contiguous()

    # TileOPs forward
    fwd_op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    o_tile, _S, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)

    cos = _cosine_sim(o_tile, o_fla_bhsd)
    print(f"\n[FWD] B={batch} H={heads} S={seq_len} cosine(TileOPs, FLA) = {cos:.6f}")
    assert cos >= 0.999, f"FWD cosine similarity too low: {cos:.6f}"

    # Performance: TileOPs
    def tileops_fn():
        return fwd_op.forward(q, k, v, g, beta)
    result = bm.profile(tileops_fn)
    BenchmarkReport.record("gated_deltanet_fla_validation_fwd", locals(), result, tag="tileops")

    # Performance: FLA
    def fla_fn():
        return chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
    result_fla = bm.profile(fla_fn)
    BenchmarkReport.record("gated_deltanet_fla_validation_fwd", locals(), result_fla, tag="fla")


# =============================================================================
# Backward validation + benchmark
# =============================================================================

@GatedDeltaNetFlaValidationFixture
def test_gated_deltanet_fla_validation_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetBwdValidationBenchmark(test)

    q = (torch.randn(batch, heads, seq_len, dim_k, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    k = (torch.randn(batch, heads, seq_len, dim_k, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    v = (torch.randn(batch, heads, seq_len, dim_v, device="cuda", dtype=dtype) * 0.1).detach().requires_grad_(True)
    g = (-torch.rand(batch, heads, seq_len, device="cuda", dtype=dtype)).detach().requires_grad_(True)
    beta = (torch.rand(batch, heads, seq_len, device="cuda", dtype=dtype) * 0.5).detach().requires_grad_(True)
    do = torch.randn(batch, heads, seq_len, dim_v, device="cuda", dtype=dtype) * 0.1

    scale = dim_k ** -0.5

    # FLA reference: fwd + bwd
    q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q.data, k.data, v.data, g.data, beta.data)
    q_fla = q_fla.detach().requires_grad_(True)
    k_fla = k_fla.detach().requires_grad_(True)
    v_fla = v_fla.detach().requires_grad_(True)
    g_fla = g_fla.detach().requires_grad_(True)
    beta_fla = beta_fla.detach().requires_grad_(True)
    do_fla = do.permute(0, 2, 1, 3).contiguous()

    o_fla, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
    o_fla.backward(do_fla)

    dq_fla = q_fla.grad.permute(0, 2, 1, 3).contiguous()
    dk_fla = k_fla.grad.permute(0, 2, 1, 3).contiguous()
    dv_fla = v_fla.grad.permute(0, 2, 1, 3).contiguous()
    dg_fla = g_fla.grad.permute(0, 2, 1).contiguous()
    dbeta_fla = beta_fla.grad.permute(0, 2, 1).contiguous()

    # TileOPs: fwd then bwd
    fwd_op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q.data, k.data, v.data, g.data, beta.data)

    bwd_op = GatedDeltaNetBwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    dq_t, dk_t, dv_t, dg_t, dbeta_t = bwd_op.forward(do, q.data, k.data, v.data, g.data, beta.data, S_fwd)

    grad_cosines = {
        "dq": _cosine_sim(dq_t, dq_fla),
        "dk": _cosine_sim(dk_t, dk_fla),
        "dv": _cosine_sim(dv_t, dv_fla),
        "dg": _cosine_sim(dg_t, dg_fla),
        "dbeta": _cosine_sim(dbeta_t, dbeta_fla),
    }
    print(f"\n[BWD] B={batch} H={heads} S={seq_len}")
    for name, cos in grad_cosines.items():
        print(f"  cosine {name} = {cos:.6f}")
        assert cos >= 0.999, f"BWD cosine similarity for {name} too low: {cos:.6f}"

    # Performance: TileOPs bwd only
    def tileops_bwd():
        return bwd_op.forward(do, q.data, k.data, v.data, g.data, beta.data, S_fwd)
    result = bm.profile(tileops_bwd)
    BenchmarkReport.record("gated_deltanet_fla_validation_bwd", locals(), result, tag="tileops")

    # Performance: FLA fwd+bwd
    q_f2 = q_fla.data.detach().requires_grad_(True)
    k_f2 = k_fla.data.detach().requires_grad_(True)
    v_f2 = v_fla.data.detach().requires_grad_(True)
    g_f2 = g_fla.data.detach().requires_grad_(True)
    beta_f2 = beta_fla.data.detach().requires_grad_(True)

    def fla_fwdbwd():
        q_f2.grad = k_f2.grad = v_f2.grad = g_f2.grad = beta_f2.grad = None
        o, _ = chunk_gated_delta_rule(q_f2, k_f2, v_f2, g_f2, beta_f2, scale=scale)
        o.backward(do_fla, retain_graph=True)

    latency = _do_bench(fla_fwdbwd, warmup=100, rep=100, backend='cupti')
    if latency <= 0:
        latency = _do_bench(fla_fwdbwd, warmup=100, rep=100, backend='event')
    result_fla = {"latency_ms": latency}
    flops = bm.calculate_flops()
    if flops is not None:
        result_fla["tflops"] = flops / latency * 1e-9
    memory = bm.calculate_memory()
    if memory is not None:
        result_fla["bandwidth_tbs"] = memory / latency * 1e-9
    BenchmarkReport.record("gated_deltanet_fla_validation_bwd", locals(), result_fla, tag="fla")


# =============================================================================
# Combined fwd+bwd validation + benchmark
# =============================================================================

@GatedDeltaNetFlaValidationFixture
def test_gated_deltanet_fla_validation_fwdbwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdBwdValidationBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    scale = DK ** -0.5

    # TileOPs: combined fwd+bwd via GatedDeltaNetOp
    op = GatedDeltaNetOp(B, H, S, DK, DV, BC, dtype)

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

    latency_tile = _do_bench(tileops_fwdbwd, warmup=50, rep=100, backend='cupti')
    if latency_tile <= 0:
        latency_tile = _do_bench(tileops_fwdbwd, warmup=50, rep=100, backend='event')
    result = {"latency_ms": latency_tile}
    flops = bm.calculate_flops()
    if flops is not None:
        result["tflops"] = flops / latency_tile * 1e-9
    memory = bm.calculate_memory()
    if memory is not None:
        result["bandwidth_tbs"] = memory / latency_tile * 1e-9
    BenchmarkReport.record("gated_deltanet_fla_validation_fwdbwd", locals(), result, tag="tileops")

    # FLA: fwd+bwd via autograd
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

    latency_fla = _do_bench(fla_fwdbwd, warmup=50, rep=100, backend='cupti')
    if latency_fla <= 0:
        latency_fla = _do_bench(fla_fwdbwd, warmup=50, rep=100, backend='event')
    result_fla = {"latency_ms": latency_fla}
    if flops is not None:
        result_fla["tflops"] = flops / latency_fla * 1e-9
    if memory is not None:
        result_fla["bandwidth_tbs"] = memory / latency_fla * 1e-9
    BenchmarkReport.record("gated_deltanet_fla_validation_fwdbwd", locals(), result_fla, tag="fla")

    print(f"\n[FWD+BWD] B={B} H={H} S={S} TileOPs={latency_tile:.3f}ms FLA={latency_fla:.3f}ms "
          f"ratio={latency_tile/latency_fla:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
