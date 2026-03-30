"""Benchmark: TileOPs Gated DeltaNet vs FLA chunk_gated_delta_rule.

Compares forward and backward latency across sequence lengths and dtypes.

When FLA is not installed, benchmarks still run using a pure-torch reference
implementation as baseline (tagged "baseline"), so the nightly CI is never
blocked by a missing optional dependency.

Layout convention:
    TileOPs uses BHSD: q/k [B, H, S, DK], v [B, H, S, DV], g/beta [B, H, S].
    FLA uses BTHK:     q/k [B, T, H, K],  v [B, T, H, V],  g/beta [B, T, H].
    Tensors are permuted before calling FLA to ensure both implementations
    compute the same function.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_chunkwise_bwd import _autograd_bwd_ref
from tests.ops.test_gated_deltanet_chunkwise_fwd import GatedDeltaNetFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None


def _to_fla_layout(q, k, v, g, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


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
            # chunk_size=32
            #(2, 1024, 4, 64, 64, 32, torch.float32, False),
            #(2, 2048, 4, 64, 64, 32, torch.float32, False),
            #(2, 4096, 4, 64, 64, 32, torch.float32, False),
            #(2, 1024, 4, 64, 64, 32, torch.float16, False),
            #(2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            #(2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            #(2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
            # chunk_size=64
            #(2, 1024, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 32768, 4, 64, 64, 64, torch.float16, False),
            #(2, 1024, 4, 64, 64, 64, torch.bfloat16, False),
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
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if chunk_gated_delta_rule is not None:
        # --- FLA (BTHK) ---
        q, k, v, g, beta = inputs
        scale = dim_k ** -0.5
        q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)

        def fla_fwd():
            return chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)

        result_fla = bm.profile(fla_fwd)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch reference baseline ---
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


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
            # chunk_size=32
            #(2, 1024, 4, 64, 64, 32, torch.float32, False),
            #(2, 2048, 4, 64, 64, 32, torch.float32, False),
            #(2, 4096, 4, 64, 64, 32, torch.float32, False),
            #(2, 1024, 4, 64, 64, 32, torch.float16, False),
            #(2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            #(2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            #(2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
            # chunk_size=64
            #(2, 1024, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            #(2, 1024, 4, 64, 64, 64, torch.bfloat16, False),
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
    BenchmarkReport.record(bwd_op, locals(), result, tag="tileops")

    if chunk_gated_delta_rule is not None:
        # --- FLA: bwd only via autograd (BTHK layout) ---
        scale = DK ** -0.5
        q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)
        do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

        q_fla = q_fla.detach().requires_grad_(True)
        k_fla = k_fla.detach().requires_grad_(True)
        v_fla = v_fla.detach().requires_grad_(True)
        g_fla = g_fla.detach().requires_grad_(True)
        beta_fla = beta_fla.detach().requires_grad_(True)

        # Run fwd once to build computation graph, then time only backward
        o_fla, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)

        def fla_bwd():
            q_fla.grad = k_fla.grad = v_fla.grad = g_fla.grad = beta_fla.grad = None
            o_fla.backward(do_fla, retain_graph=True)
            return q_fla.grad, k_fla.grad, v_fla.grad

        result_fla = bm.profile(fla_bwd)
        BenchmarkReport.record(bwd_op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_bwd():
            return _autograd_bwd_ref(do, q, k, v, g, beta, BC)
        result_bl = bm.profile(torch_bwd)
        BenchmarkReport.record(bwd_op, locals(), result_bl, tag="torch")


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
            # chunk_size=32
            #(2, 1024, 4, 64, 64, 32, torch.float32, False),
            #(2, 2048, 4, 64, 64, 32, torch.float32, False),
            #(2, 4096, 4, 64, 64, 32, torch.float32, False),
            #(2, 1024, 4, 64, 64, 32, torch.float16, False),
            #(2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            #(2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            #(2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
            # chunk_size=64
            #(2, 1024, 4, 64, 64, 64, torch.float16, False),
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            #(2, 1024, 4, 64, 64, 64, torch.bfloat16, False),
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

    result = bm.profile_autograd(tileops_fwdbwd)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if chunk_gated_delta_rule is not None:
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
            o.backward(do_fla)
            return q_fla.grad, k_fla.grad, v_fla.grad

        result_fla = bm.profile_autograd(fla_fwdbwd)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch autograd reference baseline ---
        def torch_fwdbwd():
            return _autograd_bwd_ref(do, q.data, k.data, v.data, g.data, beta.data, BC)
        result_bl = bm.profile_autograd(torch_fwdbwd)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
