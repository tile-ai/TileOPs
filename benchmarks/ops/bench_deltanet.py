"""Benchmark: TileOPs DeltaNet (ungated) chunkwise vs FLA chunk_delta_rule.

Compares forward and backward latency across sequence lengths and dtypes.

FLA is required, not optional: this file exists to compare against chunk_delta_rule,
and a torch reference is not a comparison worth recording.

Layout convention:
    TileOPs uses BHSD: q/k [B, H, S, DK], v [B, H, S, DV], beta [B, H, S].
    FLA uses BTHK:     q/k [B, T, H, K],  v [B, T, H, V],  beta [B, T, H].
    Tensors are permuted before calling FLA to ensure both implementations
    compute the same function.
"""

from typing import Optional

import pytest
import torch
from fla.ops.delta_rule import chunk_delta_rule

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, backward_of
from tileops.ops import DeltaNetBwdOp, DeltaNetFwdOp
from workloads.linear_attention import DeltaNetFwdWorkload
from workloads.workload_base import FixtureBase


def _to_fla_layout(q, k, v, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


# Forward benchmark

class DeltaNetFwdBenchmark(BenchmarkBase[DeltaNetFwdWorkload]):

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
    test = DeltaNetFwdWorkload(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = DeltaNetFwdBenchmark(test)
    inputs = test.gen_inputs()  # q, k, v, beta (BHSD)

    # --- TileOPs (BHSD) ---
    op = DeltaNetFwdOp(chunk_size=chunk_size, tune=tune)
    functors = {"tileops": op}

    # --- FLA (BTHK) ---
    q, k, v, beta = inputs
    scale = dim_k ** -0.5
    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(q, k, v, beta)

    def fla_fwd():
        return chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)

    functors["fla"] = (fla_fwd, ())

    bm.compare(functors, *inputs, record_as=op, params=locals())


# Backward benchmark

class DeltaNetBwdBenchmark(BenchmarkBase[DeltaNetFwdWorkload]):

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
    test = DeltaNetFwdWorkload(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = DeltaNetBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get S, Aw, Au, w, u; then profile bwd only ---
    fwd_op = DeltaNetFwdOp(chunk_size=BC)
    _o, S_fwd, Aw, Au, w_fwd, u_fwd = fwd_op.forward(q, k, v, beta)

    bwd_op = DeltaNetBwdOp(chunk_size=BC, tune=tune)
    functors = {"tileops": bwd_op.forward}

    # --- FLA (BTHK layout) ---
    scale = DK ** -0.5
    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(q, k, v, beta)
    do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

    q_fla = q_fla.detach().requires_grad_(True)
    k_fla = k_fla.detach().requires_grad_(True)
    v_fla = v_fla.detach().requires_grad_(True)
    beta_fla = beta_fla.detach().requires_grad_(True)

    # Run fwd once to build the graph, then time the backward node directly
    o_fla, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)
    # One grad per forward output: fla returns (o, final_state).
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        return fla_backward(do_fla, None)

    functors["fla"] = (fla_bwd, ())

    bm.compare(functors, do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd,
               record_as=bwd_op, params=locals())
