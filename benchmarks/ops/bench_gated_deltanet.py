"""Benchmark: TileOPs Gated DeltaNet vs FLA chunk_gated_delta_rule.

Compares forward and backward latency across sequence lengths and dtypes.

FLA is required, not optional: this file exists to compare against
chunk_gated_delta_rule, and a torch reference is not a comparison worth recording.

Layout convention:
    The forward benchmark uses the shared TileOps/FLA BTHD interface:
    q/k [B, T, H, K], v [B, T, H, V], g/beta [B, T, H].
    Backward still uses the legacy TileOps BHSD interface and permutes the
    FLA inputs explicitly.
"""

from typing import Optional

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark_base import (
    BenchmarkBase,
    BenchmarkReport,
    ManifestBenchmark,
    backward_of,
)
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetBHTDFwdOp, GatedDeltaNetBTHDFwdOp, GatedDeltaNetBwdOp
from workloads.linear_attention import GatedDeltaNetFwdWorkload
from workloads.workload_base import FixtureBase


def _to_fla_layout(q, k, v, g, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


# Forward benchmark


class GatedDeltaNetFwdBenchmark(BenchmarkBase[GatedDeltaNetFwdWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 2) * elem


_FWD_OP_NAME = "GatedDeltaNetBTHDFwdOp"


def _gdn_bthd_args(workload: dict) -> tuple[int, int, int, int, int, int]:
    """Constructor arguments for one manifest workload row, token-major."""
    batch, seq_len, heads, dim_k = workload["q_shape"]
    dim_v = workload["v_shape"][3]
    return batch, heads, seq_len, dim_k, dim_v, workload.get("chunk_size", 64)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune",
    manifest_params(load_workloads(_FWD_OP_NAME), _gdn_bthd_args, tune=False),
)
def test_gated_deltanet_vs_fla_fwd(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdWorkload(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    q, k, v, g, beta = test.gen_inputs()  # BHSD
    # Both sides take the token-major interface; the conversion is outside the
    # timed region.
    bthd = _to_fla_layout(q, k, v, g, beta)

    op = GatedDeltaNetBTHDFwdOp(chunk_size=chunk_size, tune=tune)
    bm = ManifestBenchmark(_FWD_OP_NAME, op, test)

    def fla_fwd():
        return chunk_gated_delta_rule(*bthd, scale=1.0)

    bm.compare({"tileops": (op, bthd), "fla": (fla_fwd, ())}, record_as=op, params=locals())


# Backward benchmark


class GatedDeltaNetBwdBenchmark(BenchmarkBase[GatedDeltaNetFwdWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 4) * elem


class GatedDeltaNetVsFlaBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                # chunk_size=32
                (2, 4096, 4, 64, 64, 32, torch.float16, False),
                (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
                # chunk_size=64
                (2, 2048, 4, 64, 64, 64, torch.float16, False),
                (2, 4096, 4, 64, 64, 64, torch.float16, False),
                (2, 8192, 4, 64, 64, 64, torch.float16, False),
                (2, 16384, 4, 64, 64, 64, torch.float16, False),
            ],
        ),
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
    test = GatedDeltaNetFwdWorkload(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get S, then profile bwd only ---
    fwd_op = GatedDeltaNetBHTDFwdOp(chunk_size=BC)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)

    bwd_op = GatedDeltaNetBwdOp(chunk_size=BC, tune=tune)
    functors = {"tileops": bwd_op.forward}

    # --- FLA (BTHK layout) ---
    scale = DK**-0.5
    q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)
    do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

    q_fla = q_fla.detach().requires_grad_(True)
    k_fla = k_fla.detach().requires_grad_(True)
    v_fla = v_fla.detach().requires_grad_(True)
    g_fla = g_fla.detach().requires_grad_(True)
    beta_fla = beta_fla.detach().requires_grad_(True)

    o_fla, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        return fla_backward(do_fla, None)

    functors["fla"] = (fla_bwd, ())
    bm.compare(functors, do, q, k, v, g, beta, S_fwd, record_as=bwd_op, params=locals())
