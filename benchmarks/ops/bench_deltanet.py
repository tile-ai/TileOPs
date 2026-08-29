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

import pytest
import torch
from fla.ops.delta_rule import chunk_delta_rule

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    backward_of,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import DeltaNetBwdOp, DeltaNetFwdOp
from workloads.linear_attention import DeltaNetFwdWorkload


def _to_fla_layout(q, k, v, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


# Forward benchmark

_FWD_OP_NAME = "DeltaNetFwdOp"
_BWD_OP_NAME = "DeltaNetBwdOp"


def _deltanet_args(workload: dict) -> tuple[int, int, int, int, int, int]:
    """Constructor arguments for one manifest workload row."""
    batch, heads, seq_len, dim_k = workload["q_shape"]
    dim_v = workload["v_shape"][3]
    return batch, seq_len, heads, dim_k, dim_v, workload.get("chunk_size", 64)


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
    workload_params(load_workloads(_FWD_OP_NAME), then_dtype(_deltanet_args, tune=False)),
)
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
    inputs = test.gen_inputs()  # q, k, v, beta (BHSD)

    # --- TileOPs (BHSD) ---
    op = DeltaNetFwdOp(chunk_size=chunk_size, tune=tune)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    # --- FLA (BTHK) ---
    q, k, v, beta = inputs
    scale = dim_k**-0.5
    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(q, k, v, beta)

    def fla_fwd():
        return chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)

    functors["fla"] = (fla_fwd, ())

    bm.compare(functors, *inputs, record_as=op, params=locals())


# Backward benchmark


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
    workload_params(load_workloads(_BWD_OP_NAME), then_dtype(_deltanet_args, tune=False)),
)
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
    bm = ManifestBenchmark(bwd_op, test)
    functors = {"tileops": bwd_op.forward}

    # --- FLA (BTHK layout) ---
    scale = DK**-0.5
    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(q, k, v, beta)
    do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

    q_fla = q_fla.detach().requires_grad_(True)
    k_fla = k_fla.detach().requires_grad_(True)
    v_fla = v_fla.detach().requires_grad_(True)
    beta_fla = beta_fla.detach().requires_grad_(True)

    o_fla, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=scale)
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        return fla_backward(do_fla, None)

    functors["fla"] = (fla_bwd, ())

    bm.compare(
        functors, do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd, record_as=bwd_op, params=locals()
    )
