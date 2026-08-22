"""Benchmark: TileOPs GLA vs FLA chunk_gla.

Compares forward and backward latency across sequence lengths and dtypes.

FLA is required, not optional: this file exists to compare against chunk_gla, and a
torch reference is not a comparison worth recording.

Layout convention:
    Both TileOPs and FLA use BTHD: q/k [B, T, H, K], v [B, T, H, V], g [B, T, H, K].
"""

import pytest
import torch
from fla.ops.gla import chunk_gla

from benchmarks.benchmark_base import ManifestBenchmark, backward_of
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GLABwdOp, GLAFwdOp
from workloads.linear_attention import GLAChunkwiseWorkload

_FWD_OP_NAME = "GLAFwdOp"
_BWD_OP_NAME = "GLABwdOp"


def _gla_args(workload: dict) -> tuple[int, int, int, int, int, int, bool]:
    """Constructor arguments for one manifest workload row.

    A row that declares ``initial_state_shape`` seeds the recurrence; one that does
    not starts from zeros, which is the other half of the optional input's contract.
    """
    batch, seq_len, heads, dim_k = workload["q_shape"]
    dim_v = workload["v_shape"][3]
    return (
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        workload.get("chunk_size", 64),
        "initial_state_shape" in workload,
    )


def _gla_bwd_args(workload: dict) -> tuple[int, int, int, int, int, int]:
    """Constructor arguments for one manifest workload row; the backward takes no state."""
    batch, seq_len, heads, dim_k = workload["q_shape"]
    return batch, seq_len, heads, dim_k, workload["v_shape"][3], workload.get("chunk_size", 64)


# Forward benchmark


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, chunk_size, has_initial_state, dtype, tune",
    manifest_params(load_workloads(_FWD_OP_NAME), _gla_args, tune=False),
)
def test_gla_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    has_initial_state: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GLAChunkwiseWorkload(
        batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, has_initial_state
    )
    inputs = test.gen_inputs()

    # --- TileOPs ---
    scale = dim_k**-0.5
    op = GLAFwdOp(chunk_size=chunk_size, scale=scale, output_final_state=True, tune=tune)
    bm = ManifestBenchmark(_FWD_OP_NAME, op, test)
    functors = {"tileops": op.forward}

    # --- FLA ---
    q, k, v, g, initial_state = inputs

    def fla_fwd():
        return chunk_gla(q, k, v, g, scale=scale, initial_state=initial_state)

    functors["fla"] = (fla_fwd, ())

    bm.compare(functors, *inputs, record_as=op, params=locals())


# Backward benchmark


@pytest.mark.xfail(
    reason="TileLang emits a WGMMA descriptor for a B operand whose layout the "
    "assert rejects: 'Not a canonical GMMA_MN layout'. Fails on main too.",
    strict=False,
)
@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
    manifest_params(load_workloads(_BWD_OP_NAME), _gla_bwd_args, tune=False),
)
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

    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size
    scale = K**-0.5

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
    bm = ManifestBenchmark(_BWD_OP_NAME, bwd_op, test)
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
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        return fla_backward(do_fla, None)

    functors["fla"] = (fla_bwd, ())

    bm.compare(functors, record_as=bwd_op, params=locals())


# Combined fwd+bwd benchmark
