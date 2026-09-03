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

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    backward_of,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import (
    GatedDeltaNetAutogradOp,
    GatedDeltaNetBHTDFwdOp,
    GatedDeltaNetBTHDFwdOp,
    GatedDeltaNetBwdOp,
)
from workloads.linear_attention import GatedDeltaNetFwdWorkload


def _to_fla_layout(q, k, v, g, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


def _gdn_bhtd_args(workload: dict) -> tuple[int, int, int, int, int, int]:
    """Constructor arguments for one manifest workload row, head-major."""
    batch, heads, seq_len, dim_k = workload["q_shape"]
    dim_v = workload["v_shape"][3]
    return batch, heads, seq_len, dim_k, dim_v, workload.get("chunk_size", 64)


def _gdn_bthd_args(workload: dict) -> tuple[int, int, int, int, int, int]:
    """Constructor arguments for one manifest workload row, token-major."""
    batch, seq_len, heads, dim_k = workload["q_shape"]
    dim_v = workload["v_shape"][3]
    return batch, heads, seq_len, dim_k, dim_v, workload.get("chunk_size", 64)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune",
    workload_params(load_workloads(GatedDeltaNetBTHDFwdOp), then_dtype(_gdn_bthd_args, tune=False)),
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
    bm = ManifestBenchmark(op, test)

    def fla_fwd():
        return chunk_gated_delta_rule(*bthd, scale=1.0)

    bm.compare({"tileops": (op, bthd), "fla": (fla_fwd, ())})


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune",
    workload_params(load_workloads(GatedDeltaNetBHTDFwdOp), then_dtype(_gdn_bhtd_args, tune=False)),
)
def test_gated_deltanet_bhtd_vs_fla_fwd(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """The head-major forward, which is what the backward reads its state from."""
    test = GatedDeltaNetFwdWorkload(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    inputs = test.gen_inputs()  # BHSD, the order this op takes
    bthd = _to_fla_layout(*inputs)

    op = GatedDeltaNetBHTDFwdOp(chunk_size=chunk_size, tune=tune)
    bm = ManifestBenchmark(op, test)

    def fla_fwd():
        return chunk_gated_delta_rule(*bthd, scale=1.0)

    bm.compare({"tileops": (op, inputs), "fla": (fla_fwd, ())})


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune",
    workload_params(load_workloads(GatedDeltaNetBwdOp), then_dtype(_gdn_bhtd_args, tune=False)),
)
def test_gated_deltanet_vs_fla_bwd(
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
    bm = ManifestBenchmark(bwd_op, test)
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
    bm.compare(functors, do, q, k, v, g, beta, S_fwd)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune",
    workload_params(
        load_workloads(GatedDeltaNetAutogradOp), then_dtype(_gdn_bhtd_args, tune=False)
    ),
)
def test_gated_deltanet_vs_fla_autograd(
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

    inputs = test.gen_inputs()

    op = GatedDeltaNetAutogradOp(chunk_size=chunk_size, tune=tune)
    bm = ManifestBenchmark(op, test)

    scale = dim_k**-0.5
    fla_inputs = _to_fla_layout(*inputs)

    def fla_fwd():
        return chunk_gated_delta_rule(*fla_inputs, scale=scale)

    bm.compare({"tileops": op, "fla": (fla_fwd, ())}, *inputs)
