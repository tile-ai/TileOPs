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
from fla.ops.delta_rule import chunk_delta_rule

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops import DeltaNetAutogradFwdOp, DeltaNetBwdOp, DeltaNetFwdOp, DeltaNetInferenceFwdOp
from workloads.linear_attention import DeltaNetChunkwiseCall, DeltaNetInferenceCall


def _to_fla_layout(q, k, v, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


@pytest.mark.parametrize("call", manifest_calls(DeltaNetInferenceFwdOp))
def test_deltanet_dense_prefill_bench(call) -> None:
    workload = DeltaNetInferenceCall(call)
    inputs = workload.gen_inputs()
    op = DeltaNetInferenceFwdOp(**workload.arguments())
    dtype = inputs[0].dtype
    assert_matches_reference(op, workload.ref_program, *inputs, **reference_tolerance(dtype))
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": workload.ref_program}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(DeltaNetFwdOp))
def test_deltanet_vs_fla_fwd(call) -> None:
    test = DeltaNetChunkwiseCall(call)
    inputs = test.gen_inputs()  # q, k, v, beta (BHSD)
    op = DeltaNetFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)

    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(*inputs)

    # FLA has no API returning the chunk buffers the backward reads; it computes o.
    def fla_fwd():
        return chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)

    bm.compare({"tileops": op, "fla": (fla_fwd, ())}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(DeltaNetBwdOp))
def test_deltanet_vs_fla_bwd(call) -> None:
    test = DeltaNetChunkwiseCall(call)
    do, q, k, v, beta, *_saved = test.gen_inputs()

    # The saved buffers are the forward's, so the backward reads what it would in training.
    fwd_op = DeltaNetFwdOp(test.arguments()["chunk_size"])
    _o, S, Aw, Au, w, u = fwd_op(q, k, v, beta)

    bwd_op = DeltaNetBwdOp(**test.arguments())
    bm = ManifestBenchmark(bwd_op, test)

    # --- FLA (BTHK layout) ---
    q_fla, k_fla, v_fla, beta_fla = (
        t.detach().requires_grad_(True) for t in _to_fla_layout(q, k, v, beta)
    )
    do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]
    o_fla, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        dq, dk, dv, dbeta = fla_backward(do_fla, None)[:4]
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), dbeta.transpose(1, 2)

    bm.compare({"tileops": bwd_op, "fla": (fla_bwd, ())}, do, q, k, v, beta, S, Aw, Au, w, u)


@pytest.mark.parametrize("call", manifest_calls(DeltaNetAutogradFwdOp))
def test_deltanet_vs_fla_autograd(call) -> None:
    test = DeltaNetChunkwiseCall(call)
    inputs = test.gen_inputs()
    op = DeltaNetAutogradFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)

    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(*inputs)

    def fla_fwd():
        return chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)[0]

    bm.compare({"tileops": op, "fla": (fla_fwd, ())}, *inputs)
