"""Benchmarks for the DeltaNet ops (chunkwise forward and backward, inference, decode), one case per
manifest call, against FLA and torch.
"""

import dataclasses

import pytest

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    compiled_reference,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from benchmarks.verification import Exact, Partial
from tileops.ops import (
    DeltaNetChunkBwdOp,
    DeltaNetChunkFwdOp,
    DeltaNetInferenceFwdOp,
    DeltaNetRecurrentFwdOp,
)
from workloads.linear_attention.deltanet import (
    DeltaNetChunkwiseCall,
    DeltaNetDecodeCall,
    DeltaNetInferenceCall,
    deltanet_autograd_bwd_torch,
    deltanet_differentiable_fwd_torch,
)


# Chunkwise: FLA's chunk_delta_rule is required. TileOPs uses BHSD (q/k [B, H, S, DK], v [B, H, S, DV],
# beta [B, H, S]) and FLA BTHK, so inputs are permuted before calling FLA.
def _to_fla_layout(q, k, v, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


@pytest.mark.parametrize("call", manifest_calls(DeltaNetInferenceFwdOp))
def test_deltanet_inference_bench(call) -> None:
    workload = DeltaNetInferenceCall(call)
    inputs = workload.gen_inputs()
    op = DeltaNetInferenceFwdOp(**workload.arguments())
    ManifestBenchmark(op, workload).compare(
        {"tileops": op, "fla": workload.ref_program},
        *inputs,
        evidence={"tileops": Exact(**reference_tolerance(inputs[0].dtype))},
    )


@pytest.mark.parametrize("call", manifest_calls(DeltaNetChunkFwdOp))
def test_deltanet_vs_fla_fwd(call) -> None:
    from fla.ops.delta_rule import chunk_delta_rule

    workload = DeltaNetChunkwiseCall(call)
    inputs = workload.gen_inputs()  # q, k, v, beta (BHSD)
    op = DeltaNetChunkFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)

    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(*inputs)

    # FLA has no API returning the chunk buffers the backward reads; it computes o.
    def fla_fwd():
        return chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)

    chunked = Partial(
        outputs=1,
        reason="S, Aw, Au, w and u are the chunk buffers the backward reads; the "
        "differentiable reference computes o without materialising them",
        rtol=2e-2,
        atol=2e-2,
        reference=lambda q, k, v, beta: deltanet_differentiable_fwd_torch(
            q.float(), k.float(), v.float(), beta.float(), workload.arguments()["chunk_size"]
        ).to(q.dtype),
    )
    fla_layout = dataclasses.replace(
        chunked, reference=lambda *a, _r=chunked.reference: _r(*a).permute(0, 2, 1, 3).contiguous()
    )
    bm.compare(
        {"tileops": op, "fla": (fla_fwd, ())},
        *inputs,
        evidence={"tileops": chunked, "fla": fla_layout},
    )


@pytest.mark.parametrize("call", manifest_calls(DeltaNetChunkBwdOp))
def test_deltanet_vs_fla_bwd(call) -> None:
    from fla.ops.delta_rule import chunk_delta_rule

    workload = DeltaNetChunkwiseCall(call)
    do, q, k, v, beta, *_saved = workload.gen_inputs()

    # The saved buffers are the forward's, so the backward reads what it would in training.
    fwd_op = DeltaNetChunkFwdOp(workload.arguments()["chunk_size"])
    _o, S, Aw, Au, w, u = fwd_op(q, k, v, beta)

    bwd_op = DeltaNetChunkBwdOp(**workload.arguments())
    bm = ManifestBenchmark(bwd_op, workload)

    q_fla, k_fla, v_fla, beta_fla = (
        t.detach().requires_grad_(True) for t in _to_fla_layout(q, k, v, beta)
    )
    do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]
    o_fla, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        dq, dk, dv, dbeta = fla_backward(do_fla, None)[:4]
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), dbeta.transpose(1, 2)

    autograd = Exact(
        rtol=2e-2,
        atol=2e-2,
        reference=lambda do, q, k, v, beta, *_saved: tuple(
            t.to(q.dtype)
            for t in deltanet_autograd_bwd_torch(
                do, q, k, v, beta, workload.arguments()["chunk_size"]
            )
        ),
    )
    bm.compare(
        {"tileops": bwd_op, "fla": (fla_bwd, ())},
        do,
        q,
        k,
        v,
        beta,
        S,
        Aw,
        Au,
        w,
        u,
        evidence={"tileops": autograd, "fla": autograd},
    )


@pytest.mark.parametrize("call", manifest_calls(DeltaNetRecurrentFwdOp))
def test_deltanet_decode_bench(call) -> None:
    workload = DeltaNetDecodeCall(call)
    inputs = workload.gen_inputs()
    op = DeltaNetRecurrentFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    bm.compare(
        {
            "tileops": op,
            "torch": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )
