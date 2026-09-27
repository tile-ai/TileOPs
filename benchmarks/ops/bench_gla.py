"""Benchmarks for the GLA ops (chunkwise forward and backward, inference, decode), one case per manifest call,
against FLA and torch.
"""

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops import GLABwdOp, GLADecodeFwdOp, GLAFwdOp, GLAInferenceFwdOp
from workloads.linear_attention import GLAChunkwiseCall, GLADecodeCall, GLAInferenceCall

try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None


# Chunkwise: FLA's chunk_gla is required; a torch reference is not a comparison worth recording.
# TileOPs and FLA both use BTHD: q/k [B, T, H, K], v [B, T, H, V], g [B, T, H, K].
@pytest.mark.parametrize("call", manifest_calls(GLAFwdOp))
def test_gla_fwd_bench(call) -> None:
    from fla.ops.gla import chunk_gla

    workload = GLAChunkwiseCall(call)
    q, k, v, g, initial_state = inputs = workload.gen_inputs()
    op = GLAFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)

    def fla_fwd():
        return chunk_gla(
            q, k, v, g, scale=op.scale, initial_state=initial_state, output_final_state=True
        )

    bm.compare({"tileops": op, "fla": (fla_fwd, ())}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(GLABwdOp))
def test_gla_bwd_bench(call) -> None:
    from fla.ops.gla import chunk_gla

    workload = GLAChunkwiseCall(call)
    q, k, v, g, _h, do, _dht = workload.gen_inputs()
    arguments = workload.arguments()

    # The per-chunk states are the forward's, so the backward reads what it would in training.
    fwd_op = GLAFwdOp(arguments["chunk_size"], arguments["scale"])
    fwd_op(q, k, v, g)
    (fwd_kernel,) = fwd_op.built_kernels("GLAFwdKernel").values()
    h = fwd_kernel._h_out
    dht = torch.zeros_like(_dht)

    bwd_op = GLABwdOp(**arguments)
    bm = ManifestBenchmark(bwd_op, workload)

    # FLA's backward recomputes h internally (not saved from fwd), so this measures
    # bwd + h recomputation, not pure bwd.
    q_fla, k_fla, v_fla, g_fla = (t.float().detach().requires_grad_(True) for t in (q, k, v, g))
    o_fla, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=bwd_op.scale)
    fla_backward = backward_of(o_fla)
    do_fla = do.float()

    def fla_bwd():
        return fla_backward(do_fla, None)[:4]

    bm.compare({"tileops": bwd_op, "fla": (fla_bwd, ())}, q, k, v, g, h, do, dht)


@pytest.mark.parametrize("call", manifest_calls(GLAInferenceFwdOp))
def test_gla_inference_dense_prefill_bench(call) -> None:
    workload = GLAInferenceCall(call)
    inputs = workload.gen_inputs()
    op = GLAInferenceFwdOp(**workload.arguments())
    tolerance = reference_tolerance(inputs[0].dtype)
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": workload.ref_program}, *inputs)


# Decode: against FLA's fused_recurrent_gla at T=1 when it is installed, and torch.


@pytest.mark.parametrize("call", manifest_calls(GLADecodeFwdOp))
def test_gla_decode_bench(call) -> None:
    workload = GLADecodeCall(call)
    inputs = workload.gen_inputs()
    op = GLADecodeFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    if fused_recurrent_gla is not None:
        q, k, v, gk, state = inputs
        q_fla, k_fla, v_fla, gk_fla = (t.unsqueeze(1) for t in (q, k, v, gk))

        def fla_decode():
            o, new_state = fused_recurrent_gla(
                q_fla,
                k_fla,
                v_fla,
                gk=gk_fla,
                scale=op.scale,
                initial_state=state.contiguous(),
                output_final_state=True,
            )
            return o.squeeze(1), new_state.to(state.dtype)

        functors["fla"] = (fla_decode, ())

    functors["torch"] = workload.ref_program
    functors[TORCH_COMPILE_TAG] = compiled_reference(workload.ref_program)

    bm.compare(functors, *inputs)
