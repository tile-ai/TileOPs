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

from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops import GLABwdOp, GLAFwdOp
from workloads.linear_attention import GLAChunkwiseCall


@pytest.mark.parametrize("call", manifest_calls(GLAFwdOp))
def test_gla_fwd_bench(call) -> None:
    test = GLAChunkwiseCall(call)
    q, k, v, g, initial_state = inputs = test.gen_inputs()
    op = GLAFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)

    def fla_fwd():
        return chunk_gla(
            q, k, v, g, scale=op.scale, initial_state=initial_state, output_final_state=True
        )

    bm.compare({"tileops": op, "fla": (fla_fwd, ())}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(GLABwdOp))
def test_gla_bwd_bench(call) -> None:
    test = GLAChunkwiseCall(call)
    q, k, v, g, _h, do, _dht = test.gen_inputs()
    arguments = test.arguments()

    # The per-chunk states are the forward's, so the backward reads what it would in training.
    fwd_op = GLAFwdOp(arguments["chunk_size"], arguments["scale"])
    fwd_op(q, k, v, g)
    (fwd_kernel,) = fwd_op.built_kernels("GLAFwdKernel").values()
    h = fwd_kernel._h_out
    dht = torch.zeros_like(_dht)

    bwd_op = GLABwdOp(**arguments)
    bm = ManifestBenchmark(bwd_op, test)

    # FLA's backward recomputes h internally (not saved from fwd), so this measures
    # bwd + h recomputation, not pure bwd.
    q_fla, k_fla, v_fla, g_fla = (t.float().detach().requires_grad_(True) for t in (q, k, v, g))
    o_fla, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=bwd_op.scale)
    fla_backward = backward_of(o_fla)
    do_fla = do.float()

    def fla_bwd():
        return fla_backward(do_fla, None)[:4]

    bm.compare({"tileops": bwd_op, "fla": (fla_bwd, ())}, q, k, v, g, h, do, dht)
