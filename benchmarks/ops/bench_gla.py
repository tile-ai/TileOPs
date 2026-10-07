"""Benchmarks for the GLA ops (chunkwise forward and backward, inference, decode), one case per manifest call,
against FLA and torch.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    backward_of,
    compiled_reference,
)
from tileops.ops import GLAChunkBwdOp, GLAChunkFwdOp, GLAInferenceFwdOp, GLARecurrentFwdOp

try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None


# Chunkwise: FLA's chunk_gla is required; a torch reference is not a comparison worth recording.
# TileOPs and FLA both use BTHD: q/k [B, T, H, K], v [B, T, H, V], g [B, T, H, K].
@pytest.mark.parametrize("case", bench.cases(GLAChunkFwdOp), ids=lambda case: case.id)
def test_gla_fwd_bench(case) -> None:
    from fla.ops.gla import chunk_gla

    q, k, v, g, initial_state = case.inputs
    op = GLAChunkFwdOp(**case.arguments)

    def fla_fwd():
        return chunk_gla(
            q, k, v, g, scale=op.scale, initial_state=initial_state, output_final_state=True
        )

    bench.Runner(op, case).compare({"fla": bench.Implementation(run=fla_fwd, args=())})


@pytest.mark.parametrize("case", bench.cases(GLAChunkBwdOp), ids=lambda case: case.id)
def test_gla_bwd_bench(case) -> None:
    from fla.ops.gla import chunk_gla

    q, k, v, g, _h, do, _dht = case.inputs
    bwd_op = GLAChunkBwdOp(**case.arguments)
    q_fla, k_fla, v_fla, g_fla = (t.float().detach().requires_grad_(True) for t in (q, k, v, g))
    o_fla, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=bwd_op.scale)
    fla_backward = backward_of(o_fla)
    do_fla = do.float()

    def fla_bwd():
        return fla_backward(do_fla, None)[:4]

    bench.Runner(bwd_op, case).compare({"fla": bench.Implementation(run=fla_bwd, args=())})


@pytest.mark.parametrize("case", bench.cases(GLAInferenceFwdOp), ids=lambda case: case.id)
def test_gla_inference_bench(case) -> None:
    op = GLAInferenceFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"fla": case.reference})


# Decode: against FLA's fused_recurrent_gla at T=1 when it is installed, and torch.


@pytest.mark.parametrize("case", bench.cases(GLARecurrentFwdOp), ids=lambda case: case.id)
def test_gla_decode_bench(case) -> None:
    op = GLARecurrentFwdOp(**case.arguments)
    implementations = {}

    if fused_recurrent_gla is not None:
        q, k, v, gk, state = case.inputs
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

        implementations["fla"] = bench.Implementation(run=fla_decode, args=())

    implementations["torch"] = case.reference
    implementations[TORCH_COMPILE_TAG] = compiled_reference(case.reference)

    bench.Runner(op, case).compare(implementations)
