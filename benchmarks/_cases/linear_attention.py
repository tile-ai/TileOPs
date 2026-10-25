"""Case factories of the linear_attention family."""

import torch

from benchmarks._cases import Entry
from workloads.linear_attention.deltanet import (
    DeltaNetChunkwiseCall,
    DeltaNetDecodeCall,
    DeltaNetFwdCall,
)
from workloads.linear_attention.gdn import GDNFwdCall
from workloads.linear_attention.gla import GLAChunkwiseCall, GLADecodeCall, GLAFwdCall
from workloads.linear_attention.kda import KDAFwdCall


def _deltanet_bwd_inputs(workload: DeltaNetChunkwiseCall) -> tuple:
    """The backward's inputs with the saved tensors a forward run produces for them."""
    from tileops.linear_attention import DeltaNetChunkFwdOp

    do, q, k, v, beta, *_saved = workload.gen_inputs()
    fwd_op = DeltaNetChunkFwdOp(workload.arguments()["chunk_size"])
    _o, h, Aw, Au, w, u = fwd_op(q, k, v, beta)
    return do, q, k, v, beta, h, Aw, Au, w, u


def _gla_bwd_inputs(workload: GLAChunkwiseCall) -> tuple:
    """The backward's inputs with the chunk states a forward run produces and a zero final-state
    gradient; the forward starts from a random state when the backward has one."""
    from tileops.linear_attention import GLAChunkFwdOp

    do, _dfinal_state, q, k, v, g, _h = workload.gen_inputs()
    arguments = workload.arguments()
    fwd_op = GLAChunkFwdOp(arguments["chunk_size"], arguments["scale"])
    initial_state = (
        torch.randn_like(_dfinal_state) * 0.1 if arguments["has_initial_state"] else None
    )
    _o, _final_state, h = fwd_op(q, k, v, g, initial_state)
    dfinal_state = torch.zeros_like(_dfinal_state)
    return do, dfinal_state, q, k, v, g, h


class _GLAChunkFwdAgainstFLA(GLAChunkwiseCall):
    """The chunkwise forward as its benchmark compares it with FLA's chunk_gla."""

    def verification(self, *inputs):
        from workloads.numerics import Partial

        tol = {torch.float32: 1e-2, torch.float16: 5e-2, torch.bfloat16: 1e-1}[inputs[0].dtype]
        return Partial(2, "h: FLA chunk_gla returns no chunk states", rtol=tol, atol=tol)


ENTRIES = {
    "DeltaNetFwdOp": Entry(DeltaNetFwdCall),
    "DeltaNetChunkFwdOp": Entry(DeltaNetChunkwiseCall),
    "DeltaNetChunkBwdOp": Entry(DeltaNetChunkwiseCall, inputs=_deltanet_bwd_inputs),
    "DeltaNetRecurrentFwdOp": Entry(DeltaNetDecodeCall),
    "GDNFwdOp": Entry(GDNFwdCall),
    "GLAChunkFwdOp": Entry(_GLAChunkFwdAgainstFLA),
    "GLAChunkBwdOp": Entry(GLAChunkwiseCall, inputs=_gla_bwd_inputs),
    "GLAFwdOp": Entry(GLAFwdCall),
    "GLARecurrentFwdOp": Entry(GLADecodeCall),
    "KDAFwdOp": Entry(KDAFwdCall),
}
