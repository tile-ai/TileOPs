"""Case factories of the linear_attention family."""

import torch

from benchmarks._cases import Entry
from workloads.linear_attention.deltanet import (
    DeltaNetChunkwiseCall,
    DeltaNetDecodeCall,
    DeltaNetInferenceCall,
)
from workloads.linear_attention.gdn import GDNFwdCall
from workloads.linear_attention.gla import GLAChunkwiseCall, GLADecodeCall, GLAInferenceCall
from workloads.linear_attention.kda import KDAFwdCall


def _deltanet_bwd_inputs(workload: DeltaNetChunkwiseCall) -> tuple:
    """The backward's inputs with the saved tensors a forward run produces for them."""
    from tileops.ops import DeltaNetChunkFwdOp

    do, q, k, v, beta, *_saved = workload.gen_inputs()
    fwd_op = DeltaNetChunkFwdOp(workload.arguments()["chunk_size"])
    _o, S, Aw, Au, w, u = fwd_op(q, k, v, beta)
    return do, q, k, v, beta, S, Aw, Au, w, u


def _gla_bwd_inputs(workload: GLAChunkwiseCall) -> tuple:
    """The backward's inputs with the chunk states a forward run produces and a zero final-state gradient."""
    from tileops.ops import GLAChunkFwdOp

    q, k, v, g, _h, do, _dht = workload.gen_inputs()
    arguments = workload.arguments()
    fwd_op = GLAChunkFwdOp(arguments["chunk_size"], arguments["scale"])
    fwd_op(q, k, v, g)
    (fwd_kernel,) = fwd_op.built_kernels("gla_fwd").values()
    h = fwd_kernel._h_out
    dht = torch.zeros_like(_dht)
    return q, k, v, g, h, do, dht


ENTRIES = {
    "DeltaNetInferenceFwdOp": Entry(DeltaNetInferenceCall),
    "DeltaNetChunkFwdOp": Entry(DeltaNetChunkwiseCall),
    "DeltaNetChunkBwdOp": Entry(DeltaNetChunkwiseCall, inputs=_deltanet_bwd_inputs),
    "DeltaNetRecurrentFwdOp": Entry(DeltaNetDecodeCall),
    "GDNFwdOp": Entry(GDNFwdCall),
    "GLAChunkFwdOp": Entry(GLAChunkwiseCall),
    "GLAChunkBwdOp": Entry(GLAChunkwiseCall, inputs=_gla_bwd_inputs),
    "GLAInferenceFwdOp": Entry(GLAInferenceCall),
    "GLARecurrentFwdOp": Entry(GLADecodeCall),
    "KDAFwdOp": Entry(KDAFwdCall),
}
