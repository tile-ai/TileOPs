"""Call workloads of the linear_attention family."""

import torch

from workloads.linear_attention.deltanet import (
    DeltaNetChunkwiseCall,
    DeltaNetDecodeCall,
    DeltaNetFwdCall,
)
from workloads.linear_attention.gdn import GDNFwdCall
from workloads.linear_attention.gla import GLAChunkwiseCall, GLADecodeCall, GLAFwdCall
from workloads.linear_attention.kda import KDAFwdCall


class _GLAChunkFwdAgainstFLA(GLAChunkwiseCall):
    """The chunkwise forward as its benchmark compares it with FLA's chunk_gla."""

    def verification(self, *inputs):
        from workloads.numerics import Partial

        tol = {torch.float32: 1e-2, torch.float16: 5e-2, torch.bfloat16: 1e-1}[inputs[0].dtype]
        return Partial(2, "h: FLA chunk_gla returns no chunk states", rtol=tol, atol=tol)


WORKLOADS = {
    "DeltaNetFwdOp": DeltaNetFwdCall,
    "DeltaNetChunkFwdOp": DeltaNetChunkwiseCall,
    "DeltaNetChunkBwdOp": DeltaNetChunkwiseCall,
    "DeltaNetRecurrentFwdOp": DeltaNetDecodeCall,
    "GDNFwdOp": GDNFwdCall,
    "GLAChunkFwdOp": _GLAChunkFwdAgainstFLA,
    "GLAChunkBwdOp": GLAChunkwiseCall,
    "GLAFwdOp": GLAFwdCall,
    "GLARecurrentFwdOp": GLADecodeCall,
    "KDAFwdOp": KDAFwdCall,
}
