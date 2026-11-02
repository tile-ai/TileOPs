"""Call workloads of the mamba family."""

from workloads.mamba import (
    CausalConv1dFwdCall,
    CausalConv1dUpdateFwdCall,
    Mamba2FwdCall,
    SelectiveScanFwdCall,
    SSDChunkCouplingFwdCall,
    SSDChunkCumsumFwdCall,
    SSDChunkScanFwdCall,
    SSDChunkStateFwdCall,
    SSDDecodeFwdCall,
    SSDStatePassingFwdCall,
)

WORKLOADS = {
    "CausalConv1dFwdOp": CausalConv1dFwdCall,
    "CausalConv1dUpdateFwdOp": CausalConv1dUpdateFwdCall,
    "SSDChunkCouplingFwdOp": SSDChunkCouplingFwdCall,
    "SSDChunkCumsumFwdOp": SSDChunkCumsumFwdCall,
    "SSDChunkScanFwdOp": SSDChunkScanFwdCall,
    "SSDChunkStateFwdOp": SSDChunkStateFwdCall,
    "SSDStatePassingFwdOp": SSDStatePassingFwdCall,
    "SSDRecurrentFwdOp": SSDDecodeFwdCall,
    "Mamba2FwdOp": Mamba2FwdCall,
    "SelectiveScanFwdOp": SelectiveScanFwdCall,
}
