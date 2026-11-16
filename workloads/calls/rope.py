"""Call workloads of the rope family."""

from workloads.rope import RoPECall, RoPECosSinCall

WORKLOADS = {
    "RoPEFwdOp": RoPECall,
    "RoPELlama31FwdOp": RoPECall,
    "YaRNFwdOp": RoPECall,
    "LongRoPEFwdOp": RoPECall,
    "RoPECosSinFwdOp": RoPECosSinCall,
}
