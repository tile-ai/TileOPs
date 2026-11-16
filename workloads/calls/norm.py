"""Call workloads of the normalization family."""

from workloads.norm import (
    BatchNormBwdCall,
    FusedAddRMSNormPerTokenQuantCall,
    FusedQKNormRoPECall,
    NormCall,
    RMSNormGatedCall,
    RunningStatsCall,
)

WORKLOADS = {
    **{
        name: NormCall
        for name in (
            "RMSNormFwdOp",
            "FusedAddRMSNormFwdOp",
            "LayerNormFwdOp",
            "FusedAddLayerNormFwdOp",
            "AdaLayerNormFwdOp",
            "AdaLayerNormZeroFwdOp",
            "GroupNormFwdOp",
        )
    },
    "BatchNormFwdOp": RunningStatsCall,
    "BatchNormBwdOp": BatchNormBwdCall,
    "InstanceNormFwdOp": RunningStatsCall,
    "FusedQKNormRoPEFwdOp": FusedQKNormRoPECall,
    "FusedAddRMSNormPerTokenQuantFwdOp": FusedAddRMSNormPerTokenQuantCall,
    "RMSNormGatedFwdOp": RMSNormGatedCall,
}
