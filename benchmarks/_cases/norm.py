"""Case factories of the normalization family."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.norm import (
    BatchNormBwdCall,
    FusedAddRMSNormPerTokenQuantCall,
    FusedQKNormRoPECall,
    NormCall,
    RunningStatsCall,
    batch_norm_forward_result,
    fused_qk_norm_rope_result,
)


def _batch_norm_fwd(op, case) -> Implementation:
    """The op on private running statistics, exposing them beside its output."""
    return private_inputs(lambda *a: batch_norm_forward_result(op, *a), case.inputs, 1, 2)


def _instance_norm_fwd(op, case) -> Implementation:
    """The op on private running statistics, which it updates when it uses input statistics."""
    return private_inputs(op, case.inputs, 1, 2)


def _fused_qk_norm_rope(op, case) -> Implementation:
    """The op on a private qkv it normalizes and rotates in place, exposing that qkv."""
    return private_inputs(lambda *a: fused_qk_norm_rope_result(op, *a), case.inputs, 0)


ENTRIES = {
    **{
        name: Entry(NormCall)
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
    "BatchNormFwdOp": Entry(RunningStatsCall, binder=_batch_norm_fwd),
    "BatchNormBwdOp": Entry(BatchNormBwdCall),
    "InstanceNormFwdOp": Entry(RunningStatsCall, binder=_instance_norm_fwd),
    "FusedQKNormRoPEFwdOp": Entry(FusedQKNormRoPECall, binder=_fused_qk_norm_rope),
    "FusedAddRMSNormPerTokenQuantFwdOp": Entry(FusedAddRMSNormPerTokenQuantCall),
}
