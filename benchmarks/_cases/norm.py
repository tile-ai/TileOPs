"""How the normalization family's benchmarks call their ops."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.norm import batch_norm_forward_result, fused_qk_norm_rope_result


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
    "BatchNormFwdOp": Entry(binder=_batch_norm_fwd),
    "InstanceNormFwdOp": Entry(binder=_instance_norm_fwd),
    "FusedQKNormRoPEFwdOp": Entry(binder=_fused_qk_norm_rope),
}
