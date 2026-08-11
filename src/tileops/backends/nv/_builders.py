"""One ``get_kernel`` per op this backend serves.

Each takes the op's inputs as :class:`~tileops.backend.InputSpec` descriptions and its
manifest params, and returns something callable with the tensors those specs describe.
Everything private to this backend — which role to use, what the kernel constructor wants —
lives in here.
"""

from __future__ import annotations

import math

from tileops.backend import InputSpec, Kernel

from ._load import kernel_class


def rms_norm(x: InputSpec, weight: InputSpec, *, normalized_shape, eps) -> Kernel:
    """Build the RMS norm kernel for a call shaped like *x* and *weight*.

    The kernel works on a 2-D ``(M, N)`` view, which is what the op hands it, so *x* here
    already describes the flattened tensor.
    """
    m = x.shape[0]
    n = math.prod(normalized_shape)
    return kernel_class("RMSNormFwdOp", "rms_norm")(m, n, eps, x.dtype)


def fused_add_rms_norm(x: InputSpec, residual: InputSpec, weight: InputSpec, *, eps) -> Kernel:
    """Build the fused residual-add + RMS norm kernel. *x* describes the 2-D view."""
    m, n = x.shape
    return kernel_class("FusedAddRMSNormFwdOp", "fused_add_rms_norm")(m, n, eps, x.dtype)


def layer_norm(x: InputSpec, weight: InputSpec, bias: InputSpec, *, normalized_shape, eps) -> Kernel:
    """Build the layer norm kernel. *x* describes the 2-D view the op hands over."""
    m = x.shape[0]
    n = math.prod(normalized_shape)
    return kernel_class("LayerNormFwdOp", "layer_norm")(m, n, eps, x.dtype)


def ada_layer_norm(x: InputSpec, scale: InputSpec, shift: InputSpec, *, eps) -> Kernel:
    """Build the adaptive layer norm kernel, gateless."""
    m, n = x.shape
    return kernel_class("AdaLayerNormFwdOp", "ada_layer_norm")(
        m, n, eps, x.dtype, has_gate=False
    )


def ada_layer_norm_zero(
    x: InputSpec, scale: InputSpec, shift: InputSpec, gate: InputSpec, *, eps
) -> Kernel:
    """Build the adaptive layer norm kernel with its gate. One class serves both ops."""
    m, n = x.shape
    return kernel_class("AdaLayerNormZeroFwdOp", "ada_layer_norm")(
        m, n, eps, x.dtype, has_gate=True
    )


def fused_add_layer_norm(
    x: InputSpec, residual: InputSpec, weight: InputSpec, bias: InputSpec, *, eps
) -> Kernel:
    """Build the fused residual-add + layer norm kernel."""
    m, n = x.shape
    return kernel_class("FusedAddLayerNormFwdOp", "fused_add_layer_norm")(m, n, eps, x.dtype)


def group_norm(x: InputSpec, weight: InputSpec, bias: InputSpec, *, num_groups, eps) -> Kernel:
    """Build the group norm kernel. Channels come from *weight*, which is shaped ``(C,)``."""
    m, d = x.shape
    channels = weight.shape[0]
    return kernel_class("GroupNormFwdOp", "group_norm")(
        m, d, eps, x.dtype, num_groups, channels // num_groups
    )


def group_norm_no_affine(x: InputSpec, *, num_groups, eps) -> Kernel:
    """Build the affine-free group norm kernel. Groups are already in the row layout."""
    m, d = x.shape
    return kernel_class("GroupNormNoAffineFwdOp", "group_norm_no_affine")(m, d, eps, x.dtype)


def instance_norm(
    x: InputSpec, weight: InputSpec, bias: InputSpec, *, use_input_stats, momentum, eps
) -> Kernel:
    """Build instance norm out of the group norm kernel: one group per channel."""
    m, d = x.shape
    channels = weight.shape[0]
    return kernel_class("InstanceNormFwdOp", "group_norm")(
        m, d, eps, x.dtype, channels, 1
    )


def instance_norm_no_affine(x: InputSpec, *, use_input_stats, momentum, eps) -> Kernel:
    """Build the affine-free instance norm kernel."""
    m, d = x.shape
    return kernel_class("InstanceNormNoAffineFwdOp", "instance_norm_no_affine")(
        m, d, eps, x.dtype
    )


def batch_norm(
    x: InputSpec,
    weight: InputSpec,
    bias: InputSpec,
    running_mean: InputSpec,
    running_var: InputSpec,
    *,
    training,
    momentum,
    eps,
) -> Kernel:
    """Build the batch norm kernel for the mode this op runs in.

    Train and infer are separate kernels here; which one serves the call is decided inside
    this backend, which is the whole point of the callback. *x* describes the ``(C, L)``
    view the op hands over.
    """
    channels, length = x.shape
    if training:
        return kernel_class("BatchNormFwdOp", "fwd_train_kernel")(
            channels, length, x.dtype, eps, momentum
        )
    return kernel_class("BatchNormFwdOp", "fwd_infer_kernel")(
        channels, length, x.dtype, eps
    )


def batch_norm_bwd(
    grad_out: InputSpec, x: InputSpec, weight: InputSpec, mean: InputSpec, rstd: InputSpec
) -> Kernel:
    """Build the batch norm backward kernel."""
    channels, length = x.shape
    return kernel_class("BatchNormBwdOp", "bwd_kernel")(channels, length, x.dtype)


#: op name -> its ``get_kernel``. An op is served by this backend once it appears here;
#: :mod:`._bindings` knowing about an op is not enough, because the kernel's constructor
#: has to be spoken to correctly and that is per-op work.
BUILDERS = {
    "AdaLayerNormFwdOp": ada_layer_norm,
    "AdaLayerNormZeroFwdOp": ada_layer_norm_zero,
    "BatchNormBwdOp": batch_norm_bwd,
    "BatchNormFwdOp": batch_norm,
    "FusedAddLayerNormFwdOp": fused_add_layer_norm,
    "FusedAddRMSNormFwdOp": fused_add_rms_norm,
    "GroupNormFwdOp": group_norm,
    "GroupNormNoAffineFwdOp": group_norm_no_affine,
    "InstanceNormFwdOp": instance_norm,
    "InstanceNormNoAffineFwdOp": instance_norm_no_affine,
    "LayerNormFwdOp": layer_norm,
    "RMSNormFwdOp": rms_norm,
}
