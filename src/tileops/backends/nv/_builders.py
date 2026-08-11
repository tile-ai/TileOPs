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


class _Adapter:
    """A kernel called in manifest order, wrapping one that wants something else.

    The protocol hands every kernel its op's inputs in the manifest's order and dtypes. A
    kernel of this backend's own is free to want another order, an extra cast, or fewer
    tensors; reconciling the two is this backend's work and stops here.
    """

    def __init__(self, kernel, adapt):
        self._kernel = kernel
        self._adapt = adapt

    def __call__(self, *tensors):
        return self._kernel(*self._adapt(*tensors))

    def autotune(self, *example):
        tune = getattr(self._kernel, "autotune", None)
        if tune is None:
            return
        tune(*self._adapt(*example)) if example else tune()


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


def instance_norm_no_affine(
    x: InputSpec, running_mean: InputSpec, running_var: InputSpec, *,
    use_input_stats, momentum, eps,
) -> Kernel:
    """Build the affine-free instance norm kernel.

    It computes from *x* alone; the running stats arrive because the op declares them and
    are dropped here, which is this backend's business rather than the op's.
    """
    m, d = x.shape
    kernel = kernel_class("InstanceNormNoAffineFwdOp", "instance_norm_no_affine")(
        m, d, eps, x.dtype
    )
    return _Adapter(kernel, lambda x_2d, _mean, _var: (x_2d,))


def batch_norm(
    x: InputSpec,
    running_mean: InputSpec,
    running_var: InputSpec,
    weight: InputSpec,
    bias: InputSpec,
    *,
    training,
    momentum,
    eps,
) -> Kernel:
    """Build the batch norm kernel for the mode this op runs in.

    Train and infer are separate kernels here, and which one serves the call is decided
    inside this backend — the whole point of the callback. *x* describes the ``(C, L)`` view
    the op hands over.

    These kernels want the affine terms first and in fp32, so the adapter puts them there.
    Reshaping the call to what a kernel accepts is a backend's job; the boundary stays in
    the manifest's order.
    """
    channels, length = x.shape
    role = "fwd_train_kernel" if training else "fwd_infer_kernel"
    args = (channels, length, x.dtype, eps, momentum) if training else (
        channels, length, x.dtype, eps)
    kernel = kernel_class("BatchNormFwdOp", role)(*args)
    return _Adapter(
        kernel,
        lambda x_cl, mean, var, w, b: (x_cl, w.float(), b.float(), mean, var),
    )


def batch_norm_bwd(
    grad_out: InputSpec, x: InputSpec, weight: InputSpec, mean: InputSpec, rstd: InputSpec
) -> Kernel:
    """Build the batch norm backward kernel, which wants its affine term in fp32."""
    channels, length = x.shape
    kernel = kernel_class("BatchNormBwdOp", "bwd_kernel")(channels, length, x.dtype)
    return _Adapter(
        kernel, lambda go, x_cl, w, mean, rstd: (go, x_cl, w.float(), mean, rstd)
    )


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
