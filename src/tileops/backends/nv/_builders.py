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


#: op name -> its ``get_kernel``. An op is served by this backend once it appears here;
#: :mod:`._bindings` knowing about an op is not enough, because the kernel's constructor
#: has to be spoken to correctly and that is per-op work.
BUILDERS = {
    "FusedAddRMSNormFwdOp": fused_add_rms_norm,
    "RMSNormFwdOp": rms_norm,
}
