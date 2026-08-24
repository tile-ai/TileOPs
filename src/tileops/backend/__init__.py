"""The backend protocol: everything a backend distribution needs, and nothing else.

A *target* is the name a backend gives its own set of kernels. A backend joins TileOPs with
four names and one entry point::

    from tileops.backend import TensorSpec, register_detector, register_kernel_builder
    from .kernels import AcmeRMSNorm

    register_detector(target="acme", detect=lambda device: device.type == "acme")

    def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
        return AcmeRMSNorm(normalized_shape, eps, x.dtype)

    register_kernel_builder(op="RMSNormFwdOp", target="acme", build_kernel=build_rms_norm)

and, in its pyproject::

    [project.entry-points."tileops.backends"]
    acme = "tileops_acme"

``build_kernel``'s signature is the op's manifest signature: one argument per
``signature.inputs`` entry in that order, with an ``optional: true`` input the call did not
pass arriving as ``None``, then params under ``signature.params`` names. Choosing among its own
kernels, constructing, caching and compiling all happen inside it, so nothing here names a
kernel class, a specialization axis, a priority or a fallback: this layer picks a target,
the target picks a kernel.

Depends on torch only — importing this does not import tilelang.

`tileops.backend.protocol` is what crosses the boundary,
`tileops.backend.errors` what goes wrong, `tileops.backend.registry` the tables,
`tileops.backend.dispatch` how they are read. ``select_target``,
``detect_target`` and ``registered_kernel_builder`` live in ``dispatch`` but are not
exported: only the op layer reads the tables, and a second public path to them is a second
thing to keep consistent.
"""

from .dispatch import (
    default_target,
    load_failures,
    registered_targets,
    set_default_target,
)
from .errors import (
    AmbiguousTargetError,
    BackendError,
    OpNotAvailableError,
    UnknownTargetError,
)
from .protocol import BUILTIN, BuildKernel, KernelResult, Target, TensorSpec
from .registry import register_detector, register_kernel_builder

__all__ = [
    "BUILTIN",
    "AmbiguousTargetError",
    "BackendError",
    "BuildKernel",
    "KernelResult",
    "OpNotAvailableError",
    "Target",
    "TensorSpec",
    "UnknownTargetError",
    "default_target",
    "load_failures",
    "register_detector",
    "register_kernel_builder",
    "registered_targets",
    "set_default_target",
]
