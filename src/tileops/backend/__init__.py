"""The backend protocol: everything a backend distribution needs, and nothing else.

A backend ships TileLang kernels for one *target* — the family of devices those kernels run
on. It joins TileOPs by implementing two callbacks at its module top level::

    from tileops.backend import register, register_detector
    from .kernels import RMSNormKernel

    register_detector(target="acme", detect=lambda device: device.type == "acme")
    register(op="RMSNormFwdOp", target="acme",
             get_kernel=lambda x, weight, **p: RMSNormKernel(p["normalized_shape"], x.dtype))

and declaring the entry point that imports that module::

    [project.entry-points."tileops.backends"]
    acme = "tileops_acme"

Choosing among implementations, constructing, caching and compiling all happen inside
``get_kernel``. So nothing here names a kernel class, a dispatch key, a priority or a
fallback: dispatch answers *who* does it, never *how*.

Depends on torch only — importing this does not import tilelang.

:mod:`~tileops.backend.protocol` is what crosses the boundary,
:mod:`~tileops.backend.errors` what goes wrong, :mod:`~tileops.backend.registry` the tables,
:mod:`~tileops.backend.dispatch` how they are read.
"""

from .dispatch import (
    default_target,
    detect_target,
    load_failures,
    registered_targets,
    registrations,
    resolve_get_kernel,
    select_target,
    set_default_target,
)
from .errors import (
    AmbiguousTargetError,
    BackendError,
    BackendLoadFailure,
    OpNotAvailableError,
    UnknownTargetError,
)
from .protocol import DetectFn, GetKernelFn, InputSpec, Kernel, KernelResult
from .registry import ENTRY_POINT_GROUP, register, register_detector

__all__ = [
    "ENTRY_POINT_GROUP",
    "AmbiguousTargetError",
    "BackendError",
    "BackendLoadFailure",
    "DetectFn",
    "GetKernelFn",
    "InputSpec",
    "Kernel",
    "KernelResult",
    "OpNotAvailableError",
    "UnknownTargetError",
    "default_target",
    "detect_target",
    "load_failures",
    "register",
    "register_detector",
    "registered_targets",
    "registrations",
    "resolve_get_kernel",
    "select_target",
    "set_default_target",
]
