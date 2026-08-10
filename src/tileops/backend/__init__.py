"""The backend protocol: everything a backend distribution needs, and nothing else.

A backend is a separate distribution shipping TileLang kernels for one *target* — the
family of devices those kernels run on. It joins TileOPs by implementing two callbacks and
registering them from its module top level::

    from tileops.backend import register, register_detector
    from .kernels import RMSNormKernel

    register_detector(target="acme", detect=lambda device: device.type == "acme")
    register(op="RMSNormFwdOp", target="acme",
             get_kernel=lambda x, weight, **p: RMSNormKernel(p["normalized_shape"], x.dtype))

and declaring the entry point that imports that module::

    [project.entry-points."tileops.backends"]
    acme = "tileops_acme"

Choosing among implementations, constructing, caching and compiling all happen inside
``get_kernel``; TileOPs takes the result. So nothing here names a kernel class, a dispatch
key, a priority or a fallback: dispatch answers *who* does it, never *how*.

Depends on torch only — importing this does not import tilelang.

Where things live: :mod:`~tileops.backend.protocol` is what crosses the boundary,
:mod:`~tileops.backend.errors` is what goes wrong, :mod:`~tileops.backend.registry` holds
the tables and fills them, :mod:`~tileops.backend.dispatch` reads them.
"""

from .dispatch import (
    default_target,
    get_kernel_for,
    load_errors,
    registered,
    registered_targets,
    resolve_target,
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
    "get_kernel_for",
    "load_errors",
    "register",
    "register_detector",
    "registered",
    "registered_targets",
    "resolve_target",
    "select_target",
    "set_default_target",
]
