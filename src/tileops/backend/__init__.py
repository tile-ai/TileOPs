"""The backend protocol: everything a backend distribution needs, and nothing else.

A *target* is the name a backend gives its own set of kernels. A backend joins TileOPs with
four names and one entry point::

    from tileops.backend import TensorSpec, register_detector, register_kernel_builder
    from acme_tileops._kernels import AcmeRMSNorm

    register_detector(target="acme", detect=lambda device: device.type == "acme")

    def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
        return AcmeRMSNorm(normalized_shape, eps, x.dtype)

    register_kernel_builder(op="RMSNormFwdOp", target="acme", build_kernel=build_rms_norm)

and, in its pyproject::

    [project.entry-points."tileops.backends"]
    acme = "tileops_acme"

A target serves the whole op. ``build_kernel`` takes one positional argument per
``signature.inputs`` entry in that order, a ``kw_only`` one included, with an
``optional: true`` input the call did not pass arriving as ``None``, then the
construction params under ``signature.params`` names. The kernel it returns is called with
those tensors in the same order, plus a caller-supplied output buffer such as ``out`` by
keyword. Every tensor is on one device, every input the call does not write is contiguous,
and the call meets the input dtypes and shape rules the manifest states.

Choosing among its own kernels, constructing, caching and compiling all happen inside it, so
nothing here names a kernel class, a specialization axis, a priority or a fallback: this layer
picks a target, the target picks a kernel.

The op layer reads four facts of the call's device for selection: ``arch``,
``calibration``, ``sm_count`` and ``smem_budget``. TileOPs reads CUDA's; a backend for
another device type registers its own with ``register_device_facts``. Without it a device
reports none, and an implementation declaring ``supported_archs`` is never available there.
The build and tuning run with the call's device current, through ``torch.accelerator``, for
a device of the process's accelerator type.

The kernel ``build_kernel`` returns may define two members an in-tree ``Kernel`` has:

- ``request_tune()``: called while the op is in tuned mode, with the kernel's device
  current. Every ``Op.request_tune()`` calls it on the kernels built so far, once per device
  a kernel was built for; a kernel built later receives it before its first call. It is
  idempotent per device. A tuning it defers to a launch completes before any CUDA graph
  capture. What it raises is the error of the call that built the kernel, or of
  ``Op.request_tune()``. A kernel without it makes a tuning request warn.
- ``config``: a non-empty dict. ``Op.kernel_config()`` reports it where the op holds no
  non-empty config of its own.

Depends on torch only — importing this does not import tilelang.

Every name a backend uses is exported here; the submodules are internal.
"""

from tileops.backend._dispatch import (
    default_target,
    load_failures,
    registered_targets,
    set_default_target,
)
from tileops.backend._errors import (
    AmbiguousTargetError,
    BackendError,
    OpNotAvailableError,
    UnknownTargetError,
)
from tileops.backend._protocol import (
    BUILTIN,
    BuildKernel,
    DeviceFacts,
    KernelResult,
    ReadDeviceFacts,
    Target,
    TensorSpec,
)
from tileops.backend._registry import (
    register_detector,
    register_device_facts,
    register_kernel_builder,
)

__all__ = [
    "BUILTIN",
    "AmbiguousTargetError",
    "BackendError",
    "BuildKernel",
    "DeviceFacts",
    "KernelResult",
    "OpNotAvailableError",
    "ReadDeviceFacts",
    "Target",
    "TensorSpec",
    "UnknownTargetError",
    "default_target",
    "load_failures",
    "register_detector",
    "register_device_facts",
    "register_kernel_builder",
    "registered_targets",
    "set_default_target",
]
