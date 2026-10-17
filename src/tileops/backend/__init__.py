"""The backend protocol: everything a backend distribution needs, and nothing else.

A *target* is the name a backend gives its own set of kernels. A backend joins TileOPs with
four names and one entry point::

    from tileops.backend import TensorSpec, register_detector, register_kernel_builder
    from acme_tileops.kernels import AcmeRMSNorm

    register_detector(target="acme", detect=lambda device: device.type == "acme")

    def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
        return AcmeRMSNorm(normalized_shape, eps, x.dtype)

    register_kernel_builder(op="RMSNormFwdOp", target="acme", build_kernel=build_rms_norm)

and, in its pyproject::

    [project.entry-points."tileops.backends"]
    acme = "tileops_acme"

A target serves the whole op. ``build_kernel``'s signature is the op's ``forward``
signature: one argument per ``signature.inputs`` entry, with an ``optional: true`` input the call did not pass arriving as ``None``, then the
construction params under ``signature.params`` names. The kernel it returns is called with
those tensors in the same order, plus a caller-supplied output buffer such as ``out`` by
keyword. Every tensor is on one device, every input the call does not write is contiguous,
and the call meets the input dtypes and shape rules the manifest states.

Choosing among its own kernels, constructing, caching and compiling all happen inside it, so
nothing here names a kernel class, a specialization axis, a priority or a fallback: this layer
picks a target, the target picks a kernel.

A backend that serves part of one kernel interface instead registers an implementation with
``register_kernel_type``. It inherits the interface and joins the in-tree implementations
of every instance of the op; a call it does not serve stays with them.

While building and tuning, the op layer makes a call's device current through
``torch.accelerator``: a device of the process's accelerator type, CUDA or a backend's, and
no other. Selection reads four facts of the call's device: ``arch``, ``calibration``, ``sm_count`` and ``smem_budget``. TileOPs reads CUDA's; a backend for another device type states how its
devices report them with ``register_device_facts``. Without it a device reports none, and
an implementation that declares ``supported_archs`` is never available there.

The kernel ``build_kernel`` returns may also define ``request_tune()`` and ``config``, the
two members an in-tree ``Kernel`` has. While the op is in tuned mode, every
``Op.request_tune()`` calls it on the kernels built so far, once for each device a kernel
was built for, and every kernel built later receives it before its first call; the device
is current each time. A later request calls it again, so it is idempotent per device: one
object built for two devices tunes on each. Tuning it defers to a launch completes in a
call made before any CUDA graph capture. What it raises is the error of the call that built
the kernel, or of ``Op.request_tune()``. ``config``, a dict, is what ``Op.kernel_config()``
reports. A kernel without ``request_tune()`` makes a tuning request warn.

Depends on torch only — importing this does not import tilelang.

`tileops.backend.protocol` is what crosses the boundary,
`tileops.backend.errors` what goes wrong, `tileops.backend.registry` the tables,
`tileops.backend.dispatch` how they are read. ``select_target``,
``detect_target``, ``registered_kernel_builder`` and ``device_facts_of`` live in
``dispatch`` but are not exported: only the op layer reads the tables, and a second public
path to them is a second thing to keep consistent.
"""

from tileops.backend.dispatch import (
    default_target,
    load_failures,
    registered_targets,
    set_default_target,
)
from tileops.backend.errors import (
    AmbiguousTargetError,
    BackendError,
    OpNotAvailableError,
    UnknownTargetError,
)
from tileops.backend.protocol import (
    BUILTIN,
    BuildKernel,
    DeviceFacts,
    KernelResult,
    ReadDeviceFacts,
    Target,
    TensorSpec,
)
from tileops.backend.registry import (
    register_detector,
    register_device_facts,
    register_kernel_builder,
    register_kernel_type,
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
    "register_kernel_type",
    "registered_targets",
    "set_default_target",
]
