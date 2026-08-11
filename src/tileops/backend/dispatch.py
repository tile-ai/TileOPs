"""Reading the tables: which target serves a call, and which builder to hand it to."""

from __future__ import annotations

import torch

from . import registry
from .errors import AmbiguousTargetError, BackendError, UnknownTargetError
from .protocol import BUILTIN, BuildKernel, DetectFn, Target


def detect_target(device: torch.device) -> str | None:
    """Return the target whose kernels are written for *device*, or None if there is none.

    Nothing claiming *device* is the normal case — it means no third-party backend is
    installed for this hardware, so the in-tree implementation runs. Only a genuine
    conflict raises.

    *device* is passed through untouched — neither ``.type`` read nor mapped to hardware,
    since that knowledge is exactly what is being delegated.

    Not memoized, and not locked. An op settles which target serves it once per instance, so
    each detector is asked once per op rather than once per call; a table to remember that
    would cost a lock, an invalidation rule, and a staleness question, to save a handful of
    one-line predicates. Staying lock-free also matters because this runs inside a
    dynamo-traced ``forward`` the first time an op is called, and dynamo cannot trace
    entering a lock.

    Raises:
        AmbiguousTargetError: More than one target claimed it. Pass ``target=`` to choose.
        BackendError: A detector raised instead of answering.
    """
    registry.ensure_loaded()
    claimed = [t for t, d in list(registry.DETECTORS.items()) if _claims(t, d, device)]
    if not claimed:
        return None  # no backend serves this device; the in-tree implementation does
    if len(claimed) > 1:
        raise AmbiguousTargetError(
            f"device {device} is claimed by {sorted(claimed)}; pass target= to "
            f"choose{registry.load_failure_suffix()}"
        )
    return claimed[0]


def _claims(target: str, detect: DetectFn, device: torch.device) -> bool:
    """Ask one detector, blaming the right distribution when it misbehaves.

    Letting the exception through unlabelled would point the user at TileOPs; stepping over
    it would silently hand the device to somebody else.
    """
    try:
        return detect(device)
    except Exception as exc:
        raise BackendError(
            f"detector for target {target!r} ({registry.describe(detect)}) raised on "
            f"device {device}: {exc!r}. A detector must return False for devices it does "
            f"not serve.{registry.load_failure_suffix()}"
        ) from exc


def select_target(requested: Target, device: torch.device | None) -> Target:
    """Decide which target serves this call, in the one place that decides it.

    *requested* wins, then the process default, then detection. Every op comes through
    here; a copy of this order per op class is how the three drift apart.

    Args:
        requested: The op's ``target=``. A name is honoured as given and not checked
            against *device* — naming a target is how a caller overrides detection.
            :data:`~.protocol.BUILTIN` forces the in-tree implementation.
        device: Where to detect from, or None when the call has no tensor input.

    Returns:
        A target name, :data:`~.protocol.BUILTIN`, or ``None``. The last two both mean
        "run the in-tree implementation"; they differ only in how that was decided, which
        the op layer needs in order to know whether to remember the answer.

    Raises:
        UnknownTargetError: A named target registered nothing.
    """
    registry.ensure_loaded()  # even the no-device answer must be able to blame a bad wheel
    if requested is not None:
        if requested is not BUILTIN and requested not in registry.known_targets():
            raise UnknownTargetError(
                f"no backend registered target {requested!r}; known targets: "
                f"{sorted(registry.known_targets())}{registry.load_failure_suffix()}")
        return requested
    if registry.default_target is not None:
        return registry.default_target
    if device is None:
        return None
    return detect_target(device)


def registered_kernel_builder(op: str, target: str) -> BuildKernel | None:
    """Return *target*'s ``build_kernel`` for *op*, or None when it registered none.

    The caller decides what a missing one means; see :class:`~.errors.OpNotAvailableError`
    for why it is never a fallback to the in-tree implementation.
    """
    registry.ensure_loaded()
    return registry.BUILDERS.get((op, target))


def registered_targets(op: str | None = None) -> list[str]:
    """Targets registered for *op*; every known target when *op* is None."""
    registry.ensure_loaded()
    if op is None:
        return sorted(registry.known_targets())
    return sorted(target for name, target in registry.BUILDERS if name == op)


def set_default_target(target: Target) -> None:
    """Route ops with no explicit ``target=`` to *target*.

    ``None`` restores detection; :data:`~.protocol.BUILTIN` turns replacement off
    process-wide. One setting, and no environment variable: configuration that changes
    which kernel runs should be visible in the program.

    Raises:
        UnknownTargetError: *target* is a name no backend registered.
    """
    registry.ensure_loaded()
    if (target is not None and target is not BUILTIN
            and target not in registry.known_targets()):
        raise UnknownTargetError(
            f"no backend registered target {target!r}; known targets: "
            f"{sorted(registry.known_targets())}{registry.load_failure_suffix()}")
    registry.default_target = target


def default_target() -> Target:
    """The process-wide default, or None when the device decides."""
    return registry.default_target


def load_failures() -> tuple[str, ...]:
    """Backends that failed to import and were skipped, one line each.

    Every error above points here, so a broken wheel cannot present itself as "no target
    serves this device".
    """
    registry.ensure_loaded()
    return tuple(registry.LOAD_FAILURES)
