"""Reading the tables: which target serves a call, and which builder to hand it to."""

from __future__ import annotations

import torch

from . import registry
from .errors import AmbiguousTargetError, BackendError, UnknownTargetError
from .protocol import BUILTIN, BuildKernel, DetectFn, Target


def detect_target(device: torch.device) -> str | None:
    """Return the target whose kernels are written for *device*, or None if there is none.

    None is the normal case: no backend is installed for this hardware, so the in-tree
    implementation runs. *device* is passed through untouched — reading ``.type`` is the
    vendor knowledge being delegated.

    Neither memoized nor locked: a target is settled once per op instance, so each detector
    is asked once per op. A table would cost a lock, an invalidation rule and a staleness
    question — and the lock cannot be traced by dynamo, which sees an op's first call.

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

    *requested* wins, then the process default, then detection. A copy of this order per op
    class is how the three drift apart.

    Args:
        requested: The op's ``target=``, honoured as named and not checked against *device*.
            :data:`~.protocol.BUILTIN` forces the in-tree implementation.
        device: Where to detect from, or None when the call has no tensor input.

    Returns:
        A target name, :data:`~.protocol.BUILTIN`, or ``None``. The last two both run the
        in-tree implementation and differ only in how that was decided, which is what tells
        the op layer whether to remember the answer.

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
    """Return *target*'s ``build_kernel`` for *op*, or None when it registered none."""
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

    ``None`` restores detection; :data:`~.protocol.BUILTIN` turns replacement off. No
    environment variable: what decides which kernel runs should be visible in the program.

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

    Every error above points here, so a broken wheel cannot present itself as something else.
    """
    registry.ensure_loaded()
    return tuple(registry.LOAD_FAILURES)
