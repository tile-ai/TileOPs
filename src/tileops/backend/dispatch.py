"""Reading the tables: which target serves a call, and which callback to hand it to."""

from __future__ import annotations

import torch

from . import registry
from .errors import (
    AmbiguousTargetError,
    BackendError,
    BackendLoadFailure,
    OpNotAvailableError,
    UnknownTargetError,
)
from .protocol import DetectFn, GetKernelFn


def detect_target(device: torch.device) -> str:
    """Return the target serving *device*.

    *device* is passed through untouched — neither ``.type`` read nor mapped to hardware,
    since that knowledge is exactly what is being delegated. ``cuda`` and ``cuda:0`` are
    therefore separate memo entries; normalizing them would mean materializing the device
    to learn its index. Memoizing an unindexed device is sound because a target claims a
    device *type*, not an index.

    Raises:
        UnknownTargetError: Nothing claimed *device*.
        AmbiguousTargetError: More than one did. Pass ``target=`` to bypass detection.
        BackendError: A detector raised instead of answering.
    """
    registry.ensure_loaded()
    with registry.LOCK:
        # Under the lock, memo included: a backend arriving concurrently clears it. Cold
        # path -- the op layer memoizes the kernel, so this is not per call.
        cached = registry.RESOLVED.get(device)
        if cached is not None:
            return cached
        claimed = [t for t, d in registry.DETECTORS.items() if _claims(t, d, device)]
        if not claimed:
            raise UnknownTargetError(
                f"no registered target claims device {device}; registered targets: "
                f"{sorted(registry.DETECTORS)}{registry.load_failure_suffix()}"
            )
        if len(claimed) > 1:
            raise AmbiguousTargetError(
                f"device {device} is claimed by {sorted(claimed)}; pass target= to "
                f"choose{registry.load_failure_suffix()}"
            )
        # Neither failure above is memoized: a later registration can change both.
        registry.RESOLVED[device] = claimed[0]
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
            f"not serve."
        ) from exc


def select_target(explicit: str | None, device: torch.device | None) -> str:
    """Decide which target serves this call, in the one place that decides it.

    *explicit* wins, then the process default, then detection. Every op comes through here;
    a copy of this order per op class is how the three drift apart.

    Args:
        explicit: The op's ``target=``, honoured as named and not checked against *device*
            — naming a target is how a caller overrides detection.
        device: Where to detect from, or None when the call has no tensor input.
    """
    registry.ensure_loaded()  # even the no-device error must be able to blame a bad wheel
    if explicit is not None:
        return explicit
    if registry.default_target is not None:
        return registry.default_target
    if device is None:
        raise UnknownTargetError(
            "this call has no tensor input, so there is no device to detect from; pass "
            f"target= or set tileops.set_default_target(){registry.load_failure_suffix()}"
        )
    return detect_target(device)


def resolve_get_kernel(op: str, target: str) -> GetKernelFn:
    """Return *target*'s ``get_kernel`` for *op*.

    Raises:
        OpNotAvailableError: That cell is empty. Never falls back to another target:
            running somewhere else silently makes "where did this run" unanswerable.
    """
    registry.ensure_loaded()
    try:
        return registry.KERNELS[(op, target)]
    except KeyError:
        raise OpNotAvailableError(
            f"no get_kernel registered for {(op, target)}; registered targets for this "
            f"op: {registered_targets(op)}{registry.load_failure_suffix()}"
        ) from None


def registrations() -> frozenset[tuple[str, str]]:
    """Every registered ``(op, target)``.

    Keys only: handing out the callbacks would open a second way to reach a ``get_kernel``
    beside :func:`resolve_get_kernel`.
    """
    registry.ensure_loaded()
    return frozenset(registry.KERNELS)


def registered_targets(op: str | None = None) -> list[str]:
    """Targets registered for *op*; every known target when *op* is None."""
    registry.ensure_loaded()
    if op is None:
        return sorted(registry.known_targets())
    return sorted(target for name, target in registry.KERNELS if name == op)


def set_default_target(target: str | None) -> None:
    """Route ops with no explicit ``target=`` to *target*; None restores detection.

    One process-wide setting, and no environment variable: configuration that changes which
    kernel runs should be visible in the program.
    """
    registry.ensure_loaded()
    if target is not None and target not in registry.known_targets():
        raise UnknownTargetError(
            f"no backend registered target {target!r}; known targets: "
            f"{sorted(registry.known_targets())}{registry.load_failure_suffix()}"
        )
    registry.default_target = target


def default_target() -> str | None:
    """The process-wide default, or None when the device decides."""
    return registry.default_target


def load_failures() -> tuple[BackendLoadFailure, ...]:
    """Backends that failed to import and were skipped.

    Every error above points here, so a broken wheel cannot present itself as "no target
    claimed this device".
    """
    registry.ensure_loaded()
    return tuple(registry.LOAD_FAILURES)
