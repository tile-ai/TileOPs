"""The two tables, and how they get filled.

Module-level state, because registration happens as mutually unaware distributions get
imported: one process-wide place is the only place they can meet.
"""

from __future__ import annotations

import threading
import traceback
import warnings
from importlib.metadata import entry_points
from typing import Callable, NamedTuple

import torch

from .errors import BackendError, BackendLoadFailure
from .protocol import DetectFn, GetKernelFn

#: The value names a *module*; importing it must perform the registration.
ENTRY_POINT_GROUP = "tileops.backends"

DETECTORS: dict[str, DetectFn] = {}
KERNELS: dict[tuple[str, str], GetKernelFn] = {}
RESOLVED: dict[torch.device, str] = {}
LOAD_ERRORS: list[BackendLoadFailure] = []

#: Which target ops use when they name none. There is deliberately no constant here: a
#: hardcoded ``"nv"`` would make detection unreachable and name one backend inside a
#: neutral layer. Kept beside the tables so the load transaction and test isolation cover
#: it.
default_target: str | None = None

#: Set only once every entry point has been tried, so no thread sees a half-built registry.
_loaded = False

#: Reentrant: discovery holds it while importing backend modules, whose top level calls
#: register() and takes it again.
LOCK = threading.RLock()

#: Set on the thread running discovery, which must read the partial registry it is
#: building while every other thread waits on the lock for the finished one.
_LOADING = threading.local()


def register(op: str, target: str, get_kernel: GetKernelFn) -> None:
    """Record *get_kernel* as how *target* computes *op*.

    Args:
        op: The op's manifest key, e.g. ``"RMSNormFwdOp"``.
        target: The target this backend serves.
        get_kernel: Callback two. Must be lazy: importing the calling module must not
            compile anything.

    Raises:
        BackendError: ``(op, target)`` is already claimed.
    """
    with LOCK:
        existing = KERNELS.get((op, target))
        if existing is not None:
            raise BackendError(
                f"{(op, target)} is already registered to {describe(existing)}; "
                f"{describe(get_kernel)} cannot take it. Overwriting silently would make "
                f"'which one ran' unanswerable."
            )
        KERNELS[(op, target)] = get_kernel


def register_detector(target: str, detect: DetectFn) -> None:
    """Record *detect* as how to recognize a device belonging to *target*.

    Raises:
        BackendError: *target* already has a detector.
    """
    with LOCK:
        existing = DETECTORS.get(target)
        if existing is not None:
            raise BackendError(
                f"target {target!r} already has detector {describe(existing)}; "
                f"{describe(detect)} cannot replace it."
            )
        DETECTORS[target] = detect
        RESOLVED.clear()  # a new detector can change an answer already given


def describe(fn: Callable) -> str:
    """Name *fn* well enough to act on, module included."""
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn)}"


def known_targets() -> set[str]:
    """Every target that registered anything.

    Kernels without a detector are legitimate: such a target never wins by detection but
    stays reachable by explicit ``target=``.
    """
    return set(DETECTORS) | {target for _, target in KERNELS}


def ensure_loaded() -> None:
    """Import every declared backend module, once.

    Lazy by necessity: enumerating at ``import tileops`` would pull in every installed
    backend, and tilelang with them.
    """
    global _loaded
    if _loaded:  # fast path: one bool read, no lock
        return
    if getattr(_LOADING, "active", False):
        return  # this thread is mid-discovery and may read what it has registered
    with LOCK:
        if _loaded:
            return
        _LOADING.active = True
        try:
            failed = _load_all()
            _loaded = True
        finally:
            _LOADING.active = False
    # Warned only now: under ``-W error`` a warning raises, and warning before discovery is
    # published means the next call redoes it and double-records every failure.
    for failure in failed:
        warnings.warn(
            f"TileOPs backend {failure.name!r} ({failure.entry_point}) failed to load and "
            f"was skipped: {failure.error} See tileops.backend.load_errors().",
            RuntimeWarning,
            stacklevel=3,
        )


def _load_all() -> list[BackendLoadFailure]:
    """Load every entry point, returning those that failed. Caller holds the lock."""
    failed = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        # All-or-nothing per entry point: a backend registering eighty ops and then raising
        # would leave the registry advertising ops it never finished initializing.
        checkpoint = snapshot()
        try:
            ep.load()
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not win
            restore(checkpoint)
            failure = BackendLoadFailure(
                name=ep.name,
                entry_point=ep.value,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            LOAD_ERRORS.append(failure)
            failed.append(failure)
        except BaseException:
            restore(checkpoint)  # an interrupt is not a backend being broken
            raise
    return failed


def load_error_suffix() -> str:
    """Append to any error, so a broken wheel is never invisible."""
    if not LOAD_ERRORS:
        return ""
    return f" ({len(LOAD_ERRORS)} backend(s) failed to load; see tileops.backend.load_errors())"


class RegistryState(NamedTuple):
    """What :func:`snapshot` captures, named so :func:`restore` cannot mis-order it."""

    detectors: dict[str, DetectFn]
    kernels: dict[tuple[str, str], GetKernelFn]
    resolved: dict[torch.device, str]
    load_errors: list[BackendLoadFailure]
    default_target: str | None
    loaded: bool


def snapshot() -> RegistryState:
    """Capture the registry. Backs both the load transaction and test isolation.

    Not exported: a public save/restore would invite production code to swap registries at
    runtime.
    """
    with LOCK:
        return RegistryState(
            detectors=dict(DETECTORS),
            kernels=dict(KERNELS),
            resolved=dict(RESOLVED),
            load_errors=list(LOAD_ERRORS),
            default_target=default_target,
            loaded=_loaded,
        )


def restore(state: RegistryState) -> None:
    """Undo everything since the matching :func:`snapshot`."""
    global default_target, _loaded
    with LOCK:
        DETECTORS.clear()
        DETECTORS.update(state.detectors)
        KERNELS.clear()
        KERNELS.update(state.kernels)
        RESOLVED.clear()
        RESOLVED.update(state.resolved)
        LOAD_ERRORS[:] = state.load_errors
        default_target = state.default_target
        _loaded = state.loaded
