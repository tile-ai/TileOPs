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

from .errors import BackendError
from .protocol import BuildKernel, DetectFn, Target

#: The value names a *module*; importing it must perform the registration.
ENTRY_POINT_GROUP = "tileops.backends"

DETECTORS: dict[str, DetectFn] = {}
BUILDERS: dict[tuple[str, str], BuildKernel] = {}

#: One line per backend that failed to import. Strings rather than records: they are read
#: to be printed, and a type crossing the boundary should earn its place.
LOAD_FAILURES: list[str] = []

#: Which target ops use when they name none. There is deliberately no constant here: the
#: in-tree implementation is not a target (see the RFC's §3.4), so "no default" means "do
#: not replace anything". Kept beside the tables so the load transaction and test isolation
#: cover it.
default_target: Target = None

#: Set only once every entry point has been tried, so no thread sees a half-built registry.
_loaded = False

#: Reentrant: discovery holds it while importing backend modules, whose top level calls
#: the register functions and takes it again.
LOCK = threading.RLock()

#: Set on the thread running discovery, which must read the partial registry it is
#: building while every other thread waits on the lock for the finished one.
_LOADING = threading.local()


def register_detector(target: str, detect: DetectFn) -> None:
    """Record *detect* as how *target* recognizes its devices.

    Args:
        target: The name this backend gives its set of kernels.
        detect: Answers "is this the kind of device my kernels are written for". Return
            ``False`` for devices it does not serve rather than raising. Whether a
            particular call is supported belongs in ``build_kernel``, not here.

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


def register_kernel_builder(op: str, target: str, build_kernel: BuildKernel) -> None:
    """Record *build_kernel* as how *target* builds a kernel for *op*.

    Args:
        op: The op's manifest key, e.g. ``"RMSNormFwdOp"``.
        target: The name this backend gives its set of kernels.
        build_kernel: Called with a :class:`~.protocol.TensorSpec` per input and the op's
            params by keyword. Must be lazy: importing the calling module must not compile
            anything.

    Raises:
        BackendError: ``(op, target)`` is already claimed. A target belongs to one
            distribution, so a second claim means two installed packages both say they are
            it — a misinstall, not a race to arbitrate.
    """
    with LOCK:
        existing = BUILDERS.get((op, target))
        if existing is not None:
            raise BackendError(
                f"{(op, target)} is already registered to {describe(existing)}; "
                f"{describe(build_kernel)} cannot take it. A target belongs to one "
                f"distribution; two packages claiming it is a misinstall."
            )
        BUILDERS[(op, target)] = build_kernel


def describe(fn: Callable) -> str:
    """Name *fn* well enough to act on, module included."""
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn)}"


def known_targets() -> set[str]:
    """Every target that registered anything.

    Builders without a detector are legitimate: such a target never wins by detection but
    stays reachable by an explicit ``target=``.
    """
    return set(DETECTORS) | {target for _, target in BUILDERS}


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
            f"TileOPs backend failed to load and was skipped: {failure} "
            f"See tileops.backend.load_failures().",
            RuntimeWarning,
            stacklevel=3,
        )


def _load_all() -> list[str]:
    """Load every entry point, returning the failures. Caller holds the lock.

    Enumerated in a fixed order so that the failure records and their warnings come out the
    same way every run.
    """
    failed = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: (e.name, e.value)):
        # All-or-nothing per entry point: a backend registering eighty ops and then raising
        # would leave the registry advertising ops it never finished initializing.
        checkpoint = snapshot()
        try:
            ep.load()
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not win
            restore(checkpoint)
            reason = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failure = f"{ep.name} ({ep.value}): {reason}"
            LOAD_FAILURES.append(failure)
            failed.append(failure)
        except BaseException:
            restore(checkpoint)  # an interrupt is not a backend being broken
            raise
    return failed


def load_failure_suffix() -> str:
    """Append to any error, so a broken wheel is never invisible."""
    if not LOAD_FAILURES:
        return ""
    return (f" ({len(LOAD_FAILURES)} backend(s) failed to load; "
            f"see tileops.backend.load_failures())")


class RegistryState(NamedTuple):
    """What :func:`snapshot` captures, named so :func:`restore` cannot mis-order it."""

    detectors: dict[str, DetectFn]
    builders: dict[tuple[str, str], BuildKernel]
    load_failures: list[str]
    default_target: Target
    loaded: bool


def snapshot() -> RegistryState:
    """Capture the registry. Backs both the load transaction and test isolation.

    Not exported: a public save/restore would invite production code to swap registries at
    runtime.
    """
    with LOCK:
        return RegistryState(
            detectors=dict(DETECTORS),
            builders=dict(BUILDERS),
            load_failures=list(LOAD_FAILURES),
            default_target=default_target,
            loaded=_loaded,
        )


def restore(state: RegistryState) -> None:
    """Undo everything since the matching :func:`snapshot`."""
    global default_target, _loaded
    with LOCK:
        DETECTORS.clear()
        DETECTORS.update(state.detectors)
        BUILDERS.clear()
        BUILDERS.update(state.builders)
        LOAD_FAILURES[:] = state.load_failures
        default_target = state.default_target
        _loaded = state.loaded
