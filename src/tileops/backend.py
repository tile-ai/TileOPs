"""The backend protocol: everything a backend distribution needs and nothing else.

A backend is a separate distribution shipping TileLang kernels for one *target* — a
family of devices its kernels can run on (``nv``, ``musa``, ``ascend``). It joins
TileOPs by implementing two callbacks and registering them:

- ``detect(device) -> bool`` answers whether a device is this target.
- ``get_kernel(*inputs, **params) -> Kernel`` returns something that computes one op.

Choosing among several implementations, constructing, caching and compiling all happen
inside ``get_kernel``; TileOPs takes the result. So this module names no kernel, no
kernel class, no dispatch key, no priority and no capability: dispatch answers *who
does it*, never *how*.

This module depends on ``torch`` only. A backend importing it does not import tilelang.
"""

from __future__ import annotations

import threading
import traceback
import warnings
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, NamedTuple, Protocol, Union

import torch

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

#: Entry point group a backend declares to be discovered. The value names a *module*;
#: importing it must perform the registration.
ENTRY_POINT_GROUP = "tileops.backends"


# --------------------------------------------------------------------------------------
# What crosses the boundary
# --------------------------------------------------------------------------------------


class InputSpec(NamedTuple):
    """What one input tensor *is*, without the tensor.

    A backend needs the properties of this call to build a kernel — never the data, and
    never the tensor's identity. This description also serves as the op layer's memo
    key, so comparing it costs nothing beyond the lookup that has to happen anyway.

    Device, dtype and shape are the whole description because the op layer hands kernels
    contiguous tensors only. Stride would otherwise belong here; the calling convention
    is what makes it redundant, not an assumption about what kernels read.
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "InputSpec":
        """Describe *tensor*."""
        return InputSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


#: What one call returns: a single tensor, several, or nothing when the op only writes
#: into its inputs. ``torch.library.custom_op`` cannot express a return value aliasing
#: an input, so a purely mutating op returns ``None`` and the op layer adds the
#: chaining convenience above this boundary.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]


class Kernel(Protocol):
    """What ``get_kernel`` returns. A structural convention — do not subclass.

    Deliberately not ``runtime_checkable``: on a protocol whose only member is
    ``__call__``, an isinstance check reduces to "is it callable", which would read as a
    conformance test while proving nothing.

    Positional argument order is the manifest's ``signature.inputs`` declaration order.

    An optional ``autotune(*example: torch.Tensor) -> None`` method may be present:
    ``Op.autotune()`` calls it when it is, and does nothing when it is not. Tuning needs
    real tensors to measure, and tensors only exist once the kernel does — which is why
    it is a method here rather than a third registered callback.
    """

    def __call__(self, *tensors: torch.Tensor) -> KernelResult: ...


#: Callback one: is this device my target?
#:
#: Answer, do not raise: ``False`` for a device this target does not serve. A detector
#: that raises stops dispatch with a :class:`BackendError` naming it, because guessing
#: past it would hand the device to somebody else. It must also register nothing while
#: it runs -- it is being asked about a table it is not allowed to change. Called once
#: per distinct device, then cached.
DetectFn = Callable[[torch.device], bool]

#: Callback two: give me something that computes this op.
#:
#: Called as ``get_kernel(*inputs, **params)`` where *inputs* are :class:`InputSpec`
#: values in the manifest's ``signature.inputs`` order and *params* are the manifest's
#: ``signature.params`` by keyword, already validated. The precise keyword names are
#: per-op, which the type system cannot express — hence ``...``.
GetKernelFn = Callable[..., "Kernel"]


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class BackendError(Exception):
    """Base class for every error this layer raises."""


class UnknownTargetError(BackendError):
    """No target claimed this device, or a target was named that nothing registered."""


class AmbiguousTargetError(BackendError):
    """More than one detector claimed the same device."""


class OpNotAvailableError(BackendError):
    """The target registered no ``get_kernel`` for this op."""


@dataclass(frozen=True)
class BackendLoadFailure:
    """One backend distribution that failed to import, kept so it stays visible.

    A record, not a raisable: a broken plugin must not abort discovery, so the failure
    is collected instead of thrown.
    """

    name: str
    entry_point: str
    error: str

    def __str__(self) -> str:
        return f"{self.name} ({self.entry_point}): {self.error}"


# --------------------------------------------------------------------------------------
# The two tables
# --------------------------------------------------------------------------------------
#
# Module-level private state, reached only through the functions below. Registration
# happens as mutually unaware distributions are imported, so they can only meet at one
# process-wide place -- the shape torch.library's operator table and pytest's plugin
# manager already have. Lookup is on the call path, so a plain dict is the floor.

_DETECTORS: dict[str, DetectFn] = {}
_KERNELS: dict[tuple[str, str], GetKernelFn] = {}

#: device -> target, memoized. Cleared whenever a detector is added, since a new
#: detector can change the answer for a device already asked about.
_RESOLVED: dict[torch.device, str] = {}

#: Bumped by every register_detector. resolve_target probes outside the lock, so it
#: re-reads this before trusting what it computed: a detector arriving mid-probe would
#: otherwise let a now-ambiguous device be cached as unambiguous.
_DETECTOR_VERSION = 0

#: How many times resolve_target re-probes before calling the table unstable. Legitimate
#: churn is over in one retry: registration happens as modules import, not per call.
_MAX_PROBE_RETRIES = 8

_LOAD_ERRORS: list[BackendLoadFailure] = []

#: Process-wide default target, or None to detect from the device. Starts unset: "nv is
#: the default" is a consequence of the nv detector claiming cuda devices, not a
#: constant here. A hardcoded default would make detection dead code and would force a
#: machine carrying only a third-party backend to configure its way out.
_DEFAULT_TARGET: str | None = None

#: True only once every entry point has been tried. Publishing it earlier would let
#: another thread observe a half-built registry -- and memoize a kernel from a backend
#: that goes on to fail and be rolled back.
_loaded = False

#: Reentrant on purpose: _ensure_loaded holds it while importing backend modules, and
#: those imports call register() from their module top level, which takes it again.
_LOCK = threading.RLock()

#: Set on the thread currently running discovery. That thread must see the partial
#: registry -- it is the one building it -- while every other thread waits on the lock
#: for the finished article. A global flag cannot tell those two cases apart.
_LOADING = threading.local()


# --------------------------------------------------------------------------------------
# Registration -- what a backend calls
# --------------------------------------------------------------------------------------


def register(op: str, target: str, get_kernel: GetKernelFn) -> None:
    """Record *get_kernel* as how *target* computes *op*.

    Args:
        op: The op's manifest key, e.g. ``"RMSNormFwdOp"``.
        target: The target id this backend serves, e.g. ``"musa"``.
        get_kernel: Callback two. Must be lazy — importing the module that calls this
            must not compile anything.

    Raises:
        BackendError: Another registration already holds ``(op, target)``. Registering
            the *identical* callable again is a no-op — two entry points naming one
            module, or a module that registers the same function twice. Anything else is
            a conflict, including ``importlib.reload``, which produces a new function
            object: a reloaded backend must reset the registry rather than re-register
            over itself.
    """
    key = (op, target)
    with _LOCK:
        existing = _KERNELS.get(key)
        if existing is not None:
            if existing is get_kernel:
                return
            raise BackendError(
                f"{key} is already registered to {_describe(existing)}; "
                f"{_describe(get_kernel)} cannot take it. Silently overwriting would "
                f"make 'which one ran' unanswerable."
            )
        _KERNELS[key] = get_kernel


def register_detector(target: str, detect: DetectFn) -> None:
    """Record *detect* as how to recognize a device belonging to *target*.

    Raises:
        BackendError: *target* already has a different detector.
    """
    with _LOCK:
        existing = _DETECTORS.get(target)
        if existing is not None:
            if existing is detect:
                return
            raise BackendError(
                f"target {target!r} already has detector {_describe(existing)}; "
                f"{_describe(detect)} cannot replace it."
            )
        global _DETECTOR_VERSION
        _DETECTORS[target] = detect
        _RESOLVED.clear()
        _DETECTOR_VERSION += 1


def _describe(fn: Callable) -> str:
    """Name *fn* well enough to act on, module included."""
    module = getattr(fn, "__module__", None) or "?"
    return f"{module}.{getattr(fn, '__qualname__', repr(fn))}"


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def _ensure_loaded() -> None:
    """Import every declared backend module, once.

    Lazy by necessity: enumerating at ``import tileops`` would pull in every installed
    backend, and with them tilelang, defeating the packaging boundary.
    """
    global _loaded
    if _loaded:  # fast path: one bool read, no lock
        return
    if getattr(_LOADING, "active", False):
        return  # this thread is mid-discovery; it may read what it has registered
    with _LOCK:
        if _loaded:
            return
        _LOADING.active = True
        try:
            failed = _load_all()
            _loaded = True  # published last: readers wait on the lock until it is true
        finally:
            _LOADING.active = False
    # Warn only after discovery is published. Under ``-W error`` a warning raises, and
    # warning any earlier would leave discovery unfinished -- so the next call would redo
    # it and record every failure twice.
    for failure in failed:
        warnings.warn(
            f"TileOPs backend {failure.name!r} ({failure.entry_point}) failed to load "
            f"and was skipped: {failure.error} See tileops.backend.load_errors().",
            RuntimeWarning,
            stacklevel=3,
        )


def _load_all() -> list[BackendLoadFailure]:
    """Load every entry point and return the ones that failed. Caller holds the lock."""
    failed: list[BackendLoadFailure] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        # One entry point's load is all-or-nothing. A backend registering 80 ops and
        # then raising would otherwise leave those 80 behind, and a half-registered
        # backend is worse than an absent one: registered() would advertise ops whose
        # distribution never finished initializing.
        checkpoint = _snapshot()
        try:
            ep.load()
        except BaseException as exc:  # noqa: BLE001 - one bad plugin must not win
            _restore(checkpoint)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failure = BackendLoadFailure(
                name=ep.name,
                entry_point=ep.value,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            # Recorded immediately, not at the end: a later entry point importing may ask
            # load_errors() during its own load, and must see the failures so far.
            _LOAD_ERRORS.append(failure)
            failed.append(failure)
    return failed


def load_errors() -> tuple[BackendLoadFailure, ...]:
    """Backends that failed to import and were skipped.

    A broken wheel must not present itself as "no target claimed this device", so every
    error below points here.
    """
    _ensure_loaded()
    return tuple(_LOAD_ERRORS)


def _load_error_suffix() -> str:
    """Append to any error message, so a broken wheel is never invisible."""
    if not _LOAD_ERRORS:
        return ""
    return f" ({len(_LOAD_ERRORS)} backend(s) failed to load; see tileops.backend.load_errors())"


# --------------------------------------------------------------------------------------
# Lookup -- what the op layer calls
# --------------------------------------------------------------------------------------


def resolve_target(device: torch.device) -> str:
    """Return the target id serving *device*.

    TileOPs does not parse *device* — it neither reads ``.type`` nor maps it to
    hardware. That knowledge is exactly what is being delegated, so the device is a
    query condition here, not an answer: it is the cache key and the callback's
    argument, passed through untouched. A backend needing more (a vendor runtime probe,
    an environment variable) takes it inside its own callback.

    ``cuda`` and ``cuda:0`` are separate cache entries, and neither is normalized into
    the other. Normalizing would mean materializing the device to learn its index, which
    is interpretation. Two entries for one GPU cost nothing; a wrong answer would.

    Memoizing an unindexed device is sound because a target claims a device *type*, not
    an index: every index of one type answers the same, so ``torch.cuda.set_device``
    moving what ``cuda`` means cannot change which target claims it. A backend that
    needs to answer per index is asked with indexed devices — tensors always carry one.

    Raises:
        UnknownTargetError: No detector claimed *device*.
        AmbiguousTargetError: More than one did. Pass ``target=`` explicitly to bypass
            detection entirely.
    """
    _ensure_loaded()
    cached = _RESOLVED.get(device)
    if cached is not None:
        return cached

    for _ in range(_MAX_PROBE_RETRIES):
        # Snapshot under the lock, then probe outside it: detect() is a backend's code
        # and must not run while holding a lock every registration needs. Re-read the
        # version afterwards, because a detector registered mid-probe changes the answer.
        with _LOCK:
            version = _DETECTOR_VERSION
            detectors = list(_DETECTORS.items())
        claimed = _claimants(device, detectors)
        with _LOCK:
            if version == _DETECTOR_VERSION:
                return _record(device, claimed)
            # The table moved under us; ask again. Retries are bounded because
            # registration is an import-time act: a detector that registers while being
            # probed would loop forever, and that is a contract violation worth naming
            # rather than a hang worth waiting out.
    raise BackendError(
        f"the detector table kept changing while resolving device {device} "
        f"({_MAX_PROBE_RETRIES} attempts). A detector must not register anything while "
        f"it is being asked."
    )


def _claimants(device: torch.device, detectors: list[tuple[str, DetectFn]]) -> list[str]:
    """Ask each detector about *device*.

    A detector must answer, not raise. One that raises is a bug in that backend, and
    guessing past it would silently hand the device to somebody else, so the failure is
    reported against the target that owns it.
    """
    claimed = []
    for target, detect in detectors:
        try:
            if detect(device):
                claimed.append(target)
        except Exception as exc:
            raise BackendError(
                f"detector for target {target!r} ({_describe(detect)}) raised on device "
                f"{device}: {exc!r}. A detector must return False for devices it does "
                f"not serve."
            ) from exc
    return claimed


def _record(device: torch.device, claimed: list[str]) -> str:
    """Memoize and return the sole claimant, or explain why there is not one.

    Caller holds the lock. Neither "nobody claimed it" nor "several did" is cached: both
    can be resolved by a later registration.
    """
    if not claimed:
        raise UnknownTargetError(
            f"no registered target claims device {device}; "
            f"registered targets: {sorted(_DETECTORS)}{_load_error_suffix()}"
        )
    if len(claimed) > 1:
        raise AmbiguousTargetError(
            f"device {device} is claimed by {sorted(claimed)}; pass target= to choose"
            f"{_load_error_suffix()}"
        )
    _RESOLVED[device] = claimed[0]
    return claimed[0]


def select_target(explicit: str | None, device: torch.device | None) -> str:
    """Decide which target serves this call, in the one place that decides it.

    Priority: *explicit* wins, then the process-wide default, then detection from
    *device*. Every op goes through here — duplicating this order across op classes is
    how the three sources drift apart.

    Args:
        explicit: The op's ``target=``, or None when it was not given. Honoured as
            named: it is not checked against *device*, because naming a target is how a
            caller overrides detection.
        device: The device to detect from, used only when neither of the first two
            applies. None when the call has no tensor input to take a device from, which
            leaves nothing to detect and makes ``target=`` the only way to say where to
            run.
    """
    if explicit is not None:
        return explicit
    if _DEFAULT_TARGET is not None:
        return _DEFAULT_TARGET
    if device is None:
        raise UnknownTargetError(
            "this call has no tensor input, so there is no device to detect from; "
            "pass target= or set tileops.set_default_target()"
        )
    return resolve_target(device)


def get_kernel_for(op: str, target: str) -> GetKernelFn:
    """Return *target*'s ``get_kernel`` for *op*.

    Raises:
        OpNotAvailableError: That cell is empty. Never falls back to another target —
            silently running somewhere else makes "where did this run" unanswerable.
    """
    _ensure_loaded()
    try:
        return _KERNELS[(op, target)]
    except KeyError:
        raise OpNotAvailableError(
            f"no get_kernel registered for {(op, target)}; "
            f"registered targets for this op: {registered_targets(op)}"
            f"{_load_error_suffix()}"
        ) from None


def registered() -> frozenset[tuple[str, str]]:
    """Every registered ``(op, target)``, for introspection and error messages.

    Keys only. Handing out the callbacks would open a second way to reach a
    ``get_kernel`` beside :func:`get_kernel_for`, and two lookup paths is one too many
    for the only thing on the call path.
    """
    _ensure_loaded()
    return frozenset(_KERNELS)


def registered_targets(op: str | None = None) -> list[str]:
    """Targets registered for *op*; every known target when *op* is None."""
    _ensure_loaded()
    if op is None:
        return sorted(_known_targets())
    return sorted(target for name, target in _KERNELS if name == op)


def _known_targets() -> set[str]:
    """Every target that registered anything.

    A target with kernels but no detector is legitimate — it is reachable by explicit
    ``target=`` and simply never wins by detection.
    """
    _ensure_loaded()
    return set(_DETECTORS) | {target for _, target in _KERNELS}


# --------------------------------------------------------------------------------------
# The process-wide default
# --------------------------------------------------------------------------------------


def set_default_target(target: str | None) -> None:
    """Route every op with no explicit ``target=`` to *target*; None restores detection.

    One process-wide setting, deliberately: it decides which kernels get built, and
    threading it through call sites would put it on the hot path. There is no
    environment variable — configuration that changes which kernel runs should be
    visible in the program.
    """
    global _DEFAULT_TARGET
    if target is not None and target not in _known_targets():
        raise UnknownTargetError(
            f"no backend registered target {target!r}; "
            f"known targets: {sorted(_known_targets())}{_load_error_suffix()}"
        )
    _DEFAULT_TARGET = target


def default_target() -> str | None:
    """The process-wide default, or None when the device decides."""
    return _DEFAULT_TARGET


# --------------------------------------------------------------------------------------
# Test isolation -- internal, deliberately not exported
# --------------------------------------------------------------------------------------


def _snapshot() -> tuple:
    """Capture the whole registry so a test can put it back.

    Without this, one test registering a fake target changes how every later test
    behaves, and the suite's result depends on its order. Internal: a public
    save/restore would invite production code to swap registries at runtime.
    """
    with _LOCK:
        return (
            dict(_DETECTORS),
            dict(_KERNELS),
            dict(_RESOLVED),
            list(_LOAD_ERRORS),
            _DEFAULT_TARGET,
            _loaded,
            _DETECTOR_VERSION,
        )


def _restore(state: tuple) -> None:
    """Undo everything since the matching :func:`_snapshot`."""
    global _DEFAULT_TARGET, _loaded, _DETECTOR_VERSION
    detectors, kernels, resolved, errors, default, loaded, version = state
    with _LOCK:
        _DETECTORS.clear()
        _DETECTORS.update(detectors)
        _KERNELS.clear()
        _KERNELS.update(kernels)
        _RESOLVED.clear()
        _RESOLVED.update(resolved)
        _LOAD_ERRORS[:] = errors
        _DEFAULT_TARGET = default
        _loaded = loaded
        # The saved value is not restored: the counter only has to be monotonic, and a
        # rollback is a change like any other, so a probe in flight must not trust what
        # it computed against the table this just replaced.
        del version
        _DETECTOR_VERSION += 1
