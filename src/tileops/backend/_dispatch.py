"""Reading the tables: which target serves a call, and which builder to hand it to."""

from __future__ import annotations

from typing import NamedTuple

import torch

from tileops.backend import _registry as registry
from tileops.backend._errors import AmbiguousTargetError, BackendError, UnknownTargetError
from tileops.backend._protocol import BUILTIN, BuildKernel, DetectFn, DeviceFacts, Target

# What a device reports when nobody registered its type: no architecture, compute units or
# shared memory.
_NO_DEVICE_FACTS = DeviceFacts(arch=-1, sm_count=0, smem_budget=0)


class TuningFacts(NamedTuple):
    """The facts TileOPs' own CUDA kernels tune by, beside `.protocol.DeviceFacts`.

    Only in-tree call specs and build targets read them, so a backend never supplies them;
    a device other than CUDA reports `_NO_TUNING_FACTS`.
    """

    # The key of the calibrated board (``tileops._utils.calibration_key``), or ``None``.
    calibration: str | None
    threads_per_sm: int
    regs_per_sm: int
    # Shared memory of one SM, in bytes.
    smem_per_sm: int
    l2_bytes: int


_NO_TUNING_FACTS = TuningFacts(
    calibration=None, threads_per_sm=0, regs_per_sm=0, smem_per_sm=0, l2_bytes=0
)


def detect_target(device: torch.device) -> str | None:
    """Return the target whose kernels are written for *device*, or None if there is none.

    None means no backend is installed for this hardware, so the in-tree implementation runs.
    *device* is passed through untouched: reading ``.type`` is the vendor knowledge being
    delegated. Neither memoized nor locked — a target is settled once per op instance, and
    an op's first call can be inside a region dynamo traces.

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
                f"{sorted(registry.known_targets())}{registry.load_failure_suffix()}"
            )
        return requested
    if registry.default_target is not None:
        return registry.default_target
    if device is None:
        return None
    return detect_target(device)


def device_facts_of(device: torch.device | None) -> DeviceFacts:
    """Return the facts kernel selection reads about *device*.

    CUDA's are read by TileOPs; another device type's by the reader its backend registered.
    ``None`` names the current device of this process's accelerator. A device of a type
    nobody registered, CPU included, reports none.

    The reader receives *device* with its index filled in where it is the accelerator's
    type; a device that numbers none, such as ``cpu``, reaches it as it is.

    Raises:
        BackendError: The reader raised or returned something other than valid
            `.protocol.DeviceFacts`.
    """
    return device_facts_and_tuning(device)[0]


def device_facts_and_tuning(device: torch.device | None) -> tuple[DeviceFacts, TuningFacts]:
    """Return `device_facts_of` *device*, and its `TuningFacts`.

    Raises:
        BackendError: What `device_facts_of` raises.
    """
    from tileops._utils import device_facts
    from tileops._utils.device import current_device, on_current_index

    device = current_device() if device is None else on_current_index(device)
    if device is None:
        return _NO_DEVICE_FACTS, _NO_TUNING_FACTS
    if device.type == "cuda":
        facts = device_facts(device.index)
        return (
            DeviceFacts(arch=facts.arch, sm_count=facts.sm_count, smem_budget=facts.smem_budget),
            TuningFacts(
                calibration=facts.calibration,
                threads_per_sm=facts.threads_per_sm,
                regs_per_sm=facts.regs_per_sm,
                smem_per_sm=facts.smem_per_sm,
                l2_bytes=facts.l2_bytes,
            ),
        )
    registry.ensure_loaded()
    read = registry.DEVICE_FACTS.get(device.type)
    if read is None:
        return _NO_DEVICE_FACTS, _NO_TUNING_FACTS
    try:
        facts = read(device)
    except Exception as exc:
        raise BackendError(
            f"device facts reader {registry.describe(read)} raised on {device}: "
            f"{exc!r}{registry.load_failure_suffix()}"
        ) from exc
    problem = _invalid_facts(facts)
    if problem is not None:
        raise BackendError(
            f"device facts reader {registry.describe(read)} returned {facts!r} for "
            f"{device}: {problem}"
        )
    return facts, _NO_TUNING_FACTS


def _invalid_facts(facts: object) -> str | None:
    """Why *facts* is not a `.protocol.DeviceFacts` selection can read, or None."""
    if not isinstance(facts, DeviceFacts):
        return "a reader returns DeviceFacts"
    for name in ("arch", "sm_count", "smem_budget"):
        value = getattr(facts, name)
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{name} is an int"
    if facts.sm_count < 0 or facts.smem_budget < 0:
        return "sm_count and smem_budget are not negative"
    return None


def registered_kernel_builder(op: str, target: str) -> BuildKernel | None:
    """Return *target*'s ``build_kernel`` for *op*, or None when it registered none."""
    registry.ensure_loaded()
    return registry.BUILDERS.get((op, target))


def registered_targets(op: str | None = None) -> list[str]:
    """Targets registered for *op*; every known target when *op* is None."""
    registry.ensure_loaded()
    if op is None:
        return sorted(registry.known_targets())
    with registry._LOCK:
        cells = list(registry.BUILDERS)
    return sorted(target for name, target in cells if name == op)


def set_default_target(target: Target) -> None:
    """Route ops with no explicit ``target=`` to *target*.

    ``None`` restores detection; :data:`~.protocol.BUILTIN` turns replacement off. No
    environment variable: what decides which kernel runs should be visible in the program.

    The default is process-global. It affects only op instances not yet bound to a target:
    an instance binds on its first call and keeps that target. A backend must not call this
    while it is being imported; the choice belongs to the program.

    Raises:
        UnknownTargetError: *target* is a name no backend registered.
    """
    registry.ensure_loaded()
    if target is not None and target is not BUILTIN and target not in registry.known_targets():
        raise UnknownTargetError(
            f"no backend registered target {target!r}; known targets: "
            f"{sorted(registry.known_targets())}{registry.load_failure_suffix()}"
        )
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
