"""Shared device facts for call records."""

import dataclasses
import enum

import torch

__all__ = ["CallSpec"]

# The facts a call spec derives from its device.
_DEVICE_FACTS = ("arch", "calibration", "sm_count", "smem_budget")


class _DeviceFact:
    """A device fact of a call record: the value the caller stated, else the device's own.

    A fact the caller did not state is read from ``device`` the first time it is read.
    Hashing and comparing a record read none, so a record that is only looked up never
    queries the device.
    """

    def __init__(self, name: str) -> None:
        self.stored = f"_{name}"

    def __get__(self, record: "CallSpec | None", owner: type) -> object:
        if record is None:
            return self
        facts = record.__dict__
        if self.stored not in facts:
            record._read_device_facts()
        return facts[self.stored]

    def __set__(self, record: "CallSpec", value: object) -> None:
        if value is not self:
            record.__dict__[self.stored] = value


@dataclasses.dataclass(frozen=True)
class CallSpec:
    """A call spec: the immutable facts of one call and the device it runs on.

    A family subclasses this and adds every fact its implementations read in ``applies`` /
    ``refusal`` / ``entry_for``, and nothing a tensor's contents decide. Equality and the
    hash cover those facts and ``device``, normalized to an explicit type and index.

    The device facts (``arch``, ``calibration``, ``sm_count``, ``smem_budget``) are derived from ``device``
    and take no part in equality: one left unstated is read from ``device`` when selection
    or a builder first reads it, which the dispatcher does only on a miss. A caller may state
    them to ask ``select_implementation`` about a device it is not on; ``kernel_for`` refuses
    such a call spec, since it keys what it builds by ``device``.
    """

    arch: int = dataclasses.field(default=_DeviceFact("arch"), compare=False)
    # The key of the calibrated board the device belongs to (``tileops.utils.calibration_key``),
    # or ``None``. A family's fitted tuning data is keyed by it.
    calibration: "str | None" = dataclasses.field(default=_DeviceFact("calibration"), compare=False)
    sm_count: int = dataclasses.field(default=_DeviceFact("sm_count"), compare=False)
    # Shared memory one block may take after opting in, in bytes.
    smem_budget: int = dataclasses.field(default=_DeviceFact("smem_budget"), compare=False)
    # The device whose facts decide selection. ``None`` reads the current device.
    device: "torch.device | None" = None
    # FIXME(staged-rollout): tuning policy travels on the record of an unmigrated call.
    #
    # Broken invariant: a call spec carries call facts only (ops-design.md § Kernel selection).
    # Why: unmigrated ops and their kernels still pass ``tune`` through the record.
    # Cleanup: when every op declares ``interfaces``, delete this field.
    tune: bool = dataclasses.field(default=False, compare=False)

    def __post_init__(self) -> None:
        stated = frozenset(f for f in _DEVICE_FACTS if f"_{f}" in vars(self))
        object.__setattr__(self, "stated_device_facts", stated)
        device = self.device
        if device is None:
            return
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)

    def _read_device_facts(self) -> None:
        """Fill the device facts the caller left unstated from ``device``.

        A device other than a CUDA one has none: no architecture, board, SM or shared memory.
        A record that named no device has none either where the process has no CUDA device.
        """
        from tileops.utils import device_facts

        if (self.device is not None and self.device.type != "cuda") or (
            self.device is None and not torch.cuda.is_available()
        ):
            arch, calibration, sm_count, smem_budget = -1, None, 0, 0
        else:
            arch, calibration, sm_count, smem_budget = device_facts(
                self.device.index if self.device is not None else None
            )
        facts = self.__dict__
        facts.setdefault("_arch", arch)
        facts.setdefault("_calibration", calibration)
        facts.setdefault("_sm_count", sm_count)
        facts.setdefault("_smem_budget", smem_budget)

    def refuse_unkeyable(self) -> None:
        """Raise unless every compared field is an immutable value.

        A tensor, a mutable container or a policy object compares by identity or not at
        all, so a record holding one would miss the cache or return another call's entry.

        Raises:
            TypeError: A compared field holds something other than a scalar, a dtype, a
                device, an enum, a frozen dataclass, or a tuple or frozenset of those.
        """
        keyable = (type(None), bool, int, float, str, torch.dtype, torch.device, enum.Enum)
        pending = [(f.name, getattr(self, f.name)) for f in dataclasses.fields(self) if f.compare]
        while pending:
            name, value = pending.pop()
            if isinstance(value, (tuple, frozenset)):
                pending.extend((name, item) for item in value)
            elif dataclasses.is_dataclass(value) and type(value).__dataclass_params__.frozen:
                pending.extend(
                    (name, getattr(value, f.name)) for f in dataclasses.fields(value) if f.compare
                )
            elif not isinstance(value, keyable):
                raise TypeError(
                    f"{type(self).__name__}.{name} holds a {type(value).__name__}, which "
                    f"cannot key a dispatch cache; a call spec holds immutable values only"
                )

    def __str__(self) -> str:
        """The facts of the call, without the fields nobody set.

        A selection failure names the call, and a record of mostly default
        fields buries the device facts that decided it.
        """
        facts = {name: getattr(self, name) for name in _DEVICE_FACTS}
        default = type(self)(**facts, device=self.device)
        stated = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if f.name not in (*_DEVICE_FACTS, "device", "tune")
            and getattr(self, f.name) != getattr(default, f.name)
        ]
        return ", ".join([*(f"{name}={value}" for name, value in facts.items()), *stated])
