"""Shared device facts for call records."""

import dataclasses
import functools

import torch

__all__ = ["CallSpec", "device_facts"]


@functools.lru_cache(maxsize=16)
def device_facts(index: "int | None") -> "tuple[int, bool, int]":
    """``(arch, h200, sm_count)`` of a device, cached per index.

    A record is built on the per-call path, so the three probes run once per
    call unless they are cached here; they cost more host time than the kernel
    a selection they feed can save.
    """
    from tileops.utils import get_sm_count, get_sm_version, is_h200

    return get_sm_version(index), is_h200(index), get_sm_count(index)


@dataclasses.dataclass(frozen=True)
class CallSpec:
    """What a call runs on: the device facts every family's record carries.

    ``Op.select_kernel_key`` asks each implementation for its ``refusal(call)``,
    and a refusal reads these. A caller that states them gets what it stated; a
    record that states none reads them when it is built, which is when the call
    is made rather than when the op is constructed.

    A family subclasses this and adds every fact its implementations read in
    ``applies`` / ``refusal`` / ``entry_for`` or take as a construction argument,
    and nothing a tensor's contents decide.
    """

    arch: int = -1
    h200: bool = False
    sm_count: int = 0
    # The device whose facts decide selection. ``None`` reads the current device.
    device: "torch.device | None" = None
    # Whether the kernel built for this call tunes itself. A construction argument
    # wherever a kernel takes one, so it belongs to the call rather than beside it. No
    # part of comparison: tuning changes how fast a kernel runs, not what it computes.
    tune: bool = dataclasses.field(default=False, compare=False)

    def __post_init__(self) -> None:
        if self.arch >= 0 and self.sm_count > 0:
            return
        index = self.device.index if self.device is not None else None
        arch, h200, sm_count = device_facts(index)
        if self.arch < 0:
            object.__setattr__(self, "arch", arch)
            object.__setattr__(self, "h200", h200)
        if self.sm_count <= 0:
            object.__setattr__(self, "sm_count", sm_count)

    def __str__(self) -> str:
        """The facts of the call, without the fields nobody set.

        A selection failure names the call, and a record of mostly default
        fields buries the device facts that decided it.
        """
        default = type(self)(
            arch=self.arch, h200=self.h200, sm_count=self.sm_count, device=self.device
        )
        stated = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if f.name not in ("arch", "h200", "sm_count", "device", "tune")
            and getattr(self, f.name) != getattr(default, f.name)
        ]
        return ", ".join(
            [f"arch={self.arch}", f"h200={self.h200}", f"sm_count={self.sm_count}", *stated]
        )
