"""Shared device facts for call records."""

import dataclasses

import torch

__all__ = ["CallSpec"]


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
        from tileops.utils import device_facts

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
