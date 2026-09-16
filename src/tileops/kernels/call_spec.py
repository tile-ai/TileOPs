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

    def __post_init__(self) -> None:
        index = self.device.index if self.device is not None else None
        if self.arch < 0:
            from tileops.utils import get_sm_version, is_h200

            object.__setattr__(self, "arch", get_sm_version(index))
            object.__setattr__(self, "h200", is_h200(index))
        if self.sm_count <= 0:
            from tileops.utils import get_sm_count

            object.__setattr__(self, "sm_count", get_sm_count(index))

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
            if f.name not in ("arch", "h200", "sm_count", "device")
            and getattr(self, f.name) != getattr(default, f.name)
        ]
        return ", ".join(
            [f"arch={self.arch}", f"h200={self.h200}", f"sm_count={self.sm_count}", *stated]
        )
