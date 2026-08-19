"""Shared device facts for call records."""

import dataclasses

__all__ = ["CallSpec"]


@dataclasses.dataclass(frozen=True)
class CallSpec:
    """What a call runs on: the device facts every family's record carries.

    ``Op.select_kernel_key`` asks each implementation for its ``refusal(call)``,
    and a refusal reads ``arch``. A caller that states ``arch`` and ``h200`` gets
    what it stated; a record that states neither reads them when it is built,
    which is when the call is made rather than when the op is constructed.

    A family subclasses this and adds its own semantic fields.
    """

    arch: int = -1
    h200: bool = False

    def __post_init__(self) -> None:
        if self.arch < 0:
            from tileops.utils import get_sm_version, is_h200

            object.__setattr__(self, "arch", get_sm_version())
            object.__setattr__(self, "h200", is_h200())

    def __str__(self) -> str:
        """The facts of the call, without the fields nobody set.

        A selection failure names the call, and a record of mostly default
        fields buries the two that decided it.
        """
        default = type(self)(arch=self.arch, h200=self.h200)
        stated = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if f.name not in ("arch", "h200") and getattr(self, f.name) != getattr(default, f.name)
        ]
        return ", ".join([f"arch={self.arch}", f"h200={self.h200}", *stated])
