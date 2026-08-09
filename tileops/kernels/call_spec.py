"""The base every family's call record shares.

``Op.select_kernel_key`` asks each implementation ``supports(call)``, and
``Kernel.supports`` reads one field: ``arch``. Resolving it belongs here rather
than in each family, so a family's record carries only its own semantic fields.
See docs/design/ops-design.md § Kernel selection.
"""

import dataclasses

__all__ = ["CallSpec"]


@dataclasses.dataclass(frozen=True)
class CallSpec:
    """What a call runs on, as the facts selection filters against.

    ``arch`` and ``h200`` describe the device the call will run on, read when the
    call is made rather than when the op is constructed: an op constructs on a
    machine that cannot run it and is refused when a kernel is selected. A caller
    that states ``arch`` gets what it stated, which is how a test drives another
    architecture without a device.

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
            if f.name not in ("arch", "h200")
            and getattr(self, f.name) != getattr(default, f.name)
        ]
        return ", ".join([f"arch={self.arch}", f"h200={self.h200}", *stated])
