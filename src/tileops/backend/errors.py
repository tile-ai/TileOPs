"""What this layer raises, and the one thing it records instead of raising."""

from __future__ import annotations

from dataclasses import dataclass


class BackendError(Exception):
    """Base class for every error this layer raises."""


class UnknownTargetError(BackendError):
    """No target claimed this device, or a named target registered nothing."""


class AmbiguousTargetError(BackendError):
    """More than one detector claimed the same device."""


class OpNotAvailableError(BackendError):
    """The target registered no ``get_kernel`` for this op."""


@dataclass(frozen=True)
class BackendLoadFailure:
    """A backend that failed to import.

    A record rather than a raisable: one broken distribution must not stop the others
    from loading, so the failure is collected and reported afterwards.
    """

    name: str
    entry_point: str
    error: str

    def __str__(self) -> str:
        return f"{self.name} ({self.entry_point}): {self.error}"
