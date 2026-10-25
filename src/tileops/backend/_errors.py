"""What this layer raises."""

from __future__ import annotations


class BackendError(Exception):
    """Base class for every error this layer raises."""


class UnknownTargetError(BackendError):
    """A named target is not registered."""


class AmbiguousTargetError(BackendError):
    """More than one detector claimed the same device."""


class OpNotAvailableError(BackendError, ValueError):
    """No implementation serves this call: the selected target cannot serve the op, or no
    kernel accepts the call's arguments. A `ValueError`, as arguments no kernel accepts are.
    """
