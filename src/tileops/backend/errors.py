"""What this layer raises."""

from __future__ import annotations


class BackendError(Exception):
    """Base class for every error this layer raises."""


class UnknownTargetError(BackendError):
    """A named target is not registered."""


class AmbiguousTargetError(BackendError):
    """More than one detector claimed the same device."""


class OpNotAvailableError(BackendError):
    """The selected target cannot serve this op.

    Either it registered no ``build_kernel`` for it, or the op has not been given the
    tensors to describe yet. Neither falls back to the in-tree implementation: a target is
    selected because the device belongs to other hardware, where in-tree kernels cannot
    launch at all.
    """
