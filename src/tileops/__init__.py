"""TileOPs: TileLang kernels for efficient LLM inference.

The names below are what a *caller* needs to steer dispatch. A *backend* imports
:mod:`tileops.backend` instead — that module is the whole protocol.

They resolve on first access rather than at import. Importing this package therefore
costs nothing: no torch, no backend discovery, no tilelang. The manifest tooling reads
YAML and must keep working in an environment with none of them installed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type checkers and IDEs, which do not run __getattr__
    from tileops.backend import (
        AmbiguousTargetError,
        BackendError,
        OpNotAvailableError,
        UnknownTargetError,
        default_target,
        load_errors,
        registered_targets,
        set_default_target,
    )

#: name -> the module it comes from. Every entry must be importable without tilelang.
_LAZY = dict.fromkeys(
    (
        "AmbiguousTargetError",
        "BackendError",
        "OpNotAvailableError",
        "UnknownTargetError",
        "default_target",
        "load_errors",
        "registered_targets",
        "set_default_target",
    ),
    "tileops.backend",
)

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """Import the module owning *name* on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # later reads hit the module dict, not this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
