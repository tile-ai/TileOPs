"""TileOPs: TileLang kernels for efficient LLM inference.

The names below steer dispatch; a *backend* imports :mod:`tileops.backend` instead. They
resolve on first access (PEP 562), so importing this package pulls in neither torch nor any
backend — the manifest tooling reads YAML where neither is installed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
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
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # later reads hit the module dict, not this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
