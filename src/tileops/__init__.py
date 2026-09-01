"""TileOPs: efficient LLM inference built on TileLang.

Ops are reached in two levels, `tileops.<family>.<Op>`:

    from tileops.elementwise import HardtanhFwdOp

`_FAMILIES` lists the families. Each is a module of its own; `tileops.ops` is where
the classes are implemented, and is not the path to import them from.

The names below steer dispatch; a *backend* imports `tileops.backend` instead. Every
name here, families included, resolves on first access (PEP 562), so importing this
package pulls in neither torch nor any backend — the manifest tooling reads YAML where
neither is installed. Reaching into a family is what imports torch.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from tileops.backend import (
        BUILTIN,
        AmbiguousTargetError,
        BackendError,
        OpNotAvailableError,
        UnknownTargetError,
        default_target,
        load_failures,
        registered_targets,
        set_default_target,
    )

    from . import (
        attention,
        convolution,
        elementwise,
        fft,
        gemm,
        linear_attention,
        mamba,
        moe,
        norm,
        pool,
        quantization,
        reduction,
        rope,
        sequence_modeling,
    )
    from .ops.op_base import Op

# One entry per op family, ordered as `tileops.ops.__all__` groups them.
_FAMILIES = (
    "elementwise",
    "reduction",
    "norm",
    "quantization",
    "gemm",
    "pool",
    "convolution",
    "fft",
    "moe",
    "rope",
    "attention",
    "linear_attention",
    "mamba",
    "sequence_modeling",
)

_LAZY = {
    **dict.fromkeys(
        (
            "BUILTIN",
            "AmbiguousTargetError",
            "BackendError",
            "OpNotAvailableError",
            "UnknownTargetError",
            "default_target",
            "load_failures",
            "registered_targets",
            "set_default_target",
        ),
        "tileops.backend",
    ),
    "Op": "tileops.ops.op_base",
}

__all__ = sorted({*_LAZY, *_FAMILIES})


def __getattr__(name: str) -> Any:
    import importlib

    if name in _FAMILIES:
        # `__package__`, not `__name__`: a tool may load this file as `tileops.__init__`.
        value = importlib.import_module(f".{name}", __package__)
    else:
        module = _LAZY.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # later reads hit the module dict, not this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY, *_FAMILIES})
