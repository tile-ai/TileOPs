"""TileOPs: TileLang kernels for efficient LLM inference.

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
        dropout,
        elementwise,
        fft,
        fp8_lightning_indexer,
        fp8_quant,
        gemm,
        linear_attention,
        mamba,
        moe,
        norm,
        pool,
        reduction,
        rope,
        sequence_modeling,
        topk_selector,
    )
    from .ops.op_base import Op

# One entry per op family, ordered simple to composite as `tileops.ops.__all__` groups
# them: the pointwise transforms first, the sequence-model kernels last. The docs
# site's API Reference pages follow this order, but not one page per family: a family
# holding unrelated algorithms is split across pages, and a single-op family can be a
# section of a related page. `tests/test_public_api.py` checks the tuple against the
# tree and against that grouping; the docs site's order is not enforceable from here.
_FAMILIES = (
    "elementwise",
    "dropout",
    "reduction",
    "norm",
    "fp8_quant",
    "topk_selector",
    "gemm",
    "pool",
    "convolution",
    "fft",
    "moe",
    "rope",
    "attention",
    "fp8_lightning_indexer",
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
    # The base class every op derives from, for callers writing one of their own.
    "Op": "tileops.ops.op_base",
}

__all__ = sorted({*_LAZY, *_FAMILIES})


def __getattr__(name: str) -> Any:
    import importlib

    if name in _FAMILIES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        module = _LAZY.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # later reads hit the module dict, not this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY, *_FAMILIES})
