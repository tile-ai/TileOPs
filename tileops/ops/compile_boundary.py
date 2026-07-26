"""Opaque dispatch boundary for torch.compile.

Invariant: a dynamo-traced ``Op.forward`` must not construct kernels or
enter a TileLang builder. Lazy-dispatch ops route ``forward`` through a
``torch.library.custom_op`` whose eager body resolves the instance here
and runs the untraced path (cache lookup, kernel construction, launch).

``Op.dispatch_kernel`` registers every conforming op at ``__init__``
time; weak references keep the registry from extending lifetimes.
"""

import weakref

_OP_REGISTRY: "weakref.WeakValueDictionary[int, object]" = weakref.WeakValueDictionary()


def register_instance(op: object) -> int:
    """Register ``op`` and return the key its dispatch custom op passes back."""
    key = id(op)
    _OP_REGISTRY[key] = op
    return key


def get_instance(key: int) -> object:
    """Resolve a key registered by :func:`register_instance`."""
    return _OP_REGISTRY[key]
