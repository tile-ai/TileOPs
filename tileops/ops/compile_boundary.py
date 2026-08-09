"""Opaque dispatch boundary for torch.compile.

Invariant: a dynamo-traced ``Op.forward`` must not construct kernels or
enter a TileLang builder. Lazy-dispatch ops route ``forward`` through a
``torch.library.custom_op`` whose eager body resolves the instance here
and runs the untraced path (cache lookup, kernel construction, launch).

``Op.dispatch_kernel`` registers every conforming op at ``__init__``
time; weak references keep the registry from extending lifetimes. Keys
are strings because dynamo treats string custom-op arguments as static
constants — an ``int`` key is generalized to an unhashable ``SymInt``
once a second instance compiles through the same frame.

Being a constant is also why keys must never repeat: inductor bakes the
fake's output shape into the artifact, so an op reaching a used key
inherits the first one's shapes. ``id()`` is an address and gets reissued.
"""

import itertools
import weakref

_OP_REGISTRY: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()
_KEY_COUNTER = itertools.count()


def register_instance(op: object) -> str:
    """Register ``op`` and return the key its dispatch custom op passes back."""
    key = f"op{next(_KEY_COUNTER)}"
    _OP_REGISTRY[key] = op
    return key


def get_instance(key: str) -> object:
    """Resolve a key registered by :func:`register_instance`."""
    return _OP_REGISTRY[key]
