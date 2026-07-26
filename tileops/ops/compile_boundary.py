"""Opaque dispatch boundary for torch.compile.

A dynamo-traced ``Op.forward`` must not construct kernels or enter a
TileLang builder: kernel-cache misses build TileLang programs through
machinery dynamo cannot trace. Ops that resolve kernels lazily at call
time therefore route ``forward`` through a ``torch.library.custom_op``
whose eager body looks the instance up here and runs the untraced eager
path, so cache lookup, kernel construction, and launch stay outside the
graph.

Instances are registered by ``Op.dispatch_kernel`` (every conforming op
calls it from ``__init__``); the registry holds weak references and never
extends instance lifetime.
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
