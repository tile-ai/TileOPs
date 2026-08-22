"""Lets a torch.compile dispatch boundary find the op instance behind it.

An operator's schema has no type for an op object, so ``self`` cannot cross the boundary.
The op passes its key instead, and the operator body trades the key back for the instance:

    class FooOp(Op):
        def forward(self, x):
            return _foo(x, self._instance_key)

    @torch.library.custom_op("tileops::foo", mutates_args=())
    def _foo(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        return get_instance(instance_key)._eager_forward(x)

``Op.dispatch_kernel`` assigns ``self._instance_key`` during ``__init__``, so an op gets a
key without writing any registration code. Keys read as ``RMSNormFwdOp#3``, so a key in a
graph dump or traceback says whose it is.

The invariant this exists to keep: a dynamo-traced ``forward`` must not construct kernels or
enter a TileLang builder. Everything past the operator body is untraced, so cache lookup,
kernel construction and launch belong there.

Two properties are load-bearing. The key is a ``str`` because dynamo treats string operator
arguments as compile-time constants, while an ``int`` is generalized to an unhashable
``SymInt``. A key is never reused, not even after its op is collected, because inductor bakes
the fake's output shape into the artifact.
"""

import itertools
import weakref

_OP_REGISTRY: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()
_KEY_COUNTER = itertools.count()


def register_instance(op: object) -> str:
    """Register ``op`` and return the key its dispatch custom op passes back."""
    # ``#`` cannot appear in a class name, so no two classes can produce the same key.
    key = f"{type(op).__name__}#{next(_KEY_COUNTER)}"
    _OP_REGISTRY[key] = op
    return key


def get_instance(key: str) -> object:
    """Resolve a key registered by :func:`register_instance`."""
    return _OP_REGISTRY[key]
