"""Lets a torch.compile dispatch boundary find the op instance behind it.

An operator's schema has no type for an op object, so ``self`` cannot cross the boundary.
The op passes its key instead, and the operator body trades the key back for the instance
and runs the op's ``forward`` inside it:

    @torch.library.custom_op("tileops::foo", mutates_args=())
    def _foo(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        op = get_instance(instance_key)
        return op._serve((x,), op.forward)

    # What the generated ``FooOp._call_boundary`` does, which ``FooOp.__call__`` calls:
    _foo(x, self._instance_key)

``Op.__init__`` assigns ``self._instance_key``, so an op gets a
key without writing any registration code. Keys read as ``RMSNormFwdOp#3``, so a key in a
graph dump or traceback says whose it is.

The invariant this exists to keep: a dynamo trace must not construct kernels or enter a
TileLang builder. ``forward`` runs inside the operator body, which is untraced, so cache
lookup, kernel construction and launch belong there.

Two properties are load-bearing. The key is a ``str`` because dynamo treats string operator
arguments as compile-time constants, while an ``int`` is generalized to an unhashable
``SymInt``. A key must never come back meaning a different instance, in this process or a
later one, because inductor bakes the fake's output shape into an artifact it caches on
disk: a key that came back meaning a different instance would resolve to that shape.
"""

import itertools
import uuid
import weakref

_OP_REGISTRY: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()
_KEY_COUNTER = itertools.count()
# FIXME(staged-rollout): a per-process uuid buys that uniqueness by giving up disk-cache
# reuse entirely.
#
# Broken invariant: the key should be the op's specialization, so that two interchangeable
#   instances share one artifact. It is an identity instead, so no two ever share, and no
#   process reuses a previous one's compile.
# Why: the specialization is `_manifest_params()` digested, and no op on this boundary has
#   had its param set audited as complete; an incomplete one shares an artifact silently.
# Cleanup: key on the digest, which `Op.__init__` can compute since every op assigns its
#   params before calling it, once every op on this boundary has a param set audited as
#   complete, with a digest that refuses a value it cannot hash stably.
_RUN = uuid.uuid4().hex


def register_instance(op: object) -> str:
    """Register ``op`` and return the key its dispatch custom op passes back."""
    # ``#`` cannot appear in a class name, so no two classes can produce the same key.
    key = f"{type(op).__name__}#{_RUN}-{next(_KEY_COUNTER)}"
    _OP_REGISTRY[key] = op
    return key


def get_instance(key: str) -> object:
    """Resolve a key registered by `register_instance`."""
    return _OP_REGISTRY[key]
