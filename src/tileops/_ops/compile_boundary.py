"""Resolve an op instance inside a ``torch.compile`` custom-op boundary.

The schema passes a string key because it cannot carry ``self`` and Dynamo keeps
string arguments constant. The registered instance runs outside tracing, keeping
kernel construction out of the graph. Keys are process-unique so cached artifacts
cannot resolve a shape to a different instance.
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
    """Resolve a key registered by `register_instance`.

    Raises:
        RuntimeError: The op instance was garbage-collected while a compiled callable still
            holds its key.
    """
    op = _OP_REGISTRY.get(key)
    if op is None:
        raise RuntimeError(
            f"the op instance {key.partition('#')[0]} that this compiled callable calls no "
            f"longer exists; keep the op instance alive as long as the compiled callable"
        )
    return op
