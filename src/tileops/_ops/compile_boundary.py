"""Resolve an op instance inside a ``torch.compile`` custom-op boundary.

The schema passes a string key because it cannot carry ``self`` and Dynamo keeps string
arguments constant. The registered instance runs outside tracing, keeping kernel
construction out of the graph.

Inductor caches a compiled graph on disk with the fake's output shapes in it, and the key is
one of the graph's constants. A key is ``<class>#<state>-<n>``: the class's qualified name, a
digest of the state the fake reads, and the instance's ordinal among the instances of its class
in this process. The ordinal resolves a key to one instance; the digest makes two processes
agree on a key only when their instances' fakes agree, so a later process reuses an earlier
one's graphs. A state holding a value with no stable encoding is digested with a token of this
process, so graphs holding its key are never reused by another.

Assigning a param re-keys its op, so a compiled callable retraces; a param edited in place
leaves the key stale and is refused when a compile traces the op.
"""

import enum
import hashlib
import itertools
import sys
import uuid
import weakref
from collections.abc import Iterable

import torch

from tileops.manifest._values import ADTValue

_OP_REGISTRY: "weakref.WeakValueDictionary[str, object]" = weakref.WeakValueDictionary()
_ORDINALS: dict[str, "itertools.count[int]"] = {}
_RUN = uuid.uuid4().hex


def register_instance(op: object) -> str:
    """Register ``op`` and return its identity, ``<class>-<n>``, which `instance_key` extends."""
    name = _qualified_name(type(op))
    identity = f"{name}-{next(_ORDINALS.setdefault(name, itertools.count()))}"
    _OP_REGISTRY[identity] = op
    return identity


def instance_key(identity: str, state: Iterable[object]) -> str:
    """The key of the instance registered as *identity* while it holds *state*."""
    name, _, ordinal = identity.rpartition("-")
    local = [name.startswith("~")]
    text = "\0".join(_encode(v, local) for v in state)
    if local[0]:
        text = f"{_RUN}\0{text}"
    return f"{name}#{hashlib.sha256(text.encode()).hexdigest()[:32]}-{ordinal}"


def get_instance(key: str) -> object:
    """Resolve a key `instance_key` returned.

    Raises:
        RuntimeError: The op instance was garbage-collected while a compiled callable still
            holds its key.
    """
    # The digest and ordinal hold neither `#` nor `-`; a class name may hold both.
    name, _, rest = key.rpartition("#")
    op = _OP_REGISTRY.get(f"{name}-{rest.rpartition('-')[2]}")
    if op is None:
        raise RuntimeError(
            f"the op instance {name.rpartition('.')[2]} that this compiled callable calls no "
            f"longer exists; keep the op instance alive as long as the compiled callable"
        )
    return op


def _qualified_name(cls: type) -> str:
    """``module.qualname`` when that path names *cls*; otherwise it prefixed with ``~``, which
    marks a class another process may give the same name to a different body."""
    name = f"{cls.__module__}.{cls.__qualname__}"
    found = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        found = getattr(found, part, None)
    return name if found is cls else f"~{name}"


def _encode(value: object, local: list) -> str:
    """*value* as text equal across processes exactly when the values are equal.

    A value nothing describes beyond its identity is encoded by its `id`, and *local* is set,
    so that the text is compared only within this process.
    """
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return f"{kind.__name__}:{value!r}"
    if kind in (torch.dtype, torch.device):
        return f"{kind.__name__}:{value}"
    if kind in (torch.Tensor, torch.nn.Parameter):
        # The fake reads a construction-time tensor's layout, never its contents.
        return f"Tensor:{tuple(value.shape)}:{value.stride()}:{value.dtype}:{value.device}"
    if isinstance(value, (type, enum.Enum)):
        cls = value if isinstance(value, type) else kind
        name = _qualified_name(cls)
        local[0] |= name.startswith("~")
        if isinstance(value, type):
            return name
        # Expressions read a member's value, so it is part of what the fake reads.
        return f"{name}.{value.name}={_encode(value.value, local)}"
    if isinstance(value, (list, tuple, dict)):
        if isinstance(value, dict):
            items = (f"{_encode(k, local)}={_encode(v, local)}" for k, v in value.items())
        else:
            items = (_encode(v, local) for v in value)
        text = f"({','.join(items)})"
        if kind in (list, tuple, dict):
            return f"{kind.__name__}{text}"
        # A subclass is read through its contents and its own fields.
        return f"{_encode(kind, local)}{text}{_encode(getattr(value, '__dict__', {}), local)}"
    if kind is ADTValue:
        return f"ADT:{value.adt}.{value.kind}{_encode(value.fields, local)}"
    fields = getattr(value, "__dict__", None)
    # A builtin base, such as an `int` subclass's, holds state its `__dict__` does not show.
    builtin = any(c.__module__ == "builtins" for c in kind.__mro__[:-1])
    if type(fields) is not dict or builtin or hasattr(kind, "__slots__"):
        local[0] = True
        return f"@{id(value)}"
    return f"{_encode(kind, local)}{_encode(fields, local)}"
