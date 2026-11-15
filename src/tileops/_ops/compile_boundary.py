"""Resolve an op instance inside a ``torch.compile`` custom-op boundary.

The schema passes a string key because it cannot carry ``self`` and Dynamo keeps string
arguments constant. The registered instance runs outside tracing, keeping kernel
construction out of the graph.

Inductor caches a compiled graph on disk with the fake's output shapes in it, and the key is
one of the graph's constants. A key is ``<class>#<state>-<n>``: the class's qualified name, a
digest of the state the fake reads, and the instance's ordinal among the instances of its class
in this process. The ordinal resolves a key to one instance; the digest makes two processes
agree on a key only when their instances' fakes agree, so a later process reuses an earlier
one's graphs. A class another process may define otherwise is digested with a token of this
process, so graphs holding its key are never reused by another.

Assigning a param re-keys its op, so a compiled callable retraces; a param edited in place
leaves the key stale, which a compile tracing the op refuses. A state holding a value the
encoding does not read keys on a per-process identity: never shared, and never re-keyed.
"""

import dataclasses
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
    encoder = _Encoder(local=name.startswith("~"))
    try:
        text = "\0".join(encoder.encode(v, 0) for v in state)
    except _OpaqueStateError:
        text = _RUN
    if encoder.local:
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
    """``module.qualname`` when that path names *cls*, else that name prefixed with ``~``.

    The ``~`` marks a class another process may give the same name to a different body.
    """
    name = f"{cls.__module__}.{cls.__qualname__}"
    found = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        found = getattr(found, part, None)
    return name if found is cls else f"~{name}"


class _OpaqueStateError(Exception):
    """The state holds a value the encoding does not read."""


class _Encoder:
    """Encodes values as text equal across processes exactly when the values are equal.

    It reads the values a manifest param holds: scalars, dtypes, devices, strided tensor
    layouts, enum members, types, lists, tuples, dicts, ADT literals and dataclasses, nested at
    most `_MAX_DEPTH` deep. Any other value, a subclass of one of these among them, raises
    `_OpaqueStateError`. A class another process may define otherwise sets `local`.
    """

    # Past this nesting the encoding would approach the interpreter's recursion limit.
    _MAX_DEPTH = 64
    # Stands for a dataclass field never set.
    _UNSET = object()

    def __init__(self, local: bool) -> None:
        """Start with `local` already set when the class itself is local."""
        self.local = local
        # The containers being encoded, outermost first; meeting one again is a cycle.
        self._open: list[object] = []

    def encode(self, value: object, depth: int) -> str:
        """*value*, found *depth* levels down, as text.

        Raises:
            _OpaqueStateError: *value* holds a value the encoding does not read.
        """
        if depth > self._MAX_DEPTH:
            raise _OpaqueStateError
        depth += 1
        kind = type(value)
        if kind is int:
            # Hexadecimal has no digit limit, where `repr` refuses a long integer.
            return f"int:{value:x}"
        if value is None or kind in (bool, float, str):
            return f"{kind.__name__}:{value!r}"
        if kind in (torch.dtype, torch.device):
            return f"{kind.__name__}:{value}"
        if isinstance(value, (type, enum.Enum)):
            cls = value if isinstance(value, type) else kind
            name = _qualified_name(cls)
            self.local |= name.startswith("~")
            if isinstance(value, type):
                return name
            # Expressions read a member's value, so it is part of what the fake reads.
            return f"{name}.{value.name}={self.encode(value.value, depth)}"
        if kind not in (list, tuple, dict, ADTValue) and not dataclasses.is_dataclass(value):
            raise _OpaqueStateError
        for index, open_value in enumerate(self._open):
            if open_value is value:
                return f"^{index}"
        self._open.append(value)
        try:
            if kind is dict:
                items = (
                    f"{self.encode(k, depth)}={self.encode(v, depth)}" for k, v in value.items()
                )
                return f"dict({','.join(items)})"
            if kind in (list, tuple):
                return f"{kind.__name__}({','.join(self.encode(v, depth) for v in value)})"
            # An attribute assigned on an ADT literal shadows the field of its name, so all of
            # them count; a dataclass field never set is marked apart from any value.
            items = (
                vars(value).items()
                if kind is ADTValue
                else (
                    (f.name, getattr(value, f.name, self._UNSET)) for f in dataclasses.fields(value)
                )
            )
            fields = ",".join(
                f"{n}=" + ("unset" if v is self._UNSET else self.encode(v, depth)) for n, v in items
            )
            return f"{self.encode(kind, depth)}({fields})"
        finally:
            self._open.pop()
