"""Values of parameter types (docs/design/manifest.md § Parameters, § Algebraic Data Types).

`convert` checks one value against a parameter `type` and returns it as expressions read it:
a default, a workload row value, or what a constructor received.
"""

from __future__ import annotations

import importlib

from .dtype_rules import DTYPE_BITS
from .expr import evaluate, parse
from .kinds import split_union

__all__ = ["ADTValue", "convert", "is_integer", "python_class"]


class ADTValue:
    """An ADT literal as expressions read it: `v.kind`, and each field as `v.<field>`."""

    def __init__(self, adt: str, literal: dict):
        ((kind, fields),) = literal.items()
        self.__dict__.update(adt=adt, kind=kind, fields=dict(fields or {}))

    def __getattr__(self, name):
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            raise AttributeError(name) from None


def python_class(path: str):
    """The class a `python` path names."""
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def is_integer(value, minimum=None) -> bool:
    """Whether `value` is an `int`, not a `bool`, and at least `minimum` when one is given."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


class _Rejected:
    """A member that does not take a value, and why, where there is more to say."""

    def __init__(self, reason: str = ""):
        self.reason = reason


_REJECT = _Rejected()


def convert(value, type_text: object, adts: dict):
    """`value` as its parameter `type` takes it; `ValueError` naming the reason when none does.

    A sequence becomes a list, or a tuple for a `tuple[...]` member; a dtype stays its name;
    an ADT literal `{ctor: {field: value}}` becomes an `ADTValue`, and a Python object is
    checked against its constructor's class.
    """
    reasons = []
    for member in split_union(str(type_text)):
        converted = _member(member.replace(" ", ""), value, adts)
        if not isinstance(converted, _Rejected):
            return converted
        if converted.reason:
            reasons.append(converted.reason)
    why = f" ({'; '.join(reasons)})" if reasons else ""
    raise ValueError(f"{value!r} is not a {type_text}{why}")


def _member(member: str, value, adts: dict):  # noqa: C901 - one case per type form
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    simple = {
        "None": value is None,
        "bool": isinstance(value, bool),
        "int": is_integer(value),
        "float": number,
        "Number": isinstance(value, (int, float)),
        "str": isinstance(value, str),
        "dict": isinstance(value, dict),
        "torch.dtype": isinstance(value, str) and value in DTYPE_BITS,
    }
    if member in simple:
        return value if simple[member] else _REJECT
    if member.startswith("'"):
        return value if value == member[1:-1] else _REJECT
    if member in DTYPE_BITS:
        return value if value == member else _REJECT
    if member in adts:
        try:
            return _adt(member, value, adts)
        except ValueError as exc:
            return _Rejected(str(exc))
    for head in ("list[", "tuple["):
        if member.startswith(head):
            return _sequence(member, head, value, adts)
    return value


def _sequence(member: str, head: str, value, adts: dict):
    if not isinstance(value, (list, tuple)):
        return _REJECT
    inner = member[len(head) : -1]
    items = [inner] * len(value)
    if head == "tuple[" and not inner.endswith(",..."):
        items = inner.split(",")
        if len(items) != len(value):
            return _REJECT
    elif head == "tuple[":
        items = [inner.removesuffix(",...")] * len(value)
    out = []
    for text, item in zip(items, value, strict=True):
        converted = next(
            (c for m in text.split("|") if not isinstance(c := _member(m, item, adts), _Rejected)),
            _REJECT,
        )
        if isinstance(converted, _Rejected):
            return _REJECT
        out.append(converted)
    return tuple(out) if head == "tuple[" else out


def _field_ok(text: str, value) -> bool:
    if text.startswith("'"):
        return value in [m[1:-1] for m in split_union(text)]
    if text in ("bool", "Bool"):
        return isinstance(value, bool)
    return is_integer(value, 0 if text == "Dim" else None)


def _adt(name: str, value, adts: dict):
    """An ADT literal checked against its constructor's fields and invariant, as an
    `ADTValue`; a Python object checked against its constructor's class, as itself."""
    ctors = adts[name]["sum"]
    if isinstance(value, dict):
        if len(value) != 1 or next(iter(value)) not in ctors:
            raise ValueError(f"{value!r} is not one of the constructors {sorted(ctors)}")
        ((ctor, fields),) = value.items()
        fields = dict(fields or {})
        view = ADTValue(name, value)
    else:
        ctor = getattr(value, "kind", None)
        if ctor not in ctors or not isinstance(value, python_class(ctors[ctor]["python"])):
            raise ValueError(f"{value!r} is not an object of a {name} constructor")
        declared = ctors[ctor].get("fields") or {}
        missing = [f for f in declared if not hasattr(value, f)]
        if missing:
            raise ValueError(f"{name}: {ctor} lacks fields {missing}")
        fields = {f: getattr(getattr(value, f), "value", getattr(value, f)) for f in declared}
        view = value
    declared = ctors[ctor].get("fields") or {}
    if set(fields) != set(declared):
        raise ValueError(f"{name}: {ctor} takes fields {sorted(declared)}, got {sorted(fields)}")
    for f, v in fields.items():
        decl = declared[f]
        text = str(decl.get("type") if isinstance(decl, dict) else decl)
        if not _field_ok(text, v):
            raise ValueError(f"{name}: {ctor}.{f} = {v!r} is not a {text}")
    invariant = ctors[ctor].get("invariant")
    if invariant is not None and not evaluate(parse(invariant), fields, f"{name}.{ctor} invariant"):
        raise ValueError(f"{name}: {ctor} invariant {invariant} fails")
    return view
