"""Kinds of the manifest expression language (docs/design/manifest.md § Signature).

A `Kind` is structured: a sequence carries its element kind and fixed length, a string its
literal set, a union its members. `parse_type` reads a parameter's `type` and `parse_spec` a kind
as the specification writes it; `fits`, `join` and `comparable` compose them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dtype_rules import DTYPE_BITS

__all__ = ["Kind", "common", "comparable", "fits", "join", "ordered", "parse_spec", "parse_type"]


@dataclass(frozen=True)
class Kind:
    # One of Int, Bool, Str, DType, Seq, Maybe, Union, ADT, None, Value.
    tag: str
    # Int: never negative (`Dim`).
    nonneg: bool = False
    # Str: its literals; DType: its members; None where any value is possible.
    values: frozenset | None = None
    # Seq: its element kind (None for the empty tuple); Maybe: its payload.
    item: Kind | None = None
    # Seq: its fixed length, where known.
    length: int | None = None
    members: tuple = field(default=())
    # ADT: its name.
    name: str | None = None

    def __str__(self) -> str:  # noqa: C901 - one case per tag
        if self.tag == "Int":
            return "Dim" if self.nonneg else "Int"
        if self.tag == "Str":
            return " | ".join(f"'{v}'" for v in sorted(self.values)) if self.values else "Str"
        if self.tag == "Seq":
            if self.item is None:
                return "Seq[?]"
            text = "Shape" if self.item == DIM else f"Seq[{self.item}]"
            return f"{text} of {self.length}" if self.length is not None else text
        if self.tag == "Maybe":
            return f"Maybe[{self.item}]"
        if self.tag == "Union":
            return " | ".join(map(str, self.members))
        if self.tag == "ADT":
            return self.name
        return "NoneType" if self.tag == "None" else self.tag

    def payload(self) -> Kind:
        return self.item if self.tag == "Maybe" else self

    def sequence(self) -> Kind | None:
        """This kind as one sequence kind: a union of sequences joins its members; else None."""
        if self.tag == "Seq":
            return self
        if self.tag == "Union" and all(m.tag == "Seq" for m in self.members):
            view = self.members[0]
            for m in self.members[1:]:
                view = join(view, m)
            return view
        return None


INT, DIM, BOOL = Kind("Int"), Kind("Int", nonneg=True), Kind("Bool")
STR, DTYPE, NONE, VALUE = Kind("Str"), Kind("DType"), Kind("None"), Kind("Value")


def seq(item: Kind | None, length: int | None = None) -> Kind:
    return Kind("Seq", item=item, length=length)


def literals(values) -> Kind:
    return Kind("Str", values=frozenset(values))


def dtypes(values) -> Kind:
    return Kind("DType", values=frozenset(values))


def adt(name: str) -> Kind:
    return Kind("ADT", name=name)


def union(*kinds: Kind) -> Kind:
    """The union of `kinds`: flattened, string literals and dtype members merged, `None` to Maybe."""
    flat: list[Kind] = []
    for k in kinds:
        flat.extend(k.members if k.tag == "Union" else [k.item, NONE] if k.tag == "Maybe" else [k])
    optional = any(k.tag == "None" for k in flat)
    rest: list[Kind] = []
    for k in (k for k in flat if k.tag != "None"):
        if k in rest:
            continue
        same = next((r for r in rest if r.tag == k.tag and k.tag in ("Str", "DType")), None)
        if same is not None:
            merged = None if same.values is None or k.values is None else same.values | k.values
            rest[rest.index(same)] = Kind(k.tag, values=merged)
        elif k.tag == "Int" and any(r.tag == "Int" for r in rest):
            rest = [r for r in rest if r.tag != "Int"] + [INT]
        else:
            rest.append(k)
    inner = rest[0] if len(rest) == 1 else Kind("Union", members=tuple(rest)) if rest else None
    if inner is None:
        return NONE
    return Kind("Maybe", item=inner) if optional else inner


def _split(text: str) -> list[str]:
    """The top-level members of `A | B`, splitting only outside brackets and quotes."""
    parts, depth, start, quoted = [], 0, 0, False
    for i, c in enumerate(text):
        quoted ^= c == "'"
        depth += 0 if quoted else {"[": 1, "]": -1}.get(c, 0)
        if c == "|" and depth == 0 and not quoted:
            parts.append(text[start:i].strip())
            start = i + 1
    return [*parts, text[start:].strip()]


def _quoted(text: str) -> bool:
    return len(text) >= 2 and text[0] == text[-1] == "'" and "'" not in text[1:-1]


def _bracket(text: str, head: str) -> str | None:
    """`X` of `head[X]`, or None when `text` is not that form."""
    if text.startswith(f"{head}[") and text.endswith("]"):
        return text[len(head) + 1 : -1]
    return None


def parse_type(text: object, adts: dict) -> Kind:  # noqa: C901 - one case per table-3 row
    """A parameter `type` as a kind; `ValueError` if malformed."""
    members = _split(str(text).strip())
    if len(members) > 1:
        return union(*(parse_type(m, adts) for m in members))
    t = members[0]
    simple = {"int": INT, "bool": BOOL, "None": NONE, "torch.dtype": DTYPE, "str": STR}
    if t in simple:
        return simple[t]
    if t in ("float", "Number", "dict", "torch.Tensor"):
        return VALUE
    if _quoted(t):
        return literals([t[1:-1]])
    if t in DTYPE_BITS:
        return dtypes([t])
    if t in adts:
        return adt(t)
    # A sequence holds ints, fixed-length when a tuple lists them.
    if _bracket(t, "list") == "int":
        return seq(INT)
    if (inner := _bracket(t, "tuple")) is not None:
        items = _commas(inner)
        if items == ["int", "..."]:
            return seq(INT)
        if items and all(i == "int" for i in items):
            return seq(INT, len(items))
        raise ValueError(f"{text!r}: a tuple type is tuple[int, ...] or lists int items")
    if t.replace(".", "_").isidentifier():
        return VALUE  # another Python object
    raise ValueError(f"{text!r} is not a parameter type")


def _commas(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        depth += {"[": 1, "]": -1}.get(c, 0)
        if c == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    return [*parts, text[start:].strip()]


def parse_spec(text: str, adts: dict | None = None) -> Kind:  # noqa: C901 - one case per form
    """A kind as the tables write it: `Dim`, `Seq[Int]`, `Maybe[X]`, `DType[a | b]`, unions."""
    adts = adts or {}
    members = _split(text.strip())
    if len(members) > 1 and not all(_quoted(m) for m in members):
        return union(*(parse_spec(m, adts) for m in members))
    if all(_quoted(m) for m in members):
        return literals(m[1:-1] for m in members)
    t = members[0]
    simple = {
        "Int": INT,
        "Dim": DIM,
        "Bool": BOOL,
        "Str": STR,
        "DType": DTYPE,
        "None": NONE,
        "Value": VALUE,
        "Shape": seq(DIM),
        "Seq": seq(VALUE),
        "Axes": union(INT, seq(INT), NONE),
        "ADT": Kind("ADT"),
    }
    if t in simple:
        return simple[t]
    if t in adts:
        return adt(t)
    if (inner := _bracket(t, "DType")) is not None:
        names = [m.strip() for m in inner.split("|")]
        if all(n in DTYPE_BITS for n in names):
            return dtypes(names)
    if (inner := _bracket(t, "Seq")) is not None:
        return seq(parse_spec(inner, adts))
    if (inner := _bracket(t, "Maybe")) is not None:
        return Kind("Maybe", item=parse_spec(inner, adts))
    raise ValueError(f"{text!r} is not a kind")


def fits(kind: Kind | None, expected: Kind) -> bool:  # noqa: C901 - one case per tag
    """Whether every value of `kind` is a value of `expected`; an unknown kind was reported."""
    if kind is None:
        return True
    if kind.tag == "Union":
        return all(fits(m, expected) for m in kind.members)
    if kind.tag == "Maybe":
        return fits(NONE, expected) and fits(kind.item, expected)
    if expected.tag == "Union":
        return any(fits(kind, m) for m in expected.members)
    if expected.tag == "Maybe":
        return kind.tag == "None" or fits(kind, expected.item)
    if expected.tag == "Value":
        return True
    if kind.tag != expected.tag:
        return False
    if kind.tag == "Int":
        return True  # a negative value on a `Dim` axis is caught at run time
    if kind.tag in ("Str", "DType"):
        return expected.values is None or (
            kind.values is not None and kind.values <= expected.values
        )
    if kind.tag == "Seq":
        if expected.length is not None and kind.length not in (None, expected.length):
            return False
        return kind.item is None or fits(kind.item, expected.item)
    if kind.tag == "ADT":
        return expected.name is None or kind.name == expected.name
    return True


def join(a: Kind | None, b: Kind | None) -> Kind | None:
    """The kind of a value that is either `a` or `b`: a conditional's arms, a tuple's items."""
    if a is None or b is None:
        return None
    if a == b:
        return a
    if a.tag == b.tag == "Int":
        return INT
    if a.tag == b.tag == "Seq":
        if a.item is None or b.item is None:
            return seq(a.item or b.item, a.length if a.length == b.length else None)
        return seq(join(a.item, b.item), a.length if a.length == b.length else None)
    return union(a, b)


def common(a: Kind | None, b: Kind | None) -> Kind | None:
    """The common kind of a conditional's arms; None when they have none.

    `Dim` and `Int` give `Int`; literal sets and dtype sets merge; a `None` arm makes a Maybe.
    """
    if a is None or b is None:
        return None
    if a == b:
        return a
    if NONE in (a, b) or "Maybe" in (a.tag, b.tag):
        payload = (
            common(a.payload(), b.payload()) if NONE not in (a, b) else (a if b == NONE else b)
        )
        return None if payload is None else union(payload, NONE)
    if a.tag == b.tag and a.tag in ("Int", "Seq", "Str", "DType"):
        return join(a, b) if a.tag in ("Int", "Seq") else union(a, b)
    return None


def ordered(a: Kind | None, b: Kind | None) -> bool:
    """Whether `a < b` is defined for every pair of members: both Int, or both a sequence."""
    if a is None or b is None or VALUE in (a, b):
        return True
    return all(
        x.tag == y.tag and x.tag in ("Int", "Seq")
        for x in (a.members if a.tag == "Union" else (a,))
        for y in (b.members if b.tag == "Union" else (b,))
    )


def comparable(a: Kind | None, b: Kind | None) -> bool:
    """Whether `a == b` can hold for some values of the two kinds."""
    if a is None or b is None or VALUE in (a, b):
        return True
    for x in a.members or (a,):
        for y in b.members or (b,):
            if x.tag == y.tag == "Str" and x.values and y.values:
                if x.values & y.values:
                    return True
            elif x.tag == y.tag or {x.tag, y.tag} <= {"Maybe", "None"}:
                return True
            if "Maybe" in (x.tag, y.tag) and comparable(x.payload(), y.payload()):
                return True
    return False
