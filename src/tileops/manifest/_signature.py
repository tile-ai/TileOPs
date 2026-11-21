"""A manifest signature and what one discriminant point selects (docs/design/manifest.md § Signature).

`parse_signature` reads an entry's `signature`; `branch` folds it at a point: the shapes and
dtypes of the tensors a call there passes and emits, its refinements and the `let`s they reach.
`unification` orders the inference plan. Nothing here imports an op.
"""

from __future__ import annotations

import ast
import copy
import functools
import itertools
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tileops.manifest._dtype_rules import DTYPE_BITS
from tileops.manifest._expr import (
    OPEN,
    KindEnv,
    SignatureError,
    Substitute,
    bind,
    callee,
    fold,
    infer_kinds,
    member_kind,
    names,
    parse,
    value_at,
    value_names,
)
from tileops.manifest._kinds import (
    BOOL,
    DTYPE,
    NONE,
    VALUE,
    Kind,
    dtypes,
    fits,
    parse_spec,
    parse_type,
    split_union,
    union,
)
from tileops.manifest.primitives import DETERMINING_PREDICATES, PRIMITIVE_KINDS, PRIMITIVES

__all__ = [
    "ITEM",
    "MEMBERS",
    "REQUIRED",
    "SLOT_TYPES",
    "TENSOR_MEMBERS",
    "Signature",
    "SignatureBranch",
    "SignatureError",
    "Step",
    "Tensor",
    "adt_fields",
    "branch",
    "call_scalar_lets",
    "complete_point",
    "components",
    "discriminant_axes",
    "discriminant_groups",
    "dtype_kind",
    "elementwise",
    "expand",
    "field_kind",
    "holds",
    "items",
    "kind_env",
    "localized",
    "member_at",
    "member_of",
    "members",
    "output_emitted",
    "param_kind",
    "parse_signature",
    "points",
    "reach",
    "read_signature",
    "rejecting_rule",
    "tensor_passed",
    "unification",
]


# The members an argument type may hold, as `member(x)` names them.
MEMBERS = ("Tensor", "list[Tensor]", "int", "float", "Number", "bool", "None")
# The members that hold tensors: an argument holding one takes part in tensor inference.
TENSOR_MEMBERS = frozenset({"Tensor", "list[Tensor]"})
# A declaration without `default`: the argument must be passed.
REQUIRED = object()
# The operator-schema type of each member's slot, and its slot-name suffix.
SLOT_TYPES = {
    "Tensor": ("Tensor", ""),
    "list[Tensor]": ("Tensor[]", ""),
    "int": ("SymInt", "__int"),
    "float": ("float", "__float"),
    "Number": ("Scalar", "__number"),
    "bool": ("bool", "__bool"),
}


def members(type_text: object) -> tuple[str, ...]:
    """The members of an argument `type`, in declared order; `SignatureError` outside `MEMBERS`."""
    found = tuple(m.replace(" ", "") for m in split_union(str(type_text).strip()))
    bad = [m for m in found if m not in MEMBERS]
    if bad or len(set(found)) != len(found):
        raise SignatureError(f"type {type_text!r} is not a union of distinct members of {MEMBERS}")
    if "Number" in found and {"int", "float"} & set(found):
        raise SignatureError(f"type {type_text!r}: Number already holds int and float")
    return found


def member_of(value: object, members: tuple[str, ...]) -> str | None:
    """The member of `members` that holds `value`; None when none does.

    A symbolic `torch.SymBool`, `SymInt` or `SymFloat` is held as the scalar it stands for.
    """
    torch = sys.modules.get("torch")
    symbolic = (torch.SymBool, torch.SymInt, torch.SymFloat) if torch is not None else ()
    if value is None:
        found = ["None"]
    elif isinstance(value, (list, tuple)):
        found = ["list[Tensor]"]
    elif torch is not None and isinstance(value, torch.Tensor):
        found = ["Tensor"]
    elif isinstance(value, (bool, *symbolic[:1])):
        found = ["bool"]
    elif isinstance(value, (int, *symbolic[1:2])):
        # A `float` member takes an int, as Python's `float` annotation and the `float` slot do.
        found = ["int", "Number", "float"]
    elif isinstance(value, (float, *symbolic[2:])):
        found = ["float", "Number"]
    else:
        found = []
    return next((m for m in found if m in members), None)


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: str
    # The argument type; `Tensor` stands for this declaration's `Tensor[dtype, shape]`.
    type: str = "Tensor"
    nullable: str | None = None
    buffer: bool = False
    mutated: bool | str = False
    values: str | None = None
    requires: tuple[str, ...] = ()
    cpu: bool = False
    write_only: bool = False
    alias: str | None = None
    contiguous: bool = False
    kw_only: bool = False
    overlaps: tuple[str, ...] = ()
    default: object = REQUIRED

    @property
    def members(self) -> tuple[str, ...]:
        return members(self.type)

    @property
    def sequence(self) -> bool:
        """Whether the argument is a tensor sequence, `list[Tensor]`."""
        return "list[Tensor]" in self.members

    @property
    def slots(self) -> list[tuple[str, str, str]]:
        """The operator-schema slots, each `(type, name, member)`.

        One member takes one slot as it is; several members take a nullable slot per member
        that is not None.
        """
        nullable = len(self.members) > 1
        return [
            (f"{SLOT_TYPES[m][0]}{'?' if nullable else ''}", f"{self.name}{SLOT_TYPES[m][1]}", m)
            for m in self.members
            if m != "None"
        ]

    @property
    def scalars(self) -> tuple[str, ...]:
        """The members that are neither a tensor nor None."""
        return tuple(m for m in self.members if m not in TENSOR_MEMBERS and m != "None")

    @property
    def derived(self) -> bool:
        """Whether a determining predicate fixes this tensor's contents from other tensors."""
        return any(r.split("(", 1)[0].strip() in DETERMINING_PREDICATES for r in self.requires)


@dataclass
class Signature:
    """An entry's parsed `signature`: its indices, parameters, tensors, `let`s and rules."""

    name: str
    forall: dict[str, str]
    params: dict[str, dict[str, Any]]
    inputs: dict[str, Tensor]
    outputs: dict[str, Tensor]
    types: dict[str, dict[str, Any]]
    let: dict[str, str]
    rules: list[str]
    adts: dict[str, dict[str, Any]] = field(default_factory=dict)
    dtype_combos: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scalar_inputs(self) -> set[str]:
        """The call-time inputs with a scalar member."""
        return {t for t, d in self.inputs.items() if d.scalars}

    def written(self, point: dict[str, object], present: set[str]) -> frozenset[str]:
        """The inputs a call at `point` writes: `mutated` where its condition holds, `write_only`."""
        return frozenset(
            t.name
            for t in self.inputs.values()
            if t.name in present
            and (
                t.write_only
                or t.mutated is True
                or (isinstance(t.mutated, str) and value_at(parse(t.mutated), point) is True)
            )
        )

    def forward_arguments[T](self, inputs: Iterable[T]) -> tuple[tuple[T, ...], dict[str, T]]:
        """*inputs*, call-time inputs in `inputs` order, as `forward` takes them.

        Returns the positional ones in order, and each `kw_only` one by name. *inputs* may stop
        before the last input; the ones it leaves out are not passed.
        """
        inputs = tuple(inputs)
        if len(inputs) > len(self.inputs):
            raise TypeError(
                f"{self.name}: {len(inputs)} inputs, the signature has {len(self.inputs)}"
            )
        if not any(t.kw_only for t in self.inputs.values()):
            return inputs, {}
        named = list(zip(list(self.inputs.values())[: len(inputs)], inputs, strict=True))
        return (
            tuple(v for t, v in named if not t.kw_only),
            {t.name: v for t, v in named if t.kw_only},
        )

    def kind(self, name: str) -> Kind | None:
        """The kind of index or parameter `name`; None for any other name."""
        if name in self.forall:
            return _spec(self.forall[name])
        if name in self.params:
            return param_kind(self.params[name].get("type"), self.adts)
        return None


def param_kind(type_text: object, adts: dict) -> Kind:
    """The kind a parameter `type` maps to; `Value` when the type is malformed."""
    try:
        return parse_type(type_text, adts)
    except ValueError:
        return VALUE


@functools.lru_cache(maxsize=1024)
def _spec(text: str) -> Kind:
    return parse_spec(text)


def _spec_or_none(text: str) -> Kind | None:
    try:
        return parse_spec(text)
    except ValueError:
        return None


def field_kind(decl: object) -> Kind | None:
    """The kind an ADT field declares, bare or as `{type, python}`: Dim, Int, Bool or an enum."""
    text = str(decl.get("type") if isinstance(decl, dict) else decl)
    kind = _spec_or_none({"int": "Int", "bool": "Bool"}.get(text, text))
    if kind is None or not (kind.tag in ("Int", "Bool") or (kind.tag == "Str" and kind.values)):
        return None
    return kind


# The prefix of the name an element's item takes in that element's unification.
ITEM = "_item_"


def elementwise(sig: Signature) -> frozenset:
    """The indices a `list[Tensor]` declaration reads per element.

    These are the indices of a `Seq` kind other than a generator's value list.
    """
    return frozenset(n for n, k in sig.forall.items() if k.startswith("Seq[") and k != "Seq[Int]")


def items(node: ast.expr, each: frozenset) -> list[str]:
    """The element-wise indices `node` reads, in order of first reading."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in each and n.id not in found:
            found.append(n.id)
    return found


def localized(node: ast.expr, each) -> ast.expr:
    """An element type with each element-wise index `X` read as the element's item `_item_X`."""
    return ast.fix_missing_locations(
        Substitute({x: ast.Name(f"{ITEM}{x}", ast.Load()) for x in each}).visit(copy.deepcopy(node))
    )


def _identifier(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier()


def _tensor(name: str, decl: object) -> Tensor:
    if (
        not isinstance(decl, dict)
        or not isinstance(decl.get("dtype"), str)
        or not isinstance(decl.get("shape"), str)
    ):
        raise SignatureError(f"tensor {name!r} needs string `dtype` and `shape`")
    try:
        members(decl.get("type", "Tensor"))
    except SignatureError as exc:
        raise SignatureError(f"tensor {name!r}: {exc}") from None
    return Tensor(
        name,
        decl["dtype"],
        decl["shape"],
        decl.get("type", "Tensor"),
        decl.get("nullable"),
        decl.get("buffer") == "out",
        decl.get("mutated", False),
        decl.get("values"),
        tuple(decl.get("requires", ()) or ()),
        decl.get("device") == "cpu",
        decl.get("write_only") is True,
        decl.get("alias"),
        decl.get("contiguous") is True,
        decl.get("kw_only") is True,
        tuple(decl.get("overlaps") or ()),
        decl.get("default", REQUIRED),
    )


def read_signature(name: str, entry: dict, adts: dict) -> tuple[Signature, list[str]]:
    """A `Signature` of the well-formed declarations, and why each other one was left out."""
    sig = entry.get("signature")
    if not isinstance(sig, dict):
        raise SignatureError("`signature` must be a mapping")
    problems, sections = [], {}
    for key in ("forall", "params", "inputs", "outputs", "types", "let"):
        section = {} if sig.get(key) is None else sig[key]
        if not isinstance(section, dict):
            problems.append(f"`{key}` must be a mapping")
            section = {}
        bad = sorted((k for k in section if not _identifier(k)), key=repr)
        if bad:
            problems.append(f"`{key}` names {[repr(k) for k in bad]}, which are not identifiers")
        sections[key] = {k: v for k, v in section.items() if _identifier(k)}
    lists = {}
    for key in ("shape_rules", "dtype_combos"):
        lists[key] = [] if sig.get(key) is None else sig[key]
        if not isinstance(lists[key], list):
            problems.append(f"`{key}` must be a list")
            lists[key] = []
    forall = {}
    for index, kind in sections["forall"].items():
        dtype = _spec_or_none(str(kind))
        elem = _spec_or_none(str(kind)[4:-1]) if str(kind).startswith("Seq[DType[") else None
        if kind in ("Dim", "Shape", "Seq[Int]", "Seq[Dim]", "Seq[Shape]") or any(
            d is not None and d.tag == "DType" and d.values for d in (dtype, elem)
        ):
            forall[index] = kind
        else:
            problems.append(f"forall index {index!r} has unknown kind {kind!r}")
    params, tensors = {}, {"inputs": {}, "outputs": {}}
    for p, decl in sections["params"].items():
        if isinstance(decl, dict) and (
            "shape" in decl or re.search(r"\bTensor\b", str(decl.get("type", "")))
        ):
            problems.append(f"parameter {p!r} declares a tensor: a tensor is a call-time input")
        elif isinstance(decl, dict) and "type" in decl:
            params[p] = decl
        else:
            problems.append(f"parameter {p!r} needs a `type`")
    for group in ("inputs", "outputs"):
        for n, decl in sections[group].items():
            try:
                tensor = _tensor(n, decl)
            except SignatureError as exc:
                problems.append(str(exc))
                continue
            tensors[group][n] = tensor
    return Signature(
        name,
        forall,
        params,
        tensors["inputs"],
        tensors["outputs"],
        dict(sections["types"]),
        dict(sections["let"]),
        list(lists["shape_rules"]),
        adts,
        list(lists["dtype_combos"]),
    ), problems


def parse_signature(name: str, entry: dict, adts: dict) -> Signature:
    """Build a `Signature` from a manifest entry; raises `SignatureError` on a malformed field."""
    sig, problems = read_signature(name, entry, adts)
    if problems:
        raise SignatureError(problems[0])
    return sig


def call_scalar_lets(sig: Signature) -> set[str]:
    """The `let`s that read a call-time scalar member, directly or through another `let`."""
    scalars = sig.scalar_inputs
    reads = {}
    for n, e in sig.let.items():
        try:
            reads[n] = value_names(parse(e))
        except SignatureError:
            reads[n] = set()
    found: set[str] = set()
    while more := {n for n, r in reads.items() if n not in found and r & (scalars | found)}:
        found |= more
    return found


# ---------------------------------------------------------------- kinds


def adt_fields(adts: dict) -> dict:
    """ADT name -> constructor -> field -> kind."""
    return {
        name: {
            ctor: {f: field_kind(d) for f, d in ((spec or {}).get("fields") or {}).items()}
            for ctor, spec in ((adt or {}).get("sum") or {}).items()
        }
        for name, adt in adts.items()
    }


def kind_env(
    sig: Signature, lets: dict[str, ast.expr], kinds: dict | None = None
) -> tuple[KindEnv, list[str]]:
    """The `KindEnv` of `sig`, and the misuse inferring its lets finds.

    The env holds every index, parameter and `let`, lets in dependency order. `kinds`
    overrides what the declarations give.
    """
    tags = {t.name: t.members for t in sig.inputs.values()}
    # A call-time argument with a scalar member is a value where narrowed to that member.
    arguments = {t: member_kind(m) for t, m in tags.items() if sig.inputs[t].scalars}
    base = {n: sig.kind(n) for n in (*sig.forall, *sig.params)} | arguments | (kinds or {})
    env = KindEnv(
        base,
        adt_fields(sig.adts),
        frozenset(sig.types),
        frozenset({*sig.inputs, *sig.outputs})
        | ({"out"} if any(t.buffer for t in sig.outputs.values()) else set()),
        frozenset(p for p in sig.params if sig.kind(p).tag == "Maybe"),
        tags=tags,
    )
    errors: list[str] = []
    pending = list(lets)
    while pending:
        ready = [n for n in pending if not names(lets[n]) & set(pending) - {n}] or pending[:1]
        for n in ready:
            kind, found = infer_kinds(lets[n], env, f"let {n}")
            env.kinds[n] = kind
            errors += found
            pending.remove(n)
    return env, errors


def dtype_kind(
    sig: Signature, node: ast.expr, errors: list[str], each: frozenset = frozenset(), top=True
) -> Kind | None:
    """The kind of a dtype expression; misuse goes to `errors`.

    An element-wise index in `each` is its item, a `list[Tensor]` element's dtype; any other
    `Seq[DType]` index is a list of dtypes, which only a primitive argument takes.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return NONE
    if isinstance(node, ast.Name):
        if node.id in DTYPE_BITS and node.id not in {*sig.forall, *sig.params}:
            return dtypes([node.id])
        kind = sig.kind(node.id)
        if kind is not None and kind.tag == "Seq" and kind.item is not None:
            if node.id in each:
                return kind.item
            if not top:
                return kind
        if kind is not None and kind.payload().tag == "DType":
            return kind
        errors.append(f"{node.id!r} is not a DType index, dtype parameter or registered dtype")
        return None
    name = callee(node.func) if isinstance(node, ast.Call) else None
    formal, result = PRIMITIVE_KINDS.get(name, ((), None))
    if result != "DType":
        errors.append(f"{ast.unparse(node)!r} is not a dtype expression")
        return None
    if len(node.args) != len(formal) or node.keywords:
        errors.append(f"{name} takes {len(formal)} arguments")
        return DTYPE
    domains = []
    for arg, wanted in zip(node.args, formal, strict=True):
        kind = dtype_kind(sig, arg, errors, each, top=False)
        if kind is not None and not fits(kind, _spec(wanted)):
            errors.append(f"argument {ast.unparse(arg)} has kind {kind}, expected {wanted}")
        domains.append(_dtype_values(kind))
    if any(d is None for d in domains):
        return DTYPE
    # Every argument has finite dtypes, so the result's are the primitive over their product;
    # a combination outside the primitive's domain yields nothing, and raises at the call.
    values = set()
    for combo in itertools.product(*domains):
        try:
            values.add(PRIMITIVES[name](*combo))
        except ValueError:
            continue
    if not values:
        errors.append(f"{ast.unparse(node)} has no value over its arguments' dtypes")
        return DTYPE
    return union(*(NONE if v is None else dtypes([v]) for v in values))


def _dtype_values(kind: Kind | None) -> set | None:
    """The values of a dtype expression's kind, None among them; None when unbounded.

    A list of dtypes takes every non-empty subset of its item's, in sorted order: the
    primitives taking one are symmetric.
    """
    if kind is None:
        return None
    if kind.tag == "Seq":
        item = _dtype_values(kind.item)
        if item is None or None in item:
            return None
        pool = sorted(item)
        return {combo for n in range(1, len(pool) + 1) for combo in itertools.combinations(pool, n)}
    if kind.tag == "None":
        return {None}
    payload = kind.payload()
    if payload.tag != "DType" or payload.values is None:
        return None
    return set(payload.values) | ({None} if kind.tag == "Maybe" else set())


# ---------------------------------------------------------------- discriminants


def _finite_axis(p: str, kind: Kind, adts: dict) -> tuple[str | None, list] | None:
    """The values a `Bool`, enum, `Maybe` presence or ADT named *p* takes, as a discriminant axis."""
    if kind == BOOL:
        return (p, [False, True])
    if kind.tag == "Str" and kind.values:
        return (p, sorted(kind.values))
    if kind.tag == "Maybe":
        payload = _finite_axis(p, kind.item, adts) if kind.item.tag == "ADT" else None
        if payload is None:
            return (f"present({p})", [False, True])
        return (
            None,
            [{f"present({p})": False}] + [{f"present({p})": True, **v} for v in payload[1]],
        )
    if kind.tag != "ADT":
        return None
    out = []
    for ctor, spec in (adts[kind.name].get("sum", {}) or {}).items():
        enums = []
        for f, k in (spec.get("fields", {}) or {}).items():
            fk = field_kind(k)
            if fk is not None and fk.tag == "Str":
                enums.append((f, sorted(fk.values)))
            elif fk == BOOL:
                enums.append((f, [False, True]))
        for combo in itertools.product(*[v for _, v in enums]):
            out.append(
                {
                    f"{p}.kind": ctor,
                    **{f"{p}.{f}": c for (f, _), c in zip(enums, combo, strict=True)},
                }
            )
    return (None, out)


def discriminant_axes(sig: Signature) -> dict[str, tuple[str | None, list]]:
    """Each discriminant, by the parameter or tensor it belongs to: its point key and values.

    A key of None marks an ADT axis whose values are partial points.
    """
    axes: dict[str, tuple[str | None, list]] = {}
    for p in sig.params:
        if (axis := _finite_axis(p, sig.kind(p), sig.adts)) is not None:
            axes[p] = axis
    for t in sig.inputs.values():
        if len(t.members) > 1:
            axes[t.name] = (f"member({t.name})", list(t.members))
    if any(t.buffer for t in sig.outputs.values()):
        axes["out"] = ("present(out)", [False, True])
    return axes


def discriminant_groups(sig: Signature) -> list[list[tuple[str | None, list]]]:
    """Discriminant axes grouped by dependency.

    A tensor, a refinement and a `let` each join every name they read; two axes are dependent
    when a chain of shared names links them. An axis nothing reads takes no part.
    """
    group: dict[str, str] = {}

    def root(n: str) -> str:
        while group.setdefault(n, n) != n:
            n = group[n]
        return n

    def link(members: set[str]) -> None:
        first, *rest = sorted(members) or [None]
        for n in rest:
            group[root(n)] = root(first)

    for t in (*sig.inputs.values(), *sig.outputs.values()):
        reads = {t.name} | ({"out"} if t.buffer else set())
        for text in (t.shape, t.dtype, t.nullable, t.mutated):
            if isinstance(text, str):
                reads |= names(parse(text))
        link(reads)
    for r in sig.rules:
        link(names(parse(r)))
    for n, e in sig.let.items():
        link({n} | names(parse(e)))
    axes = discriminant_axes(sig)
    groups: dict[str, list] = {}
    for a in sorted(axes):
        if a in group:
            groups.setdefault(root(a), []).append(axes[a])
    return list(groups.values())


def points(axes: list[tuple[str | None, list]]):
    """Every combination of the values of `axes`, as a point."""
    for combo in itertools.product(*[v for _, v in axes]):
        point = {}
        for (key, _), value in zip(axes, combo, strict=True):
            point.update(value if key is None else {key: value})
        yield point


def _inlined(sig: Signature, node: ast.expr) -> ast.expr:
    """`node` with every `let` it reads replaced by its definition, transitively."""
    for _ in range(len(sig.let) + 1):
        read = names(node) & set(sig.let)
        if not read:
            break
        node = Substitute({n: parse(sig.let[n]) for n in read}).visit(copy.deepcopy(node))
    return node


def _invariant_fails(adts: dict, p: str, kind: Kind, point: dict) -> str | None:
    """The invariant of the constructor ADT-valued *p* holds at `point`, when it is false there.

    For a `Maybe[ADT]` it is the payload's, where it is present.
    """
    if kind.tag == "Maybe" and kind.item.tag == "ADT":
        if point.get(f"present({p})") is not True:
            return None
        kind = kind.item
    ctor = point.get(f"{p}.kind")
    declared = adts.get(kind.name, {}) if kind.tag == "ADT" else {}
    invariant = declared.get("sum", {}).get(ctor, {}).get("invariant")
    if invariant is None:
        return None
    fields = {k[len(p) + 1 :]: v for k, v in point.items() if k.startswith(f"{p}.")}
    if _closed_value(parse(invariant), fields, f"{p} invariant") is False:
        return f"{p}: {invariant}"
    return None


def _closed_value(node: ast.expr, point: dict, where: str):
    """The value of `node` at `point`, or `OPEN` when it reads a name `point` does not fix."""
    return OPEN if names(bind(node, point)) else value_at(node, point, where)


def rejecting_rule(sig: Signature, point: dict) -> str | None:
    """The first domain restriction false at `point`: a refinement reading only discriminants.

    What a refinement reads is judged with the `let`s it reads inlined.
    """
    for i, text in enumerate(sig.rules):
        if _closed_value(_inlined(sig, parse(text)), point, f"shape_rules[{i}]") is False:
            return text
    for p in sig.params:
        failed = _invariant_fails(sig.adts, p, sig.kind(p), point)
        if failed is not None:
            return failed
    return None


def holds(expr: bool | str | None, point: dict) -> bool:
    """A presence or effect condition at `point`; it must read only discriminants."""
    if expr is None or isinstance(expr, bool):
        return expr is not False
    value = _closed_value(parse(expr), point, repr(expr))
    if value is OPEN:
        raise SignatureError(f"{expr!r} reads more than discriminants")
    return bool(value)


def complete_point(sig: Signature, point: dict, strict: bool = True) -> dict:
    """`point` with `present(t)` for every tensor.

    An input's comes from its member, `member(t)` being an axis of its own when its type has
    several; an output's comes from its `nullable` condition.

    With `strict=False` a presence `point` does not settle is left unsettled instead of raising.
    """
    point = dict(point)
    for t in sig.inputs.values():
        if len(t.members) == 1:
            point.setdefault(f"member({t.name})", t.members[0])
        tag = point.get(f"member({t.name})")
        if tag is not None:
            point.setdefault(f"present({t.name})", tag != "None")
    pending = {
        t.name: t.nullable if t.name in sig.outputs else True
        for t in (*sig.inputs.values(), *sig.outputs.values())
        if f"present({t.name})" not in point
    }
    lets = {}
    for n, e in sig.let.items():
        try:
            lets[n] = parse(e)
        except SignatureError:
            continue
    while pending or lets:
        settled = {}
        for name in list(lets):
            value = value_at(lets[name], point, f"let {name}")
            if value is not OPEN:
                settled[name] = value
                del lets[name]
        for name, cond in pending.items():
            if cond is True:
                value = OPEN  # its own discriminant axis, which `point` lacks
            elif not isinstance(cond, str):
                value = True
            else:
                value = _closed_value(
                    _inlined(sig, parse(cond)), point, f"tensor {name!r} presence"
                )
            if value is not OPEN:
                settled[f"present({name})"] = bool(value)
        if not settled and not pending:
            return point
        if not settled:
            if not strict:
                return point
            raise SignatureError(f"presence of {sorted(pending)} reads more than discriminants")
        point.update(settled)
        pending = {n: c for n, c in pending.items() if f"present({n})" not in point}
    return point


def member_at(tensor: Tensor, point: dict) -> str | None:
    """The member a call-time input holds at `point`; None when unsettled."""
    members_ = tensor.members
    return members_[0] if len(members_) == 1 else point.get(f"member({tensor.name})")


def tensor_passed(tensor: Tensor, point: dict) -> bool:
    """Whether a call at `point` passes this declaration's tensor or tensor sequence."""
    return member_at(tensor, point) in TENSOR_MEMBERS


def output_emitted(tensor: Tensor, point: dict) -> bool:
    """Whether a call at `point` returns an output."""
    return holds(tensor.nullable, point)


# ---------------------------------------------------------------- type families


def components(value: object) -> list:
    """A family's `match` or a case's `when` as its list of components."""
    return value if isinstance(value, list) else [value]


def _matches(pattern, value) -> bool:
    if pattern == "_":
        return True
    if isinstance(pattern, dict):
        if not isinstance(value, tuple):
            return False
        ((ctor, fields),) = pattern.items()
        kind, body = value
        return ctor == kind and (fields == "_" or all(body.get(f) == v for f, v in fields.items()))
    return pattern == value


def _discriminant(node: ast.expr, point: dict):
    text = ast.unparse(node)
    if f"{text}.kind" in point:
        prefix = f"{text}."
        return point[f"{text}.kind"], {
            k[len(prefix) :]: v
            for k, v in point.items()
            if k.startswith(prefix) and k != f"{text}.kind"
        }
    value = value_at(node, point, text)
    if value is OPEN:
        raise SignatureError(f"{text!r} is not a finite discriminant")
    return value


def _case(fam: str, family: dict, sub: dict, point: dict) -> str:
    """The `is` of the one case of *family* whose pattern matches its `match` at *point*."""
    got = []
    for k in components(family["match"]):
        node = Substitute(sub).visit(parse(k))
        absent = isinstance(node, ast.Name) and point.get(f"present({node.id})") is False
        # An absent payload has no value; only `_` matches it.
        got.append(OPEN if absent else _discriminant(node, point))
    got = tuple(got)
    hits = []
    for case in family.get("cases", []) or []:
        pattern = components(case["when"])
        if len(pattern) != len(got):
            raise SignatureError(
                f"{fam}: case {pattern} has {len(pattern)} components, match has {len(got)}"
            )
        if all(
            p == "_" or (g is not OPEN and _matches(p, g))
            for p, g in zip(pattern, got, strict=True)
        ):
            hits.append(case["is"])
    if len(hits) != 1:
        raise SignatureError(f"{fam}: {len(hits)} cases match {got}, need exactly one")
    return hits[0]


def expand(sig: Signature, shape: str, point: dict) -> ast.List:
    """A shape term with every type-family application replaced by the branch `point` selects."""
    node = parse(shape)
    while (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in sig.types
    ):
        family = sig.types[node.value.id]
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        formal = list(family.get("params", {}) or {})
        if len(args) != len(formal):
            raise SignatureError(f"{node.value.id} takes {len(formal)} arguments, got {len(args)}")
        sub = dict(zip(formal, args, strict=True))
        node = Substitute(sub).visit(parse(_case(node.value.id, family, sub, point)))
    if not isinstance(node, ast.List):
        raise SignatureError(f"shape {shape!r} is not a list or a type-family application")
    return node


# ---------------------------------------------------------------- one branch


@dataclass
class SignatureBranch:
    """What one discriminant point selects, folded at that point."""

    # The shape and dtype of every passed input, and every
    # emitted output.
    shapes: dict[str, ast.List]
    dtypes: dict[str, ast.expr]
    # Refinements, and the `let` definitions relevance reaches.
    rules: list[ast.expr]
    lets: dict[str, ast.expr]
    # Every name the branch's shapes, dtypes, refinements and reached `let`s read.
    relevant: set[str]
    # Each passed tensor sequence's element-wise indices, in order of first reading.
    sequences: dict[str, tuple[str, ...]] = field(default_factory=dict)


def reach(sig: Signature, point: dict, relevant: set[str], lets: dict) -> None:
    """Add to `lets` every `let` `relevant` reaches, folded at `point`, and what they read."""
    while more := sorted((relevant & set(sig.let)) - set(lets)):
        for n in more:
            lets[n] = fold(parse(sig.let[n]), point, f"let {n}")
            relevant |= names(lets[n])


def branch(sig: Signature, point: dict, complete: bool = True) -> SignatureBranch:
    """The branch at `point`; with `complete=False` a tensor whose family has no case is left out.

    A refinement is folded before its names count, so one whose guard folds to true there makes
    nothing relevant.
    """
    shapes, dtypes_, sequences = {}, {}, {}
    for t in (*sig.inputs.values(), *sig.outputs.values()):
        if not (output_emitted(t, point) if t.name in sig.outputs else tensor_passed(t, point)):
            continue
        try:
            shapes[t.name] = fold(expand(sig, t.shape, point), point, f"tensor {t.name!r} shape")
        except SignatureError:
            if complete:
                raise
            continue
        dtypes_[t.name] = fold(parse(t.dtype), point, f"tensor {t.name!r} dtype")
        if t.name in sig.inputs and tensor_passed(t, point) and t.sequence:
            each = elementwise(sig)
            sequences[t.name] = tuple(
                dict.fromkeys(items(shapes[t.name], each) + items(dtypes_[t.name], each))
            )
    rules = [fold(parse(r), point, f"shape_rules[{i}]") for i, r in enumerate(sig.rules)]
    relevant = set().union(
        set(), *(names(n) for n in (*shapes.values(), *dtypes_.values(), *rules))
    )
    relevant -= set(DTYPE_BITS)
    lets: dict[str, ast.expr] = {}
    reach(sig, point, relevant, lets)
    return SignatureBranch(shapes, dtypes_, rules, lets, relevant, sequences)


# ---------------------------------------------------------------- inference


def _affine_in(node: ast.expr, var: str) -> bool:
    """Whether `node` is `var`, `a * var + e`, `var + e` or `var - e`.

    `a` does not read `var`: a positive integer literal, or a known quantity whose sign the
    call checks.
    """
    if isinstance(node, ast.Name):
        return node.id == var
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = var in names(node.left), var in names(node.right)
        if left and not right:
            return _affine_in(node.left, var)
        return right and not left and isinstance(node.op, ast.Add) and _affine_in(node.right, var)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        literal = isinstance(node.left, ast.Constant)
        positive = isinstance(node.left.value, int) and node.left.value > 0 if literal else True
        return positive and var not in names(node.left) and _affine_in(node.right, var)
    return False


@dataclass(frozen=True)
class Step:
    """One step of an inference plan.

    `let` defines `name`; `bind` solves `name` from element `index` of `tensor`'s shape;
    `check` compares that element, whose names are all known, with the tensor. `before` and
    `after` are the elements on either side whose lengths place it, when those are known.

    On a tensor sequence, `first` makes its element 0 the `tensor` the following steps read,
    solving shared names no other tensor solves; where it is empty, the element 0 of the first
    nonempty sequence of `items` stands in. `each` runs the plan `sub` on every element,
    checking the items already known and collecting the others into the indices `items`.
    """

    action: str
    name: str | None = None
    tensor: str | None = None
    index: int | None = None
    before: tuple | None = None
    after: tuple | None = None
    sub: tuple = ()
    items: tuple = ()

    @property
    def solved(self) -> tuple:
        """The names this step solves; a `first` step solves none itself."""
        if self.action == "first":
            return ()
        return (self.name,) if self.name else self.items


def unification(
    shapes: dict[str, ast.List],
    known: set[str],
    lets: dict[str, ast.expr],
    sequences: dict | None = None,
) -> list[Step]:
    """The order in which `shapes` and `lets` solve their indices, starting from `known`.

    `sequences` maps each tensor sequence among `shapes` to its element-wise indices; its shape
    is its element type. An element-wise index is solved per element, and known once every
    element is solved. A sequence's element 0 solves shared names only when nothing else
    progresses.
    """
    sequences = sequences or {}
    local = {t: localized(shapes[t], each) for t, each in sequences.items()}
    owned = {t: items(shapes[t], frozenset(each)) for t, each in sequences.items()}
    known, steps = set(known), []
    pool = {t: n for t, n in shapes.items() if t not in sequences}
    done: set = set()
    collected, first = set(), []
    while True:
        progress = _solve_axes(pool, known, lets, steps, done)
        for t in sequences:
            if t in collected:
                continue
            mine = {f"{ITEM}{x}" for x in owned[t]}
            outer = {f"{ITEM}{x}" for x in owned[t] if x in known}
            base = (known - {n for n in known if n.startswith(ITEM)}) | outer
            if names(local[t]) - mine - base:
                continue
            sub = unification({t: local[t]}, base, {})
            if not mine <= base | {s.name for s in sub if s.name}:
                continue
            fresh = tuple(x for x in owned[t] if x not in known)
            steps.append(Step("each", None, t, sub=tuple(sub), items=fresh))
            known |= set(fresh)
            collected.add(t)
            progress = True
        if progress:
            continue
        idle = [t for t in sequences if t not in collected and t not in first]
        if not idle:
            break
        chosen = idle[0]
        # A sequence whose element type differs only in element-wise names solves the same
        # shared names from its element 0, where the chosen one is empty.
        alternates = tuple(
            u for u in idle[1:] if _shared_form(local[u]) == _shared_form(local[chosen])
        )
        first.append(chosen)
        steps.append(Step("first", None, chosen, items=alternates))
        pool[chosen] = local[chosen]
    return steps


def _shared_form(node: ast.expr) -> str:
    """*node* with every element-wise item name replaced by one placeholder."""
    renamed = Substitute(
        {n: ast.Name(ITEM, ast.Load()) for n in names(node) if n.startswith(ITEM)}
    ).visit(copy.deepcopy(node))
    return ast.dump(renamed)


def _solve_axes(
    shapes: dict[str, ast.List], known: set[str], lets: dict, steps: list, done: set
) -> bool:
    """Run the axis rules of table 19 over `shapes` and `lets` to a fixpoint.

    `steps` and `known` are extended in place; the result is whether anything was solved or
    checked.
    """
    any_progress, progress = False, True
    while progress:
        progress = False
        for n, node in lets.items():
            if n not in known and names(node) <= known:
                steps.append(Step("let", n))
                known.add(n)
                progress = True
        for t, node in shapes.items():
            sized = [not isinstance(e, ast.Starred) or names(e.value) <= known for e in node.elts]
            for i, e in enumerate(node.elts):
                if (t, i) in done:
                    continue
                before = tuple(node.elts[:i]) if all(sized[:i]) else None
                after = tuple(node.elts[i + 1 :]) if all(sized[i + 1 :]) else None
                value = e.value if isinstance(e, ast.Starred) else e
                free = names(value) - known
                if isinstance(e, ast.Starred):
                    one_open = isinstance(value, ast.Name) and sized.count(False) == 1
                    if before is None or after is None or (free and not one_open):
                        continue
                    name = value.id if free else None
                elif before is None and after is None:
                    continue
                elif not free:
                    name = None
                elif len(free) == 1 and _affine_in(value, next(iter(free))):
                    name = next(iter(free))
                else:
                    continue
                steps.append(Step("bind" if name else "check", name, t, i, before, after))
                done.add((t, i))
                if name:
                    known.add(name)
                progress = any_progress = True
    return any_progress
