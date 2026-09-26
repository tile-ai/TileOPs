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
from dataclasses import dataclass, field

from .dtype_rules import DTYPE_BITS
from .expr import (
    OPEN,
    KindEnv,
    SignatureError,
    Substitute,
    bind,
    callee,
    fold,
    infer_kinds,
    names,
    parse,
    value_at,
)
from .kinds import BOOL, DTYPE, NONE, VALUE, Kind, dtypes, fits, parse_spec, parse_type, union
from .primitives import PRIMITIVE_KINDS, PRIMITIVES

__all__ = [
    "Signature",
    "SignatureBranch",
    "SignatureError",
    "Step",
    "Tensor",
    "adt_fields",
    "branch",
    "complete_point",
    "discriminant_axes",
    "discriminant_groups",
    "dtype_kind",
    "expand",
    "field_kind",
    "holds",
    "is_legacy",
    "kind_env",
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


# FIXME(staged-rollout): an entry still written in the legacy manifest form.
#
# Broken invariant: every entry is a parametric signature (docs/design/manifest.md).
# Why: the migration converts one family per PR, and every consumer reads both forms meanwhile.
# Cleanup: delete this predicate and every legacy branch that calls it once no entry has `source`.
def is_legacy(entry: object) -> bool:
    """Whether an entry is in the legacy form: it declares `source`."""
    return isinstance(entry, dict) and "source" in entry


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: str
    optional: "bool | str" = False
    nullable: str | None = None
    buffer: bool = False
    mutated: "bool | str" = False
    values: str | None = None
    requires: tuple[str, ...] = ()
    cpu: bool = False
    write_only: bool = False
    alias: str | None = None
    contiguous: bool = False


@dataclass
class Signature:
    name: str
    forall: dict[str, str]
    params: dict[str, dict]
    ctor_tensors: dict[str, Tensor]
    inputs: dict[str, Tensor]
    outputs: dict[str, Tensor]
    types: dict[str, dict]
    let: dict[str, str]
    rules: list[str]
    adts: dict[str, dict] = field(default_factory=dict)
    dtype_combos: list[dict] = field(default_factory=list)

    @property
    def call_tensors(self) -> dict[str, Tensor]:
        """Tensors unification reads: construction-time tensors, then call-time inputs."""
        return {**self.ctor_tensors, **self.inputs}

    def kind(self, name: str) -> Kind | None:
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


def _identifier(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier()


def _tensor(name: str, decl: object) -> Tensor:
    if (
        not isinstance(decl, dict)
        or not isinstance(decl.get("dtype"), str)
        or not isinstance(decl.get("shape"), str)
    ):
        raise SignatureError(f"tensor {name!r} needs string `dtype` and `shape`")
    return Tensor(
        name,
        decl["dtype"],
        decl["shape"],
        decl.get("optional", False),
        decl.get("nullable"),
        decl.get("buffer") == "out",
        decl.get("mutated", False),
        decl.get("values"),
        tuple(decl.get("requires", ()) or ()),
        decl.get("device") == "cpu",
        decl.get("write_only") is True,
        decl.get("alias"),
        decl.get("contiguous") is True,
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
        if kind in ("Dim", "Shape", "Seq[Int]") or (
            dtype is not None and dtype.tag == "DType" and dtype.values
        ):
            forall[index] = kind
        else:
            problems.append(f"forall index {index!r} has unknown kind {kind!r}")
    params, ctor_tensors, tensors = {}, {}, {"inputs": {}, "outputs": {}}
    for p, decl in sections["params"].items():
        if isinstance(decl, dict) and "shape" in decl:
            tensors.setdefault("params", {})[p] = decl
        elif isinstance(decl, dict) and "type" in decl:
            params[p] = decl
        else:
            problems.append(f"parameter {p!r} needs a `type`")
    for group in ("params", "inputs", "outputs"):
        source = tensors.get("params", {}) if group == "params" else sections[group]
        for n, decl in source.items():
            try:
                tensor = _tensor(n, decl)
            except SignatureError as exc:
                problems.append(str(exc))
                continue
            (ctor_tensors if group == "params" else tensors[group])[n] = tensor
    return Signature(
        name,
        forall,
        params,
        ctor_tensors,
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
    """The `KindEnv` of `sig`: every index, parameter and `let`, lets in dependency order, and
    the misuse inferring the lets finds. `kinds` overrides what the declarations give."""
    base = {n: sig.kind(n) for n in (*sig.forall, *sig.params)} | (kinds or {})
    env = KindEnv(
        base,
        adt_fields(sig.adts),
        frozenset(sig.types),
        frozenset({*sig.call_tensors, *sig.outputs})
        | ({"out"} if any(t.buffer for t in sig.outputs.values()) else set()),
        frozenset(
            p for p in sig.params if param_kind(sig.params[p].get("type"), sig.adts).tag == "Maybe"
        ),
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


def dtype_kind(sig: Signature, node: ast.expr, errors: list[str]) -> Kind | None:
    """The kind of a dtype expression; misuse goes to `errors`."""
    if isinstance(node, ast.Constant) and node.value is None:
        return NONE
    if isinstance(node, ast.Name):
        if node.id in DTYPE_BITS and node.id not in {*sig.forall, *sig.params}:
            return dtypes([node.id])
        kind = sig.kind(node.id)
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
        kind = dtype_kind(sig, arg, errors)
        if kind is not None and not fits(kind, _spec(wanted)):
            errors.append(f"argument {ast.unparse(arg)} has kind {kind}, expected {wanted}")
        domains.append(_dtype_values(kind))
    if any(d is None for d in domains):
        return DTYPE
    # Every argument has finite dtypes, so the result's are the primitive over their product.
    values = {PRIMITIVES[name](*combo) for combo in itertools.product(*domains)}
    return union(*(NONE if v is None else dtypes([v]) for v in values))


def _dtype_values(kind: Kind | None) -> set | None:
    """The values of a dtype expression's kind, None among them; None when unbounded."""
    if kind is None:
        return None
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
        payload = _finite_axis(f"{p}.value", kind.item, adts) if kind.item.tag == "ADT" else None
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
    for p, decl in sig.params.items():
        kind = param_kind(decl.get("type"), sig.adts)
        if (axis := _finite_axis(p, kind, sig.adts)) is not None:
            axes[p] = axis
    for t in sig.call_tensors.values():
        if t.optional is True:
            axes[t.name] = (f"present({t.name})", [False, True])
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

    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        reads = {t.name} | ({"out"} if t.buffer else set())
        for text in (t.shape, t.dtype, t.optional, t.nullable, t.mutated):
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
    """The invariant of the constructor ADT-valued *p* holds at `point`, when it is false there;
    a `Maybe[ADT]` payload's, where it is present."""
    if kind.tag == "Maybe" and kind.item.tag == "ADT":
        if point.get(f"present({p})") is not True:
            return None
        p, kind = f"{p}.value", kind.item
    ctor = point.get(f"{p}.kind")
    declared = adts.get(kind.name, {}) if kind.tag == "ADT" else {}
    invariant = declared.get("sum", {}).get(ctor, {}).get("invariant")
    if invariant is None:
        return None
    fields = {k[len(p) + 1 :]: v for k, v in point.items() if k.startswith(f"{p}.")}
    node = parse(invariant)
    if not names(bind(node, fields)) and value_at(node, fields, f"{p} invariant") is False:
        return f"{p}: {invariant}"
    return None


def rejecting_rule(sig: Signature, point: dict) -> str | None:
    """The first domain restriction false at `point`: a refinement reading only discriminants.

    What a refinement reads is judged with the `let`s it reads inlined.
    """
    for i, text in enumerate(sig.rules):
        rule = _inlined(sig, parse(text))
        if not names(bind(rule, point)) and value_at(rule, point, f"shape_rules[{i}]") is False:
            return text
    for p, decl in sig.params.items():
        failed = _invariant_fails(sig.adts, p, param_kind(decl.get("type"), sig.adts), point)
        if failed is not None:
            return failed
    return None


def holds(expr: "bool | str | None", point: dict) -> bool:
    """A presence or effect condition at `point`; it must read only discriminants."""
    if expr is None or isinstance(expr, bool):
        return expr is not False
    node = parse(expr)
    value = OPEN if names(bind(node, point)) else value_at(node, point, repr(expr))
    if value is OPEN:
        raise SignatureError(f"{expr!r} reads more than discriminants")
    return bool(value)


def complete_point(sig: Signature, point: dict, strict: bool = True) -> dict:
    """`point` with `present(t)` for every tensor, from its `optional` or `nullable` condition.

    A required input and an output that is not nullable are present; an `optional: true`
    tensor's presence is its own axis. With `strict=False` a presence `point` does not settle
    is left unsettled instead of raising.
    """
    point = dict(point)
    pending = {
        t.name: t.nullable if t.name in sig.outputs else t.optional
        for t in (*sig.call_tensors.values(), *sig.outputs.values())
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
            node = _inlined(sig, parse(cond)) if isinstance(cond, str) else None
            if cond is True:
                value = OPEN  # its own discriminant axis, which `point` lacks
            elif node is None:
                value = True
            elif names(bind(node, point)):
                value = OPEN
            else:
                value = value_at(node, point, f"tensor {name!r} presence")
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


def tensor_passed(tensor: Tensor, point: dict) -> bool:
    """Whether a call at `point` passes an input or construction-time tensor."""
    if tensor.optional is True:
        return point.get(f"present({tensor.name})", False)
    return holds(tensor.optional or None, point)


def output_emitted(tensor: Tensor, point: dict) -> bool:
    """Whether a call at `point` returns an output."""
    return holds(tensor.nullable, point)


# ---------------------------------------------------------------- type families


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
    keys = family["match"] if isinstance(family["match"], list) else [family["match"]]
    got = []
    for k in keys:
        node = Substitute(sub).visit(parse(k))
        absent = (
            isinstance(node, ast.Attribute)
            and node.attr == "value"
            and isinstance(node.value, ast.Name)
            and point.get(f"present({node.value.id})") is False
        )
        # An absent payload has no value; only `_` matches it.
        got.append(OPEN if absent else _discriminant(node, point))
    got = tuple(got)
    hits = []
    for case in family.get("cases", []) or []:
        pattern = case["when"] if isinstance(case["when"], list) else [case["when"]]
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

    # The shape and dtype of every passed input and construction-time tensor, and every
    # emitted output.
    shapes: dict[str, ast.List]
    dtypes: dict[str, ast.expr]
    # Refinements, and the `let` definitions relevance reaches.
    rules: list[ast.expr]
    lets: dict[str, ast.expr]
    # Every name the branch's shapes, dtypes, refinements and reached `let`s read.
    relevant: set[str]


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
    shapes, dtypes_ = {}, {}
    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        if not (output_emitted(t, point) if t.name in sig.outputs else tensor_passed(t, point)):
            continue
        try:
            shapes[t.name] = fold(expand(sig, t.shape, point), point, f"tensor {t.name!r} shape")
        except SignatureError:
            if complete:
                raise
            continue
        dtypes_[t.name] = fold(parse(t.dtype), point, f"tensor {t.name!r} dtype")
    rules = [fold(parse(r), point, f"shape_rules[{i}]") for i, r in enumerate(sig.rules)]
    relevant = set().union(
        set(), *(names(n) for n in (*shapes.values(), *dtypes_.values(), *rules))
    )
    relevant -= set(DTYPE_BITS)
    lets: dict[str, ast.expr] = {}
    reach(sig, point, relevant, lets)
    return SignatureBranch(shapes, dtypes_, rules, lets, relevant)


# ---------------------------------------------------------------- inference


def _affine_in(node: ast.expr, var: str) -> bool:
    """`var`, `a * var + e`, `var + e` or `var - e` for a positive integer literal `a`."""
    if isinstance(node, ast.Name):
        return node.id == var
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = var in names(node.left), var in names(node.right)
        if left and not right:
            return _affine_in(node.left, var)
        return right and not left and isinstance(node.op, ast.Add) and _affine_in(node.right, var)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return (
            isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, int)
            and node.left.value > 0
            and _affine_in(node.right, var)
        )
    return False


@dataclass(frozen=True)
class Step:
    """One step of an inference plan.

    `let` defines `name`; `bind` solves `name` from element `index` of `tensor`'s shape;
    `check` compares that element, whose names are all known, with the tensor. `before` and
    `after` are the elements on either side whose lengths place it, when those are known.
    """

    action: str
    name: str | None = None
    tensor: str | None = None
    index: int | None = None
    before: tuple | None = None
    after: tuple | None = None


def unification(
    shapes: dict[str, ast.List], known: set[str], lets: dict[str, ast.expr]
) -> list[Step]:
    """The order in which `shapes` and `lets` solve their indices, starting from `known`."""
    known, steps, done = set(known), [], set()
    progress = True
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
                progress = True
    return steps
