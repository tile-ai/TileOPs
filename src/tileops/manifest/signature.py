"""Parse and statically check a manifest signature (docs/design/manifest.md § Signature, § Validation).

`check_entry` returns the static diagnostics: name categories and kinds, type
families, the inference plan, `let` cycles, and the closed expression language. Every expression
is parsed once; a field that fails to parse is reported and skipped by the checks that read it.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import functools
import importlib
import itertools
import math
from dataclasses import dataclass, field

from .dtype_rules import DTYPE_BITS
from .kinds import (
    BOOL,
    DTYPE,
    INT,
    NONE,
    VALUE,
    Kind,
    _quoted,
    _split,
    adt,
    common,
    comparable,
    dtypes,
    fits,
    join,
    literals,
    ordered,
    parse_spec,
    parse_type,
    seq,
    union,
)
from .primitives import PRIMITIVE_KINDS, PRIMITIVES, namespace

__all__ = [
    "DISCRIMINANT_LIMIT",
    "Signature",
    "SignatureError",
    "check_adts",
    "check_entry",
    "parse_signature",
    "effect_errors",
    "roofline_plan",
    "signature_schema_errors",
]

# Discriminant combinations above this count draw an advisory diagnostic.
DISCRIMINANT_LIMIT = 256

_SIGNATURE_KEYS = frozenset(
    {"forall", "params", "inputs", "outputs", "types", "let", "shape_rules", "dtype_combos"}
)
# Each tensor key: the values it takes, and what they are.
_TENSOR_FIELDS: dict[str, tuple] = {
    "dtype": (lambda v: isinstance(v, str), "a dtype expression"),
    "shape": (lambda v: isinstance(v, str), "a shape"),
    "optional": (lambda v: v is True or isinstance(v, str), "true or a presence condition"),
    "nullable": (lambda v: isinstance(v, str), "a presence condition"),
    "mutated": (lambda v: isinstance(v, (bool, str)), "a bool or a discriminant condition"),
    "write_only": (lambda v: isinstance(v, bool), "a bool"),
    "contiguous": (lambda v: isinstance(v, bool), "a bool"),
    "alias": (lambda v: isinstance(v, str), "an input name"),
    "buffer": (lambda v: v == "out", "'out'"),
    "device": (lambda v: v == "cpu", "'cpu'"),
    "values": (lambda v: isinstance(v, str), "a generator call"),
    "requires": (lambda v: isinstance(v, list), "a list of predicate calls"),
}
_COMMON_TENSOR_KEYS = {"dtype", "shape", "contiguous"}
# The tensor keys each role admits.
_ROLE_KEYS = {
    "params": frozenset(_COMMON_TENSOR_KEYS | {"optional", "device", "values", "requires"}),
    "inputs": frozenset(
        _COMMON_TENSOR_KEYS | {"optional", "mutated", "write_only", "device", "values", "requires"}
    ),
    "outputs": frozenset(_COMMON_TENSOR_KEYS | {"nullable", "buffer", "alias"}),
}
_ADT_CTOR_KEYS = frozenset({"fields", "invariant", "python"})
_PARAM_KEYS = frozenset({"type", "default", "kw_only"})
_FAMILY_KEYS = frozenset({"params", "match", "cases"})
_COMPREHENSION_CALLEES = frozenset({"all", "sum", "max", "min"})
_NODES = (
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.GeneratorExp,
    ast.comprehension,
    ast.keyword,
)


class SignatureError(ValueError):
    """A declaration outside the schema; the message names it."""


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


def _spec_or_none(text: str) -> Kind | None:
    try:
        return parse_spec(text)
    except ValueError:
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


def _formal(text: object, adts: dict) -> Kind:
    """A kind as a table, a family formal or a primitive signature writes it."""
    return adt(str(text)) if str(text) in adts else parse_spec(str(text), adts)


def _enum_values(type_text: object) -> list[str]:
    return [m[1:-1] for m in _split(str(type_text))]


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


def _read_signature(name: str, entry: dict, adts: dict) -> tuple[Signature, list[str]]:
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
    sig, problems = _read_signature(name, entry, adts)
    if problems:
        raise SignatureError(problems[0])
    return sig


# ---------------------------------------------------------------- parsing and the language


@functools.lru_cache(maxsize=4096)
def _parsed(text: str) -> ast.expr | str:
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        return f"cannot parse {text!r}: {exc.msg}"


def _parse(text: object) -> ast.expr:
    """A fresh copy of `text`'s tree; each distinct string is parsed once."""
    if not isinstance(text, str):
        raise SignatureError(f"expression {text!r} is not a string")
    node = _parsed(text)
    if isinstance(node, str):
        raise SignatureError(node)
    return copy.deepcopy(node)


def _parses(text: str) -> bool:
    try:
        _parse(text)
    except SignatureError:
        return False
    return True


def _callee(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def names(node: ast.AST) -> set[str]:
    """Free names an expression reads: callees excluded, comprehension variables in their scope."""
    if isinstance(node, ast.Name):
        return set() if node.id == "inf" else {node.id}
    if isinstance(node, ast.Call):
        free = set() if _callee(node.func) else names(node.func)
        return free.union(*(names(a) for a in (*node.args, *(k.value for k in node.keywords))))
    if isinstance(node, ast.GeneratorExp):
        free, bound = set(), set()
        for gen in node.generators:
            free |= names(gen.iter) - bound
            bound |= {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
            free |= set().union(set(), *(names(c) for c in gen.ifs)) - bound
        return free | (names(node.elt) - bound)
    return set().union(set(), *(names(c) for c in ast.iter_child_nodes(node)))


class _Unpresent(ast.NodeTransformer):
    """Drop `present(x)`, which reads `x`'s presence and not its value."""

    def visit_Call(self, node):
        if _callee(node.func) == "present":
            return ast.Constant(True)
        return self.generic_visit(node)


def _language_errors(node: ast.expr, shape: bool) -> list[str]:
    """Constructs outside the expression language; a shape's outer list is allowed."""
    errors = []
    root = node
    comprehension_args = {
        id(c.args[0])
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and _callee(c.func) in _COMPREHENSION_CALLEES and c.args
    }
    for sub in ast.walk(node):
        if shape and sub is root and isinstance(sub, ast.List):
            continue
        if (
            shape
            and isinstance(sub, ast.Starred)
            and isinstance(root, ast.List)
            and sub in root.elts
        ):
            continue
        if isinstance(sub, ast.Subscript) and shape and sub is root:
            continue
        if not isinstance(sub, _NODES):
            errors.append(f"{type(sub).__name__} is outside the expression language")
        elif isinstance(sub, ast.Constant) and not (
            sub.value is None
            or isinstance(sub.value, (bool, int, str))
            or (isinstance(sub.value, float) and math.isfinite(sub.value))
        ):
            errors.append(f"literal {sub.value!r} is outside the expression language")
        elif isinstance(sub, ast.GeneratorExp) and id(sub) not in comprehension_args:
            errors.append("a comprehension is only an argument of all, sum, max or min")
        elif isinstance(sub, ast.GeneratorExp) and (
            len(sub.generators) != 1
            or sub.generators[0].ifs
            or sub.generators[0].is_async
            or not isinstance(sub.generators[0].target, ast.Name)
        ):
            errors.append("a comprehension is `f(x) for x in s`: one name, no `if`, no `async`")
        elif (
            isinstance(sub, ast.Call)
            and _callee(sub.func) != "present"
            and _callee(sub.func) not in PRIMITIVE_KINDS
        ):
            errors.append(
                f"{_callee(sub.func) or ast.unparse(sub.func)} is not a built-in primitive"
            )
    return errors


# ---------------------------------------------------------------- kinds


class _Kinds:
    """Infers expression kinds and records misuse."""

    def __init__(
        self, sig: Signature, env: dict, where: str, ctors: dict | None = None, point: bool = False
    ):
        self.sig, self.env, self.where, self.errors = sig, env, where, []
        # The constructor each ADT-valued name holds on the branch being checked, where known.
        self.ctors = ctors or {}
        # On one discriminant point, where dead arms are folded away, constants decide domains.
        self.point = point

    def error(self, message: str) -> None:
        self.errors.append(f"{self.where}: {message}")

    def expect(self, node: ast.expr, expected: Kind) -> None:
        kind = self.of(node)
        if not fits(kind, expected):
            self.error(f"{ast.unparse(node)} has kind {kind}, expected {expected}")

    def of(self, node: ast.expr) -> Kind | None:  # noqa: C901 - one case per node kind
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return BOOL
            if isinstance(v, int):
                return Kind("Int", nonneg=v >= 0)
            return literals([v]) if isinstance(v, str) else NONE if v is None else VALUE
        if isinstance(node, ast.Name):
            if node.id == "inf":
                return VALUE
            if node.id in self.sig.types:
                self.error(f"type family {node.id} is not a value")
            return self.env.get(node.id)
        if isinstance(node, ast.BinOp):
            left, right = self.of(node.left), self.of(node.right)
            if VALUE in (left, right):
                return VALUE
            for side, kind in ((node.left, left), (node.right, right)):
                if not fits(kind, INT):
                    self.error(f"{ast.unparse(side)} has kind {kind}, expected an integer")
            nonneg = all(k is not None and k.tag == "Int" and k.nonneg for k in (left, right))
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                divisor = node.right.value if isinstance(node.right, ast.Constant) else None
                if self.point and isinstance(divisor, int) and divisor <= 0:
                    self.error(f"{ast.unparse(node)} divides by {divisor}")
                nonneg = nonneg and isinstance(divisor, int) and divisor > 0
            return Kind("Int", nonneg=nonneg and not isinstance(node.op, ast.Sub))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                self.expect(node.operand, BOOL)
                return BOOL
            operand = self.of(node.operand)
            if operand != VALUE and not fits(operand, INT):
                self.error(f"{ast.unparse(node.operand)} has kind {operand}, expected an integer")
            return VALUE if operand == VALUE else INT
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self.expect(value, BOOL)
            return BOOL
        if isinstance(node, ast.Compare):
            self._compare(node)
            return BOOL
        if isinstance(node, ast.IfExp):
            self.expect(node.test, BOOL)
            body, orelse = self.of(node.body), self.of(node.orelse)
            kind = common(body, orelse)
            if kind is None and None not in (body, orelse):
                self.error(f"{ast.unparse(node)} has arms of kinds {body} and {orelse}")
            return kind
        if isinstance(node, ast.Tuple):
            kinds = [self.of(e) for e in node.elts]
            if not kinds:
                return seq(None, 0)
            item = kinds[0]
            for k in kinds[1:]:
                item = join(item, k)
            return seq(item, len(kinds))
        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.GeneratorExp):
            gen = node.generators[0]
            if not isinstance(gen.target, ast.Name):
                return None  # reported by the language check
            iterable = self.of(gen.iter)
            if iterable is not None and iterable.sequence() is None:
                self.error(f"{ast.unparse(gen.iter)} has kind {iterable}, not a sequence")
            iterable = iterable and iterable.sequence()
            element = iterable.item if iterable is not None else None
            env = {**self.env, gen.target.id: element}
            inner = _Kinds(self.sig, env, self.where, self.ctors, self.point)
            kind = inner.of(node.elt)
            self.errors += inner.errors
            return seq(kind, iterable.length if iterable is not None else None)
        if isinstance(node, ast.Call):
            return self._call(node)
        return None

    def _subscript(self, node: ast.Subscript) -> Kind | None:
        """A subscript of a sequence; on a union, of every member, whose results are joined."""
        value = self.of(node.value)
        members = value.members if value is not None and value.tag == "Union" else (value,)
        if value is not None and any(m.tag != "Seq" for m in members):
            self.error(f"{ast.unparse(node.value)} has kind {value}, not a sequence")
            return None
        parts = (
            (node.slice.lower, node.slice.upper, node.slice.step)
            if isinstance(node.slice, ast.Slice)
            else (node.slice,)
        )
        for part in parts:
            if part is not None:
                self.expect(part, INT)
        if value is None:
            return None
        results = [self._item(node, m, parts) for m in members]
        if any(r is None for r in results):
            return None
        kind = results[0]
        for r in results[1:]:
            kind = join(kind, r)
        return kind

    def _item(self, node: ast.Subscript, value: Kind, parts: tuple) -> Kind | None:
        if isinstance(node.slice, ast.Slice):
            ends = [p.value if isinstance(p, ast.Constant) else _OPEN if p else None for p in parts]
            closed = value.length is not None and _OPEN not in ends and ends[2] != 0
            length = len(range(*slice(*ends).indices(value.length))) if closed else None
            return seq(value.item, length)
        index = node.slice.value if isinstance(node.slice, ast.Constant) else None
        length = value.length
        if length is not None and isinstance(index, int) and not -length <= index < length:
            self.error(
                f"{ast.unparse(node)} is outside the {length} items of {ast.unparse(node.value)}"
            )
        return value.item

    def _compare(self, node: ast.Compare) -> None:
        if any(
            isinstance(e, ast.Constant) and e.value is None for e in (node.left, *node.comparators)
        ):
            self.error(f"{ast.unparse(node)} compares with None; presence is present(v)")
        left = self.of(node.left)
        for op, operand in zip(node.ops, node.comparators, strict=True):
            right = self.of(operand)
            if isinstance(op, (ast.In, ast.NotIn)):
                if right is not None and right.sequence() is None:
                    self.error(f"{ast.unparse(operand)} has kind {right}, not a sequence")
                elif right is not None and not comparable(left, right.sequence().item):
                    self.error(f"{ast.unparse(node)} tests a {left} against {right}")
                elif (
                    right is not None
                    and left is not None
                    and left.tag == "Str"
                    and left.values
                    and (item := right.sequence().item) is not None
                    and item.tag == "Str"
                    and item.values
                    and not item.values <= left.values
                ):
                    self.error(f"{ast.unparse(node)} tests literals {left} never takes")
            elif not comparable(left, right) or (
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and not ordered(left, right)
            ):
                self.error(f"{ast.unparse(node)} compares {left} with {right}")
            left = right

    def _attribute(self, node: ast.Attribute) -> Kind | None:
        receiver = node.value
        if (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "value"
            and isinstance(receiver.value, ast.Name)
        ):
            # A field of `v.value` is a field of v's payload; v names the branch facts.
            receiver = receiver.value
            kind = self._attribute(node.value)
        else:
            kind = self.env.get(receiver.id) if isinstance(receiver, ast.Name) else None
            if kind is not None and kind.tag == "Maybe" and node.attr == "value":
                return kind.item
            declared_maybe = self.sig.kind(receiver.id) if isinstance(receiver, ast.Name) else None
            if kind == NONE and declared_maybe is not None and declared_maybe.tag == "Maybe":
                return None  # an absent payload's read is `_read_errors`'
        payload = receiver is not node.value
        if kind is not None and kind.tag == "ADT":
            ctors = self.sig.adts[kind.name]["sum"]
            # A payload's branch facts are keyed `v.value`.
            name = f"{receiver.id}.value" if payload else receiver.id
            chosen = [self.ctors[name]] if name in self.ctors else list(ctors)
            if node.attr == "kind":
                return literals(chosen)
            fixed = self.ctors.get(f"{name}.{node.attr}", _OPEN)
            if fixed is not _OPEN:
                return BOOL if isinstance(fixed, bool) else literals([fixed])
            declared = [
                (ctors[c].get("fields") or {})[node.attr]
                for c in chosen
                if node.attr in (ctors[c].get("fields") or {})
            ]
            if declared:
                return union(*(_field_kind(d) for d in declared))
            if name in self.ctors:
                if not self.point:  # at a discriminant point `_read_errors` reports it
                    self.error(
                        f"{ast.unparse(node)} reads a field constructor {self.ctors[name]!r} lacks"
                    )
                return None
        self.error(f"attribute {ast.unparse(node)} is not a declared field")
        return None

    def _call(self, node: ast.Call, table: dict = PRIMITIVE_KINDS) -> Kind | None:
        """The result kind of a call to an entry of *table*, its arguments bound and checked."""
        callee = _callee(node.func)
        if callee == "present":
            known = {*self.sig.call_tensors, *self.sig.outputs}
            known |= {n for n, k in self.env.items() if k is not None and k.tag == "Maybe"}
            known |= {"out"} if any(t.buffer for t in self.sig.outputs.values()) else set()
            if (
                len(node.args) != 1
                or node.keywords
                or not isinstance(node.args[0], ast.Name)
                or node.args[0].id not in known
            ):
                self.error(f"{ast.unparse(node)} names no tensor, `Maybe` parameter or `out`")
            return BOOL
        if callee not in table:
            return None
        formal, result = table[callee]
        if formal and formal[0].endswith("*"):
            slots = [(None, formal[0][:-1])] * len(node.args)
        else:
            slots = [tuple(f.split("=")) if "=" in f else (None, f) for f in formal]
        bound: dict[int, ast.expr] = dict(enumerate(node.args))
        if len(node.args) > len(slots):
            self.error(f"{callee} takes {len(slots)} arguments, got {len(node.args)}")
        for kw in node.keywords:
            index = next((i for i, (name, _) in enumerate(slots) if name == kw.arg), None)
            if index is None or index in bound:
                self.error(f"{callee} takes no keyword {kw.arg!r} here")
            else:
                bound[index] = kw.value
        missing = [i for i, (name, _) in enumerate(slots) if name is None and i not in bound]
        if missing:
            self.error(f"{callee} misses argument {missing[0] + 1} of {len(slots)}")
        kinds = {
            i: self._argument(arg, _formal(slots[i][1], self.sig.adts))
            for i, arg in sorted(bound.items())
            if i < len(slots)
        }
        self._domain(callee, bound, kinds)
        result_kind = _formal(result, self.sig.adts)
        if callee == "ceil_div" and all(k is not None and k.nonneg for k in kinds.values()):
            return Kind("Int", nonneg=True)
        return result_kind

    def _domain(self, callee: str, bound: dict, kinds: dict) -> None:
        """A primitive's domain, where constants and fixed lengths decide it."""
        constant = {i: a.value for i, a in bound.items() if isinstance(a, ast.Constant)}
        if callee == "per_axis" and 0 in kinds:
            value, n = kinds[0], constant.get(2)
            length = value.length if value is not None and value.tag == "Seq" else None
            if length is not None and isinstance(n, int) and n != length:
                self.error(f"{ast.unparse(bound[0])} holds {length} items, not {n}")
        if not self.point:
            return  # the checks below read constants a dead arm may hold
        if callee == "reduced" and 0 in kinds and 1 in bound:
            rank = kinds[0].length if kinds[0] is not None else None
            axes = [
                a.value
                for a in getattr(bound[1], "elts", [bound[1]])
                if isinstance(a, ast.Constant)
            ]
            low, high = (-1, 1) if rank == 0 else (-(rank or 0), rank)
            bad = [
                a for a in axes if isinstance(a, int) and rank is not None and not low <= a < high
            ]
            if bad:
                self.error(f"reduced axes {bad} are outside rank {rank}")
        if callee == "ceil_div" and isinstance(constant.get(1), int) and constant[1] <= 0:
            self.error(f"ceil_div needs a positive divisor, got {constant[1]}")
        if callee != "per_axis" or 0 not in kinds:
            return
        value, n, i = kinds[0], constant.get(2), constant.get(1)
        fallback = kinds.get(3, NONE)
        if (
            value is not None
            and value.tag == "None"
            and fallback is not None
            and fallback.tag == "None"
        ):
            self.error(f"per_axis of {ast.unparse(bound[0])}, which is None here, needs a fallback")
            return
        if isinstance(n, int) and isinstance(i, int) and not 0 <= i < n:
            self.error(f"per_axis item {i} is outside the {n} items")

    def _argument(self, arg: ast.expr, expected: Kind) -> Kind | None:
        kind = self.of(arg)
        if expected.tag == "ADT" and expected.name is None:
            if kind is not None and kind.tag != "ADT":
                self.error(f"{ast.unparse(arg)} is not an ADT value")
        elif expected.tag == "Str" and expected.values and kind is not None and kind.tag == "Str":
            if kind.values is None or not kind.values <= expected.values:
                self.error(f"{ast.unparse(arg)} is not one of {expected}")
        elif not fits(kind, expected):
            self.error(f"{ast.unparse(arg)} has kind {kind}, expected {expected}")
        return kind


# ---------------------------------------------------------------- discriminants


class _Bind(ast.NodeTransformer):
    """Replace what a point fixes — names, ADT fields, `present(...)` — with constants."""

    def __init__(self, point: dict):
        self.point = point

    def visit_Name(self, node):
        return ast.Constant(self.point[node.id]) if node.id in self.point else node

    def visit_Attribute(self, node):
        key = ast.unparse(node)
        return ast.Constant(self.point[key]) if key in self.point else self.generic_visit(node)

    def visit_Call(self, node):
        key = ast.unparse(node)
        return ast.Constant(self.point[key]) if key in self.point else self.generic_visit(node)


_OPEN = object()
_NAMESPACE = namespace()
_UNFOLDED = (ast.Constant, ast.Name, ast.List, ast.Starred, ast.Slice, ast.GeneratorExp)


def _bound(node: ast.expr, point: dict) -> ast.expr:
    return ast.fix_missing_locations(_Bind(point).visit(copy.deepcopy(node)))


class _Fold(ast.NodeTransformer):
    """Evaluate closed subexpressions; drop the constant operands of `and`, `or` and conditionals."""

    def visit(self, node):
        if isinstance(node, ast.BoolOp):
            # Python's order: a decisive constant ends evaluation, but what ran before it stays.
            decisive, rest = isinstance(node.op, ast.Or), []
            for value in node.values:
                value = self.visit(value)
                if not isinstance(value, ast.Constant):
                    rest.append(value)
                elif bool(value.value) is decisive:
                    if not rest:
                        return ast.Constant(decisive)
                    rest.append(value)
                    break
            if len(rest) <= 1:
                return rest[0] if rest else ast.Constant(not decisive)
            node.values = rest
            return node
        if isinstance(node, ast.IfExp):
            test = self.visit(node.test)
            if isinstance(test, ast.Constant):
                return self.visit(node.body if test.value else node.orelse)
            node.test, node.body, node.orelse = test, self.visit(node.body), self.visit(node.orelse)
            return node
        node = self.generic_visit(node)
        if isinstance(node, ast.expr) and not isinstance(node, _UNFOLDED) and not names(node):
            code = compile(ast.Expression(ast.fix_missing_locations(node)), "<manifest>", "eval")
            try:
                return ast.Constant(eval(code, dict(_NAMESPACE)))  # noqa: S307
            except Exception as exc:
                raise SignatureError(f"{ast.unparse(node)} raises {exc}") from None
        return node


def fold(node: ast.expr, point: dict) -> ast.expr:
    """`node` on the branch `point` selects, with what that branch fixes evaluated."""
    return ast.fix_missing_locations(_Fold().visit(_bound(node, point)))


def _value_at(node: ast.expr, point: dict):
    """The value of `node` at `point`, or `_OPEN` when it reads more than discriminants."""
    folded = fold(node, point)
    return folded.value if isinstance(folded, ast.Constant) else _OPEN


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
    points = []
    for ctor, spec in (adts[kind.name].get("sum", {}) or {}).items():
        enums = []
        for f, k in (spec.get("fields", {}) or {}).items():
            field_kind = _field_kind(k)
            if field_kind is not None and field_kind.tag == "Str":
                enums.append((f, sorted(field_kind.values)))
            elif field_kind == BOOL:
                enums.append((f, [False, True]))
        for combo in itertools.product(*[v for _, v in enums]):
            points.append(
                {
                    f"{p}.kind": ctor,
                    **{f"{p}.{f}": c for (f, _), c in zip(enums, combo, strict=True)},
                }
            )
    return (None, points)


def _discriminant_axes(sig: Signature) -> dict[str, tuple[str | None, list]]:
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


def _discriminant_groups(sig: Signature) -> list[list[tuple[str | None, list]]]:
    """Discriminant axes grouped by dependency.

    A tensor, a refinement and a `let` each join every name they read; two axes are dependent
    when a chain of shared names links them. An axis nothing reads takes no part.
    """
    group: dict[str, str] = {}

    def root(n: str) -> str:
        while group.setdefault(n, n) != n:
            n = group[n]
        return n

    def join(members: set[str]) -> None:
        first, *rest = sorted(members) or [None]
        for n in rest:
            group[root(n)] = root(first)

    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        reads = {t.name} | ({"out"} if t.buffer else set())
        for text in (t.shape, t.dtype, t.optional, t.nullable, t.mutated):
            if isinstance(text, str):
                reads |= names(_parse(text))
        join(reads)
    for r in sig.rules:
        join(names(_parse(r)))
    for n, e in sig.let.items():
        join({n} | names(_parse(e)))
    axes = _discriminant_axes(sig)
    groups: dict[str, list] = {}
    for a in sorted(axes):
        if a in group:
            groups.setdefault(root(a), []).append(axes[a])
    return list(groups.values())


def _points(group: list[tuple[str | None, list]]):
    for combo in itertools.product(*[v for _, v in group]):
        point = {}
        for (key, _), value in zip(group, combo, strict=True):
            point.update(value if key is None else {key: value})
        yield point


def _inlined(sig: Signature, node: ast.expr) -> ast.expr:
    """`node` with every `let` it reads replaced by its definition, transitively."""
    for _ in range(len(sig.let) + 1):
        read = names(node) & set(sig.let)
        if not read:
            break
        node = _Substitute({n: _parse(sig.let[n]) for n in read}).visit(copy.deepcopy(node))
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
    node = _parse(invariant)
    if not names(_bound(node, fields)) and _value_at(node, fields) is False:
        return f"{p}: {invariant}"
    return None


def rejecting_rule(sig: Signature, point: dict) -> str | None:
    """The first domain restriction false at `point`: a refinement reading only discriminants.

    What a refinement reads is judged with the `let`s it reads inlined.
    """
    for text in sig.rules:
        rule = _inlined(sig, _parse(text))
        if not names(_bound(rule, point)) and _value_at(rule, point) is False:
            return text
    for p, decl in sig.params.items():
        failed = _invariant_fails(sig.adts, p, param_kind(decl.get("type"), sig.adts), point)
        if failed is not None:
            return failed
    return None


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


class _Substitute(ast.NodeTransformer):
    def __init__(self, sub):
        self.sub = sub

    def visit_Name(self, node):
        return copy.deepcopy(self.sub.get(node.id, node))


def _discriminant(node: ast.expr, point: dict):
    text = ast.unparse(node)
    if f"{text}.kind" in point:
        prefix = f"{text}."
        return point[f"{text}.kind"], {
            k[len(prefix) :]: v
            for k, v in point.items()
            if k.startswith(prefix) and k != f"{text}.kind"
        }
    value = _value_at(node, point)
    if value is _OPEN:
        raise SignatureError(f"{text!r} is not a finite discriminant")
    return value


def _case(fam: str, family: dict, sub: dict, point: dict) -> str:
    """The `is` of the one case of *family* whose pattern matches its `match` at *point*."""
    keys = family["match"] if isinstance(family["match"], list) else [family["match"]]
    got = []
    for k in keys:
        node = _Substitute(sub).visit(_parse(k))
        absent = (
            isinstance(node, ast.Attribute)
            and node.attr == "value"
            and isinstance(node.value, ast.Name)
            and point.get(f"present({node.value.id})") is False
        )
        # An absent payload has no value; only `_` matches it.
        got.append(_OPEN if absent else _discriminant(node, point))
    got = tuple(got)
    hits = []
    for case in family.get("cases", []) or []:
        pattern = case["when"] if isinstance(case["when"], list) else [case["when"]]
        if len(pattern) != len(got):
            raise SignatureError(
                f"{fam}: case {pattern} has {len(pattern)} components, match has {len(got)}"
            )
        if all(
            p == "_" or (g is not _OPEN and _matches(p, g))
            for p, g in zip(pattern, got, strict=True)
        ):
            hits.append(case["is"])
    if len(hits) != 1:
        raise SignatureError(f"{fam}: {len(hits)} cases match {got}, need exactly one")
    return hits[0]


def expand(sig: Signature, shape: str, point: dict) -> ast.List:
    """A shape term with every type-family application replaced by the branch `point` selects."""
    node = _parse(shape)
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
        node = _Substitute(sub).visit(_parse(_case(node.value.id, family, sub, point)))
    if not isinstance(node, ast.List):
        raise SignatureError(f"shape {shape!r} is not a list or a type-family application")
    return node


def _holds(expr: "bool | str | None", point: dict) -> bool:
    if expr is None or isinstance(expr, bool):
        return expr is not False
    node = _parse(expr)
    value = _OPEN if names(_bound(node, point)) else _value_at(node, point)
    if value is _OPEN:
        raise SignatureError(f"{expr!r} reads more than discriminants")
    return bool(value)


def complete_point(sig: Signature, point: dict, strict: bool = True) -> dict:
    """`point` with `present(t)` for every tensor, from its `optional` or `nullable` condition.

    A required input and an output that is not nullable are present. With `strict=False` a
    condition reading an axis `point` lacks is left unsettled instead of raising.
    """
    point = dict(point)
    pending = {
        t.name: t.nullable if t.name in sig.outputs else t.optional
        for t in (*sig.call_tensors.values(), *sig.outputs.values())
        if f"present({t.name})" not in point
    }
    lets = {n: _parse(e) for n, e in sig.let.items() if isinstance(e, str) and _parses(e)}
    while pending or lets:
        settled = {}
        for name in list(lets):
            value = _value_at(lets[name], point)
            if value is not _OPEN:
                settled[name] = value
                del lets[name]
        for name, cond in pending.items():
            node = _inlined(sig, _parse(cond)) if isinstance(cond, str) else None
            if node is None:
                value = True
            elif names(_bound(node, point)):
                value = _OPEN
            else:
                value = _value_at(node, point)
            if value is not _OPEN:
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


def _passed(sig: Signature, tensor: Tensor, point: dict) -> bool:
    if tensor.optional is True:
        return point.get(f"present({tensor.name})", False)
    return _holds(tensor.optional or None, point)


def _emitted(sig: Signature, tensor: Tensor, point: dict) -> bool:
    return _holds(tensor.nullable, point)


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


@dataclass
class Branch:
    """What one discriminant point selects, folded at that point."""

    # The shape of every passed input and emitted output.
    shapes: dict[str, ast.List]
    # Refinements, and the `let` definitions relevance reaches.
    rules: list[ast.expr]
    lets: dict[str, ast.expr]
    # Every name the branch's shapes, dtypes, refinements and reached `let`s read.
    relevant: set[str]


def branch(sig: Signature, point: dict, complete: bool = True) -> Branch:
    """The branch at `point`; with `complete=False` a tensor whose family has no case is left out."""
    shapes, dtypes = {}, []
    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        if not (_emitted(sig, t, point) if t.name in sig.outputs else _passed(sig, t, point)):
            continue
        try:
            shapes[t.name] = fold(expand(sig, t.shape, point), point)
        except SignatureError:
            if complete:
                raise
            continue
        dtypes.append(fold(_parse(t.dtype), point))
    rules = [fold(_parse(r), point) for r in sig.rules]
    written = [_bound(_parse(r), point) for r in sig.rules]
    relevant = set().union(set(), *(names(n) for n in (*shapes.values(), *dtypes, *written)))
    relevant -= set(DTYPE_BITS)
    lets: dict[str, ast.expr] = {}
    while more := sorted((relevant & set(sig.let)) - set(lets)):
        for n in more:
            lets[n] = fold(_parse(sig.let[n]), point)
            relevant |= names(lets[n])
    return Branch(shapes, rules, lets, relevant)


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


def _inference_errors(sig: Signature, point: dict, b: Branch) -> list[str]:
    """Why the inference plan fails at `point`, or nothing."""
    solvable = {n for n, k in sig.forall.items() if k != "Seq[Int]"}
    inputs = {t: node for t, node in b.shapes.items() if t in sig.call_tensors}
    known = set(sig.params) | {k for k in point if "(" not in k and "." not in k}
    dtypes = [fold(_parse(sig.call_tensors[t].dtype), point) for t in inputs]
    known |= {d.id for d in dtypes if isinstance(d, ast.Name)} & solvable
    known |= {s.name for s in unification(inputs, known, b.lets) if s.name}
    return [
        f"index {u!r} cannot be solved from the inputs"
        for u in sorted((b.relevant & solvable) - known)
    ]


def _read_errors(sig: Signature, point: dict, b: Branch) -> list[str]:
    """Reads the branch forbids: `v.value` where `present(v)` is false, a field its constructor lacks."""
    errors = []
    nodes = {
        **{f"tensor {t!r} shape": n for t, n in b.shapes.items()},
        **{f"shape_rules[{i}]": n for i, n in enumerate(b.rules)},
        **{
            f"let {n}": fold(_parse(e), point)
            for n, e in sig.let.items()
            if isinstance(e, str) and _parses(e)
        },
    }
    for where, node in nodes.items():
        errors += _branch_reads(where, node, point, sig.kind, sig.adts)
    return errors


def _branch_reads(where: str, node: ast.expr, point: dict, kind_of, adts: dict) -> list[str]:
    """Reads `point` forbids in `node`: `v.value` where `present(v)` is false, a field the
    constructor `point` fixes lacks."""
    errors = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Attribute):
            continue
        receiver = n.value
        payload = (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "value"
            and isinstance(receiver.value, ast.Name)
        )
        if not (isinstance(receiver, ast.Name) or payload):
            continue
        v, f = ast.unparse(receiver), n.attr
        if f == "value" and point.get(f"present({v})") is False:
            errors.append(f"{where} reads {v}.value where present({v}) is false")
        ctor = point.get(f"{v}.kind")
        kind = kind_of(receiver.value.id if payload else v)
        kind = kind.payload() if kind is not None and payload else kind
        adt = adts[kind.name]["sum"] if kind is not None and kind.tag == "ADT" else {}
        if ctor is not None and f != "kind" and f not in (adt[ctor].get("fields") or {}):
            errors.append(f"{where} reads {v}.{f}, which constructor {ctor!r} lacks")
    return errors


def _cycle(graph: dict[str, set[str]]) -> list[str] | None:
    done, active = set(), []

    def visit(n):
        if n in active:
            return [*active[active.index(n) :], n]
        if n in done:
            return None
        active.append(n)
        for m in sorted(graph.get(n, ())):
            found = visit(m)
            if found:
                return found
        active.pop()
        done.add(n)
        return None

    for n in sorted(graph):
        found = visit(n)
        if found:
            return found
    return None


# ---------------------------------------------------------------- checks


def _dtype_kind(sig: Signature, node: ast.expr, errors: list[str]) -> Kind | None:
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
    callee = _callee(node.func) if isinstance(node, ast.Call) else None
    formal, result = PRIMITIVE_KINDS.get(callee, ((), None))
    if result != "DType":
        errors.append(f"{ast.unparse(node)!r} is not a dtype expression")
        return None
    if len(node.args) != len(formal) or node.keywords:
        errors.append(f"{callee} takes {len(formal)} arguments")
        return DTYPE
    domains = []
    for arg, wanted in zip(node.args, formal, strict=True):
        kind = _dtype_kind(sig, arg, errors)
        if kind is not None and not fits(kind, _spec(wanted)):
            errors.append(f"argument {ast.unparse(arg)} has kind {kind}, expected {wanted}")
        domains.append(_dtype_values(kind))
    if any(d is None for d in domains):
        return DTYPE
    # Every argument has finite dtypes, so the result's are the primitive over their product.
    values = {PRIMITIVES[callee](*combo) for combo in itertools.product(*domains)}
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


def _dtype_errors(sig: Signature, t: Tensor, node: ast.expr) -> list[str]:
    errors: list[str] = []
    kind = _dtype_kind(sig, node, errors)
    if kind is not None and kind.tag == "Maybe":
        errors.append(f"{ast.unparse(node)} may be absent; use coalesce_dtype")
    return [f"tensor {t.name!r} dtype: {e}" for e in errors]


def _shape_errors(kinds: _Kinds, node: ast.List) -> list[str]:
    for axis in node.elts:
        if isinstance(axis, ast.Starred):
            kinds.expect(axis.value, seq(INT))
        else:
            kinds.expect(axis, INT)
    return kinds.errors


def _application_errors(kinds: _Kinds, node: ast.Subscript) -> list[str]:
    family = kinds.sig.types[node.value.id]
    args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    formal = family.get("params") or {}
    if len(args) != len(formal):
        kinds.error(f"{node.value.id} takes {len(formal)} arguments, got {len(args)}")
        return kinds.errors
    for arg, kind in zip(args, formal.values(), strict=True):
        kinds._argument(arg, _formal(kind, kinds.sig.adts))
    return kinds.errors


def signature_schema_errors(sig: dict) -> list[str]:
    """Keys a fixed section of `signature` does not define, and tensor fields of the wrong type."""
    malformed = [
        f"`{key}` must be a mapping"
        for key in ("forall", "params", "inputs", "outputs", "types", "let")
        if sig.get(key) is not None and not isinstance(sig[key], dict)
    ]
    sig = {k: ({} if f"`{k}` must be a mapping" in malformed else v) for k, v in sig.items()}
    sections = [("signature", sig, _SIGNATURE_KEYS)]
    tensors = [
        (g, f"{g}.{n}", d) for g in ("inputs", "outputs") for n, d in (sig.get(g) or {}).items()
    ]
    for n, d in (sig.get("params") or {}).items():
        if isinstance(d, dict) and "shape" in d:
            tensors.append(("params", f"params.{n}", d))
        else:
            sections.append((f"params.{n}", d, _PARAM_KEYS))
    sections += [(where, d, _ROLE_KEYS[role]) for role, where, d in tensors]
    sections += [(f"types.{n}", d, _FAMILY_KEYS) for n, d in (sig.get("types") or {}).items()]
    errors = malformed + [
        f"{where}: unknown key {k!r}"
        for where, d, keys in sections
        if isinstance(d, dict)
        for k in sorted(set(d) - keys, key=repr)
    ]
    for _, where, d in tensors:
        for key, (ok, what) in _TENSOR_FIELDS.items():
            if isinstance(d, dict) and key in d and not ok(d[key]):
                errors.append(f"{where}.{key}: {d[key]!r} is not {what}")
    return errors


def _default_fits(value, type_text: str, adts: dict) -> bool:
    """Whether a default is a value of its `type`: a `float` or `Number` member takes numbers."""
    for member in _split(type_text):
        if member in ("float", "Number"):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
        elif _literal_fits(value, parse_type(member, adts), adts):
            return True
    return False


def _literal_fits(value, kind: Kind, adts: dict) -> bool:  # noqa: C901 - one case per tag
    """Whether a YAML literal is a value of `kind`, as a default is written."""
    if kind.tag in ("Union", "Maybe"):
        members = kind.members if kind.tag == "Union" else (kind.item, NONE)
        return any(_literal_fits(value, m, adts) for m in members)
    if kind.tag == "Value":
        return True
    if kind.tag == "None":
        return value is None
    if kind.tag == "Bool":
        return isinstance(value, bool)
    if kind.tag == "Int":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (value >= 0 or not kind.nonneg)
        )
    if kind.tag in ("Str", "DType"):
        known = kind.values if kind.tag == "Str" or kind.values else DTYPE_BITS
        return isinstance(value, str) and (known is None or value in known)
    if kind.tag == "Seq":
        return (
            isinstance(value, list)
            and kind.length in (None, len(value))
            and all(_literal_fits(v, kind.item, adts) for v in value)
        )
    if kind.tag == "ADT":
        ctors = adts[kind.name]["sum"]
        if not (isinstance(value, dict) and len(value) == 1 and next(iter(value)) in ctors):
            return False
        ((ctor, fields),) = value.items()
        declared = ctors[ctor].get("fields") or {}
        fits_fields = (
            isinstance(fields or {}, dict)
            and set(fields or {}) == set(declared)
            and all(
                (k := _field_kind(declared[f])) is not None and _literal_fits(v, k, adts)
                for f, v in (fields or {}).items()
            )
        )
        invariant = ctors[ctor].get("invariant")
        return fits_fields and (
            invariant is None or _value_at(_parse(invariant), fields or {}) is True
        )
    return False


def _field_type(decl: object) -> str:
    """An ADT field's kind text, written bare or as `{type: ..., python: ...}`."""
    return str(decl.get("type") if isinstance(decl, dict) else decl)


def _enum_text(text: str) -> bool:
    """`'a' | 'b'`: a union of quoted string literals."""
    return all(_quoted(m) for m in _split(text))


def _field_kind(decl: object) -> Kind | None:
    """The kind an ADT field declares: Dim, Int, Bool or an enum; None for anything else."""
    text = _field_type(decl)
    if _enum_text(text):
        return literals(_enum_values(text))
    return {
        "Dim": Kind("Int", nonneg=True),
        "Int": INT,
        "int": INT,
        "Bool": BOOL,
        "bool": BOOL,
    }.get(text)


def _identifier(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier()


def _qualified(name: object) -> bool:
    return (
        isinstance(name, str) and len(parts := name.split(".")) > 1 and all(map(_identifier, parts))
    )


def _adt_errors(name: object, adt: object) -> list[str]:
    """One ADT: constructors with their `python` class, field kinds, invariants in the language."""
    ctors = adt.get("sum") if isinstance(adt, dict) else None
    if not _identifier(name) or not isinstance(ctors, dict) or not ctors or set(adt) != {"sum"}:
        return [f"adt {name!r}: needs an identifier and exactly a non-empty `sum` of constructors"]
    errors = []
    for ctor, spec in ctors.items():
        where = f"adt {name}.{ctor}"
        if not _identifier(ctor) or not isinstance(spec, dict) or set(spec) - _ADT_CTOR_KEYS:
            errors.append(f"{where}: needs an identifier and keys among {sorted(_ADT_CTOR_KEYS)}")
            continue
        if not _qualified(spec.get("python")):
            errors.append(f"{where}: `python` must name the constructor's class, `module.Class`")
        fields = spec.get("fields", {})
        if not isinstance(fields, dict):
            errors.append(f"{where}: `fields` must be a mapping")
            continue
        for f, k in fields.items():
            mapping = isinstance(k, dict)
            if (
                not _identifier(f)
                or _field_kind(k) is None
                or (mapping and (set(k) != {"type", "python"} or not _qualified(k["python"])))
            ):
                errors.append(f"{where}.{f}: needs an identifier and a Dim, Int, Bool or enum kind")
        if "invariant" not in spec:
            continue
        try:
            node = _parse(spec["invariant"])
        except SignatureError as exc:
            errors.append(f"{where} invariant: {exc}")
            continue
        errors += [f"{where} invariant: {e}" for e in _language_errors(node, False)]
        errors += [
            f"{where} invariant: name {n!r} is not a field"
            for n in sorted(names(node) - set(fields))
        ]
        env = {f: _field_kind(k) for f, k in fields.items()}
        kinds = _Kinds(Signature(str(name), {}, {}, {}, {}, {}, {}, {}, []), env, where)
        kinds.expect(node, BOOL)
        errors += kinds.errors
    return errors


def check_adts(adts: object) -> tuple[dict, list[str]]:
    """The ADTs of `types.yaml` that are well formed, and the diagnostics of the others."""
    if not isinstance(adts, dict):
        return {}, ["adts: must be a mapping"]
    accepted, errors = {}, []
    for name, declared in sorted(adts.items(), key=lambda kv: repr(kv[0])):
        found = _adt_errors(name, declared)
        errors += found
        if not found:
            accepted[name] = declared
    return accepted, errors


def _pattern_errors(where: str, pattern: object, kind: Kind, adts: dict) -> list[str]:
    """A `when` component against its discriminant's kind: a sealed ADT, `Bool` or an enum."""
    if pattern == "_":
        return []
    if kind.tag == "ADT":
        ctors = adts[kind.name]["sum"]
        if not isinstance(pattern, dict) or len(pattern) != 1:
            return [f"{where}: pattern {pattern!r} is not `{{ctor: fields}}`"]
        ((ctor, fields),) = pattern.items()
        if ctor not in ctors:
            return [f"{where}: {ctor!r} is not a constructor of {kind}"]
        if fields == "_":
            return []
        if not isinstance(fields, dict):
            return [f"{where}: fields of {ctor!r} must be `_` or a mapping"]
        declared = ctors[ctor].get("fields") or {}
        errors = []
        for f, v in fields.items():
            field_kind = _field_kind(declared[f]) if f in declared else None
            if field_kind is None:
                errors.append(f"{where}: {ctor!r} has no field {f!r}")
            elif field_kind.tag == "Int":
                errors.append(f"{where}: {ctor}.{f} is not a finite field; it cannot be matched")
            elif field_kind.tag == "Str" and v not in field_kind.values:
                errors.append(f"{where}: {v!r} is not a value of {ctor}.{f}")
            elif field_kind == BOOL and not isinstance(v, bool):
                errors.append(f"{where}: {ctor}.{f} is a Bool")
        return errors
    if kind == BOOL and not isinstance(pattern, bool):
        return [f"{where}: {pattern!r} is not a Bool"]
    if kind.tag == "Str" and pattern not in kind.values:
        return [f"{where}: {pattern!r} is not one of {kind}"]
    return []


def _family_env(formals: dict, adts: dict) -> dict:
    """A family's formals as a kind environment; a formal whose kind is malformed is left out."""
    env = {}
    for p, k in formals.items():
        with contextlib.suppress(ValueError):
            env[p] = _formal(k, adts)
    return env


def _case_ctors(spec: dict, case: dict) -> dict:
    """What a case's pattern fixes: each ADT formal's constructor, as `w`, and its fields, `w.f`."""
    keys = spec["match"] if isinstance(spec["match"], list) else [spec["match"]]
    patterns = case["when"] if isinstance(case["when"], list) else [case["when"]]
    facts = {}
    for k, p in zip(keys, patterns, strict=False):
        if isinstance(p, dict) and len(p) == 1:
            ((ctor, fields),) = p.items()
            facts[k] = ctor
            facts |= {f"{k}.{f}": v for f, v in fields.items()} if isinstance(fields, dict) else {}
        elif p != "_" and isinstance(k, str) and k.endswith(".kind"):
            facts[k.removesuffix(".kind")] = p
        elif p != "_" and isinstance(k, str) and "." in k:
            facts[k] = p
    return facts


def _family_errors(fam: str, spec: object, sig: Signature) -> list[str]:
    """A type family's formals and `match`, and each case pattern against the kind it matches."""
    if not isinstance(spec, dict) or "match" not in spec or not spec.get("cases"):
        return [f"type family {fam}: needs `match` and at least one case"]
    formals = {} if spec.get("params") is None else spec["params"]
    if not isinstance(formals, dict) or not isinstance(spec["cases"], list):
        return [f"type family {fam}: `params` must be a mapping and `cases` a list"]
    env = _family_env(formals, sig.adts)
    errors = [
        f"type family {fam}: formal {p!r} needs an identifier and a kind"
        for p in formals
        if not _identifier(p) or p not in env
    ]
    keys = spec["match"] if isinstance(spec["match"], list) else [spec["match"]]
    kinds = []
    for key in keys:
        where = f"type family {fam} match {key!r}"
        try:
            node = _parse(key)
        except SignatureError as exc:
            errors.append(f"{where}: {exc}")
            continue
        errors += [f"{where}: {e}" for e in _language_errors(node, False)]
        errors += [
            f"{where}: name {n!r} is not a formal" for n in sorted(names(node) - set(formals))
        ]
        kinds_pass = _Kinds(sig, env, where)
        kind = kinds_pass.of(node)
        errors += kinds_pass.errors
        if kind is not None and (
            kind == BOOL or kind.tag == "ADT" or (kind.tag == "Str" and kind.values)
        ):
            kinds.append(kind)
        else:
            errors.append(f"{where}: has kind {kind}, not Bool, an enum or an ADT")
    if errors:
        return errors
    for j, case in enumerate(spec["cases"]):
        where = f"type family {fam} case {j}"
        if not isinstance(case, dict) or set(case) != {"when", "is"}:
            errors.append(f"{where}: needs exactly `when` and `is`")
            continue
        if isinstance(case["when"], list) != isinstance(spec["match"], list):
            errors.append(f"{where}: `when` is a list exactly when `match` is")
            continue
        patterns = case["when"] if isinstance(case["when"], list) else [case["when"]]
        if len(patterns) != len(keys):
            errors.append(f"{where}: {len(patterns)} components, `match` has {len(keys)}")
            continue
        for pattern, kind in zip(patterns, kinds, strict=True):
            errors += _pattern_errors(where, pattern, kind, sig.adts)
    return errors


def _coverage_errors(fam: str, spec: dict, adts: dict) -> list[str]:
    """Cases of a family no shape applies, exhaustive and disjoint over its formals' values, each
    reading only what the point that selects it allows."""
    env = _family_env(spec.get("params") or {}, adts)
    axes = [a for p, k in env.items() if (a := _finite_axis(p, k, adts)) is not None]
    for combo in itertools.product(*(values for _, values in axes)):
        point = {}
        for (key, _), value in zip(axes, combo, strict=True):
            point.update(value if key is None else {key: value})
        if any(_invariant_fails(adts, p, k, point) for p, k in env.items()):
            continue
        try:
            case = _parse(_case(fam, spec, {}, point))
        except SignatureError as exc:
            return [f"type family {exc}"]
        reads = _branch_reads(f"type family {fam} at {point}", case, point, env.get, adts)
        if reads:
            return reads
    return []


def _combo_errors(sig: Signature) -> list[str]:
    """`dtype_combos`: equal keys, declared dtype columns, members of each column's set, no repeats."""
    rows = sig.dtype_combos
    if not all(isinstance(r, dict) and all(isinstance(v, str) for v in r.values()) for r in rows):
        return ["dtype_combos: every row maps a dtype name to a string"]
    errors = []
    if any(set(r) != set(rows[0]) for r in rows):
        errors.append("dtype_combos: rows have different keys")
    for key in sorted(set().union(set(), *rows), key=repr):
        kind = sig.kind(key)
        if kind is None or kind.tag != "DType":
            errors.append(f"dtype_combos: {key!r} is not a DType index or dtype parameter")
            continue
        members = kind.values or set(DTYPE_BITS)
        errors += [
            f"dtype_combos: {r[key]!r} is not in the set of {key!r}"
            for r in rows
            if key in r and r[key] not in members
        ]
    if len({frozenset(r.items()) for r in rows}) != len(rows):
        errors.append("dtype_combos: a row repeats")
    return errors


def kind_env(sig: Signature, lets: dict[str, ast.expr]) -> tuple[dict, list[str]]:
    """The kind of every index, parameter and `let`, lets in dependency order; their misuse."""
    env: dict[str, str | None] = {n: sig.kind(n) for n in (*sig.forall, *sig.params)}
    errors: list[str] = []
    pending = list(lets)
    while pending:
        ready = [n for n in pending if not names(lets[n]) & set(pending) - {n}] or pending[:1]
        for n in ready:
            kinds = _Kinds(sig, env, f"let {n}")
            env[n] = kinds.of(lets[n])
            errors += kinds.errors
            if env[n] is not None and env[n].payload().tag == "ADT":
                # Constructor facts are keyed by the parameter; a renamed ADT would escape them.
                errors.append(f"let {n}: holds an ADT value; read its fields where they are used")
            pending.remove(n)
    return env, errors


_ROOFLINE_KEYS = frozenset({"flops", "bytes", "func"})


def roofline_plan(
    sig: Signature | None, roofline: object, resolve: bool = True
) -> tuple[list[str], dict | None]:
    """The `roofline` field (docs/design/roofline.md) checked, and what emission reads.

    The plan is `{"func": callable}` or `{"flops": tree, "bytes": tree or None}`; it is None
    when there are diagnostics. `ix` is the signature's indices other than value lists, its
    parameters and its `let`s; an inline expression may also call `bytes(t)` and `present(t)`.
    A `func` is imported only when *resolve* is set: an entry not yet implemented may name a
    formula that does not exist yet. Without a readable signature (*sig* None) the names a
    formula reads cannot be judged, and only its form is checked.
    """
    if not isinstance(roofline, dict):
        return [f"roofline must be a mapping of {sorted(_ROOFLINE_KEYS)}"], None
    errors = [f"roofline: unknown key {k!r}" for k in sorted(set(roofline) - _ROOFLINE_KEYS)]
    if ("func" in roofline) == ("flops" in roofline) or (
        "func" in roofline and "bytes" in roofline
    ):
        errors.append("roofline gives either `flops` (and optional `bytes`) or `func`")
    if "func" in roofline:
        path, fn = roofline["func"], None
        if not (_qualified(path) and path.startswith("tileops.perf.formulas.")):
            errors.append(f"roofline.func {path!r} is not tileops.perf.formulas.<name>")
        elif resolve:
            module, _, name = path.rpartition(".")
            try:
                fn = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                errors.append(f"roofline.func {path!r} does not resolve")
            else:
                if not callable(fn):
                    errors.append(f"roofline.func {path!r} is not callable")
        return errors, (None if errors else {"func": fn})
    plan = {}
    for key in ("flops", "bytes"):
        if key not in roofline:
            continue
        try:
            node = _parse(roofline[key])
        except SignatureError as exc:
            errors.append(f"roofline.{key}: {exc}")
            continue
        unbytes = _Unbytes().visit(copy.deepcopy(node))
        errors += [f"roofline.{key}: {e}" for e in _language_errors(unbytes, False)]
        plan[key] = node
        if sig is None:
            continue
        ix = {n for n, k in sig.forall.items() if k != "Seq[Int]"} | set(sig.params) | set(sig.let)
        tensors = {*sig.call_tensors, *sig.outputs}
        maybes = {p for p in sig.params if sig.kind(p).tag == "Maybe"}
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
        for call in calls:
            callee = _callee(call.func)
            buffered = {"out"} if any(o.buffer for o in sig.outputs.values()) else set()
            allowed = tensors | (maybes | buffered if callee == "present" else set())
            if callee in ("bytes", "present") and not (
                len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in allowed
            ):
                errors.append(f"roofline.{key}: {ast.unparse(call)} names no tensor")
        read = names(_Unpresent().visit(_Unbytes().visit(copy.deepcopy(node))))
        errors += [f"roofline.{key}: name {n!r} is not in ix" for n in sorted(read - ix)]
    return errors, (None if errors else {"bytes": None, **plan})


def effect_errors(sig: Signature) -> list[str]:
    """Effect declarations: `write_only` is written, an alias names a written input,
    one output at most is buffered."""
    errors = [
        f"tensor {t.name!r}: `write_only` needs `mutated: true`"
        for t in sig.call_tensors.values()
        if t.write_only and t.mutated is not True
    ]
    for t in sig.outputs.values():
        target = sig.inputs.get(t.alias) if t.alias else None
        if t.alias and (target is None or target.mutated is False):
            errors.append(f"output {t.name!r}: alias {t.alias!r} is not an input that is written")
        if t.alias and t.buffer:
            errors.append(f"output {t.name!r}: an aliased output takes no `out` buffer")
    buffered = [t.name for t in sig.outputs.values() if t.buffer]
    if len(buffered) > 1:
        errors.append(f"outputs {buffered}: one output takes the `out` buffer, not several")
    return errors


class _Unbytes(ast.NodeTransformer):
    """Replace `bytes(t)` by a constant: it reads `t`'s size, not its value."""

    def visit_Call(self, node):
        if _callee(node.func) == "bytes":
            return ast.Constant(0)
        return self.generic_visit(node)


def check_entry(name: str, entry: dict, adts: dict) -> tuple[list[str], list[str]]:  # noqa: C901
    """Return `(errors, warnings)` for one entry; each message is prefixed with its op name.

    *adts* are the ADTs `check_adts` accepts.
    """
    try:
        sig, problems = _read_signature(name, entry, adts)
    except SignatureError as exc:
        return [f"{name}: {exc}"], []
    errors: list[str] = problems + _combo_errors(sig)
    for p, decl in sig.params.items():
        if "kw_only" in decl and not isinstance(decl["kw_only"], bool):
            errors.append(f"params.{p}: kw_only must be true or false")
        try:
            parse_type(decl.get("type"), sig.adts)
        except ValueError as exc:
            errors.append(f"params.{p}: {exc}")
            continue
        default = decl.get("default")
        if "default" in decl and not _default_fits(default, str(decl.get("type")), sig.adts):
            errors.append(f"params.{p}: default {decl['default']!r} is not a {decl.get('type')}")
    declared = [*sig.forall, *sig.params, *sig.ctor_tensors, *sig.let, *sig.inputs, *sig.outputs]
    errors += [
        f"name {d!r} is declared twice"
        for d in sorted({n for n in declared if declared.count(n) > 1})
    ]

    parsed: dict[str, ast.expr] = {}

    def parse(key: str, text: object, shape: bool = False) -> ast.expr | None:
        try:
            node = _parse(text)
        except SignatureError as exc:
            errors.append(f"{key}: {exc}")
            return None
        errors.extend(f"{key}: {e}" for e in _language_errors(node, shape))
        parsed[key] = node
        return node

    tensors = {**sig.call_tensors, **sig.outputs}
    lets = {n: node for n, e in sig.let.items() if (node := parse(f"let {n}", e)) is not None}
    rules = [
        node for i, r in enumerate(sig.rules) if (node := parse(f"shape_rules[{i}]", r)) is not None
    ]
    shapes = {t: parse(f"tensor {t!r} shape", d.shape, shape=True) for t, d in tensors.items()}
    flags = {
        (t, a): parse(f"tensor {t!r} {a}", getattr(d, a))
        for t, d in tensors.items()
        for a in ("optional", "nullable", "mutated")
        if isinstance(getattr(d, a), str)
    }
    for t, d in tensors.items():
        dtype = parse(f"tensor {t!r} dtype", d.dtype)
        if dtype is not None:
            errors += _dtype_errors(sig, d, dtype)
    malformed = {fam: _family_errors(fam, spec, sig) for fam, spec in sig.types.items()}
    errors += [e for family in malformed.values() for e in family]
    sig.types = {fam: spec for fam, spec in sig.types.items() if not malformed[fam]}
    cases = {}
    for fam, spec in sig.types.items():
        for j, case in enumerate(spec["cases"]):
            cases[(fam, j)] = parse(f"type family {fam} case {j}", case["is"], shape=True)

    env, let_errors = kind_env(sig, lets)
    errors += let_errors
    value_lists = {n for n, k in sig.forall.items() if k == "Seq[Int]"}
    # A parameter outside the type system appears only in refinements.
    untyped = {
        p
        for p, k in env.items()
        if p in sig.params
        and k is not None
        and all(
            m == VALUE or (m.tag == "Str" and not m.values)
            for m in (k.payload().members or (k.payload(),))
        )
    }
    for key, node in parsed.items():
        formals = (
            set(sig.types[key.split()[2]].get("params") or {})
            if key.startswith("type family ")
            else set()
        )
        read = names(_Unpresent().visit(copy.deepcopy(node))) - formals
        errors += [
            f"{key}: tensor {n!r} is read as a value; use its shape or present({n})"
            for n in sorted(read & set(tensors))
        ]
        if not key.startswith("shape_rules"):
            errors += [
                f"{key}: parameter {n!r} of type {sig.params[n].get('type')} takes no part in types"
                for n in sorted(read & untyped)
            ]
        scope = set(env) | set(tensors) | set(malformed)
        if key.startswith("type family "):
            scope = set(sig.types[key.split()[2]].get("params") or {}) | set(malformed)
        if key.endswith(" dtype"):
            scope |= set(DTYPE_BITS)
        errors += [f"{key}: name {n!r} is not declared" for n in sorted(names(node) - scope)]
        if names(node) & value_lists:
            errors.append(
                f"{key}: value list {sorted(names(node) & value_lists)} appears outside `values`"
            )
    for i, node in enumerate(rules):
        kinds = _Kinds(sig, env, f"shape_rules[{i}]")
        kinds.expect(node, BOOL)
        errors += kinds.errors
    for (t, a), node in flags.items():
        if node is not None:
            kinds = _Kinds(sig, env, f"tensor {t!r} {a}")
            kinds.expect(node, BOOL)
            errors += kinds.errors

    def shape_errors(kinds: _Kinds, node: ast.expr | None) -> list[str]:
        if isinstance(node, ast.List):
            return _shape_errors(kinds, node)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in malformed
        ):
            return [] if malformed[node.value.id] else _application_errors(kinds, node)
        if node is None:
            return []
        return [f"{kinds.where}: {ast.unparse(node)!r} is not a list or a type-family application"]

    for t, node in shapes.items():
        errors += shape_errors(_Kinds(sig, env, f"tensor {t!r} shape"), node)
    for (fam, j), node in cases.items():
        spec = sig.types[fam]
        local = _family_env(spec.get("params") or {}, sig.adts)
        ctors = _case_ctors(spec, spec["cases"][j])
        errors += shape_errors(_Kinds(sig, local, f"type family {fam} case {j}", ctors), node)

    let_cycle = _cycle({n: names(node) & set(lets) for n, node in lets.items()})
    if let_cycle:
        errors.append(f"let cycle: {' -> '.join(let_cycle)}")
    family_refs = {fam: set() for fam in sig.types}
    for (fam, _), node in cases.items():
        if node is not None:
            family_refs[fam] |= {
                n.value.id
                for n in ast.walk(node)
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id in sig.types
            }
    family_cycle = _cycle(family_refs)
    if family_cycle:
        errors.append(f"type family cycle: {' -> '.join(family_cycle)}")
    # A family a shape applies is covered on each point; one nothing applies, over its formals.
    applied = {
        n.value.id
        for node in shapes.values()
        if node is not None
        for n in ast.walk(node)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
    }
    while more := set().union(*(family_refs[f] for f in applied & set(family_refs))) - applied:
        applied |= more
    for fam in sorted(set(sig.types) - applied):
        errors += _coverage_errors(fam, sig.types[fam], sig.adts)
    if errors:
        return sorted({f"{name}: {e}" for e in errors}), []
    return _point_errors(sig, env)


def _point_errors(sig: Signature, env: dict) -> tuple[list[str], list[str]]:
    """Check each group of dependent discriminants on its own, the other groups held accepted."""

    def accepted(point: dict) -> bool:
        try:
            return rejecting_rule(sig, complete_point(sig, point, strict=False)) is None
        except SignatureError:
            return False

    groups = [list(_points(g)) for g in _discriminant_groups(sig)]
    held = {}
    for points in groups:
        held.update(next((p for p in points if accepted(p)), points[0]))
    warnings = []
    combinations = sum(len(points) for points in groups)
    if combinations > DISCRIMINANT_LIMIT:
        warnings.append(
            f"{sig.name}: {combinations} discriminant combinations exceed {DISCRIMINANT_LIMIT}"
        )
    errors: list[str] = []
    for part in [p for points in groups for p in points] or [{}]:
        point = {**held, **part}
        try:
            point = complete_point(sig, point)
            accepted = rejecting_rule(sig, point) is None
            for t in sig.call_tensors.values():
                _holds(t.mutated if isinstance(t.mutated, str) else None, point)
            b = branch(sig, point, complete=accepted)
            errors += _read_errors(sig, point, b)
            ctors = {k[: -len(".kind")]: v for k, v in point.items() if k.endswith(".kind")}
            # An absent `Maybe` parameter is None on this branch.
            local = {
                n: NONE
                if k is not None and k.tag == "Maybe" and point.get(f"present({n})") is False
                else k
                for n, k in env.items()
            }
            for t, node in b.shapes.items():
                errors += _shape_errors(
                    _Kinds(sig, local, f"tensor {t!r} shape", ctors, point=True), node
                )
            for n, node in b.lets.items():
                kinds = _Kinds(sig, local, f"let {n}", ctors, point=True)
                kinds.of(node)
                errors += kinds.errors
            for i, rule in enumerate(b.rules):
                kinds = _Kinds(sig, local, f"shape_rules[{i}]", ctors, point=True)
                kinds.expect(rule, BOOL)
                errors += kinds.errors
            if accepted:
                errors += _inference_errors(sig, point, b)
                errors += [
                    f"dtype_combos column {k!r} is not relevant at {point}"
                    for k in sorted(set().union(set(), *sig.dtype_combos) - b.relevant)
                ]
        except SignatureError as exc:
            errors.append(f"at {point}: {exc}")
    return sorted({f"{sig.name}: {e}" for e in errors}), warnings
