"""The closed expression language of the manifest (docs/design/manifest.md § Signature).

Parsing, the language check, kind inference over an explicit `KindEnv`, folding at a
discriminant point, and the one evaluation protocol every expression site shares: a failure
of any kind becomes an `EvaluationError` naming the declaration that failed.
"""

from __future__ import annotations

import ast
import copy
import functools
import math
from dataclasses import dataclass, field

from .kinds import (
    BOOL,
    INT,
    NONE,
    VALUE,
    Kind,
    common,
    comparable,
    exclude,
    fits,
    join,
    literals,
    ordered,
    parse_spec,
    restrict,
    seq,
    union,
)
from .primitives import PRIMITIVE_KINDS, namespace

__all__ = [
    "OPEN",
    "EvaluationError",
    "KindEnv",
    "SignatureError",
    "Substitute",
    "Unpresent",
    "argument_errors",
    "bind",
    "call_errors",
    "callee",
    "evaluate",
    "fold",
    "infer_kinds",
    "language_errors",
    "names",
    "parse",
    "value_at",
]


class SignatureError(ValueError):
    """A declaration outside the schema; the message names it."""


class EvaluationError(SignatureError):
    """An expression that failed to evaluate; the message names its declaration."""


# The value of an expression that reads more than a point fixes.
OPEN = object()

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


# ---------------------------------------------------------------- parsing and the language


@functools.lru_cache(maxsize=4096)
def _parsed(text: str) -> ast.expr | str:
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        return f"cannot parse {text!r}: {exc.msg}"


def parse(text: object) -> ast.expr:
    """A fresh copy of `text`'s tree; each distinct string is parsed once."""
    if not isinstance(text, str):
        raise SignatureError(f"expression {text!r} is not a string")
    node = _parsed(text)
    if isinstance(node, str):
        raise SignatureError(node)
    return copy.deepcopy(node)


def callee(func: ast.expr) -> str | None:
    """The name a call calls: `f` or `mod.f`; None for anything else."""
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
        free = set() if callee(node.func) else names(node.func)
        return free.union(*(names(a) for a in (*node.args, *(k.value for k in node.keywords))))
    if isinstance(node, ast.GeneratorExp):
        free, bound = set(), set()
        for gen in node.generators:
            free |= names(gen.iter) - bound
            bound |= {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
            free |= set().union(set(), *(names(c) for c in gen.ifs)) - bound
        return free | (names(node.elt) - bound)
    return set().union(set(), *(names(c) for c in ast.iter_child_nodes(node)))


class Unpresent(ast.NodeTransformer):
    """Drop `present(x)`, which reads `x`'s presence and not its value."""

    def visit_Call(self, node):
        if callee(node.func) == "present":
            return ast.Constant(True)
        return self.generic_visit(node)


class Substitute(ast.NodeTransformer):
    """Replace each name in `sub` by a copy of its tree."""

    def __init__(self, sub: dict):
        self.sub = sub

    def visit_Name(self, node):
        return copy.deepcopy(self.sub.get(node.id, node))


def language_errors(node: ast.expr, shape: bool) -> list[str]:
    """Constructs outside the expression language; a shape's outer list is allowed."""
    errors = []
    root = node
    comprehension_args = {
        id(c.args[0])
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and callee(c.func) in _COMPREHENSION_CALLEES and c.args
    }
    for sub in ast.walk(node):
        if shape and sub is root and isinstance(sub, (ast.List, ast.Subscript)):
            continue
        if (
            shape
            and isinstance(sub, ast.Starred)
            and isinstance(root, ast.List)
            and sub in root.elts
        ):
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
            and callee(sub.func) != "present"
            and callee(sub.func) not in PRIMITIVE_KINDS
        ):
            errors.append(
                f"{callee(sub.func) or ast.unparse(sub.func)} is not a built-in primitive"
            )
    return errors


# ---------------------------------------------------------------- kinds


@dataclass
class KindEnv:
    """What kind inference reads: the kind of every name, and what a branch fixes."""

    # Every index, parameter, `let` and bound variable in scope.
    kinds: dict
    # ADT name -> constructor -> field -> kind (None where the field's kind is malformed).
    adts: dict = field(default_factory=dict)
    # Type family names, which are not values.
    families: frozenset = frozenset()
    # Names `present(...)` may take besides a `Maybe` value: tensors and `out`.
    presence: frozenset = frozenset()
    # Parameters declared `Maybe`, whose `v.value` is legal where `present(v)` holds.
    maybes: frozenset = frozenset()
    # What a branch fixes: the constructor an ADT-valued name holds, keyed by the name (or
    # `v.value` for a payload), and each finite field it fixes, keyed `name.field`.
    facts: dict = field(default_factory=dict)
    # On one discriminant point a field read its constructor lacks is reported by the
    # point's read check instead.
    point: bool = False

    def narrowed(self, kinds: dict) -> KindEnv:
        return KindEnv(
            {**self.kinds, **kinds},
            self.adts,
            self.families,
            self.presence,
            self.maybes,
            self.facts,
            self.point,
        )


def infer_kinds(
    node: ast.expr, env: KindEnv, where: str, expected: Kind | None = None
) -> tuple[Kind | None, list[str]]:
    """The kind of `node` under `env`, and its misuse; with `expected`, the kind must fit it."""
    infer = _Infer(env, where)
    if expected is None:
        return infer.of(node), infer.errors
    infer.expect(node, expected)
    return None, infer.errors


def argument_errors(node: ast.expr, expected: Kind, env: KindEnv, where: str) -> list[str]:
    """`node` passed where `expected` is taken: a type-family argument."""
    infer = _Infer(env, where)
    infer.argument(node, expected)
    return infer.errors


def call_errors(node: ast.Call, table: dict, env: KindEnv, where: str) -> list[str]:
    """A call of an entry of `table`, its arguments bound and checked."""
    infer = _Infer(env, where)
    infer.call(node, table)
    return infer.errors


def _formal(text: object, adts: dict) -> Kind:
    """A kind as a table, a family formal or a primitive signature writes it."""
    return parse_spec(str(text), adts)


def _guards(node: ast.expr, env: KindEnv) -> tuple[dict, dict]:
    """The kinds a guard narrows names to where it is true, and where it is false."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        when_true, when_false = _guards(node.operand, env)
        return when_false, when_true
    if isinstance(node, ast.BoolOp):
        conjunction = isinstance(node.op, ast.And)
        facts: dict = {}
        for value in node.values:
            when_true, when_false = _guards(value, env.narrowed(facts))
            facts |= when_true if conjunction else when_false
        return (facts, {}) if conjunction else ({}, facts)
    if (
        isinstance(node, ast.Call)
        and callee(node.func) == "present"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    ):
        v = node.args[0].id
        kind = env.kinds.get(v)
        if kind is not None and kind.tag == "Maybe":
            return {v: kind.item}, {v: NONE}
        return {}, {}
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, op, right = node.left, node.ops[0], node.comparators[0]
        if isinstance(left, ast.Constant) and isinstance(op, (ast.Eq, ast.NotEq)):
            left, right = right, left
        # The subject is a name or its `.value` projection, which narrows the same name's payload.
        if (
            isinstance(left, ast.Attribute)
            and left.attr == "value"
            and isinstance(left.value, ast.Name)
        ):
            left = left.value if left.value.id in env.maybes else left
        if not isinstance(left, ast.Name):
            return {}, {}
        values = _literal_values(right, many=isinstance(op, (ast.In, ast.NotIn)))
        kind = env.kinds.get(left.id)
        if values is None or kind is None:
            return {}, {}
        inside = {left.id: restrict(kind, values)}
        outside = {left.id: exclude(kind, values)}
        if isinstance(op, (ast.Eq, ast.In)):
            return inside, outside
        if isinstance(op, (ast.NotEq, ast.NotIn)):
            return outside, inside
    return {}, {}


def _literal_values(node: ast.expr, many: bool) -> frozenset | None:
    """The literals a comparison names: one, or a literal tuple of them."""
    literal = (str, int, bool)
    if not many:
        return (
            frozenset({node.value})
            if isinstance(node, ast.Constant) and isinstance(node.value, literal)
            else None
        )
    items = (
        node.elts
        if isinstance(node, ast.Tuple)
        else [ast.Constant(v) for v in node.value]
        if isinstance(node, ast.Constant) and isinstance(node.value, tuple)
        else None
    )
    if items is None or not all(
        isinstance(i, ast.Constant) and isinstance(i.value, literal) for i in items
    ):
        return None
    return frozenset(i.value for i in items)


def _constant_kind(value) -> Kind:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return Kind("Int", nonneg=value >= 0)
    if isinstance(value, str):
        return literals([value])
    if value is None:
        return NONE
    if isinstance(value, tuple):
        # A folded literal tuple keeps its sequence kind.
        if not value:
            return seq(None, 0)
        item = _constant_kind(value[0])
        for v in value[1:]:
            item = join(item, _constant_kind(v))
        return seq(item, len(value))
    return VALUE


class _Infer:
    """Infers expression kinds and records misuse."""

    def __init__(self, env: KindEnv, where: str):
        self.env, self.where, self.errors = env, where, []

    def error(self, message: str) -> None:
        self.errors.append(f"{self.where}: {message}")

    def under(self, facts: dict) -> _Infer:
        inner = _Infer(self.env.narrowed(facts), self.where)
        inner.errors = self.errors
        return inner

    def expect(self, node: ast.expr, expected: Kind) -> None:
        kind = self.of(node)
        if not fits(kind, expected):
            self.error(f"{ast.unparse(node)} has kind {kind}, expected {expected}")

    def of(self, node: ast.expr) -> Kind | None:  # noqa: C901 - one case per node kind
        if isinstance(node, ast.Constant):
            return _constant_kind(node.value)
        if isinstance(node, ast.Name):
            if node.id == "inf":
                return VALUE
            if node.id in self.env.families:
                self.error(f"type family {node.id} is not a value")
            return self.env.kinds.get(node.id)
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
            conjunction, facts = isinstance(node.op, ast.And), {}
            for value in node.values:
                self.under(facts).expect(value, BOOL)
                when_true, when_false = _guards(value, self.env.narrowed(facts))
                facts |= when_true if conjunction else when_false
            return BOOL
        if isinstance(node, ast.Compare):
            self._compare(node)
            return BOOL
        if isinstance(node, ast.IfExp):
            self.expect(node.test, BOOL)
            when_true, when_false = _guards(node.test, self.env)
            body, orelse = (
                self.under(when_true).of(node.body),
                self.under(when_false).of(node.orelse),
            )
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
            kind = self.under({gen.target.id: element}).of(node.elt)
            return seq(kind, iterable.length if iterable is not None else None)
        if isinstance(node, ast.Call):
            return self.call(node)
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
        kind = None
        for m in members:
            if isinstance(node.slice, ast.Slice):
                ends = [
                    p.value if isinstance(p, ast.Constant) else OPEN if p else None for p in parts
                ]
                closed = m.length is not None and OPEN not in ends and ends[2] != 0
                length = len(range(*slice(*ends).indices(m.length))) if closed else None
                item = seq(m.item, length)
            else:
                item = m.item
            if item is None:
                return None
            kind = item if kind is None else join(kind, item)
        return kind

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
            elif not comparable(left, right) or (
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and not ordered(left, right)
            ):
                self.error(f"{ast.unparse(node)} compares {left} with {right}")
            left = right

    def _attribute(self, node: ast.Attribute) -> Kind | None:  # noqa: C901 - payloads and fields
        receiver = node.value
        env = self.env
        if (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "value"
            and isinstance(receiver.value, ast.Name)
        ):
            # A field of `v.value` is a field of v's payload; v names the branch facts.
            receiver = receiver.value
            kind = self._attribute(node.value)
        else:
            kind = env.kinds.get(receiver.id) if isinstance(receiver, ast.Name) else None
            maybe = isinstance(receiver, ast.Name) and receiver.id in env.maybes
            if node.attr == "value" and kind is not None and kind.tag == "Maybe":
                return kind.item
            if node.attr == "value" and maybe:
                # Narrowed by `present(v)`; an absent payload's read is the point's read check's.
                return None if kind == NONE else kind
        payload = receiver is not node.value
        if kind is not None and kind.tag == "ADT":
            ctors = env.adts.get(kind.name, {})
            # A payload's branch facts are keyed `v.value`.
            name = f"{receiver.id}.value" if payload else receiver.id
            chosen = [env.facts[name]] if name in env.facts else list(ctors)
            if node.attr == "kind":
                return literals(chosen)
            fixed = env.facts.get(f"{name}.{node.attr}", OPEN)
            if fixed is not OPEN:
                return BOOL if isinstance(fixed, bool) else literals([fixed])
            declared = [ctors[c][node.attr] for c in chosen if node.attr in ctors.get(c, {})]
            if declared:
                return union(*(d for d in declared if d is not None)) if all(declared) else None
            if name in env.facts:
                if not env.point:
                    self.error(
                        f"{ast.unparse(node)} reads a field constructor {env.facts[name]!r} lacks"
                    )
                return None
        self.error(f"attribute {ast.unparse(node)} is not a declared field")
        return None

    def call(self, node: ast.Call, table: dict = PRIMITIVE_KINDS) -> Kind | None:
        """The result kind of a call to an entry of *table*, its arguments bound and checked."""
        name = callee(node.func)
        if name == "present":
            known = self.env.presence | {
                n for n, k in self.env.kinds.items() if k is not None and k.tag == "Maybe"
            }
            known |= self.env.maybes
            if (
                len(node.args) != 1
                or node.keywords
                or not isinstance(node.args[0], ast.Name)
                or node.args[0].id not in known
            ):
                self.error(f"{ast.unparse(node)} names no tensor, `Maybe` parameter or `out`")
            return BOOL
        if name not in table:
            return None
        formal, result = table[name]
        if formal and formal[0].endswith("*"):
            slots = [(None, formal[0][:-1])] * len(node.args)
        else:
            slots = [tuple(f.split("=")) if "=" in f else (None, f) for f in formal]
        bound: dict[int, ast.expr] = dict(enumerate(node.args))
        if len(node.args) > len(slots):
            self.error(f"{name} takes {len(slots)} arguments, got {len(node.args)}")
        for kw in node.keywords:
            index = next((i for i, (n, _) in enumerate(slots) if n == kw.arg), None)
            if index is None or index in bound:
                self.error(f"{name} takes no keyword {kw.arg!r} here")
            else:
                bound[index] = kw.value
        missing = [i for i, (n, _) in enumerate(slots) if n is None and i not in bound]
        if missing:
            self.error(f"{name} misses argument {missing[0] + 1} of {len(slots)}")
        kinds = {
            i: self.argument(arg, _formal(slots[i][1], self.env.adts))
            for i, arg in sorted(bound.items())
            if i < len(slots)
        }
        if name == "ceil_div" and all(k is not None and k.nonneg for k in kinds.values()):
            return Kind("Int", nonneg=True)
        return _formal(result, self.env.adts)

    def argument(self, arg: ast.expr, expected: Kind) -> Kind | None:
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


# ---------------------------------------------------------------- points


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


_NAMESPACE = namespace()
_UNFOLDED = (ast.Constant, ast.Name, ast.List, ast.Starred, ast.Slice, ast.GeneratorExp)


def bind(node: ast.expr, point: dict) -> ast.expr:
    """A copy of `node` with what `point` fixes replaced by constants."""
    return ast.fix_missing_locations(_Bind(point).visit(copy.deepcopy(node)))


class _Fold(ast.NodeTransformer):
    """Evaluate closed subexpressions; drop the constant operands of `and`, `or` and conditionals."""

    def __init__(self, where: str):
        self.where = where

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
            return ast.Constant(evaluate(node, {}, self.where))
        return node


def fold(node: ast.expr, point: dict, where: str = "") -> ast.expr:
    """`node` on the branch `point` selects, with what that branch fixes evaluated."""
    return ast.fix_missing_locations(_Fold(where).visit(bind(node, point)))


def value_at(node: ast.expr, point: dict, where: str = ""):
    """The value of `node` at `point`, or `OPEN` when it reads more than `point` fixes."""
    folded = fold(node, point, where)
    return folded.value if isinstance(folded, ast.Constant) else OPEN


def evaluate(node: ast.expr, scope: dict, where: str, extra: dict | None = None):
    """The value of `node` over `scope`; any failure is an `EvaluationError` naming `where`."""
    try:
        code = compile(
            ast.Expression(ast.fix_missing_locations(copy.deepcopy(node))), "<manifest>", "eval"
        )
        return eval(code, {**_NAMESPACE, **(extra or {})}, dict(scope))  # noqa: S307
    except Exception as exc:  # noqa: BLE001 - every failure names its declaration
        prefix = f"{where}: " if where else ""
        raise EvaluationError(f"{prefix}{ast.unparse(node)} raises {exc}") from None
