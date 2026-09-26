"""Instantiate and check workload rows (docs/design/manifest.md § Workloads).

`instantiate` turns one row and one dtype case into a `Call`: constructor arguments and the shape,
dtype and generated values of every tensor, read off the entry's branch view at the row's point.
`Call.materialize` builds the tensors a call passes and `Call.arguments` the constructor values;
both need torch. `check_workloads` returns the row diagnostics the validator reports.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
import random
import re
from dataclasses import dataclass
from zlib import crc32

from .dtype_rules import DTYPE_BITS
from .expr import (
    EvaluationError,
    SignatureError,
    call_errors,
    callee,
    evaluate,
    fold,
    infer_kinds,
    language_errors,
    names,
    parse,
)
from .kinds import INT, seq, split_union
from .plan import EntryPlan, PlanBranch, entry_plan
from .primitives import (
    GENERATOR_KINDS,
    GENERATOR_RANKS,
    GENERATOR_SHAPES,
    GENERATORS,
    PREDICATE_KINDS,
    PREDICATE_RANKS,
    PREDICATES,
    PRIMITIVES,
    RANDOM_GENERATORS,
    WORKLOAD_SEED,
)
from .signature import (
    Signature,
    complete_point,
    discriminant_axes,
    expand,
    kind_env,
    param_kind,
    points,
    rejecting_rule,
    unification,
)
from .values import ADTValue, convert, is_integer, python_class

__all__ = ["Call", "CallView", "RowError", "TensorSpec", "check_workloads", "instantiate"]

_ROW_KEYS = frozenset({"some", "dtype_cases", "label"})
_LABEL = re.compile(r"[A-Za-z0-9._-]+")
# The integer dtypes generated metadata may take, with their ranges.
_METADATA_RANGES = {"int32": 2**31, "int64": 2**63}


class RowError(ValueError):
    """A workload row that does not state a call of its entry."""


class CallView:
    """The checked-call view a roofline `func` formula reads (docs/design/roofline.md).

    `ix` holds the parameters, the resolved indices and dtype indices, and the reached `let`s;
    `tensors` maps each present tensor to its `(shape, dtype name)`.
    """

    ix: dict
    tensors: dict
    out: bool = False

    def present(self, name: str) -> bool:
        """Whether the call passes, holds or returns tensor `name`, or passes `out`."""
        return self.out if name == "out" else name in self.tensors

    def bytes(self, name: str) -> int:
        """The bytes tensor `name` occupies."""
        shape, dtype = self.tensors[name]
        return (math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8

    def values(self, name: str) -> list:
        """The contents of metadata tensor `name`, as nested lists."""
        raise NotImplementedError


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    values: list | None = None
    cpu: bool = False


@dataclass(frozen=True)
class Call(CallView):
    case_id: str
    # Parameter values as expressions read them: a dtype by name, an ADT by its fields.
    params: dict
    specs: dict[str, TensorSpec | None]
    ix: dict = dataclasses.field(default_factory=dict)
    signature: Signature | None = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def tensors(self) -> dict:
        return {n: (s.shape, s.dtype) for n, s in self.specs.items() if s is not None}

    def values(self, name: str) -> list:
        return self.specs[name].values

    def arguments(self, tensors: dict) -> dict:
        """The op's constructor arguments, its construction-time
        tensors taken from *tensors*, as `materialize` returns them.

        A dtype is a `torch.dtype`; an ADT is its constructor's `python` class called with its
        fields as keyword arguments, an enum field built from its own `python` class.
        """
        values = {p: _python_value(self.signature, p, v) for p, v in self.params.items()}
        return values | {t: tensors[t] for t in self.signature.ctor_tensors}

    def materialize(self, device="cuda"):
        """Build the tensors the call passes: generated values where declared, random data elsewhere.

        A `device` parameter that is set decides where they go; *device* is the fallback.
        """
        import torch

        device = self.params.get("device") or device
        out = {}
        outputs = self.signature.outputs if self.signature is not None else {}
        for name, spec in self.specs.items():
            if name in outputs:
                continue
            if spec is None:
                out[name] = None
                continue
            dtype = getattr(torch, spec.dtype)
            place = "cpu" if spec.cpu else device
            if spec.values is not None:
                out[name] = torch.tensor(spec.values, dtype=dtype, device=place).reshape(spec.shape)
            elif dtype.is_floating_point or dtype.is_complex:
                out[name] = torch.randn(
                    spec.shape, device=place, dtype=dtype if dtype.is_complex else None
                ).to(dtype)
            else:
                high = 2 if dtype == torch.bool else 8
                out[name] = torch.randint(0, high, spec.shape, device=place).to(dtype)
        return out


def _python_value(sig: Signature, p: str, value):
    """One parameter value as its constructor takes it."""
    members = split_union(str(sig.params[p].get("type")))
    if isinstance(value, ADTValue):
        spec = sig.adts[value.adt]["sum"][value.kind]
        fields = {}
        for f, decl in (spec.get("fields") or {}).items():
            make = (
                python_class(decl["python"])
                if isinstance(decl, dict) and "python" in decl
                else None
            )
            fields[f] = value.fields[f] if make is None else make(value.fields[f])
        return python_class(spec["python"])(**fields)
    if (
        isinstance(value, str)
        and value in DTYPE_BITS
        and ("torch.dtype" in members or value in members)
    ):
        import torch

        return getattr(torch, value)
    return value


class _Payload(ast.NodeTransformer):
    """`v.value` of a present `Maybe` parameter is `v` itself."""

    def __init__(self, maybes):
        self.maybes = maybes

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (
            node.attr == "value"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.maybes
        ):
            return node.value
        return node


def _evaluate(sig: Signature, node: ast.expr, scope: dict, where: str):
    """`node`'s value in `scope`; a failure is a `RowError` naming `where`."""
    maybes = {
        p for p in sig.params if param_kind(sig.params[p].get("type"), sig.adts).tag == "Maybe"
    }
    tree = _Payload(maybes).visit(copy.deepcopy(node))
    try:
        return evaluate(tree, scope, where, {"present": lambda v: v is not None})
    except EvaluationError as exc:
        raise RowError(str(exc)) from None


def _convert(sig: Signature, key: str, value):
    """A row or default value as its declared kind or `type` states it; `RowError` otherwise."""
    if key in sig.forall:
        kind = sig.forall[key]
        ok = {
            "Dim": lambda v: is_integer(v, 0),
            "Shape": lambda v: isinstance(v, list) and all(is_integer(x, 0) for x in v),
            "Seq[Int]": lambda v: isinstance(v, list) and all(is_integer(x) for x in v),
        }.get(kind, lambda v: False)
        if not ok(value):
            raise RowError(f"{key} = {value!r} is not a {kind}")
        return tuple(value) if kind == "Shape" else value
    try:
        return convert(value, sig.params[key].get("type"), sig.adts)
    except ValueError as exc:
        raise RowError(f"{key} = {exc}") from None


def _shape(node: ast.List, sig: Signature, scope: dict, where: str) -> tuple[int, ...]:
    out = []
    for axis in node.elts:
        starred = isinstance(axis, ast.Starred)
        value = _evaluate(sig, axis.value if starred else axis, scope, where)
        for v in value if starred else [value]:
            if not is_integer(v, 0):
                raise RowError(
                    f"{where}: axis {ast.unparse(axis)} is {v!r}, not a non-negative integer"
                )
            out.append(v)
    return tuple(out)


def _dtype_value(node: ast.expr, scope: dict):
    """Resolve a folded dtype expression."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return scope.get(node.id, node.id)
    return PRIMITIVES[callee(node.func)](*(_dtype_value(a, scope) for a in node.args))


def _within(values, bound: int) -> bool:
    if isinstance(values, list):
        return all(_within(v, bound) for v in values)
    return is_integer(values) and -bound <= values < bound


def _point(sig: Signature, scope: dict, some: set[str]) -> dict:
    point = {}
    for p, decl in sig.params.items():
        kind = param_kind(decl.get("type"), sig.adts)
        if kind.tag == "Bool" or (kind.tag == "Str" and kind.values):
            point[p] = scope.get(p)
        elif kind.payload().tag == "ADT":
            value = scope.get(p)
            key = f"{p}.value" if kind.tag == "Maybe" else p
            if kind.tag == "Maybe":
                point[f"present({p})"] = value is not None
            if value is not None:
                point[f"{key}.kind"] = value.kind
                for f, v in value.fields.items():
                    if not is_integer(v):
                        point[f"{key}.{f}"] = v
        elif kind.tag == "Maybe":
            point[f"present({p})"] = scope.get(p) is not None
    for t in sig.call_tensors.values():
        if t.optional is True:
            point[f"present({t.name})"] = t.name in some
    if any(t.buffer for t in sig.outputs.values()):
        point["present(out)"] = False
    return complete_point(sig, point)


def _relevant(sig: Signature, b: PlanBranch) -> tuple[set[str], set[str]]:
    """Indices the row must give at the branch, and those a generator solves."""
    tensors = set(sig.call_tensors)
    free = {n for n, k in sig.forall.items() if not k.startswith("DType")}
    generated = {t: b.shapes[t] for t in b.generators}
    # What no generated shape reads, the row gives; the rest is solved in dependency order.
    shaped = set().union(set(), *(names(node) for node in generated.values()))
    need = b.relevant & free
    known = set(sig.params) | (need - shaped)
    seed, ran = set(known), {}
    while True:
        before = set(known)
        known |= {n for n, e in b.lets.items() if names(e) <= known}
        for t in generated:
            reads = set().union(set(), *(names(a) for a in b.generators[t])) - tensors
            if t not in ran and reads <= known:
                ran[t] = generated[t]
        known |= {s.name for s in unification(ran, known, {}) if s.name}
        if known == before:
            break
    return need, (known - seed) & free


def _apply(sig: Signature, steps: list, nodes: dict, shapes: dict, scope: dict, lets: dict) -> None:
    """Carry out an inference plan on concrete `shapes`, binding into `scope`."""

    def length(parts) -> int:
        return sum(
            len(_evaluate(sig, p.value, scope, "an axis count"))
            if isinstance(p, ast.Starred)
            else 1
            for p in parts
        )

    for step in steps:
        if step.action == "let":
            scope[step.name] = _evaluate(sig, lets[step.name], scope, f"let {step.name}")
            continue
        shape, e = shapes[step.tensor], nodes[step.tensor].elts[step.index]
        where = f"tensor {step.tensor!r} shape"
        start = length(step.before) if step.before is not None else None
        end = len(shape) - length(step.after) if step.after is not None else None
        if isinstance(e, ast.Starred):
            part = tuple(shape[start:end])
            if step.action == "bind":
                scope[step.name] = part
            elif part != tuple(_evaluate(sig, e.value, scope, where)):
                raise RowError(
                    f"{step.tensor} axes {start}:{end} are {part}, not {ast.unparse(e.value)}"
                )
            continue
        size = shape[start if start is not None else end - 1]
        if step.action == "check":
            if size != _evaluate(sig, e, scope, where):
                raise RowError(f"{step.tensor} axis {step.index} is {size}, not {ast.unparse(e)}")
            continue
        lo = _evaluate(sig, e, {**scope, step.name: 0}, where)
        slope = _evaluate(sig, e, {**scope, step.name: 1}, where) - lo
        if slope <= 0 or (size - lo) % slope or size < lo:
            raise RowError(
                f"{step.tensor} axis {step.index} = {size} is not {ast.unparse(e)} for a Dim"
            )
        scope[step.name] = (size - lo) // slope


def _fixed_rank(node: ast.List, length) -> int | None:
    """The rank of a shape where every splice has a fixed length, `length(splice)`; else None."""
    lengths = [length(e.value) if isinstance(e, ast.Starred) else 1 for e in node.elts]
    return None if None in lengths else sum(lengths)


def instantiate(plan: EntryPlan, row: dict, dtype_case: dict) -> Call:  # noqa: C901
    """The call one row and one dtype case state; raises `RowError` when they state none."""
    sig = plan.sig
    scope = {}
    for key, value in row.items():
        if key in _ROW_KEYS:
            continue
        if key not in sig.forall and key not in sig.params:
            raise RowError(f"row key {key!r} is neither an index nor a parameter")
        scope[key] = _convert(sig, key, value)
    for p, decl in sig.params.items():
        if p not in scope:
            if "default" not in decl:
                raise RowError(f"row misses parameter {p!r}, which has no default")
            scope[p] = _convert(sig, p, decl["default"])
    try:
        point = _point(sig, scope, set(row.get("some", [])))
        rule = rejecting_rule(sig, point)
        if rule:
            raise RowError(f"refinement fails: {rule}")
        b = plan.branch(point)
    except SignatureError as exc:
        raise RowError(str(exc)) from None

    need, solved = _relevant(sig, b)
    given = {k for k in row if k in sig.forall}
    if need - solved - given:
        raise RowError(f"row misses {sorted(need - solved - given)}")
    if given - (need - solved):
        raise RowError(f"row gives irrelevant {sorted(given - (need - solved))}")

    dtype_indices = {n for n in b.relevant if n in sig.forall and sig.kind(n).tag == "DType"}
    if set(dtype_case) != dtype_indices:
        raise RowError(
            f"dtype_cases assign {sorted(dtype_case)}, the entry's DType indices are {sorted(dtype_indices)}"
        )
    for index, dtype in dtype_case.items():
        if dtype not in sig.kind(index).values:
            raise RowError(f"{index} = {dtype} is outside {sig.forall[index]}")
    scope.update(dtype_case)
    if sig.dtype_combos:
        columns = set(sig.dtype_combos[0])
        assignment = {c: str(scope.get(c)) for c in columns}
        if assignment not in [{c: str(v) for c, v in r.items()} for r in sig.dtype_combos]:
            raise RowError(f"dtype assignment {assignment} is not a row of dtype_combos")

    generated, nodes, shapes = {}, {}, {}
    lets = dict(b.lets)
    waiting = list(b.generators)
    while True:
        known = set(scope)
        for name in [n for n, e in lets.items() if names(e) <= set(scope)]:
            scope[name] = _evaluate(sig, lets.pop(name), scope, f"let {name}")
        ready = [
            t
            for t in waiting
            if set().union(set(), *map(names, b.generators[t])) - set(sig.call_tensors)
            <= set(scope)
        ]
        for t in ready:
            waiting.remove(t)
            generated[t], shapes[t] = _generate(sig, t, b, scope)
            nodes[t] = b.shapes[t]
        _apply(sig, unification(nodes, set(scope), {}), nodes, shapes, scope, {})
        if set(scope) == known and not ready:
            break
    if waiting:
        raise RowError(f"generator arguments of {waiting} are never known")

    for i, rule in enumerate(b.rules):
        if not _evaluate(sig, rule, scope, f"shape_rules[{i}]"):
            raise RowError(f"refinement fails: {sig.rules[i]}")
    specs: dict[str, TensorSpec | None] = {}
    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        if t.name not in b.shapes:
            specs[t.name] = None
            continue
        shape = _shape(b.shapes[t.name], sig, scope, f"tensor {t.name!r} shape")
        specs[t.name] = TensorSpec(
            shape, _dtype_value(b.dtypes[t.name], scope), generated.get(t.name), t.cpu
        )
    for t, calls in b.requires.items():
        for call, text in zip(calls, sig.call_tensors[t].requires, strict=False):
            args = [
                _evaluate(sig, a, {**scope, **generated}, f"tensor {t!r} requires")
                for a in call.args
            ]
            try:
                ok = PREDICATES[callee(call.func)](generated[t], *args)
            except (TypeError, ValueError, IndexError) as exc:
                raise RowError(f"tensor {t!r}: requires {text} raises {exc}") from None
            if not ok:
                raise RowError(f"tensor {t!r}: requires {text} fails")
    mismatch = _reinfer(sig, b, specs, scope)
    if mismatch:
        raise RowError(f"inference from the instantiated inputs disagrees: {mismatch}")
    dtype_params = [
        scope[p]
        for p in sig.params
        if param_kind(sig.params[p].get("type"), sig.adts).payload().tag == "DType"
        and isinstance(scope[p], str)
    ]
    label = row.get("label", "")
    case_id = "-".join(
        [label, *(dtype_case[i] for i in sig.forall if i in dtype_case), *dtype_params]
    )
    values = {n for n, k in sig.forall.items() if k == "Seq[Int]"}
    ix = {
        n: v
        for n, v in scope.items()
        if n in sig.params or (n in sig.forall and n not in values) or n in b.lets
    }
    return Call(case_id, {p: scope[p] for p in sig.params}, specs, ix, sig)


def _generate(sig: Signature, t: str, b: PlanBranch, scope: dict) -> tuple[list, tuple]:
    """The values tensor `t`'s generator yields for this row, seeded per op and tensor, and
    their shape, checked against the rank `t` declares."""
    fn = callee(parse(sig.call_tensors[t].values).func)
    dtype = _dtype_value(b.dtypes[t], scope)
    if dtype not in _METADATA_RANGES:
        raise RowError(f"tensor {t!r}: generated values need an int32 or int64 dtype, not {dtype}")
    args = [_evaluate(sig, a, scope, f"tensor {t!r} values") for a in b.generators[t]]
    if fn in RANDOM_GENERATORS:
        args.insert(0, random.Random(WORKLOAD_SEED ^ crc32(f"{sig.name}:{t}".encode())))
    try:
        values = GENERATORS[fn](*args)
    except ValueError as exc:
        raise RowError(f"tensor {t!r}: {exc}") from None
    if not _within(values, _METADATA_RANGES[dtype]):
        raise RowError(f"tensor {t!r}: {sig.call_tensors[t].values} yields values outside {dtype}")
    shape = GENERATOR_SHAPES[fn](*(a for a in args if not isinstance(a, random.Random)))
    rank = _fixed_rank(
        b.shapes[t],
        lambda e: len(_evaluate(sig, e, scope, f"tensor {t!r} shape"))
        if names(e) <= set(scope)
        else None,
    )
    if rank is not None and rank != len(shape):
        raise RowError(
            f"tensor {t!r}: {fn} yields rank {len(shape)}, not {ast.unparse(b.shapes[t])}"
        )
    return values, shape


def _reinfer(sig: Signature, b: PlanBranch, specs: dict, scope: dict) -> dict:
    """Solve the indices again from the instantiated inputs and compare with the row's."""
    known = {
        k: v
        for k, v in scope.items()
        if k in sig.params or (k in sig.forall and sig.kind(k).tag == "DType")
    }
    nodes = {t: b.shapes[t] for t in sig.call_tensors if specs.get(t)}
    steps = unification(nodes, set(known), b.lets)
    _apply(sig, steps, nodes, {t: specs[t].shape for t in nodes}, known, b.lets)
    return {
        n: (known[n], scope[n])
        for n in sig.forall
        if n in scope and n in known and known[n] != scope[n]
    }


# ---------------------------------------------------------------- checks


def _nested(rank: int):
    """The kind of a generated tensor's contents: `rank` nested sequences of ints."""
    kind = INT
    for _ in range(rank):
        kind = seq(kind)
    return kind


def _callee_of(text: str) -> str | None:
    try:
        call = parse(text)
    except SignatureError:
        return None
    return callee(call.func) if isinstance(call, ast.Call) else None


def _declaration_errors(sig: Signature) -> list[str]:
    """Every `values` and `requires` call: a built-in callee, its arity, arguments in the language;
    and the `requires` contract on every discriminant point."""
    errors, valid = [], {}
    declared = {*sig.forall, *sig.params, *sig.let}
    env, _ = kind_env(sig, {n: parse(e) for n, e in sig.let.items()})
    generated = {
        n: GENERATOR_RANKS.get(_callee_of(d.values))
        for n, d in sig.call_tensors.items()
        if d.values is not None
    }
    for t in sig.call_tensors.values():
        declared_calls = [("values", t.values, GENERATOR_KINDS)] if t.values is not None else []
        declared_calls += [("requires", r, PREDICATE_KINDS) for r in t.requires]
        if t.requires and t.values is None:
            errors.append(f"tensor {t.name!r} has `requires` but no `values`")
        for field, text, table in declared_calls:
            where = f"tensor {t.name!r} {field}"
            try:
                call = parse(text)
            except SignatureError as exc:
                errors.append(f"{where}: {exc}")
                continue
            name = callee(call.func) if isinstance(call, ast.Call) else None
            if name not in table or call.keywords:
                errors.append(f"{where}: {text!r} is not a call of a built-in")
                continue
            scope = declared | (set(generated) if field == "requires" else set())
            found = []
            for arg in call.args:
                found += [f"{where}: {e}" for e in language_errors(arg, False)]
                found += [
                    f"{where}: name {n!r} is not declared" for n in sorted(names(arg) - scope)
                ]
            contents = {n: _nested(r) for n, r in generated.items() if r is not None}
            local = env.narrowed(contents) if field == "requires" else env
            found += call_errors(call, table, local, where)
            errors += found
            if not found:
                valid.setdefault(t.name, []).append((field, name, call))
    return errors + _requires_errors(sig, valid, env)


def _requires_errors(sig: Signature, valid: dict, env) -> list[str]:
    """On every discriminant point where its tensor is present, a `requires` reads only metadata
    tensors that are present, and its predicate reads the rank the tensor declares there, which
    must be fixed."""

    def length(e: ast.expr) -> int | None:
        kind = infer_kinds(e, env, "")[0]
        return kind.sequence().length if kind is not None and kind.sequence() is not None else None

    errors = set()
    for point in points(list(discriminant_axes(sig).values())):
        point = complete_point(sig, point, strict=False)
        for name, calls in valid.items():
            if not point.get(f"present({name})", True):
                continue
            try:
                node = fold(expand(sig, sig.call_tensors[name].shape, point), point)
            except SignatureError:
                node = None
            for field, predicate, call in calls:
                if field != "requires":
                    continue
                for m in names(call) & set(sig.call_tensors):
                    if not point.get(f"present({m})", True):
                        errors.add(
                            f"tensor {name!r}: {ast.unparse(call)} reads {m!r} where it is absent"
                        )
                rank = PREDICATE_RANKS.get(predicate)
                if rank is None or node is None:
                    continue
                declared = _fixed_rank(node, length)
                if declared is None:
                    errors.add(
                        f"tensor {name!r}: {predicate} needs a fixed-rank tensor, not {ast.unparse(node)}"
                    )
                elif declared != rank:
                    errors.add(
                        f"tensor {name!r}: {predicate} reads rank {rank}, not {ast.unparse(node)}"
                    )
    return sorted(errors)


def _row_errors(sig: Signature, row: object) -> list[str]:
    if not isinstance(row, dict):
        return ["a row must be a mapping"]
    errors = []
    label = row.get("label")
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        errors.append("`label` must be a non-empty [A-Za-z0-9._-] string")
    optional = {t.name for t in sig.call_tensors.values() if t.optional is True}
    some = row.get("some", [])
    if (
        not isinstance(some, list)
        or not all(isinstance(s, str) for s in some)
        or len(set(some)) != len(some)
    ):
        errors.append("`some` must be a list of distinct names")
    elif set(some) - optional:
        errors.append(f"`some` names {sorted(set(some) - optional)}, not `optional: true` inputs")
    cases = row.get("dtype_cases")
    if "dtype_cases" in row and not (
        isinstance(cases, list) and cases and all(isinstance(c, dict) and c for c in cases)
    ):
        errors.append("`dtype_cases` must be a non-empty list of non-empty mappings")
    return errors


def check_workloads(name: str, entry: dict, adts: dict) -> list[str]:
    """Row diagnostics for one entry, each prefixed with its op name."""
    try:
        plan = entry_plan(name, entry, adts, resolve=False)
    except SignatureError as exc:
        return [f"{name}: {exc}"]
    sig = plan.sig
    errors = _declaration_errors(sig)
    if errors:
        return [f"{name}: {e}" for e in errors]
    rows = entry.get("workloads")
    if not isinstance(rows, list) or not rows:
        return [f"{name}: `workloads` must be a non-empty list"]
    seen, passed, omitted = set(), set(), set()
    gated = [t.name for t in sig.call_tensors.values() if t.optional is not False]
    for i, row in enumerate(rows):
        problems = _row_errors(sig, row)
        if problems:
            errors += [f"row {i}: {p}" for p in problems]
            continue
        for case in row.get("dtype_cases", [{}]):
            try:
                call = instantiate(plan, row, case)
                call.arguments(dict.fromkeys(sig.ctor_tensors))
            except (RowError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"row {row['label']!r} {case}: {exc}")
                continue
            except (ImportError, AttributeError) as exc:
                errors.append(f"row {row['label']!r} {case}: a constructor class: {exc}")
                continue
            passed |= {t for t in gated if call.specs[t] is not None}
            omitted |= {t for t in gated if call.specs[t] is None}
            if call.case_id in seen:
                errors.append(f"case id {call.case_id!r} repeats")
            seen.add(call.case_id)
    if entry.get("status") == "implemented":
        errors += [f"no row passes optional tensor {t!r}" for t in gated if t not in passed]
        errors += [f"no row omits optional tensor {t!r}" for t in gated if t not in omitted]
    return [f"{name}: {e}" for e in errors]
