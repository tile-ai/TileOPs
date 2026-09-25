"""Instantiate and check workload rows (docs/design/manifest.md § Workloads).

`instantiate` turns one row and one dtype case into a `Call`: constructor arguments and the shape,
dtype and generated values of every tensor. `Call.materialize` builds the tensors a call passes and
`Call.arguments` the constructor values; both need torch.
`check_workloads` returns the row diagnostics the validator reports.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import itertools
import random
import re
from dataclasses import dataclass
from zlib import crc32

from .dtype_rules import DTYPE_BITS
from .kinds import BOOL, INT, seq
from .kinds import _split as _members
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
    namespace,
)
from .signature import (
    Signature,
    SignatureError,
    _bound,
    _discriminant_axes,
    _dtype_kind,
    _emitted,
    _enum_values,
    _field_type,
    _Kinds,
    _language_errors,
    _parse,
    _passed,
    _points,
    branch,
    complete_point,
    expand,
    kind_env,
    names,
    param_kind,
    parse_signature,
    rejecting_rule,
    unification,
)

__all__ = ["Call", "RowError", "check_workloads", "instantiate"]

_ROW_KEYS = frozenset({"some", "dtype_cases", "label"})
_LABEL = re.compile(r"[A-Za-z0-9._-]+")


class RowError(ValueError):
    """A workload row that does not state a call of its entry."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    values: list | None = None
    cpu: bool = False


@dataclass(frozen=True)
class Call:
    case_id: str
    # Parameter values as expressions read them: a dtype by name, an ADT by its fields.
    params: dict
    tensors: dict[str, TensorSpec | None]
    signature: Signature | None = dataclasses.field(default=None, repr=False, compare=False)

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
        for name, spec in self.tensors.items():
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


def _python(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _python_value(sig: Signature, p: str, value):
    """One parameter value as its constructor takes it."""
    members = _members(str(sig.params[p].get("type")))
    if isinstance(value, _ADTValue):
        spec = sig.adts[value.adt]["sum"][value.kind]
        fields = {}
        for f, decl in (spec.get("fields") or {}).items():
            make = _python(decl["python"]) if isinstance(decl, dict) and "python" in decl else None
            fields[f] = value.fields[f] if make is None else make(value.fields[f])
        return _python(spec["python"])(**fields)
    if (
        isinstance(value, str)
        and value in DTYPE_BITS
        and ("torch.dtype" in members or value in members)
    ):
        import torch

        return getattr(torch, value)
    return value


class _ADTValue:
    """An ADT literal as expressions read it: `v.kind`, and each field as `v.<field>`."""

    def __init__(self, adt: str, literal: dict):
        ((kind, fields),) = literal.items()
        self.__dict__.update(adt=adt, kind=kind, fields=dict(fields or {}))

    def __getattr__(self, name):
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            raise AttributeError(name) from None


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


def _evaluate(sig: Signature, text: str, scope: dict, point: dict):
    tree = ast.Expression(_bound(_parse(text), point))
    maybes = {
        p for p in sig.params if param_kind(sig.params[p].get("type"), sig.adts).tag == "Maybe"
    }
    tree = ast.fix_missing_locations(_Payload(maybes).visit(tree))
    try:
        return eval(
            compile(tree, "<manifest>", "eval"),
            {**namespace(), "present": lambda v: v is not None},
            dict(scope),
        )  # noqa: S307
    except (ValueError, ArithmeticError, IndexError, KeyError, TypeError) as exc:
        # A primitive or operator outside its domain names the declaration that called it.
        raise RowError(f"{text}: {exc}") from None


def _integer(value, minimum=None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _convert(sig: Signature, key: str, value):
    """A row or default value as its declared kind or `type` states it; `RowError` otherwise."""
    if key in sig.forall:
        kind = sig.forall[key]
        ok = {
            "Dim": lambda v: _integer(v, 0),
            "Shape": lambda v: isinstance(v, list) and all(_integer(x, 0) for x in v),
            "Seq[Int]": lambda v: isinstance(v, list) and all(_integer(x) for x in v),
        }.get(kind, lambda v: False)
        if not ok(value):
            raise RowError(f"{key} = {value!r} is not a {kind}")
        return tuple(value) if kind == "Shape" else value
    type_text = sig.params[key].get("type")
    return _param_value(sig, key, type_text, value)


class _Rejected:
    """A member that does not take a value, and why, where there is more to say."""

    def __init__(self, reason: str = ""):
        self.reason = reason


_REJECT = _Rejected()


def _member_value(sig: Signature, member: str, value):
    """`value` as one member of a parameter `type` takes it, or a `_Rejected`."""
    m = member.replace(" ", "")
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    simple = {
        "None": value is None,
        "bool": isinstance(value, bool),
        "int": _integer(value),
        "float": number,
        "Number": number,
        "str": isinstance(value, str),
        "dict": isinstance(value, dict),
        "torch.dtype": isinstance(value, str) and value in DTYPE_BITS,
    }
    if m in simple:
        return value if simple[m] else _REJECT
    if m.startswith("'"):
        return value if value == m[1:-1] else _REJECT
    if m in DTYPE_BITS:
        return value if value == m else _REJECT
    if m in sig.adts:
        try:
            return _adt_value(sig, m, value)
        except RowError as exc:
            return _Rejected(str(exc))
    if m.startswith(("list[int]", "tuple[int")):
        if not (isinstance(value, list) and all(map(_integer, value))):
            return _REJECT
        if m.startswith("tuple[") and "..." not in m and len(value) != m.count(",") + 1:
            return _REJECT
        return tuple(value) if m.startswith("tuple[") else list(value)
    return value


def _param_value(sig: Signature, key: str, type_text: object, value):
    """A parameter value checked against each member of its `type`; `RowError` when none takes it."""
    reasons = []
    for member in _members(str(type_text)):
        converted = _member_value(sig, member, value)
        if not isinstance(converted, _Rejected):
            return converted
        if converted.reason:
            reasons.append(converted.reason)
    why = f" ({'; '.join(reasons)})" if reasons else ""
    raise RowError(f"{key} = {value!r} is not a {type_text}{why}")


def _adt_value(sig: Signature, key: str, value) -> "_ADTValue":
    """An ADT literal `{ctor: {field: value}}` checked against its constructor's fields and
    invariant."""
    ctors = sig.adts[key]["sum"]
    if not (isinstance(value, dict) and len(value) == 1 and next(iter(value)) in ctors):
        raise RowError(f"{key} = {value!r} is not one of the constructors {sorted(ctors)}")
    ((ctor, fields),) = value.items()
    declared = ctors[ctor].get("fields") or {}
    fields = fields or {}
    if set(fields) != set(declared):
        raise RowError(f"{key}: {ctor} takes fields {sorted(declared)}, got {sorted(fields)}")
    for f, v in fields.items():
        text = _field_type(declared[f])
        ok = (
            v in _enum_values(text)
            if text.startswith("'")
            else isinstance(v, bool)
            if text in ("bool", "Bool")
            else _integer(v, 0 if text == "Dim" else None)
        )
        if not ok:
            raise RowError(f"{key}: {ctor}.{f} = {v!r} is not a {text}")
    if "invariant" in ctors[ctor] and not _evaluate(sig, ctors[ctor]["invariant"], fields, {}):
        raise RowError(f"{key}: {ctor} invariant {ctors[ctor]['invariant']} fails")
    return _ADTValue(key, value)


def _shape(node: ast.List, sig: Signature, scope: dict, point: dict) -> tuple[int, ...]:
    out = []
    for axis in node.elts:
        starred = isinstance(axis, ast.Starred)
        value = _evaluate(sig, ast.unparse(axis.value if starred else axis), scope, point)
        for v in value if starred else [value]:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise RowError(f"axis {ast.unparse(axis)} is {v!r}, not a non-negative integer")
            out.append(v)
    return tuple(out)


def _dtype_value(node: ast.expr, scope: dict):
    """Resolve a dtype expression."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return scope.get(node.id, node.id)
    return PRIMITIVES[node.func.id](*(_dtype_value(a, scope) for a in node.args))


def _int32(values) -> bool:
    if isinstance(values, list):
        return all(_int32(v) for v in values)
    return isinstance(values, int) and not isinstance(values, bool) and -(2**31) <= values < 2**31


def _point(sig: Signature, scope: dict, some: set[str]) -> dict:
    point = {}
    for p, decl in sig.params.items():
        kind = param_kind(decl.get("type"), sig.adts)
        if kind == BOOL or (kind.tag == "Str" and kind.values):
            point[p] = scope.get(p)
        elif kind.payload().tag == "ADT":
            value = scope.get(p)
            key = f"{p}.value" if kind.tag == "Maybe" else p
            if kind.tag == "Maybe":
                point[f"present({p})"] = value is not None
            if value is not None:
                point[f"{key}.kind"] = value.kind
                for f, v in value.fields.items():
                    if not isinstance(v, int) or isinstance(v, bool):
                        point[f"{key}.{f}"] = v
        elif kind.tag == "Maybe":
            point[f"present({p})"] = scope.get(p) is not None
    for t in sig.call_tensors.values():
        if t.optional is True:
            point[f"present({t.name})"] = t.name in some
    for t in sig.outputs.values():
        if t.buffer:
            point["present(out)"] = False
    return complete_point(sig, point)


def _relevant(sig: Signature, point: dict) -> tuple[set[str], set[str]]:
    """Indices the row must give at `point`, and those a generator solves."""
    b = branch(sig, point)
    generated = {
        t: b.shapes[t]
        for t in b.shapes
        if t in sig.call_tensors and sig.call_tensors[t].values is not None
    }
    tensors = set(sig.call_tensors)
    args = set().union(set(), *(names(_parse(sig.call_tensors[t].values)) for t in generated))
    reads = set().union(
        set(), *(names(_parse(r)) for t in generated for r in sig.call_tensors[t].requires)
    )
    relevant = b.relevant | ((args | reads) - tensors)
    seen: set[str] = set()
    while more := (relevant & set(sig.let)) - seen:
        seen |= more
        relevant |= set().union(set(), *(names(_parse(sig.let[n])) for n in more))
    free = {n for n, k in sig.forall.items() if not k.startswith("DType")}
    # What no generated shape reads, the row gives; the rest is solved in dependency order.
    shaped = set().union(set(), *(names(node) for node in generated.values()))
    known = set(sig.params) | ((relevant & free) - shaped)
    seed, ran = set(known), {}
    while True:
        before = set(known)
        known |= {n for n, e in sig.let.items() if names(_parse(e)) <= known}
        for t in generated:
            if t not in ran and names(_parse(sig.call_tensors[t].values)) - tensors <= known:
                ran[t] = generated[t]
        known |= {s.name for s in unification(ran, known, {}) if s.name}
        if known == before:
            break
    return relevant & free, (known - seed) & free


def _apply(
    sig: Signature, point: dict, steps: list, nodes: dict, shapes: dict, scope: dict
) -> None:
    """Carry out an inference plan on concrete `shapes`, binding into `scope`."""

    def length(parts) -> int:
        return sum(
            len(_evaluate(sig, ast.unparse(p.value), scope, point))
            if isinstance(p, ast.Starred)
            else 1
            for p in parts
        )

    for step in steps:
        if step.action == "let":
            scope[step.name] = _evaluate(sig, sig.let[step.name], scope, point)
            continue
        shape, e = shapes[step.tensor], nodes[step.tensor].elts[step.index]
        start = length(step.before) if step.before is not None else None
        end = len(shape) - length(step.after) if step.after is not None else None
        if isinstance(e, ast.Starred):
            part = tuple(shape[start:end])
            if step.action == "bind":
                scope[step.name] = part
            elif part != tuple(_evaluate(sig, ast.unparse(e.value), scope, point)):
                raise RowError(
                    f"{step.tensor} axes {start}:{end} are {part}, not {ast.unparse(e.value)}"
                )
            continue
        size = shape[start if start is not None else end - 1]
        if step.action == "check":
            if size != _evaluate(sig, ast.unparse(e), scope, point):
                raise RowError(f"{step.tensor} axis {step.index} is {size}, not {ast.unparse(e)}")
            continue
        lo = _evaluate(sig, ast.unparse(e), {**scope, step.name: 0}, point)
        slope = _evaluate(sig, ast.unparse(e), {**scope, step.name: 1}, point) - lo
        if slope <= 0 or (size - lo) % slope or size < lo:
            raise RowError(
                f"{step.tensor} axis {step.index} = {size} is not {ast.unparse(e)} for a Dim"
            )
        scope[step.name] = (size - lo) // slope


def instantiate(sig: Signature, row: dict, dtype_case: dict) -> Call:
    """The call one row and one dtype case state; raises `RowError` when they state none."""
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
    some = set(row.get("some", []))
    point = _point(sig, scope, some)
    rule = rejecting_rule(sig, point)
    if rule:
        raise RowError(f"refinement fails: {rule}")

    need, solved = _relevant(sig, point)
    given = {k for k in row if k in sig.forall}
    if need - solved - given:
        raise RowError(f"row misses {sorted(need - solved - given)}")
    if given - (need - solved):
        raise RowError(f"row gives irrelevant {sorted(given - (need - solved))}")

    dtypes = {
        n for n in branch(sig, point).relevant if n in sig.forall and sig.kind(n).tag == "DType"
    }
    if set(dtype_case) != dtypes:
        raise RowError(
            f"dtype_cases assign {sorted(dtype_case)}, the entry's DType indices are {sorted(dtypes)}"
        )
    for index, dtype in dtype_case.items():
        if dtype not in sig.forall[index][6:-1].replace(" ", "").split("|"):
            raise RowError(f"{index} = {dtype} is outside {sig.forall[index]}")
    scope.update(dtype_case)
    if sig.dtype_combos:
        columns = set(sig.dtype_combos[0])
        assignment = {c: str(scope.get(c)) for c in columns}
        if assignment not in [{c: str(v) for c, v in r.items()} for r in sig.dtype_combos]:
            raise RowError(f"dtype assignment {assignment} is not a row of dtype_combos")

    generated, nodes, shapes = {}, {}, {}
    lets = dict(sig.let)
    waiting = [
        t for t in sig.call_tensors.values() if t.values is not None and _passed(sig, t, point)
    ]
    while True:
        known = set(scope)
        for name in [n for n, e in lets.items() if names(_parse(e)) <= set(scope)]:
            scope[name] = _evaluate(sig, lets.pop(name), scope, point)
        ready = [
            t for t in waiting if names(_parse(t.values)) - set(sig.call_tensors) <= set(scope)
        ]
        for t in ready:
            waiting.remove(t)
            generated[t.name], shapes[t.name] = _generate(sig, t, scope, point)
            nodes[t.name] = expand(sig, t.shape, point)
        _apply(sig, point, unification(nodes, set(scope), {}), nodes, shapes, scope)
        if set(scope) == known and not ready:
            break
    if waiting:
        raise RowError(f"generator arguments of {[t.name for t in waiting]} are never known")

    for rule in sig.rules:
        if not _evaluate(sig, rule, scope, point):
            raise RowError(f"refinement fails: {rule}")
    tensors: dict[str, TensorSpec | None] = {}
    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        present = _emitted(sig, t, point) if t.name in sig.outputs else _passed(sig, t, point)
        if not present:
            tensors[t.name] = None
            continue
        shape = _shape(expand(sig, t.shape, point), sig, scope, point)
        dtype = _dtype_value(_parse(t.dtype), scope)
        tensors[t.name] = TensorSpec(shape, dtype, generated.get(t.name), t.cpu)
    for t in sig.call_tensors.values():
        for req in t.requires if t.name in generated else ():
            call = _parse(req)
            args = [
                _evaluate(sig, ast.unparse(a), {**scope, **generated}, point) for a in call.args
            ]
            try:
                holds = PREDICATES[ast.unparse(call.func)](generated[t.name], *args)
            except (TypeError, ValueError, IndexError) as exc:
                raise RowError(f"tensor {t.name!r}: requires {req} raises {exc}") from None
            if not holds:
                raise RowError(f"tensor {t.name!r}: requires {req} fails")
    mismatch = _reinfer(sig, point, tensors, scope)
    if mismatch:
        raise RowError(f"inference from the instantiated inputs disagrees: {mismatch}")
    label = row.get("label", "")
    case_id = "-".join([label, *(dtype_case[i] for i in sig.forall if i in dtype_case)])
    params = {p: scope[p] for p in sig.params}
    return Call(case_id, params, tensors, sig)


def _generate(sig: Signature, t, scope: dict, point: dict) -> tuple[list, tuple]:
    """The values tensor `t`'s generator yields for this row, seeded per op and tensor, and
    their shape."""
    call = _parse(t.values)
    fn = ast.unparse(call.func)
    args = [_evaluate(sig, ast.unparse(a), scope, point) for a in call.args]
    if fn in RANDOM_GENERATORS:
        args.insert(0, random.Random(WORKLOAD_SEED ^ crc32(f"{sig.name}:{t.name}".encode())))
    try:
        values = GENERATORS[fn](*args)
    except ValueError as exc:
        raise RowError(f"tensor {t.name!r}: {exc}") from None
    if not _int32(values):
        raise RowError(f"tensor {t.name!r}: {t.values} yields values outside int32")
    shape = GENERATOR_SHAPES[fn](*(a for a in args if not isinstance(a, random.Random)))
    return values, shape


def _reinfer(sig: Signature, point: dict, tensors: dict, scope: dict) -> dict:
    """Solve the indices again from the instantiated inputs and compare with the row's."""
    known = {
        k: v
        for k, v in scope.items()
        if k in sig.params or (k in sig.forall and sig.kind(k).tag == "DType")
    }
    nodes = {
        t: expand(sig, sig.call_tensors[t].shape, point) for t in sig.call_tensors if tensors.get(t)
    }
    lets = {n: _parse(e) for n, e in sig.let.items()}
    steps = unification(nodes, set(known), lets)
    _apply(sig, point, steps, nodes, {t: tensors[t].shape for t in nodes}, known)
    return {
        n: (known[n], scope[n])
        for n in sig.forall
        if n in scope and n in known and known[n] != scope[n]
    }


def _declaration_errors(sig: Signature) -> list[str]:
    """Every `values` and `requires` call: a built-in callee, its arity, arguments in the language.

    Only the well-formed calls then meet the contracts of `_generated_contract_errors`.
    """
    errors, valid = [], {}
    declared = {*sig.forall, *sig.params, *sig.let}
    lets_env, _ = kind_env(sig, {n: _parse(e) for n, e in sig.let.items()})
    ranks = {
        n: GENERATOR_RANKS.get(_callee_of(d.values))
        for n, d in sig.call_tensors.items()
        if d.values is not None
    }
    for t in sig.call_tensors.values():
        declared_calls = [("values", t.values, GENERATORS)] if t.values is not None else []
        declared_calls += [("requires", r, PREDICATES) for r in t.requires]
        if t.requires and t.values is None:
            errors.append(f"tensor {t.name!r} has `requires` but no `values`")
        for field, text, registry in declared_calls:
            where = f"tensor {t.name!r} {field}"
            try:
                call = _parse(text)
            except SignatureError as exc:
                errors.append(f"{where}: {exc}")
                continue
            callee = ast.unparse(call.func) if isinstance(call, ast.Call) else None
            if callee not in registry or call.keywords:
                errors.append(f"{where}: {text!r} is not a call of a built-in")
                continue
            scope = declared | (set(ranks) if field == "requires" else set())
            found = []
            for arg in call.args:
                found += [f"{where}: {e}" for e in _language_errors(arg, False)]
                found += [
                    f"{where}: name {n!r} is not declared" for n in sorted(names(arg) - scope)
                ]
            env = dict(lets_env)
            if field == "requires":
                env |= {n: _nested(r) for n, r in ranks.items() if r is not None}
            kinds = _Kinds(sig, env, where)
            kinds._call(call, GENERATOR_KINDS if field == "values" else PREDICATE_KINDS)
            found += kinds.errors
            errors += found
            if not found:
                valid.setdefault(t.name, []).append((field, callee, call))
    errors += _requires_presence_errors(sig, valid)
    errors += _generated_contract_errors(sig, valid, lets_env)
    return errors


def _callee_of(text: str) -> str | None:
    try:
        call = _parse(text)
    except SignatureError:
        return None
    return ast.unparse(call.func) if isinstance(call, ast.Call) else None


def _nested(rank: int):
    """The kind of a generated tensor's contents: `rank` nested sequences of ints."""
    kind = INT
    for _ in range(rank):
        kind = seq(kind)
    return kind


def _all_points(sig: Signature):
    """Every discriminant point, completed."""
    for point in _points(list(_discriminant_axes(sig).values())):
        yield complete_point(sig, point, strict=False)


def _generated_contract_errors(sig: Signature, valid: dict, env: dict) -> list[str]:
    """A generated tensor is int32 and has its generator's rank, on every branch it is passed."""
    errors = set()
    for t in sig.call_tensors.values():
        calls = {f: c for f, c, _ in valid.get(t.name, []) if f == "values"}
        if "values" not in calls:
            continue
        callee = calls["values"]
        dtype = _dtype_kind(sig, _parse(t.dtype), [])
        if dtype is None or dtype.tag != "DType" or not dtype.values or dtype.values - {"int32"}:
            errors.add(f"tensor {t.name!r}: generated values need dtype int32, not {t.dtype}")
        rank = GENERATOR_RANKS[callee]
        for field, predicate, _ in valid.get(t.name, []):
            if field == "requires" and PREDICATE_RANKS.get(predicate, rank) != rank:
                errors.add(
                    f"tensor {t.name!r}: {predicate} reads rank {PREDICATE_RANKS[predicate]}, {callee} yields {rank}"
                )
        for point in _all_points(sig):
            if not point.get(f"present({t.name})", True):
                continue
            try:
                node = expand(sig, t.shape, point)
            except SignatureError:
                continue
            if not _rank_fits(sig, env, node, rank):
                errors.add(
                    f"tensor {t.name!r}: {callee} yields rank {rank}, not {ast.unparse(node)}"
                )
    return sorted(errors)


def _rank_fits(sig: Signature, env: dict, node: ast.List, rank: int) -> bool:
    """Whether a shape list can have `rank` axes; a starred sequence counts each fixed length
    its kind's members allow, and at least zero where one is unknown."""
    options = []
    for e in node.elts:
        kind = _Kinds(sig, env, "").of(e.value) if isinstance(e, ast.Starred) else None
        members = kind.members if kind is not None and kind.tag == "Union" else (kind,)
        options.append(
            [1]
            if not isinstance(e, ast.Starred)
            else [m.length if m is not None and m.tag == "Seq" else None for m in members]
        )
    for combo in itertools.product(*options):
        known = sum(n for n in combo if n is not None)
        if known == rank or (None in combo and known < rank):
            return True
    return False


def _requires_presence_errors(sig: Signature, valid: dict) -> list[str]:
    """On every discriminant point, a `requires` reads only metadata tensors that are passed."""
    errors = set()
    for point in _all_points(sig):
        for name, calls in valid.items():
            if not point.get(f"present({name})", True):
                continue
            for field, _, call in calls:
                absent = {
                    m
                    for m in names(call) & set(sig.call_tensors)
                    if field == "requires" and not point.get(f"present({m})", True)
                }
                errors |= {
                    f"tensor {name!r}: {ast.unparse(call)} reads {m!r} where it is absent"
                    for m in absent
                }
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
        sig = parse_signature(name, entry, adts)
    except SignatureError as exc:
        return [f"{name}: {exc}"]
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
                call = instantiate(sig, row, case)
                call.arguments(dict.fromkeys(sig.ctor_tensors))
            except (RowError, SignatureError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"row {row['label']!r} {case}: {exc}")
                continue
            except (ImportError, AttributeError) as exc:
                errors.append(f"row {row['label']!r} {case}: a constructor class: {exc}")
                continue
            passed |= {t for t in gated if call.tensors[t] is not None}
            omitted |= {t for t in gated if call.tensors[t] is None}
            if call.case_id in seen:
                errors.append(f"case id {call.case_id!r} repeats")
            seen.add(call.case_id)
    if entry.get("status") == "implemented":
        errors += [f"no row passes optional tensor {t!r}" for t in gated if t not in passed]
        errors += [f"no row omits optional tensor {t!r}" for t in gated if t not in omitted]
    return [f"{name}: {e}" for e in errors]
