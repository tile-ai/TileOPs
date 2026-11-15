"""Instantiate and check workload rows (docs/design/manifest.md § Workloads).

`instantiate` turns one row and one dtype case into a `Call`: constructor arguments and the shape,
dtype and generated values of every tensor, read off the entry's branch view at the row's point.
`Call.materialize` builds the tensors a call passes and `Call.arguments` the constructor values;
both need torch. `check_workloads` returns the row diagnostics the validator reports.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from zlib import crc32

from tileops.manifest._dtype_rules import DTYPE_BITS
from tileops.manifest._expr import (
    EvaluationError,
    SignatureError,
    call_errors,
    callee,
    discriminant_names,
    evaluate,
    fold,
    infer_kinds,
    language_errors,
    names,
    parse,
    value_names,
)
from tileops.manifest._kinds import INT, _commas, seq, split_union
from tileops.manifest._plan import EntryPlan, PlanBranch, entry_plan
from tileops.manifest._signature import (
    ITEM,
    REQUIRED,
    TENSOR_MEMBERS,
    Signature,
    call_scalar_lets,
    complete_point,
    discriminant_axes,
    expand,
    items,
    kind_env,
    localized,
    member_of,
    param_kind,
    points,
    rejecting_rule,
    unification,
)
from tileops.manifest._values import ADTValue, convert, is_integer, python_class
from tileops.manifest.primitives import (
    GENERATOR_KINDS,
    GENERATOR_RANKS,
    GENERATOR_SHAPES,
    GENERATORS,
    PREDICATE_KINDS,
    PREDICATE_RANKS,
    PREDICATES,
    PRIMITIVE_KINDS,
    PRIMITIVES,
    RANDOM_GENERATORS,
    WORKLOAD_SEED,
)

if TYPE_CHECKING:
    import torch

__all__ = [
    "Call",
    "CallView",
    "Elements",
    "RowError",
    "TensorSpec",
    "call_from_tensors",
    "check_workloads",
    "instantiate",
]

_ROW_KEYS = frozenset({"some", "dtype_cases", "label"})
_LABEL = re.compile(r"[A-Za-z0-9._-]+")
# The integer dtypes generated metadata may take, with their ranges.
_METADATA_RANGES = {"int32": 2**31, "int64": 2**63}


class RowError(ValueError):
    """A workload row that does not state a call of its entry."""


@dataclass(frozen=True)
class Elements:
    """A tensor sequence's elements as a call holds them: each `(shape, dtype name)`, in order."""

    items: tuple


class CallView:
    """The checked-call view a roofline `func` formula reads (docs/design/roofline.md).

    `indices` holds the parameters, the resolved indices and dtype indices, and the reached
    `let`s; `tensors` maps each present tensor to its `(shape, dtype name)`, a tensor sequence
    to its `Elements`; `members` maps each call-time argument to the member it holds.
    """

    indices: dict[str, Any]
    tensors: dict[str, Any]
    members: dict[str, str | None] | None
    has_out: bool = False

    def present(self, name: str) -> bool:
        """`present(name)` of the call.

        A call-time argument is present when it holds a member other than None; any other
        name, when the call returns that tensor or passes `out`.
        """
        members = self.members or {}
        if name in members:
            return members[name] != "None"
        return name in self.tensors or (name == "out" and self.has_out)

    def member(self, name: str) -> str:
        """The member call-time argument `name` holds, as `member(name)` names it."""
        return self.members[name]

    def bytes(self, name: str) -> int:
        """The bytes tensor `name` occupies; a tensor sequence's, summed over its elements."""
        value = self.tensors[name]
        pairs = value.items if isinstance(value, Elements) else (value,)
        return sum((math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8 for shape, dtype in pairs)

    def metadata_values(self, name: str) -> list[Any]:
        """The contents of metadata tensor `name`, as nested lists."""
        raise NotImplementedError

    def derived(self, name: str) -> bool:
        """Whether the call moves none of input `name`.

        A determining predicate fixes its contents, so it is not read, and the call does not
        write it.
        """
        return False


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    values: list | None = None
    cpu: bool = False


@dataclass(frozen=True)
class Call(CallView):
    """One workload row instantiated for one dtype case: its parameters and tensor specs."""

    case_id: str
    # Parameter values as expressions read them: a dtype by name, an ADT by its fields.
    params: dict[str, Any]
    # A tensor sequence's spec is a tuple of its elements' specs.
    specs: dict[str, TensorSpec | tuple[TensorSpec, ...] | None]
    indices: dict[str, Any] = dataclasses.field(default_factory=dict)
    signature: Signature | None = dataclasses.field(default=None, repr=False, compare=False)
    # The inputs the call writes.
    written: frozenset[str] = frozenset()
    # Each call-time argument holding a scalar or None member, by name: what the call passes.
    scalars: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def members(self) -> dict[str, str | None]:
        """The member each call-time argument holds."""
        out = {}
        for t in self.signature.inputs.values():
            found = t.members
            if self.specs.get(t.name) is not None:
                out[t.name] = next(m for m in found if m in TENSOR_MEMBERS)
            else:
                out[t.name] = member_of(self.scalars.get(t.name), found)
        return out

    @property
    def tensors(self) -> dict[str, Any]:
        """Each present tensor's `(shape, dtype name)`, a tensor sequence's `Elements`."""
        return {
            n: Elements(tuple((e.shape, e.dtype) for e in s))
            if isinstance(s, tuple)
            else (s.shape, s.dtype)
            for n, s in self.specs.items()
            if s is not None
        }

    def metadata_values(self, name: str) -> list[Any]:
        """The generated contents of metadata tensor `name`, as nested lists."""
        return self.specs[name].values

    def derived(self, name: str) -> bool:
        """Whether a determining predicate fixes input `name` and the call does not write it."""
        tensor = self.signature.inputs.get(name) if self.signature is not None else None
        return tensor is not None and tensor.derived and name not in self.written

    def arguments(self) -> dict[str, Any]:
        """The op's constructor arguments.

        A dtype is a `torch.dtype`; an ADT is its constructor's `python` class called with its
        fields as keyword arguments, an enum field built from its own `python` class.
        """
        return {p: _python_value(self.signature, p, v) for p, v in self.params.items()}

    def materialize(self, device: str | torch.device = "cuda") -> dict[str, torch.Tensor | None]:
        """Build the tensors the call passes: generated values where declared, random data elsewhere.

        A tensor sequence is a list of its elements, each allocated on its own. A `device`
        parameter that is set decides where they go; *device* is the fallback.
        """
        device = self.params.get("device") or device
        out = {}
        outputs = self.signature.outputs if self.signature is not None else {}
        for name, spec in self.specs.items():
            if name in outputs:
                continue
            if name in self.scalars:
                out[name] = self.scalars[name]
            elif isinstance(spec, tuple):
                out[name] = [_tensor(e, device) for e in spec]
            else:
                out[name] = None if spec is None else _tensor(spec, device)
        return out


def _tensor(spec: TensorSpec, device: Any) -> Any:
    """One tensor of *spec*: its generated values where declared, random data elsewhere."""
    import torch

    dtype = getattr(torch, spec.dtype)
    place = "cpu" if spec.cpu else device
    if spec.values is not None:
        return torch.tensor(spec.values, dtype=dtype, device=place).reshape(spec.shape)
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(spec.shape, device=place, dtype=dtype if dtype.is_complex else None).to(
            dtype
        )
    high = 2 if dtype == torch.bool else 8
    return torch.randint(0, high, spec.shape, device=place).to(dtype)


def _python_value(sig: Signature, p: str, value: Any) -> Any:
    """One parameter value as its constructor takes it."""
    return _constructor_value(sig, str(sig.params[p].get("type")), value)


def _constructor_value(sig: Signature, type_text: str, value: Any) -> Any:
    """*value*, of parameter type *type_text*, as a constructor takes it.

    A sequence's elements are converted by their own type.
    """
    members = [m.replace(" ", "") for m in split_union(type_text)]
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
    if isinstance(value, (list, tuple)):
        container = next(
            (m for m in members if m.startswith(("list[", "tuple[")) and _holds(value, m, sig)),
            None,
        )
        if container is None:
            return value
        inner = container[container.index("[") + 1 : -1]
        items = _commas(inner)
        if container.startswith("list[") or items[-1] == "...":
            items = [items[0]] * len(value)
        converted = [_constructor_value(sig, t, v) for t, v in zip(items, value, strict=True)]
        return type(value)(converted)
    return value


def _holds(value: Any, type_text: str, sig: Signature) -> bool:
    """Whether parameter type *type_text* takes *value*."""
    try:
        convert(value, type_text, sig.adts)
    except ValueError:
        return False
    return True


def _evaluate(sig: Signature, node: ast.expr, scope: dict, where: str):
    """`node`'s value in `scope`; a failure is a `RowError` naming `where`."""
    try:
        return evaluate(node, scope, where, {"present": lambda v: v is not None})
    except EvaluationError as exc:
        raise RowError(str(exc)) from None


def _value_call(key: str, text: str) -> list:
    """The list a row's value-primitive call over integer literals yields; `RowError` otherwise."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        raise RowError(f"{key} = {text!r} is not an expression") from None
    name = ast.unparse(node.func) if isinstance(node, ast.Call) else None
    if name not in PRIMITIVES or PRIMITIVE_KINDS[name][1] != "Seq[Int]" or node.keywords:
        raise RowError(f"{key} = {text!r} is not a call of a Seq[Int] primitive")
    try:
        args = [ast.literal_eval(a) for a in node.args]
    except ValueError:
        raise RowError(f"{key} = {text!r}: arguments must be integer literals") from None
    if len(args) != len(PRIMITIVE_KINDS[name][0]) or not all(is_integer(a) for a in args):
        raise RowError(f"{key} = {text!r}: {name} takes {len(PRIMITIVE_KINDS[name][0])} integers")
    try:
        return PRIMITIVES[name](*args)
    except ValueError as exc:
        raise RowError(f"{key} = {exc}") from None


def _convert(sig: Signature, key: str, value):
    """A row or default value as its declared kind or `type` states it; `RowError` otherwise."""
    if key in sig.forall:
        kind = sig.forall[key]
        if kind == "Seq[Int]" and isinstance(value, str):
            value = _value_call(key, value)
        shape = lambda v: isinstance(v, list) and all(is_integer(x, 0) for x in v)  # noqa: E731
        ok = {
            "Dim": lambda v: is_integer(v, 0),
            "Shape": shape,
            "Seq[Int]": lambda v: isinstance(v, list) and all(is_integer(x) for x in v),
            "Seq[Dim]": shape,
            "Seq[Shape]": lambda v: isinstance(v, list) and all(shape(x) for x in v),
        }.get(kind, lambda v: False)
        if not ok(value):
            raise RowError(f"{key} = {value!r} is not a {kind}")
        if kind == "Seq[Shape]":
            return tuple(tuple(x) for x in value)
        return tuple(value) if kind in ("Shape", "Seq[Dim]") else value
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
            if kind.tag == "Maybe":
                point[f"present({p})"] = value is not None
            if value is not None:
                point[f"{p}.kind"] = value.kind
                for f, v in value.fields.items():
                    if not is_integer(v):
                        point[f"{p}.{f}"] = v
        elif kind.tag == "Maybe":
            point[f"present({p})"] = scope.get(p) is not None
    for t in sig.inputs.values():
        if len(t.members) > 1:
            point[f"member({t.name})"] = _member_of(t, scope, some)
    if any(t.buffer for t in sig.outputs.values()):
        point["present(out)"] = False
    return complete_point(sig, point)


def _member_of(t, scope: dict, some: set[str]) -> str:
    """The member a row's call passes for argument *t*.

    That is its tensor member where `some` names it, else the member holding the row's value
    for it, else its default's.
    """
    if t.name in some:
        return next(m for m in t.members if m in TENSOR_MEMBERS)
    if t.name in scope:
        value = scope[t.name]
    elif t.default is not REQUIRED:
        value = t.default
    else:
        raise RowError(f"row misses argument {t.name!r}: list it in `some` or give its value")
    found = member_of(value, t.members)
    if found is None or found in TENSOR_MEMBERS:
        raise RowError(f"{t.name} = {value!r} is not a non-tensor member of {t.type!r}")
    return found


def _dtype_index(sig: Signature, n: str) -> bool:
    """Whether index *n* is a `DType` index or an element-wise `Seq[DType[...]]` one."""
    return sig.forall[n].startswith(("DType", "Seq[DType"))


def _relevant(sig: Signature, b: PlanBranch) -> tuple[set[str], set[str]]:
    """Indices the row must give at the branch, and those a generator solves."""
    tensors = set(sig.inputs)
    free = {n for n in sig.forall if not _dtype_index(sig, n)}
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
            reads = set().union(set(), *(value_names(a) for a in b.generators[t])) - tensors
            if t not in ran and reads <= known:
                ran[t] = generated[t]
        known |= {n for s in unification(ran, known, {}) for n in s.solved}
        if known == before:
            break
    return need, (known - seed) & free


def _apply(
    sig: Signature,
    steps: list,
    nodes: dict,
    shapes: dict,
    scope: dict,
    lets: dict,
    sequences: dict | None = None,
) -> None:
    """Carry out an inference plan on concrete `shapes`, binding into `scope`.

    A tensor sequence's shape is the tuple of its elements' shapes; `sequences` maps it to its
    element-wise indices.
    """
    sequences = sequences or {}
    raw, nodes, shapes = nodes, dict(nodes), dict(shapes)
    elements = {t: shapes[t] for t in sequences}

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
        if step.action in ("first", "each"):
            t = step.tensor
            if step.action == "first":
                pooled = [elements[u] for u in (t, *step.items) if elements[u]]
                if not pooled:
                    names_ = " and ".join((t, *step.items))
                    raise RowError(f"{names_} empty: no element 0 solves {t}'s shared names")
                shapes[t], nodes[t] = pooled[0][0], localized(raw[t], sequences[t])
                continue
            _apply_sequence(sig, step, nodes, shapes, scope, raw[t], sequences[t], elements[t])
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


def _apply_sequence(
    sig: Signature,
    step,
    nodes: dict,
    shapes: dict,
    scope: dict,
    element: ast.List,
    each: tuple,
    elements: tuple,
) -> None:
    """Carry out one `each` step on tensor sequence `step.tensor`.

    Its element type is `element`, its element-wise indices `each` and its element shapes
    `elements`.
    """
    t, node = step.tensor, localized(element, each)
    collected = {x: [] for x in step.items}
    checked = [x for x in items(element, frozenset(each)) if x not in collected]
    for x in checked:
        if len(scope[x]) != len(elements):
            raise RowError(f"{x} has {len(scope[x])} items, {t} has {len(elements)} elements")
    for k, shape in enumerate(elements):
        local = {**scope, **{f"{ITEM}{x}": scope[x][k] for x in checked}}
        _apply(sig, list(step.sub), {t: node}, {t: shape}, local, {})
        for x in collected:
            collected[x].append(local[f"{ITEM}{x}"])
    scope.update({x: tuple(v) for x, v in collected.items()})


def _fixed_rank(node: ast.List, length) -> int | None:
    """The rank of a shape where every splice has a fixed length, `length(splice)`; else None."""
    lengths = [length(e.value) if isinstance(e, ast.Starred) else 1 for e in node.elts]
    return None if None in lengths else sum(lengths)


def instantiate(
    plan: EntryPlan, row: dict[str, Any], dtype_case: dict[str, str], *, metadata: bool = True
) -> Call:
    """The call one row and one dtype case state; raises `RowError` when they state none.

    With `metadata=False`, describe tensor shapes and dtypes without generating or
    validating their contents. Shape refinements and inference are still checked.
    """
    sig = plan.sig
    scope = {}
    for key, value in row.items():
        if key in _ROW_KEYS:
            continue
        if key in sig.inputs and len(sig.inputs[key].members) > 1:
            if key in row.get("some", ()):
                raise RowError(f"{key} is in `some`, which passes its tensor, and has a value")
            scope[key] = value
            continue
        if key not in sig.forall and key not in sig.params:
            raise RowError(f"row key {key!r} is neither an index, a parameter nor an argument")
        scope[key] = _convert(sig, key, value)
    for p, decl in sig.params.items():
        if p not in scope:
            if "default" not in decl:
                raise RowError(f"row misses parameter {p!r}, which has no default")
            scope[p] = _convert(sig, p, decl["default"])
    try:
        point = _point(sig, scope, set(row.get("some", [])))
        scope |= {
            t.name: t.default
            for t in sig.inputs.values()
            if t.name not in scope
            and len(t.members) > 1
            and point[f"member({t.name})"] not in TENSOR_MEMBERS
        }
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

    dtype_indices = {n for n in b.relevant if n in sig.forall and _dtype_index(sig, n)}
    if set(dtype_case) != dtype_indices:
        raise RowError(
            f"dtype_cases assign {sorted(dtype_case)}, the entry's DType indices are {sorted(dtype_indices)}"
        )
    for index, dtype in dtype_case.items():
        kind = sig.kind(index)
        if kind.tag == "Seq":
            if not isinstance(dtype, list) or any(d not in kind.item.values for d in dtype):
                raise RowError(f"{index} = {dtype} is not a list of {sig.forall[index][4:-1]}")
            dtype_case = {**dtype_case, index: tuple(dtype)}
        elif dtype not in kind.values:
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
            if set().union(set(), *map(names, b.generators[t])) - set(sig.inputs) <= set(scope)
        ]
        for t in ready:
            waiting.remove(t)
            generated[t], shapes[t] = _generate(sig, t, b, scope, metadata=metadata)
            nodes[t] = b.shapes[t]
        _apply(sig, unification(nodes, set(scope), {}), nodes, shapes, scope, {})
        if set(scope) == known and not ready:
            break
    if waiting:
        raise RowError(f"generator arguments of {waiting} are never known")

    for i, rule in enumerate(b.rules):
        if not _evaluate(sig, rule, scope, f"shape_rules[{i}]"):
            raise RowError(f"refinement fails: {sig.rules[i]}")
    specs: dict = {}
    for t in (*sig.inputs.values(), *sig.outputs.values()):
        if t.name not in b.shapes:
            specs[t.name] = None
            continue
        where = f"tensor {t.name!r} shape"
        if t.name in b.sequences:
            specs[t.name] = tuple(
                TensorSpec(
                    _shape(b.shapes[t.name], sig, local, where),
                    _dtype_value(b.dtypes[t.name], local),
                )
                for local in _elements(t.name, b.sequences[t.name], scope)
            )
            continue
        shape = _shape(b.shapes[t.name], sig, scope, where)
        dtype = _dtype_value(b.dtypes[t.name], scope)
        specs[t.name] = TensorSpec(shape, dtype, generated.get(t.name), t.cpu)
    scalars = {t: scope[t] for t in sig.inputs if t in scope and specs[t] is None}
    for t, calls in b.requires.items() if metadata else ():
        for call, text in zip(calls, sig.inputs[t].requires, strict=False):
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
    # Solve the indices again from the instantiated inputs alone, as the op's checks do.
    known = {
        k: v
        for k, v in scope.items()
        if k in sig.params
        or k in scalars
        or (k in sig.forall and (_dtype_index(sig, k) or sig.forall[k] == "Seq[Int]"))
    }
    solved = call_from_tensors(plan, known, {t: specs[t] for t in sig.inputs}).indices
    mismatch = {
        n: (solved[n], scope[n])
        for n in sig.forall
        if n in scope and n in solved and solved[n] != scope[n]
    }
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
        [
            label,
            # A `Seq[DType]` index's dtypes, in element order, joined by `.`.
            *(
                ".".join(dtype_case[i]) if isinstance(dtype_case[i], tuple) else dtype_case[i]
                for i in sig.forall
                if i in dtype_case
            ),
            *dtype_params,
        ]
    )
    written = sig.written(point, {n for n, spec in specs.items() if spec is not None})
    return Call(
        case_id,
        {p: scope[p] for p in sig.params},
        specs,
        _indices(sig, b, scope),
        sig,
        written,
        scalars,
    )


def _elements(t: str, each: tuple, scope: dict) -> list[dict]:
    """The scope of each element of tensor sequence *t*.

    Each binds the element-wise indices `each` to that element's items.
    """
    counts = {len(scope[x]) for x in each}
    if len(counts) != 1:
        raise RowError(f"the element-wise indices {list(each)} of {t} differ in length")
    return [{**scope, **{x: scope[x][k] for x in each}} for k in range(counts.pop())]


def call_from_tensors(plan: EntryPlan, known: dict, inputs: dict) -> Call:
    """The call of `plan` on the call tensors `inputs`.

    Its indices are solved from their shapes and dtypes as the op's checks solve them.

    `known` holds every signature parameter, as expressions read it, and may hold indices the
    tensors do not determine, such as a generator's arguments. `inputs` maps each call tensor
    to its `TensorSpec`, or None where the call does not pass it; a DType index an input's
    dtype names is that dtype. The outputs' specs follow from the solved indices.

    Raises:
        RowError: An input's shape disagrees with what is known or with another input.
    """
    sig = plan.sig
    scope = dict(known)
    # In declaration order, which inference follows.
    present = [t for t in sig.inputs if inputs.get(t) is not None]
    point = _point(sig, scope, set(present))
    b = plan.branch(point)
    sequences = {t: b.sequences[t] for t in present if t in b.sequences}
    for t in sorted(present, key=sequences.__contains__):
        node, leaves = b.dtypes[t], inputs[t] if t in sequences else (inputs[t],)
        if not (isinstance(node, ast.Name) and node.id in sig.forall):
            continue
        if node.id in sequences.get(t, ()):
            found = tuple(e.dtype for e in leaves)
            if scope.setdefault(node.id, found) != found:
                raise RowError(f"the elements of {t} differ in dtype from {node.id}")
        else:
            for e in leaves:
                if scope.setdefault(node.id, e.dtype) != e.dtype:
                    raise RowError(f"{t} dtype {e.dtype} differs from {node.id}")
    nodes = {t: b.shapes[t] for t in present}
    shapes = {
        t: tuple(tuple(e.shape) for e in inputs[t]) if t in sequences else tuple(inputs[t].shape)
        for t in present
    }
    steps = unification(nodes, set(scope), b.lets, sequences)
    _apply(sig, steps, nodes, shapes, scope, b.lets, sequences)
    specs = {t: inputs.get(t) for t in sig.inputs}
    for t in sig.outputs.values():
        specs[t.name] = (
            TensorSpec(
                _shape(b.shapes[t.name], sig, scope, f"tensor {t.name!r} shape"),
                _dtype_value(b.dtypes[t.name], scope),
                None,
                t.cpu,
            )
            if t.name in b.shapes
            else None
        )
    params = {p: scope[p] for p in sig.params}
    scalars = {t: scope[t] for t in sig.inputs if t in scope and specs[t] is None}
    return Call(
        "", params, specs, _indices(sig, b, scope), sig, sig.written(point, present), scalars
    )


def _indices(sig: Signature, b: PlanBranch, scope: dict) -> dict:
    """The `indices` a call's formulas read, as `scope` binds them.

    They are the parameters, the indices other than a generator's value sequences, and the
    `let`s.
    """
    values = {n for n, k in sig.forall.items() if k == "Seq[Int]"}
    return {
        n: v
        for n, v in scope.items()
        if n in sig.params or (n in sig.forall and n not in values) or n in b.lets
    }


def _generate(
    sig: Signature, t: str, b: PlanBranch, scope: dict, *, metadata: bool = True
) -> tuple[list | None, tuple]:
    """The values tensor `t`'s generator yields for this row, and their shape.

    The generator is seeded per op and tensor, and the shape is checked against the rank `t`
    declares.

    With `metadata=False`, the values are None and only the shape is computed.
    """
    fn = callee(parse(sig.inputs[t].values).func)
    dtype = _dtype_value(b.dtypes[t], scope)
    if dtype not in _METADATA_RANGES:
        raise RowError(f"tensor {t!r}: generated values need an int32 or int64 dtype, not {dtype}")
    args = [_evaluate(sig, a, scope, f"tensor {t!r} values") for a in b.generators[t]]
    values = None
    if metadata:
        if fn in RANDOM_GENERATORS:
            args.insert(0, random.Random(WORKLOAD_SEED ^ crc32(f"{sig.name}:{t}".encode())))
        try:
            values = GENERATORS[fn](*args)
        except ValueError as exc:
            raise RowError(f"tensor {t!r}: {exc}") from None
        if not _within(values, _METADATA_RANGES[dtype]):
            raise RowError(f"tensor {t!r}: {sig.inputs[t].values} yields values outside {dtype}")
    shape = GENERATOR_SHAPES[fn](*(a for a in args if not isinstance(a, random.Random)))
    rank = _fixed_rank(
        b.shapes[t],
        lambda e: (
            len(_evaluate(sig, e, scope, f"tensor {t!r} shape")) if names(e) <= set(scope) else None
        ),
    )
    if rank is not None and rank != len(shape):
        raise RowError(
            f"tensor {t!r}: {fn} yields rank {len(shape)}, not {ast.unparse(b.shapes[t])}"
        )
    return values, shape


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


def _dumped(text: str | None) -> str | None:
    try:
        return None if text is None else ast.dump(parse(text))
    except SignatureError:
        return None


def equal_inputs(sig: Signature) -> list[tuple[str, ...]]:
    """The groups of inputs that declare one deterministic `values`, which must be equal."""
    groups: dict[str, list[str]] = {}
    for t in sig.inputs.values():
        if _dumped(t.values) is not None and _callee_of(t.values) not in RANDOM_GENERATORS:
            groups.setdefault(_dumped(t.values), []).append(t.name)
    return [tuple(g) for g in groups.values() if len(g) > 1]


def _declaration_errors(sig: Signature) -> list[str]:
    """Diagnostics of every `values` and `requires` call.

    Each has a built-in callee, its arity and arguments in the language; the `requires`
    contract holds on every discriminant point.
    """
    errors, valid = [], {}
    declared = {*sig.forall, *sig.params, *sig.let}
    scalar_lets = call_scalar_lets(sig)
    env, _ = kind_env(sig, {n: parse(e) for n, e in sig.let.items()})
    generated = {
        n: GENERATOR_RANKS.get(_callee_of(d.values))
        for n, d in sig.inputs.items()
        if d.values is not None
    }
    for t in sig.inputs.values():
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
                read = value_names(arg)
                found += [
                    f"{where}: let {n!r} reads a call-time scalar; read it only in a refinement"
                    for n in sorted(read & scalar_lets)
                ]
                found += [f"{where}: name {n!r} is not declared" for n in sorted(read - scope)]
                found += [
                    f"{where}: {n!r} is no argument"
                    for n in sorted(discriminant_names(arg) - {*sig.inputs, *sig.params})
                ]
            contents = {n: _nested(r) for n, r in generated.items() if r is not None}
            # A `Maybe` parameter is read as its payload; where it is absent is checked per point.
            payloads = {
                n: k.item
                for n, k in env.kinds.items()
                if n in sig.params and k and k.tag == "Maybe"
            }
            local = env.narrowed(contents | payloads) if field == "requires" else env
            found += call_errors(call, table, local, where)
            errors += found
            if not found:
                valid.setdefault(t.name, []).append((field, name, call))
    return errors + _requires_errors(sig, valid, env)


def _requires_errors(sig: Signature, valid: dict, env) -> list[str]:
    """Diagnostics of the `requires` contract on every discriminant point.

    On every point a domain restriction accepts where its argument holds a tensor, a
    `requires` reads only metadata arguments holding a tensor and `Maybe` parameters that are
    present, and its predicate reads the rank the tensor declares there, which must be fixed.
    """

    def length(e: ast.expr) -> int | None:
        kind = infer_kinds(e, env, "")[0]
        return kind.sequence().length if kind is not None and kind.sequence() is not None else None

    errors = set()
    for point in points(list(discriminant_axes(sig).values())):
        point = complete_point(sig, point, strict=False)
        if rejecting_rule(sig, point) is not None:
            continue
        for name, calls in valid.items():
            if point.get(f"member({name})", "Tensor") not in TENSOR_MEMBERS:
                continue
            try:
                node = fold(expand(sig, sig.inputs[name].shape, point), point)
            except SignatureError:
                node = None
            for field, predicate, call in calls:
                if field != "requires":
                    continue
                for m in value_names(call) & {*sig.inputs, *sig.params}:
                    held = point.get(f"member({m})", "Tensor") if m in sig.inputs else None
                    if held not in (None, *TENSOR_MEMBERS) or not point.get(f"present({m})", True):
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
    # The manifest spec's cap on a label's length.
    label_max = 24
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        errors.append("`label` must be a non-empty [A-Za-z0-9._-] string")
    elif len(label) > label_max:
        errors.append(f"`label` {label!r} is longer than {label_max} characters")
    optional = {t.name for t in sig.inputs.values() if len(t.members) > 1}
    some = row.get("some", [])
    if (
        not isinstance(some, list)
        or not all(isinstance(s, str) for s in some)
        or len(set(some)) != len(some)
    ):
        errors.append("`some` must be a list of distinct names")
    elif set(some) - optional:
        errors.append(
            f"`some` names {sorted(set(some) - optional)}, not arguments with several members"
        )
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
    seen = set()
    gated = {t.name: t.members for t in sig.inputs.values() if len(t.members) > 1}
    covered: dict = {t: set() for t in gated}
    for i, row in enumerate(rows):
        problems = _row_errors(sig, row)
        if problems:
            errors += [f"row {i}: {p}" for p in problems]
            continue
        for case in row.get("dtype_cases", [{}]):
            try:
                call = instantiate(plan, row, case)
                call.arguments()
            except (RowError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"row {row['label']!r} {case}: {exc}")
                continue
            except (ImportError, AttributeError) as exc:
                errors.append(f"row {row['label']!r} {case}: a constructor class: {exc}")
                continue
            for t, found in gated.items():
                passed = call.specs[t] is not None
                covered[t].add(
                    next(m for m in found if m in TENSOR_MEMBERS)
                    if passed
                    else member_of(call.scalars.get(t), found)
                )
            if call.case_id in seen:
                errors.append(f"case id {call.case_id!r} repeats")
            seen.add(call.case_id)
    if entry.get("status") == "implemented":
        errors += [
            f"no row passes member {m!r} of {t!r}"
            for t, found in gated.items()
            for m in found
            if m not in covered[t] and _reachable(sig, t, m)
        ]
    return [f"{name}: {e}" for e in errors]


def _reachable(sig: Signature, t: str, member: str) -> bool:
    """Whether a discriminant point a domain restriction accepts has argument *t* hold *member*."""
    axes = dict(discriminant_axes(sig))
    axes[t] = (f"member({t})", [member])
    for point in points(list(axes.values())):
        try:
            if rejecting_rule(sig, complete_point(sig, point)) is None:
                return True
        except SignatureError:
            continue
    return False
