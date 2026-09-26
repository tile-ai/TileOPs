"""An entry's checked plan and its static diagnostics (docs/design/manifest.md § Validation).

`EntryPlan` holds a checked signature and its checked roofline; `EntryPlan.branch(point)` is
the one view every per-call quantity is read from. `check_entry` returns the static
diagnostics: name categories and kinds, type families, the inference plan, `let` cycles and
the closed expression language. Every expression is parsed once; a field that fails to parse
is reported and skipped by the checks that read it.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib
from dataclasses import dataclass

from .dtype_rules import DTYPE_BITS
from .expr import (
    KindEnv,
    SignatureError,
    Unpresent,
    argument_errors,
    callee,
    fold,
    infer_kinds,
    language_errors,
    names,
    parse,
)
from .kinds import BOOL, INT, NONE, VALUE, Kind, parse_spec, parse_type, seq
from .signature import (
    Signature,
    SignatureBranch,
    adt_fields,
    branch,
    complete_point,
    discriminant_groups,
    dtype_kind,
    field_kind,
    holds,
    kind_env,
    parse_signature,
    points,
    reach,
    read_signature,
    rejecting_rule,
    unification,
)

__all__ = [
    "EntryPlan",
    "PlanBranch",
    "check_adts",
    "check_entry",
    "effect_errors",
    "entry_plan",
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
_ROOFLINE_KEYS = frozenset({"flops", "bytes", "func"})
# Reserved: `forward` takes the output buffer under this name.
_RESERVED = "out"


# ---------------------------------------------------------------- the plan


class _Unbytes(ast.NodeTransformer):
    """Replace `bytes(t)` by a constant: it reads `t`'s size, not its value."""

    def visit_Call(self, node):
        if callee(node.func) == "bytes":
            return ast.Constant(0)
        return self.generic_visit(node)


def _formula_reads(node: ast.expr) -> set[str]:
    """What an inline roofline expression reads besides tensor sizes and presence."""
    return names(Unpresent().visit(_Unbytes().visit(copy.deepcopy(node))))


def _call_or_none(text: object) -> ast.Call | None:
    try:
        node = parse(text)
    except SignatureError:
        return None
    return node if isinstance(node, ast.Call) else None


@dataclass
class PlanBranch(SignatureBranch):
    """The branch view: the signature's branch plus the generator arguments, `requires` and
    roofline expressions at the same point, all folded there."""

    # Each passed generated tensor's generator arguments, and each `requires` call.
    generators: dict = None
    requires: dict = None
    # The inline roofline `{"flops": tree, "bytes": tree or None}`; None for `func` or none.
    roofline: dict | None = None
    # What a call reads: `relevant` less what only generators and `requires` read.
    called: set = None


@dataclass
class EntryPlan:
    """A checked signature and its checked roofline plan."""

    sig: Signature
    # `{"func": callable}` or `{"flops": tree, "bytes": tree or None}`; None without one.
    roofline: dict | None = None

    def branch(self, point: dict, complete: bool = True) -> PlanBranch:
        """Every per-call quantity at `point`, folded there."""
        sig = self.sig
        b = branch(sig, point, complete)
        tensors = {*sig.call_tensors, *sig.outputs, _RESERVED}
        roofline = None
        called = set(b.relevant)
        if self.roofline is not None and "flops" in self.roofline:
            roofline = {
                k: None if node is None else fold(node, point, f"roofline.{k}")
                for k, node in (
                    ("flops", self.roofline["flops"]),
                    ("bytes", self.roofline["bytes"]),
                )
            }
            called |= (
                set().union(*(_formula_reads(n) for n in roofline.values() if n is not None))
                - tensors
            )
        lets = dict(b.lets)
        reach(sig, point, called, lets)
        generators, requires, relevant = {}, {}, set(called)
        for t in sig.call_tensors.values():
            if t.name not in b.shapes:
                continue
            # A call that is not well formed is `check_workloads`' to report.
            if (call := _call_or_none(t.values)) is not None:
                generators[t.name] = [
                    fold(a, point, f"tensor {t.name!r} values") for a in call.args
                ]
                relevant |= set().union(set(), *(names(a) for a in generators[t.name])) - tensors
            requires[t.name] = []
            for call in filter(None, map(_call_or_none, t.requires)):
                call.args = [fold(a, point, f"tensor {t.name!r} requires") for a in call.args]
                requires[t.name].append(call)
            relevant |= set().union(set(), *(names(r) for r in requires[t.name])) - tensors
        reach(sig, point, relevant, lets)
        return PlanBranch(
            b.shapes, b.dtypes, b.rules, lets, relevant, generators, requires, roofline, called
        )


def entry_plan(name: str, entry: dict, adts: dict, resolve: bool = True) -> EntryPlan:
    """The plan of an entry; `SignatureError` on a malformed signature. A roofline with
    diagnostics leaves the plan without one."""
    sig = parse_signature(name, entry, adts)
    errors, roofline = roofline_plan(sig, entry.get("roofline"), resolve=resolve)
    return EntryPlan(sig, None if errors else roofline)


def roofline_plan(
    sig: Signature | None, roofline: object, resolve: bool = True
) -> tuple[list[str], dict | None]:
    """The `roofline` field (docs/design/roofline.md) checked, and what emission reads.

    The plan is `{"func": callable}` or `{"flops": tree, "bytes": tree or None}`; it is None
    when there are diagnostics. An inline expression reads the signature's indices other than
    value lists, its parameters and its `let`s, and may call `bytes(t)` and `present(t)`. A
    `func` is imported only when *resolve* is set: an entry not yet implemented may name a
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
        parts = path.split(".") if isinstance(path, str) else []
        if not (
            len(parts) > 1
            and all(p.isidentifier() for p in parts)
            and path.startswith("tileops.perf.formulas.")
        ):
            errors.append(f"roofline.func {path!r} is not tileops.perf.formulas.<name>")
        elif resolve:
            module, _, attr = path.rpartition(".")
            try:
                fn = getattr(importlib.import_module(module), attr)
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
            node = parse(roofline[key])
        except SignatureError as exc:
            errors.append(f"roofline.{key}: {exc}")
            continue
        errors += [
            f"roofline.{key}: {e}"
            for e in language_errors(_Unbytes().visit(copy.deepcopy(node)), False)
        ]
        plan[key] = node
        if sig is None:
            continue
        ix = {n for n, k in sig.forall.items() if k != "Seq[Int]"} | set(sig.params) | set(sig.let)
        tensors = {*sig.call_tensors, *sig.outputs}
        maybes = {p for p in sig.params if sig.kind(p).tag == "Maybe"}
        buffered = {_RESERVED} if any(o.buffer for o in sig.outputs.values()) else set()
        for call in (c for c in ast.walk(node) if isinstance(c, ast.Call)):
            name = callee(call.func)
            allowed = tensors | (maybes | buffered if name == "present" else set())
            if name in ("bytes", "present") and not (
                len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in allowed
            ):
                errors.append(f"roofline.{key}: {ast.unparse(call)} names no tensor")
        errors += [
            f"roofline.{key}: name {n!r} is not in ix" for n in sorted(_formula_reads(node) - ix)
        ]
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


# ---------------------------------------------------------------- ADTs and schema


def _qualified(name: object) -> bool:
    return (
        isinstance(name, str)
        and len(parts := name.split(".")) > 1
        and all(p.isidentifier() for p in parts)
    )


def _adt_errors(name: object, adt: object) -> list[str]:
    """One ADT: constructors with their `python` class, field kinds, invariants in the language."""
    ctors = adt.get("sum") if isinstance(adt, dict) else None
    if (
        not (isinstance(name, str) and name.isidentifier())
        or not isinstance(ctors, dict)
        or not ctors
        or set(adt) != {"sum"}
    ):
        return [f"adt {name!r}: needs an identifier and exactly a non-empty `sum` of constructors"]
    errors = []
    for ctor, spec in ctors.items():
        where = f"adt {name}.{ctor}"
        if (
            not (isinstance(ctor, str) and ctor.isidentifier())
            or not isinstance(spec, dict)
            or set(spec) - _ADT_CTOR_KEYS
        ):
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
                not (isinstance(f, str) and f.isidentifier())
                or field_kind(k) is None
                or (mapping and (set(k) != {"type", "python"} or not _qualified(k["python"])))
            ):
                errors.append(f"{where}.{f}: needs an identifier and a Dim, Int, Bool or enum kind")
        if "invariant" not in spec:
            continue
        try:
            node = parse(spec["invariant"])
        except SignatureError as exc:
            errors.append(f"{where} invariant: {exc}")
            continue
        errors += [f"{where} invariant: {e}" for e in language_errors(node, False)]
        errors += [
            f"{where} invariant: name {n!r} is not a field"
            for n in sorted(names(node) - set(fields))
        ]
        env = KindEnv({f: field_kind(k) for f, k in fields.items()})
        errors += infer_kinds(node, env, where, BOOL)[1]
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


# ---------------------------------------------------------------- type families


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
            fk = field_kind(declared[f]) if f in declared else None
            if fk is None:
                errors.append(f"{where}: {ctor!r} has no field {f!r}")
            elif fk.tag == "Int":
                errors.append(f"{where}: {ctor}.{f} is not a finite field; it cannot be matched")
            elif fk.tag == "Str" and v not in fk.values:
                errors.append(f"{where}: {v!r} is not a value of {ctor}.{f}")
            elif fk == BOOL and not isinstance(v, bool):
                errors.append(f"{where}: {ctor}.{f} is a Bool")
        return errors
    if kind == BOOL and not isinstance(pattern, bool):
        return [f"{where}: {pattern!r} is not a Bool"]
    if kind.tag == "Str" and pattern not in kind.values:
        return [f"{where}: {pattern!r} is not one of {kind}"]
    return []


def _family_env(formals: dict, sig: Signature) -> KindEnv:
    """A family's formals as a kind environment; a formal whose kind is malformed is left out."""
    kinds = {}
    for p, k in formals.items():
        with contextlib.suppress(ValueError):
            kinds[p] = parse_spec(str(k), sig.adts)
    return KindEnv(kinds, adt_fields(sig.adts), frozenset(sig.types))


def _case_facts(spec: dict, case: dict) -> dict:
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
    env = _family_env(formals, sig)
    errors = [
        f"type family {fam}: formal {p!r} needs an identifier and a kind"
        for p in formals
        if not (isinstance(p, str) and p.isidentifier()) or p not in env.kinds
    ]
    keys = spec["match"] if isinstance(spec["match"], list) else [spec["match"]]
    kinds = []
    for key in keys:
        where = f"type family {fam} match {key!r}"
        try:
            node = parse(key)
        except SignatureError as exc:
            errors.append(f"{where}: {exc}")
            continue
        errors += [f"{where}: {e}" for e in language_errors(node, False)]
        errors += [
            f"{where}: name {n!r} is not a formal" for n in sorted(names(node) - set(formals))
        ]
        kind, found = infer_kinds(node, env, where)
        errors += found
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
        patterns = case["when"] if isinstance(case["when"], list) else [case["when"]]
        if len(patterns) != len(keys):
            errors.append(f"{where}: {len(patterns)} components, `match` has {len(keys)}")
            continue
        for pattern, kind in zip(patterns, kinds, strict=True):
            errors += _pattern_errors(where, pattern, kind, sig.adts)
    return errors


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


# ---------------------------------------------------------------- check_entry


def _shape_errors(node: ast.expr | None, env: KindEnv, where: str, families: dict) -> list[str]:
    """A shape: each axis an integer, each spliced sequence of integers; or a family application."""
    if node is None:
        return []
    if isinstance(node, ast.List):
        errors = []
        for axis in node.elts:
            if isinstance(axis, ast.Starred):
                errors += infer_kinds(axis.value, env, where, seq(INT))[1]
            else:
                errors += infer_kinds(axis, env, where, INT)[1]
        return errors
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in families
    ):
        family = families[node.value.id]
        if family is None:
            return []  # a malformed family is reported on its own
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        formal = family.get("params") or {}
        if len(args) != len(formal):
            return [f"{where}: {node.value.id} takes {len(formal)} arguments, got {len(args)}"]
        return [
            e
            for arg, kind in zip(args, formal.values(), strict=True)
            for e in argument_errors(arg, parse_spec(str(kind), env.adts), env, where)
        ]
    return [f"{where}: {ast.unparse(node)!r} is not a list or a type-family application"]


def check_entry(name: str, entry: dict, adts: dict) -> tuple[list[str], list[str]]:  # noqa: C901
    """Return `(errors, warnings)` for one entry; each message is prefixed with its op name.

    *adts* are the ADTs `check_adts` accepts.
    """
    try:
        sig, problems = read_signature(name, entry, adts)
    except SignatureError as exc:
        return [f"{name}: {exc}"], []
    errors: list[str] = problems + _combo_errors(sig)
    for p, decl in sig.params.items():
        try:
            parse_type(decl.get("type"), sig.adts)
        except ValueError as exc:
            errors.append(f"params.{p}: {exc}")
    declared = [*sig.forall, *sig.params, *sig.ctor_tensors, *sig.let, *sig.inputs, *sig.outputs]
    errors += [
        f"name {d!r} is declared twice"
        for d in sorted({n for n in declared if declared.count(n) > 1})
    ]
    if _RESERVED in declared:
        errors.append(f"name {_RESERVED!r} is reserved for the output buffer")

    parsed: dict[str, ast.expr] = {}

    def parse_field(key: str, text: object, shape: bool = False) -> ast.expr | None:
        try:
            node = parse(text)
        except SignatureError as exc:
            errors.append(f"{key}: {exc}")
            return None
        errors.extend(f"{key}: {e}" for e in language_errors(node, shape))
        parsed[key] = node
        return node

    tensors = {**sig.call_tensors, **sig.outputs}
    lets = {n: node for n, e in sig.let.items() if (node := parse_field(f"let {n}", e)) is not None}
    rules = [
        node
        for i, r in enumerate(sig.rules)
        if (node := parse_field(f"shape_rules[{i}]", r)) is not None
    ]
    shapes = {
        t: parse_field(f"tensor {t!r} shape", d.shape, shape=True) for t, d in tensors.items()
    }
    flags = {
        (t, a): parse_field(f"tensor {t!r} {a}", getattr(d, a))
        for t, d in tensors.items()
        for a in ("optional", "nullable", "mutated")
        if isinstance(getattr(d, a), str)
    }
    for t, d in tensors.items():
        node = parse_field(f"tensor {t!r} dtype", d.dtype)
        if node is not None:
            found: list[str] = []
            kind = dtype_kind(sig, node, found)
            if kind is not None and kind.tag == "Maybe":
                found.append(f"{ast.unparse(node)} may be absent; use coalesce_dtype")
            errors += [f"tensor {t!r} dtype: {e}" for e in found]
    malformed = {fam: _family_errors(fam, spec, sig) for fam, spec in sig.types.items()}
    errors += [e for family in malformed.values() for e in family]
    sig.types = {fam: spec for fam, spec in sig.types.items() if not malformed[fam]}
    cases = {}
    for fam, spec in sig.types.items():
        for j, case in enumerate(spec["cases"]):
            cases[(fam, j)] = parse_field(f"type family {fam} case {j}", case["is"], shape=True)

    env, let_errors = kind_env(sig, lets)
    errors += let_errors
    value_lists = {n for n, k in sig.forall.items() if k == "Seq[Int]"}
    # A parameter outside the type system appears only in refinements.
    untyped = {
        p
        for p, k in env.kinds.items()
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
        read = names(Unpresent().visit(copy.deepcopy(node))) - formals
        errors += [
            f"{key}: tensor {n!r} is read as a value; use its shape or present({n})"
            for n in sorted(read & set(tensors))
        ]
        if not key.startswith("shape_rules"):
            errors += [
                f"{key}: parameter {n!r} of type {sig.params[n].get('type')} takes no part in types"
                for n in sorted(read & untyped)
            ]
        scope = set(env.kinds) | set(tensors) | set(malformed)
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
        errors += infer_kinds(node, env, f"shape_rules[{i}]", BOOL)[1]
    for (t, a), node in flags.items():
        if node is not None:
            errors += infer_kinds(node, env, f"tensor {t!r} {a}", BOOL)[1]
    families = {fam: (None if malformed[fam] else sig.types[fam]) for fam in malformed}
    for t, node in shapes.items():
        errors += _shape_errors(node, env, f"tensor {t!r} shape", families)
    for (fam, j), node in cases.items():
        spec = sig.types[fam]
        local = _family_env(spec.get("params") or {}, sig)
        local.facts = _case_facts(spec, spec["cases"][j])
        errors += _shape_errors(node, local, f"type family {fam} case {j}", families)

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
    applied = {
        n.value.id
        for node in shapes.values()
        if node is not None
        for n in ast.walk(node)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
    }
    while more := set().union(*(family_refs[f] for f in applied & set(family_refs))) - applied:
        applied |= more
    errors += [
        f"type family {fam} is applied by no shape" for fam in sorted(set(sig.types) - applied)
    ]
    if errors:
        return sorted({f"{name}: {e}" for e in errors}), []
    rules_ok, plan_roofline = roofline_plan(sig, entry.get("roofline"), resolve=False)
    return _point_errors(EntryPlan(sig, None if rules_ok else plan_roofline), env)


def inference_errors(plan: EntryPlan, point: dict, b: PlanBranch) -> list[str]:
    """Why the inference plan fails at `point`, or nothing: an index a call reads that no input,
    parameter or `let` solves."""
    sig = plan.sig
    solvable = {n for n, k in sig.forall.items() if k != "Seq[Int]"}
    inputs = {t: node for t, node in b.shapes.items() if t in sig.call_tensors}
    known = set(sig.params) | {k for k in point if "(" not in k and "." not in k}
    known |= {
        d.id for t, d in b.dtypes.items() if t in inputs and isinstance(d, ast.Name)
    } & solvable
    known |= {s.name for s in unification(inputs, known, b.lets) if s.name}
    return [
        f"index {u!r} cannot be solved from the inputs"
        for u in sorted((b.called & solvable) - known)
    ]


def _read_errors(sig: Signature, point: dict, b: PlanBranch, env: KindEnv) -> list[str]:
    """Reads the branch forbids: `v.value` where `present(v)` is false, a field its constructor lacks."""
    nodes = {
        **{f"tensor {t!r} shape": n for t, n in b.shapes.items()},
        **{f"shape_rules[{i}]": n for i, n in enumerate(b.rules)},
        **{f"let {n}": fold(parse(e), point, f"let {n}") for n, e in sig.let.items()},
    }
    errors = []
    for where, node in nodes.items():
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
            kind = sig.kind(receiver.value.id if payload else v)
            kind = kind.payload() if kind is not None and payload else kind
            fields = (
                env.adts.get(kind.name, {}).get(ctor)
                if kind is not None and kind.tag == "ADT"
                else None
            )
            if ctor is not None and f != "kind" and fields is not None and f not in fields:
                errors.append(f"{where} reads {v}.{f}, which constructor {ctor!r} lacks")
    return errors


def _point_errors(plan: EntryPlan, env: KindEnv) -> tuple[list[str], list[str]]:
    """Check each group of dependent discriminants on its own, the other groups held accepted."""
    sig = plan.sig

    def accepted(point: dict) -> bool:
        try:
            return rejecting_rule(sig, complete_point(sig, point, strict=False)) is None
        except SignatureError:
            return False

    groups = [list(points(g)) for g in discriminant_groups(sig)]
    held = {}
    for group in groups:
        held.update(next((p for p in group if accepted(p)), group[0]))
    warnings = []
    combinations = sum(len(group) for group in groups)
    if combinations > DISCRIMINANT_LIMIT:
        warnings.append(
            f"{sig.name}: {combinations} discriminant combinations exceed {DISCRIMINANT_LIMIT}"
        )
    errors: list[str] = []
    for part in [p for group in groups for p in group] or [{}]:
        point = {**held, **part}
        try:
            point = complete_point(sig, point)
            ok = rejecting_rule(sig, point) is None
            for t in sig.call_tensors.values():
                holds(t.mutated if isinstance(t.mutated, str) else None, point)
            b = plan.branch(point, complete=ok)
            errors += _read_errors(sig, point, b, env)
            facts = {k[: -len(".kind")]: v for k, v in point.items() if k.endswith(".kind")}
            # An absent `Maybe` parameter is None on this branch.
            local = env.narrowed(
                {
                    n: NONE
                    for n, k in env.kinds.items()
                    if k is not None and k.tag == "Maybe" and point.get(f"present({n})") is False
                }
            )
            local.facts, local.point = facts, True
            families = dict.fromkeys(sig.types)
            for t, node in b.shapes.items():
                errors += _shape_errors(node, local, f"tensor {t!r} shape", families)
            for n, node in b.lets.items():
                errors += infer_kinds(node, local, f"let {n}")[1]
            for i, rule in enumerate(b.rules):
                errors += infer_kinds(rule, local, f"shape_rules[{i}]", BOOL)[1]
            if ok:
                errors += inference_errors(plan, point, b)
                errors += [
                    f"dtype_combos column {k!r} is not relevant at {point}"
                    for k in sorted(set().union(set(), *sig.dtype_combos) - b.relevant)
                ]
        except SignatureError as exc:
            errors.append(f"at {point}: {exc}")
    return sorted({f"{sig.name}: {e}" for e in errors}), warnings
