"""Analyse an entry's ``roofline`` block once, for both validation and emission.

Two phases sit behind one entry point. :func:`analyze_roofline` reads the raw
block and returns every defect it can name plus, when emission is safe, a
:class:`RooflinePlan`. Code generation consumes the plan and decides nothing.

Three properties make that split hold, and each is a rule the module keeps:

Total.
    The input is whatever YAML produced -- a scalar, a list, a mapping with
    non-string keys. Nothing here raises on its input; a fact that cannot be
    read becomes a :class:`Fact` in the ``MALFORMED`` state and the rest of the
    block is still read.

Lossless.
    A fact records which of absent, malformed and valid it is. Collapsing the
    first two into "empty" is what makes a missing dependency indistinguishable
    from a satisfied one, and both :class:`Unjudged` and the decision to build a
    plan rest on telling them apart.

Accumulating.
    A defect does not stop the pass. Two expressions each wrong in their own way
    yield two diagnostics, and a judgment that cannot be reached for want of a
    fact yields an :class:`Unjudged` naming the fact rather than silence.

The module imports the standard library only: it is read by the validator, which
runs without torch, and by code generation, which ships.
"""

from __future__ import annotations

import ast
import importlib
import math
import unicodedata
from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable

# --------------------------------------------------------------------------
# Namespaces. The legal names of each layer belong to analysis, which decides
# legality; emission receives the helpers as the generated body's globals.
# --------------------------------------------------------------------------

VARS_HELPERS: dict[str, Any] = {
    "product": prod,
    "isinstance": isinstance,
    "len": len,
    "set": set,
    "tuple": tuple,
    "list": list,
    "range": range,
    "int": int,
    "float": float,
    "bool": bool,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
}

ARITHMETIC_HELPERS: dict[str, Any] = {
    "ceil": math.ceil,
    "floor": math.floor,
    "log2": math.log2,
}

VARS_ATTR_WHITELIST = frozenset({"shape", "ndim"})

_VARS_FORBIDDEN_NODES = (
    ast.Lambda,
    ast.NamedExpr,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.AsyncFunctionDef,
    ast.FunctionDef,
    ast.ClassDef,
)

# Listing what survives -- rather than what to forbid -- keeps the gate from
# drifting as new AST node kinds appear in future Python versions.
_ARITHMETIC_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.IfExp, ast.Compare,
    ast.Call, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.USub, ast.UAdd, ast.Invert, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)  # fmt: skip

# Reserved words only: a soft keyword (``match``, ``case``, ``type``, ``_``)
# binds a local.
_KEYWORDS = frozenset(__import__("keyword").kwlist)

# Names the generated body binds for itself. A declared name landing on one
# shadows it, or is shadowed by it.
EMITTER_NAMES = frozenset(
    {
        "self",
        "_flops",
        "_bytes",
        "elem_bytes",
        "out_elem_bytes",
        "_resolve_tensor_binding",
        "_output_dtype",
    }
)


def normalized(name: str) -> str:
    """The identifier Python will see.

    The parser normalizes identifiers to NFKC, so two names that differ as
    strings can be one name in the emitted body.
    """
    return unicodedata.normalize("NFKC", name)


# Predicates the manifest schema level rules on. The analysis judges them too,
# because it needs the answers to decide whether a plan can be built, and a
# consumer running alongside that level renders only what it alone owns.
SCHEMA_OWNED_CODES = frozenset(
    {
        "roofline.absent",
        "inline.missing-expressions",
        "roofline.mixed-modes",
        "vars.not-a-mapping",
        "vars.key-not-a-string",
        "vars.not-a-string",
        "vars.empty",
        "flops.empty",
        "bytes.empty",
        "func.not-a-string",
        # `_l0_signature` rules on the signature's shape.
        "signature.not-a-mapping",
        "signature.inputs.not-a-mapping",
        "signature.outputs.not-a-mapping",
        "signature.params.not-a-mapping",
        "signature.input-attributes",
        "signature.non-string-name",
    }
)


# --------------------------------------------------------------------------
# Facts, carrying where they came from
# --------------------------------------------------------------------------

ABSENT = "absent"
MALFORMED = "malformed"
VALID = "valid"


@dataclass(frozen=True)
class Fact:
    """One piece of the entry, with whether it could be read at all.

    ``value`` is meaningful only in the ``VALID`` state. In ``MALFORMED`` it
    keeps what was found, so a diagnostic can name it.
    """

    state: str
    value: Any = None

    @property
    def usable(self) -> bool:
        return self.state == VALID


# --------------------------------------------------------------------------
# What a pass produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    """One defect. ``(code, path, subject)`` is its identity.

    Two defects of one kind in one expression differ by ``subject`` -- the
    offending name or node -- so neither is lost when reports from separate
    producers are merged.
    """

    code: str
    path: str
    subject: str
    message: str
    blocking: bool = True

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.code, self.path, self.subject)


@dataclass(frozen=True)
class Unjudged:
    """A judgment not reached, and the fact whose absence prevented it."""

    missing: str
    judgment: str


@dataclass(frozen=True)
class Binding:
    """One local the generated body binds before the expressions run."""

    name: str
    kind: str  # "input" or "param"
    optional: bool = False


@dataclass(frozen=True)
class RooflinePlan:
    """Everything emission needs, with every decision already made.

    Emission walks ``bindings`` in order, binds ``elem_bytes`` and
    ``out_elem_bytes`` exactly when told to, assigns ``vars_program`` in order
    and returns the two expressions. It reads no signature and re-parses nothing.
    """

    op_name: str
    mode: str  # "inline" or "func"
    bindings: tuple[Binding, ...] = ()
    bind_elem_bytes: bool = False
    # The declared output whose dtype prices the write, or None when the
    # formula does not reference ``out_elem_bytes``.
    out_elem_bytes_output: str | None = None
    vars_program: tuple[tuple[str, str], ...] = ()
    flops_expr: str | None = None
    bytes_expr: str | None = None
    func: Callable[..., Any] | None = None
    # Kept alongside the resolved callable because the emitted docstring
    # states it.
    func_path: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    diagnostics: tuple[Diagnostic, ...] = ()
    unjudged: tuple[Unjudged, ...] = ()
    plan: RooflinePlan | None = None

    @property
    def blocking(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.blocking)


# --------------------------------------------------------------------------
# Reading the raw blocks
# --------------------------------------------------------------------------


def _mapping_fact(container: Any, key: str) -> Fact:
    """Read ``container[key]`` as a mapping, keeping why it is not one."""
    if not isinstance(container, dict):
        return Fact(ABSENT)
    if key not in container:
        return Fact(ABSENT)
    value = container[key]
    if value is None:
        return Fact(ABSENT)
    if not isinstance(value, dict):
        return Fact(MALFORMED, value)
    return Fact(VALID, value)


@dataclass
class _Pass:
    """Accumulator for one analysis. Nothing here raises on entry data."""

    op_name: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    unjudged: list[Unjudged] = field(default_factory=list)

    def report(
        self, code: str, path: str, subject: str, message: str, *, blocking: bool = True
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                code=code,
                path=path,
                subject=subject,
                message=f"{self.op_name}: {message}",
                blocking=blocking,
            )
        )

    def defer(self, missing: str, judgment: str) -> None:
        self.unjudged.append(Unjudged(missing=missing, judgment=judgment))


# --------------------------------------------------------------------------
# Expression analysis
# --------------------------------------------------------------------------


class _VarsExprWalker(ast.NodeVisitor):
    """Walk one vars-layer expression, collecting every defect it carries.

    Scope-aware: comprehensions push a child scope holding their generator
    targets, so a loop variable resolves inside the comprehension and nowhere
    else. The walk continues past a defect, so two unknown names in one
    expression are two diagnostics with different subjects.
    """

    def __init__(
        self,
        pass_: _Pass,
        path: str,
        allowed: set[str],
        input_names: set[str],
        optional_names: set[str],
        *,
        inputs_unreadable: bool = False,
    ) -> None:
        self._pass = pass_
        self._path = path
        self._optional_names = set(optional_names)
        self._input_names = set(input_names)
        # While the declared inputs cannot be read, which names are tensors is
        # not known, and the judgments that turn on it are left unjudged.
        self._inputs_unreadable = inputs_unreadable
        # Each scope maps a name to what it is. A comprehension target is a
        # local whatever an outer name of the same spelling is, so the kind
        # travels with the binding.
        outer: dict[str, str] = {}
        for name in allowed:
            outer[name] = "input" if name in self._input_names else "name"
        self._scopes: list[dict[str, str]] = [outer]

    def _report(self, code: str, subject: str, message: str) -> None:
        self._pass.report(code, self._path, subject, message)

    def _kind(self, name: str) -> str | None:
        """What this name is where it stands, innermost binding first."""
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _is_bound(self, name: str) -> bool:
        return self._kind(name) is not None

    def _collect_targets(self, target: ast.AST, scope: dict[str, str]) -> None:
        if isinstance(target, ast.Name):
            scope[target.id] = "local"
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_targets(elt, scope)
            return
        if isinstance(target, ast.Starred):
            self._collect_targets(target.value, scope)
            return
        self._report(
            "vars.comprehension-target",
            type(target).__name__,
            f"roofline.{self._path} uses unsupported comprehension target {type(target).__name__}",
        )

    def _visit_comp(self, node: ast.AST) -> None:
        # Python binds a generator's target after evaluating its iterable, so
        # ``sum(d for d in d)`` is a NameError at run time and must fail here.
        self._scopes.append({})
        try:
            for gen in node.generators:  # type: ignore[attr-defined]
                if getattr(gen, "is_async", 0):
                    # The emitted body is a plain function.
                    self._report(
                        "vars.async-comprehension",
                        "",
                        f"roofline.{self._path} uses an async comprehension, which the "
                        f"generated body cannot contain",
                    )
                self.visit(gen.iter)
                self._collect_targets(gen.target, self._scopes[-1])
                for cond in gen.ifs:
                    self.visit(cond)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)  # type: ignore[attr-defined]
        finally:
            self._scopes.pop()

    # ast.NodeVisitor dispatch hooks: names must match AST class names.
    visit_ListComp = _visit_comp  # noqa: N815
    visit_SetComp = _visit_comp  # noqa: N815
    visit_DictComp = _visit_comp  # noqa: N815
    visit_GeneratorExp = _visit_comp  # noqa: N815

    def visit_Name(self, node: ast.Name) -> None:
        kind = self._kind(node.id)
        if kind is None:
            if self._inputs_unreadable:
                return
            self._report(
                "vars.unknown-name",
                node.id,
                f"roofline.{self._path} references unknown name {node.id!r}",
            )
            return
        if kind == "input":
            self._report(
                "vars.tensor-as-value",
                node.id,
                f"roofline.{self._path} references tensor input {node.id!r} as a "
                f"bare value; access shape metadata via {node.id}.shape / "
                f"{node.id}.ndim instead",
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        base = node.value
        if (
            isinstance(base, ast.Name)
            and self._kind(base.id) == "input"
            and base.id in self._optional_names
        ):
            self._report(
                "vars.optional-read",
                base.id,
                f"roofline.{self._path} reads {base.id}.{node.attr} from an optional "
                f"input; the call may omit it, so only '{base.id} is None' / "
                f"'{base.id} is not None' is allowed here. A formula that needs its "
                f"shape uses roofline.func",
            )
            return
        if node.attr not in VARS_ATTR_WHITELIST:
            self._report(
                "vars.attribute",
                node.attr,
                f"roofline.{self._path} accesses non-whitelisted attribute "
                f"{node.attr!r}; vars-layer allows only {sorted(VARS_ATTR_WHITELIST)!r}",
            )
            return
        # ``.shape`` / ``.ndim`` are valid only directly off a declared tensor
        # input: not chained, not subscripted, not on a local.
        if not isinstance(node.value, ast.Name) or self._kind(node.value.id) != "input":
            if self._inputs_unreadable:
                return
            self._report(
                "vars.attribute-operand",
                node.attr,
                f"roofline.{self._path} accesses .{node.attr} on a non-tensor-input "
                f"operand; .shape / .ndim are valid only directly on a declared "
                f"signature.inputs name",
            )
            return

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            self._report(
                "vars.non-helper-call",
                "",
                f"roofline.{self._path} performs a non-helper call (only whitelisted "
                f"helper names may be invoked)",
            )
        # A comprehension target of the same spelling is not the helper.
        elif node.func.id not in VARS_HELPERS or self._kind(node.func.id) == "local":
            self._report(
                "vars.unknown-helper",
                node.func.id,
                f"roofline.{self._path} calls non-whitelisted name {node.func.id!r}; "
                f"vars-layer helpers are {sorted(VARS_HELPERS)!r}",
            )
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        base = node.value
        if (
            isinstance(base, ast.Name)
            and self._kind(base.id) == "input"
            and base.id in self._optional_names
        ):
            self._report(
                "vars.optional-subscript",
                base.id,
                f"roofline.{self._path} subscripts optional input {base.id!r}; the "
                f"call may omit it, so only '{base.id} is None' / "
                f"'{base.id} is not None' is allowed here",
            )
            return
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # ``X is None`` / ``X is not None`` is the one place an optional input's
        # bare name is legal; everything else is checked normally.
        skip = self._presence_operand(node)
        for child in ast.iter_child_nodes(node):
            if child is skip:
                continue
            self.visit(child)

    def _presence_operand(self, node: ast.Compare) -> ast.AST | None:
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            return None
        left, right = node.left, node.comparators[0]
        if not isinstance(right, ast.Constant) or right.value is not None:
            return None
        if (
            isinstance(left, ast.Name)
            and self._kind(left.id) == "input"
            and left.id in self._optional_names
        ):
            return left
        return None

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _VARS_FORBIDDEN_NODES):
            self._report(
                "vars.forbidden-construct",
                type(node).__name__,
                f"roofline.{self._path} uses forbidden construct {type(node).__name__}",
            )
            return
        super().generic_visit(node)


def _analyse_vars_expr(
    pass_: _Pass,
    path: str,
    expr: str,
    allowed: set[str],
    input_names: set[str],
    optional_names: set[str],
    *,
    inputs_unreadable: bool = False,
) -> None:
    """Parse and walk one vars-layer expression, reporting what it carries."""
    if not expr.strip():
        pass_.report("vars.empty", path, "", f"roofline.{path} is empty")
        return
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        pass_.report(
            "vars.syntax",
            path,
            "",
            f"roofline.{path} is not a valid Python expression ({exc})",
        )
        return
    except (RecursionError, MemoryError, ValueError) as exc:
        # Nesting deep enough to exhaust the parser is a defect in the entry.
        pass_.report(
            "vars.unparsable",
            path,
            type(exc).__name__,
            f"roofline.{path} could not be parsed ({type(exc).__name__}); the "
            f"expression is nested too deeply to analyse",
        )
        return
    try:
        _VarsExprWalker(
            pass_,
            path,
            allowed,
            input_names,
            optional_names,
            inputs_unreadable=inputs_unreadable,
        ).visit(tree)
    except RecursionError:
        pass_.report(
            "vars.unparsable",
            path,
            "RecursionError",
            f"roofline.{path} is nested too deeply to analyse",
        )


def _analyse_arithmetic_expr(
    pass_: _Pass,
    label: str,
    expr: str,
    allowed: set[str],
    unresolved: dict[str, str],
) -> None:
    """Parse and walk one arithmetic-layer expression.

    ``unresolved`` maps a missing fact to the reason its absence leaves name
    legality unsettled. While it is non-empty an unknown name draws an
    :class:`Unjudged` naming that fact rather than a false accusation, because
    the name might have been declared in the part that could not be read.
    """
    path = f"roofline.{label}"
    if not expr.strip():
        pass_.report(f"{label}.empty", path, "", f"roofline.{label} is empty")
        return
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        pass_.report(
            f"{label}.syntax",
            path,
            "",
            f"roofline.{label} is not a valid Python expression ({exc})",
        )
        return
    except (RecursionError, MemoryError, ValueError) as exc:
        pass_.report(
            f"{label}.unparsable",
            path,
            type(exc).__name__,
            f"roofline.{label} could not be parsed ({type(exc).__name__}); the "
            f"expression is nested too deeply to analyse",
        )
        return
    for node in ast.walk(tree):
        if not isinstance(node, _ARITHMETIC_ALLOWED_NODES):
            pass_.report(
                "arith.forbidden-construct",
                path,
                type(node).__name__,
                f"roofline.{label} uses forbidden construct {type(node).__name__} "
                f"(arithmetic layer permits only BinOp/UnaryOp/BoolOp/IfExp/Compare/"
                f"constants/names and calls to ceil/floor/log2)",
            )
            continue
        if isinstance(node, ast.Name) and node.id not in allowed:
            if unresolved:
                # The block that would settle this name is unreadable. One line
                # per block, raised where the plan is refused, rather than one
                # per name that could have come from it.
                continue
            pass_.report(
                "arith.unknown-name",
                path,
                node.id,
                f"roofline.{label} references unknown name {node.id!r}; allowed names "
                f"are {sorted(allowed)!r}",
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                pass_.report(
                    "arith.non-helper-call",
                    path,
                    "",
                    f"roofline.{label} performs a non-helper call (only "
                    f"ceil/floor/log2 may be invoked)",
                )
            elif node.func.id not in ARITHMETIC_HELPERS:
                pass_.report(
                    "arith.unknown-helper",
                    path,
                    node.func.id,
                    f"roofline.{label} calls non-arithmetic helper {node.func.id!r}; "
                    f"allowed callees are {sorted(ARITHMETIC_HELPERS)!r}",
                )


# --------------------------------------------------------------------------
# Mode analysis
# --------------------------------------------------------------------------


def _resolve_func(pass_: _Pass, path: Any) -> Callable[..., Any] | None:
    """Resolve ``module.path.callable``, reporting why it did not resolve."""
    where = "roofline.func"
    if not isinstance(path, str) or not path.strip():
        # The schema level rules on the field's type and emptiness.
        pass_.report(
            "func.not-a-string",
            where,
            type(path).__name__,
            f"roofline.func must be a dotted module.attr path, got {path!r}",
        )
        return None
    if "." not in path:
        pass_.report(
            "func.path",
            where,
            path,
            f"roofline.func must be a dotted module.attr path, got {path!r}",
        )
        return None
    mod_path, _, attr = path.rpartition(".")
    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:
        pass_.report(
            "func.import",
            where,
            mod_path,
            f"cannot resolve roofline.func {path!r}: import {mod_path!r} failed ({exc})",
        )
        return None
    try:
        fn = getattr(mod, attr, None)
    except Exception as exc:
        pass_.report(
            "func.attribute",
            where,
            attr,
            f"cannot resolve roofline.func {path!r}: reading {attr!r} on {mod_path!r} "
            f"raised {type(exc).__name__}: {exc}",
        )
        return None
    if not callable(fn):
        pass_.report(
            "func.not-callable",
            where,
            attr,
            f"cannot resolve roofline.func {path!r}: {attr!r} is not a callable on {mod_path!r}",
        )
        return None
    return fn


class _FreeNames(ast.NodeVisitor):
    """Names an expression reads from outside itself.

    A comprehension binds its targets in a scope of its own, so the ``x`` in
    ``len([1 for x in range(3)])`` is that comprehension's and not a declared
    input of the same name.
    """

    def __init__(self) -> None:
        self.free: set[str] = set()
        self._bound: list[set[str]] = []

    def _is_local(self, name: str) -> bool:
        return any(name in scope for scope in self._bound)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._is_local(node.id):
            self.free.add(node.id)

    def _targets(self, target: ast.AST, scope: set[str]) -> None:
        if isinstance(target, ast.Name):
            scope.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._targets(elt, scope)
        elif isinstance(target, ast.Starred):
            self._targets(target.value, scope)

    def _comp(self, node: ast.AST) -> None:
        self._bound.append(set())
        try:
            for gen in node.generators:  # type: ignore[attr-defined]
                self.visit(gen.iter)
                self._targets(gen.target, self._bound[-1])
                for cond in gen.ifs:
                    self.visit(cond)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)  # type: ignore[attr-defined]
        finally:
            self._bound.pop()

    # ast.NodeVisitor dispatch hooks: names must match AST class names.
    visit_ListComp = _comp  # noqa: N815
    visit_SetComp = _comp  # noqa: N815
    visit_DictComp = _comp  # noqa: N815
    visit_GeneratorExp = _comp  # noqa: N815


def _referenced_names(*exprs: str | None) -> set[str]:
    """Names any of the given expressions reads from outside itself.

    Analysis owns this: which locals the body binds is a judgment.
    """
    walker = _FreeNames()
    for expr in exprs:
        if not isinstance(expr, str):
            continue
        try:
            tree = ast.parse(expr, mode="eval")
        except (SyntaxError, RecursionError, MemoryError, ValueError):
            continue
        try:
            walker.visit(tree)
        except RecursionError:
            continue
    return walker.free


def _string_keys(pass_: _Pass, fact: Fact, where: str) -> tuple[list[str], bool]:
    """Declared names of one signature block, and whether all keys were usable.

    A key that is not a string, or is a keyword, cannot become a local in the
    generated body, and is reported rather than counted among the names.
    """
    if fact.state is ABSENT:
        # Nothing declared is not a defect.
        return [], True
    if not fact.usable:
        return [], False
    names: list[str] = []
    clean = True
    for key in fact.value:
        if not isinstance(key, str):
            pass_.report(
                "signature.non-string-name",
                where,
                repr(key),
                f"{where} declares {key!r}, which is not a name; a declared name binds "
                f"a local in the generated body and must be a string",
                blocking=False,
            )
            clean = False
            continue
        if not key.isidentifier() or normalized(key) in _KEYWORDS:
            pass_.report(
                "signature.unusable-name",
                where,
                key,
                f"{where} declares {key!r}, which cannot bind a local in the generated "
                f"body (not an identifier, or a Python keyword)",
                blocking=False,
            )
            clean = False
            continue
        if normalized(key) in EMITTER_NAMES:
            pass_.report(
                "signature.reserved-name",
                where,
                key,
                f"{where} declares {key!r}, which the generated body binds for itself; "
                f"a formula reading it would read that binding, not the declaration",
            )
            clean = False
            continue
        if normalized(key) in names:
            pass_.report(
                "signature.duplicate-name",
                where,
                key,
                f"{where} declares {key!r}, which Python reads as a name already "
                f"declared here; the second binding would shadow the first",
            )
            clean = False
            continue
        # The spelling Python will see. An attribute name normalizes the same
        # way, so `self.<name>` reads this name too.
        names.append(normalized(key))
    return names, clean


def _names_a_tensor(expr: str) -> bool:
    """Whether the expression reads ``.shape`` or ``.ndim`` off anything.

    Only a declared input carries those, so an expression that reads one needs
    ``signature.inputs`` to say which names are declared.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, RecursionError, MemoryError, ValueError):
        return False
    return any(
        isinstance(node, ast.Attribute) and node.attr in VARS_ATTR_WHITELIST
        for node in ast.walk(tree)
    )


def _report_malformed_signature(
    pass_: _Pass,
    inputs: Fact,
    outputs: Fact,
    params: Fact,
    *,
    needed: frozenset[str],
    signature: Any = None,
) -> None:
    """Report each signature block that could not be read.

    Whether one stops emission is not a property of the defect but of the
    formula: ``needed`` names the blocks this one reaches for, and only those
    are blocking.
    """
    if signature is not None and not isinstance(signature, dict):
        # The whole block settles nothing, so it is needed when any part is.
        pass_.report(
            "signature.not-a-mapping",
            "signature",
            type(signature).__name__,
            f"manifest signature must be a mapping, got {type(signature).__name__}",
            blocking=bool(needed),
        )
        return
    for block, fact in (("inputs", inputs), ("outputs", outputs), ("params", params)):
        if fact.state is not MALFORMED:
            continue
        pass_.report(
            f"signature.{block}.not-a-mapping",
            f"signature.{block}",
            type(fact.value).__name__,
            f"manifest signature.{block} must be a mapping, got {type(fact.value).__name__}",
            blocking=block in needed,
        )


def _analyse_inline(
    pass_: _Pass,
    op_name: str,
    roofline: dict[str, Any],
    inputs: Fact,
    outputs: Fact,
    params: Fact,
    signature: Any,
) -> RooflinePlan | None:
    """Judge an inline block and build its plan when every needed fact is there.

    Which facts are needed follows the formula, not the block: a formula that
    never says ``out_elem_bytes`` is complete without a readable ``outputs``.
    """
    flops_expr = roofline.get("flops")
    bytes_expr = roofline.get("bytes")
    exprs_usable = isinstance(flops_expr, str) and isinstance(bytes_expr, str)
    if not exprs_usable:
        pass_.report(
            "inline.missing-expressions",
            "roofline",
            "",
            "inline-mode roofline must declare both flops and bytes as strings",
        )

    raw_vars = roofline.get("vars")
    if raw_vars is not None and not isinstance(raw_vars, dict):
        pass_.report(
            "vars.not-a-mapping",
            "roofline.vars",
            type(raw_vars).__name__,
            "roofline.vars must be a mapping when present",
        )
        vars_block: dict[Any, Any] = {}
        vars_usable = False
    else:
        vars_block = raw_vars or {}
        vars_usable = True

    input_names, inputs_clean = _string_keys(pass_, inputs, "signature.inputs")
    param_names, params_clean = _string_keys(pass_, params, "signature.params")
    # Read in full, which an absent block is: declaring nothing is a complete
    # answer, while a block with one unreadable key states an incomplete one.
    inputs_ok = inputs.state is ABSENT or (inputs.usable and inputs_clean)
    params_ok = params.state is ABSENT or (params.usable and params_clean)

    # ``out_elem_bytes`` exists only for a single declared output: it resolves
    # that one output's dtype, and there is no answer for several.
    single_output = outputs.usable and len(outputs.value) == 1
    out_name: str | None = None
    if single_output:
        first = next(iter(outputs.value))
        out_name = first if isinstance(first, str) else None

    vars_allowed: set[str] = set(input_names) | set(param_names)
    vars_allowed.add("elem_bytes")
    if single_output:
        vars_allowed.add("out_elem_bytes")
    vars_allowed.update(VARS_HELPERS)

    input_name_set = set(input_names)
    # An input whose attributes could not be read states no optionality, so it
    # blocks emission only where the formula binds it.
    unreadable_attrs = {
        normalized(name)
        for name, attrs in (inputs.value.items() if inputs.usable else ())
        if isinstance(name, str) and not isinstance(attrs, dict)
    }
    for name in sorted(unreadable_attrs):
        pass_.report(
            "signature.input-attributes",
            f"signature.inputs[{name!r}]",
            name,
            f"signature.inputs[{name!r}] is not a mapping, so whether the input is "
            f"optional cannot be read",
            blocking=False,
        )
    optional_names = {
        normalized(name)
        for name, attrs in (inputs.value.items() if inputs.usable else ())
        if isinstance(name, str) and isinstance(attrs, dict) and attrs.get("optional") is True
    }

    # A missing fact leaves name legality unsettled: a name might have been
    # declared in the part that could not be read. Which fact matters differs
    # by layer -- the arithmetic layer never sees inputs -- so the two sets are
    # kept apart rather than merged into "the signature is unreadable".
    arith_unresolved: dict[str, str] = {}
    if not params_ok and params.state is not ABSENT:
        arith_unresolved["signature.params"] = "declared param names"

    vars_program: list[tuple[str, str]] = []
    for name, expr in vars_block.items():
        path = f"vars[{name!r}]"
        if not isinstance(name, str):
            pass_.report(
                "vars.key-not-a-string",
                f"roofline.{path}",
                repr(name),
                f"roofline.vars key {name!r} is not a valid Python identifier",
            )
            continue
        if not name.isidentifier():
            pass_.report(
                "vars.key-not-an-identifier",
                f"roofline.{path}",
                name,
                f"roofline.vars key {name!r} is not a valid Python identifier",
            )
            continue
        if normalized(name) in EMITTER_NAMES:
            pass_.report(
                "vars.key-reserved",
                f"roofline.{path}",
                name,
                f"roofline.vars key {name!r} is a name the generated body binds for "
                f"itself; the assignment would shadow it",
            )
            continue
        if normalized(name) in _KEYWORDS:
            pass_.report(
                "vars.key-keyword",
                f"roofline.{path}",
                name,
                f"roofline.vars key {name!r} is a Python keyword and cannot bind a local",
            )
            continue
        if not isinstance(expr, str):
            pass_.report(
                "vars.not-a-string",
                f"roofline.{path}",
                type(expr).__name__,
                f"roofline.vars[{name!r}] must be a string expression",
            )
            # Declared but unusable: still in scope, so a later entry naming
            # it does not also draw "unknown name".
            vars_allowed.add(normalized(name))
            continue
        if normalized(name) in vars_allowed:
            # The emitted body assigns ``<name> = <expr>``, shadowing the
            # colliding binding for every later expression.
            collides_with_signature = normalized(name) in input_name_set | set(param_names)
            pass_.report(
                "vars.collision",
                f"roofline.{path}",
                name,
                f"roofline.vars key {name!r} collides with an existing name "
                f"(input / param / helper / elem_bytes / earlier var)",
            )
            if not collides_with_signature:
                vars_program.append((normalized(name), expr))
            continue
        _analyse_vars_expr(
            pass_,
            path,
            expr,
            vars_allowed,
            input_name_set,
            optional_names,
            inputs_unreadable=not inputs_ok and inputs.state is not ABSENT,
        )
        vars_allowed.add(normalized(name))
        vars_program.append((normalized(name), expr))

    arith_allowed: set[str] = {name for name, _ in vars_program}
    arith_allowed.update(param_names)
    arith_allowed.add("elem_bytes")
    if single_output:
        arith_allowed.add("out_elem_bytes")
    arith_allowed.update(ARITHMETIC_HELPERS)

    if exprs_usable:
        _analyse_arithmetic_expr(pass_, "flops", flops_expr, arith_allowed, arith_unresolved)
        _analyse_arithmetic_expr(pass_, "bytes", bytes_expr, arith_allowed, arith_unresolved)

    referenced = _referenced_names(*(e for _, e in vars_program), flops_expr, bytes_expr)

    # Emission binds what the formula reads and nothing else: a param the
    # roofline never names is not bound, and need not be exposed.
    wants_out_elem_bytes = "out_elem_bytes" in referenced

    # One name, one source. The body binds inputs then params, so a name in
    # both binds twice, and a name shared with a helper binds over it. Declaring
    # such a name is allowed; reading it is not, which is where the binding
    # would be emitted.
    for name in sorted(referenced & set(input_names) & set(param_names)):
        pass_.report(
            "signature.name-in-two-blocks",
            "signature",
            name,
            f"{name!r} is declared as both an input and a param, and the formula "
            f"reads it; the emitted body would bind it twice and the second would win",
        )
    for name in sorted(referenced & (set(input_names) | set(param_names)) & set(VARS_HELPERS)):
        pass_.report(
            "signature.name-shadows-helper",
            "signature",
            name,
            f"{name!r} is declared in the signature and is also a vars-layer helper, "
            f"and the formula reads it; binding it would shadow the helper",
        )

    # Which blocks this formula reaches for. One it never names may be
    # unreadable without stopping it: reported, but not blocking. The same rule
    # governs all three.
    needed: set[str] = set()
    if wants_out_elem_bytes:
        needed.add("outputs")
    if any(_names_a_tensor(e) for _, e in vars_program):
        # Only a declared input carries ``.shape`` / ``.ndim``.
        needed.add("inputs")
    # A name nothing else accounts for is an input or a param, and which cannot
    # be settled while either is unreadable.
    accounted = (
        {n for n, _ in vars_program}
        | set(VARS_HELPERS)
        | {"elem_bytes", "out_elem_bytes"}
        | set(input_names)
        | set(param_names)
    )
    if referenced - accounted:
        if not inputs_ok:
            needed.add("inputs")
        if not params_ok:
            needed.add("params")
    _report_malformed_signature(
        pass_, inputs, outputs, params, needed=frozenset(needed), signature=signature
    )

    # A block the formula wanted and could not read is one unjudged line, raised
    # below where the plan is refused. Naming each judgment it cost would say
    # the same thing once per name.

    if not outputs.usable and wants_out_elem_bytes:
        pass_.defer(
            "signature.outputs",
            "which output prices the write for out_elem_bytes",
        )
    if wants_out_elem_bytes and single_output and out_name is None:
        pass_.report(
            "outputs.unusable-name",
            "signature.outputs",
            "",
            "the declared output has no usable name, so out_elem_bytes cannot resolve it",
        )

    # Every judgment is made here, whether or not an earlier one already refuses
    # the plan. Raising one at the point of refusal would lose it behind the
    # first refusal that happens to come first.
    #
    # A block is unreadable when it is not a mapping, and equally when a key in
    # it could not be read: both leave the declared names incomplete.
    unreadable_needed = {
        block
        for block, ok in (("inputs", inputs_ok), ("params", params_ok))
        if block in needed and not ok
    }
    for block in sorted(unreadable_needed):
        pass_.defer(
            f"signature.{block}",
            f"which names signature.{block} declares, which this formula reads",
        )
    bound_unreadable_attrs = unreadable_attrs & referenced
    if bound_unreadable_attrs:
        pass_.defer(
            "signature.inputs",
            f"whether {sorted(bound_unreadable_attrs)} are optional",
        )

    # The refusal reads what was judged; it judges nothing itself.
    if (
        unreadable_needed
        or bound_unreadable_attrs
        or any(d.blocking for d in pass_.diagnostics)
        or not exprs_usable
        or not vars_usable
        or (wants_out_elem_bytes and out_name is None)
    ):
        return None

    bindings = [
        Binding(name=n, kind="input", optional=n in optional_names)
        for n in input_names
        if n in referenced
    ]
    bindings += [Binding(name=n, kind="param") for n in param_names if n in referenced]

    return RooflinePlan(
        op_name=op_name,
        mode="inline",
        bindings=tuple(bindings),
        bind_elem_bytes="elem_bytes" in referenced,
        out_elem_bytes_output=out_name if wants_out_elem_bytes else None,
        vars_program=tuple(vars_program),
        flops_expr=flops_expr,
        bytes_expr=bytes_expr,
    )


def analyze_roofline(
    op_name: str,
    *,
    roofline: Any,
    signature: Any = None,
) -> AnalysisResult:
    """Read one ``roofline`` block and say what is wrong with it, and whether it can emit.

    Total over whatever YAML produced. Every defect it can name is a
    :class:`Diagnostic`; every judgment it could not reach is an
    :class:`Unjudged` naming the fact it wanted. ``plan`` is non-None only when
    emission is safe.
    """
    pass_ = _Pass(op_name)

    if not isinstance(roofline, dict) or not roofline:
        pass_.report(
            "roofline.absent",
            "roofline",
            "",
            "manifest roofline is missing or empty; cannot synthesize eval_roofline",
        )
        return AnalysisResult(tuple(pass_.diagnostics), tuple(pass_.unjudged), None)

    inputs = _mapping_fact(signature, "inputs")
    outputs = _mapping_fact(signature, "outputs")
    params = _mapping_fact(signature, "params")

    has_func = "func" in roofline
    has_inline = "flops" in roofline or "bytes" in roofline or "vars" in roofline
    mixed = has_func and has_inline
    if mixed:
        pass_.report(
            "roofline.mixed-modes",
            "roofline",
            "",
            "roofline cannot mix func and inline modes",
        )

    plan: RooflinePlan | None = None
    # Both halves are judged when both are present. Stopping at the mode
    # verdict would leave whichever half is also wrong unreported, which is the
    # suppression this boundary exists to remove.
    if has_func:
        if not has_inline:
            # Func mode reads no signature.
            _report_malformed_signature(
                pass_, inputs, outputs, params, needed=frozenset(), signature=signature
            )
        fn = _resolve_func(pass_, roofline["func"])
        if fn is not None and not mixed:
            plan = RooflinePlan(
                op_name=op_name,
                mode="func",
                func=fn,
                func_path=roofline["func"],
            )
    if has_inline or not has_func:
        inline_plan = _analyse_inline(pass_, op_name, roofline, inputs, outputs, params, signature)
        if not mixed:
            plan = inline_plan

    if any(d.blocking for d in pass_.diagnostics):
        plan = None
    return AnalysisResult(tuple(pass_.diagnostics), tuple(pass_.unjudged), plan)
