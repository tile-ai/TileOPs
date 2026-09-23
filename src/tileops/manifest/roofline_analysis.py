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

# A name that parses as an identifier but cannot be assigned to. Interpolating
# one into the generated body yields a SyntaxError rather than a diagnostic.
_KEYWORDS = frozenset(__import__("keyword").kwlist) | frozenset(__import__("keyword").softkwlist)


# Predicates the manifest schema level already rules on. The analysis still
# judges them -- it needs the answers to decide whether a plan can be built --
# but a consumer that runs alongside that level renders only what it alone
# owns. Partitioning by owner is what keeps one defect to one line; matching
# message text, or collapsing by code, would either miss a rewording or discard
# two real defects that happen to share a kind.
SCHEMA_OWNED_CODES = frozenset(
    {
        "inline.missing-expressions",
        "roofline.mixed-modes",
        "vars.not-a-mapping",
        "vars.key",
        "vars.not-a-string",
        "func.path",
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
    and returns the two expressions. It reads no signature and re-parses
    nothing: a plan that left any of that to be rediscovered would put analysis
    back into emission.
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
    else. Unlike a gate that raises, this keeps walking after a defect, so two
    unknown names in one expression yield two diagnostics with different
    subjects.
    """

    def __init__(
        self,
        pass_: _Pass,
        path: str,
        allowed: set[str],
        input_names: set[str],
        optional_names: set[str],
    ) -> None:
        self._pass = pass_
        self._path = path
        self._optional_names = set(optional_names)
        self._input_names = set(input_names)
        self._scopes: list[set[str]] = [set(allowed)]

    def _report(self, code: str, subject: str, message: str) -> None:
        self._pass.report(code, self._path, subject, message)

    def _is_bound(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def _collect_targets(self, target: ast.AST, scope: set[str]) -> None:
        if isinstance(target, ast.Name):
            scope.add(target.id)
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
        self._scopes.append(set())
        try:
            for gen in node.generators:  # type: ignore[attr-defined]
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
        if not self._is_bound(node.id):
            self._report(
                "vars.unknown-name",
                node.id,
                f"roofline.{self._path} references unknown name {node.id!r}",
            )
            return
        if node.id in self._input_names:
            self._report(
                "vars.tensor-as-value",
                node.id,
                f"roofline.{self._path} references tensor input {node.id!r} as a "
                f"bare value; access shape metadata via {node.id}.shape / "
                f"{node.id}.ndim instead",
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        base = node.value
        if isinstance(base, ast.Name) and base.id in self._optional_names:
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
        # ``.shape`` / ``.ndim`` are valid only taken directly off a declared
        # tensor input. Chained access, a subscripted operand and a local name
        # all reject here rather than producing a body that dies at call time.
        if not isinstance(node.value, ast.Name) or node.value.id not in self._input_names:
            self._report(
                "vars.attribute-operand",
                node.attr,
                f"roofline.{self._path} accesses .{node.attr} on a non-tensor-input "
                f"operand; .shape / .ndim are valid only directly on a declared "
                f"signature.inputs name",
            )
            return
        if not self._is_bound(node.value.id):
            self._report(
                "vars.unknown-name",
                node.value.id,
                f"roofline.{self._path} references unknown name {node.value.id!r}",
            )

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            self._report(
                "vars.non-helper-call",
                "",
                f"roofline.{self._path} performs a non-helper call (only whitelisted "
                f"helper names may be invoked)",
            )
        elif node.func.id not in VARS_HELPERS:
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
        if isinstance(base, ast.Name) and base.id in self._optional_names:
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
        if isinstance(left, ast.Name) and left.id in self._optional_names:
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
) -> None:
    """Parse and walk one vars-layer expression, reporting what it carries."""
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
    _VarsExprWalker(pass_, path, allowed, input_names, optional_names).visit(tree)


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
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        pass_.report(
            f"{label}.syntax",
            f"roofline.{label}",
            "",
            f"roofline.{label} is not a valid Python expression ({exc})",
        )
        return

    path = f"roofline.{label}"
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
                for missing in unresolved:
                    pass_.defer(
                        missing,
                        f"whether {node.id!r} in roofline.{label} names something declared",
                    )
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
    if not isinstance(path, str) or "." not in path:
        pass_.report(
            "func.path",
            where,
            "",
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


def _referenced_names(*exprs: str | None) -> set[str]:
    """Names any of the given expressions reads.

    Analysis owns this: deciding which locals the body binds is a judgment, and
    leaving it to emission would put a second parse behind the boundary.
    """
    names: set[str] = set()
    for expr in exprs:
        if not isinstance(expr, str):
            continue
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _string_keys(pass_: _Pass, fact: Fact, where: str) -> tuple[list[str], bool]:
    """Declared names of one signature block, and whether all keys were usable.

    A key that is not a string, or is a keyword, cannot become a local in the
    generated body. Reporting it here is what keeps it from reaching a name set
    and failing later as a ``TypeError`` out of ``sorted``.
    """
    if fact.state is ABSENT:
        # Nothing declared is not a defect: an op with no params declares none.
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
        if not key.isidentifier() or key in _KEYWORDS:
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
        names.append(key)
    return names, clean


def _names_a_tensor(expr: str) -> bool:
    """Whether the expression reads ``.shape`` or ``.ndim`` off anything.

    Only a declared input carries those, so an expression that reads one needs
    ``signature.inputs`` to say which names are declared.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
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
) -> None:
    """Report each signature block that could not be read.

    Whether one of these stops emission is not a property of the defect: a
    formula that never says ``out_elem_bytes`` is complete without a readable
    ``outputs``, and the same malformed block would stop a formula that does.
    ``needed`` names the blocks this formula reaches for, and only those are
    blocking.
    """
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
    # Usable means read in full: a block with one unreadable key states an
    # incomplete set of names, which settles no more than an unreadable block.
    inputs_ok = inputs.usable and inputs_clean
    params_ok = params.usable and params_clean

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
    optional_names = {
        name
        for name, attrs in (inputs.value.items() if inputs.usable else ())
        if isinstance(name, str) and isinstance(attrs, dict) and attrs.get("optional") is True
    }

    # A missing fact leaves name legality unsettled: a name might have been
    # declared in the part that could not be read. Which fact matters differs
    # by layer -- the arithmetic layer never sees inputs -- so the two sets are
    # kept apart rather than merged into "the signature is unreadable".
    vars_unresolved: dict[str, str] = {}
    if not inputs_ok and inputs.state is not ABSENT:
        vars_unresolved["signature.inputs"] = "declared input names"
    if not params_ok and params.state is not ABSENT:
        vars_unresolved["signature.params"] = "declared param names"
    arith_unresolved: dict[str, str] = {}
    if not params_ok and params.state is not ABSENT:
        arith_unresolved["signature.params"] = "declared param names"

    vars_program: list[tuple[str, str]] = []
    for name, expr in vars_block.items():
        path = f"vars[{name!r}]"
        if not isinstance(name, str) or not name.isidentifier():
            pass_.report(
                "vars.key",
                f"roofline.{path}",
                repr(name),
                f"roofline.vars key {name!r} is not a valid Python identifier",
            )
            continue
        if name in _KEYWORDS:
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
            # Declared but unusable: still in scope, so a later entry naming it
            # does not draw a second, misleading "unknown name".
            vars_allowed.add(name)
            continue
        if name in vars_allowed:
            # The emitted body assigns ``<name> = <expr>``, which would shadow
            # the colliding binding for every later expression.
            collides_with_signature = name in input_name_set or name in set(param_names)
            pass_.report(
                "vars.collision",
                f"roofline.{path}",
                name,
                f"roofline.vars key {name!r} collides with an existing name "
                f"(input / param / helper / elem_bytes / earlier var)",
            )
            if not collides_with_signature:
                vars_program.append((name, expr))
            continue
        _analyse_vars_expr(pass_, path, expr, vars_allowed, input_name_set, optional_names)
        vars_allowed.add(name)
        vars_program.append((name, expr))

    for missing, what in vars_unresolved.items():
        pass_.defer(missing, f"vars-layer name legality, for want of {what}")

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

    # Emission binds what the formula reads and nothing else: declaring a param
    # the roofline never names is legitimate, and binding it anyway would
    # require every op to expose every param.
    wants_out_elem_bytes = "out_elem_bytes" in referenced

    # Which blocks this formula reaches for. One it never names may be
    # unreadable without stopping it: the defect is still reported, just not as
    # one that prevents emission. The same rule governs all three blocks --
    # singling one out is how a formula that reads nothing from a malformed
    # block still lost its evaluator.
    needed: set[str] = set()
    if wants_out_elem_bytes:
        needed.add("outputs")
    if any(_names_a_tensor(e) for _, e in vars_program):
        # Only a declared input carries ``.shape`` / ``.ndim``.
        needed.add("inputs")
    # A name nothing else accounts for was meant to be an input or a param. Which
    # of the two cannot be settled while either is unreadable, so an unreadable
    # one is needed.
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
    _report_malformed_signature(pass_, inputs, outputs, params, needed=frozenset(needed))

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

    if any(d.blocking for d in pass_.diagnostics) or not exprs_usable or not vars_usable:
        return None
    if wants_out_elem_bytes and out_name is None:
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

    if signature is not None and not isinstance(signature, dict):
        pass_.report(
            "signature.not-a-mapping",
            "signature",
            type(signature).__name__,
            f"manifest signature must be a mapping, got {type(signature).__name__}",
        )

    inputs = _mapping_fact(signature, "inputs")
    outputs = _mapping_fact(signature, "outputs")
    params = _mapping_fact(signature, "params")

    has_func = "func" in roofline
    has_inline = "flops" in roofline or "bytes" in roofline or "vars" in roofline
    if has_func and has_inline:
        pass_.report(
            "roofline.mixed-modes",
            "roofline",
            "",
            "roofline cannot mix func and inline modes",
        )
        return AnalysisResult(tuple(pass_.diagnostics), tuple(pass_.unjudged), None)

    plan: RooflinePlan | None
    if has_func:
        # Func mode reads no signature, so an unreadable block is a defect the
        # entry still carries but not one that stops this evaluator.
        _report_malformed_signature(pass_, inputs, outputs, params, needed=frozenset())
        fn = _resolve_func(pass_, roofline["func"])
        plan = (
            None
            if fn is None
            else RooflinePlan(
                op_name=op_name,
                mode="func",
                func=fn,
                func_path=roofline["func"],
            )
        )
    else:
        plan = _analyse_inline(pass_, op_name, roofline, inputs, outputs, params)

    if any(d.blocking for d in pass_.diagnostics):
        plan = None
    return AnalysisResult(tuple(pass_.diagnostics), tuple(pass_.unjudged), plan)
