"""Synthesize ``eval_roofline`` bodies from manifest ``roofline`` entries.

The L1 ``Op`` base declares ``eval_roofline`` as a staged-rollout stub
that raises ``NotImplementedError``. Per ``docs/design/roofline.md`` §4.4,
every concrete op with ``status: implemented`` must override the stub
with a body derived from its manifest ``roofline`` block.

This module provides:

- ``synthesize_eval_roofline`` — emit an ``eval_roofline`` function from
  a manifest ``roofline`` block (inline or func mode).
- ``maybe_install_eval_roofline`` — ``Op.__init_subclass__`` hook that
  auto-applies the generated method when the subclass advertises the
  manifest metadata and does not supply its own override.

The two roofline modes follow ``docs/design/roofline.md`` §2.2 and §4.4.2:

- **Func** — ``roofline.func`` points at ``module.path.callable``. The
  emitted body is ``return <func>(self)``. Codegen resolves the dotted
  path at synthesis time so a typo fails class construction rather than
  the first benchmark call.
- **Inline** — ``roofline.vars`` (optional) + ``flops`` + ``bytes`` are
  Python expression strings. Codegen validates each expression's AST
  against the §4.4.4 namespace at synthesis time and emits a plain
  Python function body — generated ``eval_roofline`` does not parse,
  AST-analyze, or ``eval`` formula strings at call time (§4.4.6).

The L1 stub is preserved for ``status: spec-only`` entries — codegen
re-evaluates them once the status flips.
"""

from __future__ import annotations

import ast
import importlib
import math
from math import prod
from typing import Any, Callable


# Names bound into the vars-layer namespace per
# ``docs/design/roofline.md`` §4.4.4. Helper names map to their Python
# implementations; tensor/param/elem_bytes names are bound dynamically.
class _ShapeProxy:
    """Synthetic tensor stand-in exposing only ``shape`` and ``ndim``.

    Inline-mode roofline expressions reference ``<tensor>.shape`` and
    ``<tensor>.ndim``. Op classes are not required to retain the original
    tensor argument on ``self`` — many keep only derived state such as
    ``self.shape`` (the input shape tuple) or ``self.N_total`` (a flat
    element count). When the op does not store the tensor itself,
    ``_resolve_tensor_binding`` constructs a ``_ShapeProxy`` from the
    derived state so vars-layer expressions resolve uniformly.
    """

    __slots__ = ("shape", "ndim")

    def __init__(self, shape: tuple) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)


def _resolve_tensor_binding(op: Any, name: str, op_name: str) -> Any:
    """Bind ``name`` for inline-mode synthesis from op-instance state.

    Two accepted conventions, in order:

    1. ``self.<name>`` exposes ``.shape`` (a real tensor or any object
       exposing ``.shape`` / ``.ndim``).
    2. ``self.<name>_shape`` is a shape tuple/list; wrapped in a
       :class:`_ShapeProxy` for uniform ``.shape``/``.ndim`` access.

    Anything else raises :class:`ValueError` so the missing binding is
    surfaced with the op and input name rather than the vacuous
    ``'NoneType' object has no attribute 'shape'`` that would otherwise
    reach the caller from inside the generated body.

    Op-family-specific aliases (``self.shape`` / ``self.num_channels``
    / ``self.N_total``) are *not* consulted; ops opting into inline
    roofline declare bindings explicitly per ``docs/design/roofline.md``
    §4.4.3.
    """
    direct = getattr(op, name, None)
    if direct is not None and hasattr(direct, "shape"):
        return direct
    shape_attr = getattr(op, f"{name}_shape", None)
    if isinstance(shape_attr, (tuple, list)):
        return _ShapeProxy(tuple(shape_attr))
    raise ValueError(
        f"{op_name}: cannot resolve roofline input {name!r}; expected "
        f"either self.{name} (with .shape/.ndim) or self.{name}_shape "
        f"(shape tuple) on the op instance"
    )


_VARS_HELPERS: dict[str, Any] = {
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

# Arithmetic-layer helpers (a strict subset per §4.4.4).
_ARITHMETIC_HELPERS: dict[str, Any] = {
    "ceil": math.ceil,
    "floor": math.floor,
    "log2": math.log2,
}


def _resolve_func_path(path: str) -> Callable[..., Any]:
    """Resolve ``module.path.callable`` to a Python callable.

    Raises ``ValueError`` if the module or attribute is absent — codegen
    is the authoritative gate for ``func`` correctness
    (``docs/design/roofline.md`` §4.4).
    """
    if not isinstance(path, str) or "." not in path:
        raise ValueError(
            f"roofline.func must be a dotted module.attr path, got {path!r}"
        )
    mod_path, _, attr = path.rpartition(".")
    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"cannot resolve roofline.func {path!r}: import {mod_path!r} "
            f"failed ({exc})"
        ) from exc
    fn = getattr(mod, attr, None)
    if not callable(fn):
        raise ValueError(
            f"cannot resolve roofline.func {path!r}: {attr!r} is not a "
            f"callable on {mod_path!r}"
        )
    return fn


def _synthesize_func_mode(
    op_name: str, func_path: str,
) -> Callable[..., tuple[int, int]]:
    """Build an ``eval_roofline`` that delegates to a human-authored func.

    Codegen resolves the dotted path eagerly so the closure captures the
    callable directly — subsequent ``op.eval_roofline()`` calls skip the
    import machinery on the hot path. Per ``docs/design/roofline.md``
    §4.4.2, the emitted body is ``return <func>(self)``; if the author
    writes a non-``func(op)`` signature, the resulting TypeError surfaces
    to the caller as designed.
    """
    fn = _resolve_func_path(func_path)

    def eval_roofline(self):
        return fn(self)

    eval_roofline.__name__ = "eval_roofline"
    eval_roofline.__qualname__ = f"{op_name}.eval_roofline"
    eval_roofline.__doc__ = (
        f"Synthesized from manifest roofline.func={func_path!r}."
    )
    return eval_roofline


_VARS_FORBIDDEN_NODES = (
    ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await,
    ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef,
)
_VARS_ATTR_WHITELIST = frozenset({"shape", "ndim"})


class _VarsExprValidator(ast.NodeVisitor):
    """AST-check a vars-layer expression with scope-aware name resolution.

    Maintains a stack of name scopes. The outermost scope holds tensor
    bindings, params, ``elem_bytes``, earlier vars, and helper names;
    comprehensions (``ListComp`` / ``SetComp`` / ``DictComp`` /
    ``GeneratorExp``) push a child scope containing their generator
    target names so loop variables (``d`` in ``sum(d for d in x.shape)``)
    resolve cleanly without polluting the surrounding scope.
    """

    def __init__(self, op_name: str, var_name: str, allowed: set[str]) -> None:
        self.op_name = op_name
        self.var_name = var_name
        self._scopes: list[set[str]] = [set(allowed)]

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
        raise ValueError(
            f"{self.op_name}: roofline.vars[{self.var_name!r}] uses "
            f"unsupported comprehension target {type(target).__name__}"
        )

    def _visit_comp(self, node: ast.AST) -> None:
        # Walk the comprehension matching Python's scoping rules: each
        # generator's iterable is evaluated *before* its target is bound
        # (so ``sum(d for d in d)`` raises NameError at runtime in
        # Python — and must therefore fail validation here too). Bind
        # the target only after visiting its iterable, then conditions
        # and any subsequent generator iterables / body see the binding.
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
            raise ValueError(
                f"{self.op_name}: roofline.vars[{self.var_name!r}] "
                f"references unknown name {node.id!r}"
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr not in _VARS_ATTR_WHITELIST:
            raise ValueError(
                f"{self.op_name}: roofline.vars[{self.var_name!r}] "
                f"accesses non-whitelisted attribute {node.attr!r}; "
                f"vars-layer allows only "
                f"{sorted(_VARS_ATTR_WHITELIST)!r}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"{self.op_name}: roofline.vars[{self.var_name!r}] "
                f"performs a non-helper call (only whitelisted helper "
                f"names may be invoked)"
            )
        if node.func.id not in _VARS_HELPERS:
            raise ValueError(
                f"{self.op_name}: roofline.vars[{self.var_name!r}] "
                f"calls non-whitelisted name {node.func.id!r}; "
                f"vars-layer helpers are {sorted(_VARS_HELPERS)!r}"
            )
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _VARS_FORBIDDEN_NODES):
            raise ValueError(
                f"{self.op_name}: roofline.vars[{self.var_name!r}] uses "
                f"forbidden construct {type(node).__name__}"
            )
        super().generic_visit(node)


def _referenced_names(*exprs: str | None) -> set[str]:
    """Return ``Name`` ids referenced in any of the given expressions.

    Used by inline synthesis to scope locals to names the manifest
    actually reads — params declared on the op but unused by the
    roofline are not pulled in. Comprehension target names appear here
    too; that's harmless because the caller intersects with the known
    {inputs ∪ params} set, which never contains target names.
    """
    names: set[str] = set()
    for expr in exprs:
        if expr is None:
            continue
        for node in ast.walk(ast.parse(expr, mode="eval")):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _validate_vars_expr(
    op_name: str,
    var_name: str,
    expr: str,
    allowed_names: set[str],
) -> ast.Expression:
    """Parse and AST-check a vars-layer expression.

    Vars-layer permits ``.shape`` / ``.ndim`` access, small
    comprehensions, calls to whitelisted helpers, and references to
    bound names (tensors, params, ``elem_bytes``, earlier vars,
    helpers). Forbidden constructs raise ``ValueError`` so class
    construction fails before the manifest lands. Comprehension target
    names bind to a child scope so they are reachable only inside the
    comprehension's element/conditions/iterables.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{op_name}: roofline.vars[{var_name!r}] is not a valid Python "
            f"expression ({exc})"
        ) from exc
    _VarsExprValidator(op_name, var_name, allowed_names).visit(tree)
    return tree


# Positive allowlist for arithmetic-layer AST nodes. Anything else
# (tensor access, slicing, collection literals, comprehensions, lambdas,
# starred / walrus / etc.) is rejected at synthesis time. Listing what
# survives — rather than enumerating what to forbid — keeps the gate
# from drifting as new AST node kinds appear in future Python versions.
_ARITHMETIC_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.IfExp, ast.Compare,
    ast.Call,
    ast.Constant,
    ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.USub, ast.UAdd, ast.Invert, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _validate_arithmetic_expr(
    op_name: str,
    label: str,
    expr: str,
    allowed_names: set[str],
) -> ast.Expression:
    """Parse and AST-check an arithmetic-layer expression.

    Per ``docs/design/roofline.md`` §4.4.3, the arithmetic layer
    permits only numeric operations on resolved vars + ``elem_bytes``
    + ``ceil`` / ``floor`` / ``log2``. Any AST node outside
    ``_ARITHMETIC_ALLOWED_NODES`` fails synthesis.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{op_name}: roofline.{label} is not a valid Python "
            f"expression ({exc})"
        ) from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ARITHMETIC_ALLOWED_NODES):
            raise ValueError(
                f"{op_name}: roofline.{label} uses forbidden construct "
                f"{type(node).__name__} (arithmetic layer permits only "
                f"BinOp/UnaryOp/BoolOp/IfExp/Compare/constants/names "
                f"and calls to ceil/floor/log2)"
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(
                f"{op_name}: roofline.{label} references unknown name "
                f"{node.id!r}; allowed names are "
                f"{sorted(allowed_names)!r}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"{op_name}: roofline.{label} performs a non-helper "
                    f"call (only ceil/floor/log2 may be invoked)"
                )
            if node.func.id not in _ARITHMETIC_HELPERS:
                raise ValueError(
                    f"{op_name}: roofline.{label} calls non-arithmetic "
                    f"helper {node.func.id!r}; allowed callees are "
                    f"{sorted(_ARITHMETIC_HELPERS)!r}"
                )
    return tree


def _synthesize_inline_mode(
    op_name: str,
    roofline: dict[str, Any],
    signature: dict[str, Any] | None,
) -> Callable[..., tuple[int, int]]:
    """Build an ``eval_roofline`` from inline ``flops`` / ``bytes`` exprs.

    Synthesis-time steps (``docs/design/roofline.md`` §4.4.3/§4.4.4):

    1. Compute the legal name set for each layer from
       ``signature.inputs`` + ``signature.params`` + ``elem_bytes`` +
       the layer's helper table.
    2. AST-validate every ``vars`` expression against the vars-layer
       name set; each entry expands the name set for later entries.
    3. AST-validate ``flops`` and ``bytes`` against the arithmetic-layer
       name set (resolved vars + ``elem_bytes`` + ceil/floor/log2).
    4. Emit a plain Python function body whose locals are bound from
       ``self.<input>`` / ``self.<param>`` / ``self.dtype.itemsize``
       and whose vars/return statements are the original expression
       strings copied verbatim — no ``eval`` at call time.
    """
    flops_expr = roofline.get("flops")
    bytes_expr = roofline.get("bytes")
    if not isinstance(flops_expr, str) or not isinstance(bytes_expr, str):
        raise ValueError(
            f"{op_name}: inline-mode roofline must declare both "
            f"flops and bytes as strings"
        )
    vars_block = roofline.get("vars") or {}
    if not isinstance(vars_block, dict):
        raise ValueError(
            f"{op_name}: roofline.vars must be a mapping when present"
        )

    sig = signature or {}
    inputs = sig.get("inputs") or {}
    params = sig.get("params") or {}
    input_names = list(inputs.keys()) if isinstance(inputs, dict) else []
    param_names = list(params.keys()) if isinstance(params, dict) else []

    # Vars-layer legal name set: tensors + params + elem_bytes +
    # vars-layer helpers. Each successfully-parsed vars entry expands
    # the set so later entries may reference earlier locals.
    vars_allowed: set[str] = set(input_names) | set(param_names)
    vars_allowed.add("elem_bytes")
    vars_allowed.update(_VARS_HELPERS.keys())

    for name, expr in vars_block.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"{op_name}: roofline.vars key {name!r} is not a valid "
                f"Python identifier"
            )
        if not isinstance(expr, str):
            raise ValueError(
                f"{op_name}: roofline.vars[{name!r}] must be a string "
                f"expression"
            )
        _validate_vars_expr(op_name, name, expr, vars_allowed)
        vars_allowed.add(name)

    # Arithmetic-layer legal name set per §4.4.3 Block 2: "references
    # only Block 1 locals + ``elem_bytes`` + arithmetic-layer helpers".
    # Block 1 binds both vars and ``signature.params``, so params are
    # reachable here; tensor inputs are not — anything derived from a
    # tensor must surface through a vars entry.
    arith_allowed: set[str] = set(vars_block.keys())
    arith_allowed.update(param_names)
    arith_allowed.add("elem_bytes")
    arith_allowed.update(_ARITHMETIC_HELPERS.keys())
    _validate_arithmetic_expr(op_name, "flops", flops_expr, arith_allowed)
    _validate_arithmetic_expr(op_name, "bytes", bytes_expr, arith_allowed)

    # Emit a plain function body. Locals are bound from the op instance;
    # vars and the return statement are the manifest expression strings
    # copied verbatim — no parsing, no eval at call time (§4.4.6).
    src_lines: list[str] = [
        "def eval_roofline(self):",
        f'    """Synthesized from manifest inline roofline for {op_name}."""',
    ]
    # Only bind inputs / params that the roofline expressions actually
    # reference. Authoring a manifest entry that declares a param but
    # never reads it in the roofline (e.g. NanToNumFwdOp's
    # ``nan``/``posinf``/``neginf`` configure the kernel but do not
    # affect FLOPs / bytes) is legitimate; binding them anyway would
    # widen the contract to "expose every signature.params name on
    # every Op even when the roofline does not need it", which is more
    # than the design requires.
    referenced = _referenced_names(
        *vars_block.values(), flops_expr, bytes_expr,
    )
    for n in input_names:
        if n not in referenced:
            continue
        # Inputs bind from op-instance state via the resolver, which
        # accepts either ``self.<n>`` exposing ``.shape`` or
        # ``self.<n>_shape`` as a tuple. Anything else raises
        # ``ValueError`` at call time naming the missing convention.
        src_lines.append(
            f"    {n} = _resolve_tensor_binding(self, {n!r}, {op_name!r})"
        )
    for n in param_names:
        if n not in referenced:
            continue
        # ``signature.params`` referenced by the roofline must be
        # exposed as ``self.<n>``. A missing attribute surfaces an
        # ``AttributeError`` naming the op rather than a downstream
        # ``NameError`` deep in the body.
        src_lines.append(f"    {n} = self.{n}")
    src_lines.append("    elem_bytes = self.dtype.itemsize")
    for name, expr in vars_block.items():
        src_lines.append(f"    {name} = {expr}")
    src_lines.append(f"    _flops = {flops_expr}")
    src_lines.append(f"    _bytes = {bytes_expr}")
    src_lines.append("    return int(_flops), int(_bytes)")

    # The vars-layer helper namespace is exposed as module-level globals
    # of the generated function. The arithmetic-layer subset is included
    # by virtue of being a subset of the vars table.
    globs: dict[str, Any] = dict(_VARS_HELPERS)
    globs["_resolve_tensor_binding"] = _resolve_tensor_binding
    globs["__builtins__"] = {
        "int": int,
        "float": float,
        "bool": bool,
        "ValueError": ValueError,
    }

    src = "\n".join(src_lines)
    try:
        code = compile(src, f"<{op_name}.eval_roofline>", "exec")
    except SyntaxError as exc:  # pragma: no cover - validated above
        raise ValueError(
            f"{op_name}: synthesized eval_roofline body did not compile "
            f"({exc})"
        ) from exc
    local_ns: dict[str, Any] = {}
    exec(code, globs, local_ns)  # noqa: S102
    fn = local_ns["eval_roofline"]
    fn.__name__ = "eval_roofline"
    fn.__qualname__ = f"{op_name}.eval_roofline"
    return fn


def synthesize_eval_roofline(
    op_name: str,
    *,
    roofline: dict[str, Any] | None,
    signature: dict[str, Any] | None,
) -> Callable[..., tuple[int, int]]:
    """Build an ``eval_roofline`` function from a manifest roofline block.

    Args:
        op_name: Manifest op name; used in error messages and __qualname__.
        roofline: The ``roofline`` block from the manifest entry.
        signature: The ``signature`` block; consumed for inline-mode
            input/param bindings. May be ``None`` for func mode.

    Returns:
        A method-shaped callable ``eval_roofline(self) -> tuple[int, int]``.

    Raises:
        ValueError: When ``roofline`` is absent or malformed (missing
            both modes, mixing both modes, unresolvable func path,
            inline missing ``flops``/``bytes``, or an inline expression
            that fails the §4.4.3/§4.4.4 name-and-form gate).
    """
    if not isinstance(roofline, dict) or not roofline:
        raise ValueError(
            f"{op_name}: manifest roofline is missing or empty; cannot "
            f"synthesize eval_roofline"
        )
    has_func = "func" in roofline
    has_inline = "flops" in roofline or "bytes" in roofline or "vars" in roofline
    if has_func and has_inline:
        raise ValueError(
            f"{op_name}: roofline cannot mix func and inline modes"
        )
    if has_func:
        return _synthesize_func_mode(op_name, roofline["func"])
    return _synthesize_inline_mode(op_name, roofline, signature)


def _lookup_manifest_entry(op_name: str) -> dict[str, Any] | None:
    """Return the manifest entry for *op_name* or ``None`` if absent."""
    try:
        from tileops.manifest import load_manifest
    except Exception:  # noqa: BLE001
        return None
    try:
        ops = load_manifest()
    except Exception:  # noqa: BLE001
        return None
    entry = ops.get(op_name)
    if not isinstance(entry, dict):
        return None
    return entry


def maybe_install_eval_roofline(cls: type) -> None:
    """Install a synthesized ``eval_roofline`` on *cls* when warranted.

    Resolution order mirrors ``_dtype_codegen.maybe_install_validator``:

    1. Class-attached ``__manifest_roofline__`` + ``__manifest_status__``
       + ``__manifest_signature__`` (used by unit tests and bypass paths).
    2. Manifest entry whose key matches ``cls.__name__``.

    Conditions for installation:

    - Resolved status is ``"implemented"``.
    - No class in ``cls.__mro__`` other than the L1 ``Op`` base supplies
      its own ``eval_roofline`` (direct overrides on ``cls`` and inherited
      overrides on intermediate base classes such as ``UnaryOp`` are both
      honored verbatim).
    - The manifest roofline block parses successfully under
      ``synthesize_eval_roofline``.

    Synthesis failures are swallowed so an irregular manifest entry
    leaves the base stub in place rather than blocking class
    construction; the validator catches the resulting C7 error.
    """
    from tileops.ops.op_base import Op

    for base in cls.__mro__:
        if base is Op:
            break
        if "eval_roofline" in base.__dict__:
            # Manual override on cls or an intermediate base (e.g. UnaryOp,
            # SoftmaxBase) — preserve it.
            return

    roofline = getattr(cls, "__manifest_roofline__", None)
    sig = getattr(cls, "__manifest_signature__", None)
    status = getattr(cls, "__manifest_status__", None)
    if roofline is None or status is None:
        entry = _lookup_manifest_entry(cls.__name__)
        if entry is None:
            return
        roofline = entry.get("roofline")
        sig = entry.get("signature")
        status = entry.get("status")
    if status != "implemented":
        return
    if roofline is None:
        return
    try:
        fn = synthesize_eval_roofline(
            cls.__name__, roofline=roofline, signature=sig,
        )
    except ValueError:
        return
    cls.eval_roofline = fn  # type: ignore[assignment]
