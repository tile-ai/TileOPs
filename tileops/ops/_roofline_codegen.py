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


def _resolve_tensor_binding(
    op: Any, name: str, is_primary: bool,
) -> Any:
    """Bind ``name`` for inline-mode synthesis from op-instance state.

    Resolution order:

    1. ``self.<name>`` when it exposes ``.shape`` (a real tensor or
       previously-bound proxy).
    2. ``self.<name>_shape`` (explicit shape tuple convention) wrapped
       in a ``_ShapeProxy``.
    3. The primary input falls back to ``self.shape`` (canonical
       op-level shape attribute used by elementwise ops).
    4. A ``weight`` tensor falls back to ``self.num_channels`` → a 1-D
       proxy of length ``num_channels``.
    5. The primary input falls back to ``self.N_total`` → a 1-D proxy
       of length ``N_total`` (used by ops that flatten at construction).

    Returns ``None`` when no convention matches; vars expressions that
    dereference ``None.shape`` then surface ``AttributeError`` so the
    contract gap is visible rather than silently producing zero work.
    """
    direct = getattr(op, name, None)
    if direct is not None and hasattr(direct, "shape"):
        return direct
    shape_attr = getattr(op, f"{name}_shape", None)
    if isinstance(shape_attr, (tuple, list)):
        return _ShapeProxy(tuple(shape_attr))
    if is_primary:
        op_shape = getattr(op, "shape", None)
        if isinstance(op_shape, (tuple, list)):
            return _ShapeProxy(tuple(op_shape))
    if name == "weight":
        n_ch = getattr(op, "num_channels", None)
        if isinstance(n_ch, int):
            return _ShapeProxy((n_ch,))
    if is_primary:
        n_total = getattr(op, "N_total", None)
        if isinstance(n_total, int):
            return _ShapeProxy((n_total,))
    return direct


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


def _validate_vars_expr(
    op_name: str,
    var_name: str,
    expr: str,
    allowed_names: set[str],
) -> ast.Expression:
    """Parse and AST-check a vars-layer expression.

    Vars-layer permits tensor shape access (``Attribute`` + ``Subscript``),
    small comprehensions, calls to whitelisted helpers, and references
    to bound names (tensors, params, ``elem_bytes``, earlier vars,
    helpers). Forbidden constructs (``Lambda``, ``Assign``, dunder
    access, calls to unknown callees, references to unknown names)
    raise ``ValueError`` so class construction fails before the manifest
    lands.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{op_name}: roofline.vars[{var_name!r}] is not a valid Python "
            f"expression ({exc})"
        ) from exc

    # Walk the AST. ``Call`` nodes must target a whitelisted helper name
    # (no ``__import__``, no ``os.system``, no method calls on tensors).
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.NamedExpr, ast.Yield,
                             ast.YieldFrom, ast.Await,
                             ast.AsyncFunctionDef, ast.FunctionDef,
                             ast.ClassDef)):
            raise ValueError(
                f"{op_name}: roofline.vars[{var_name!r}] uses forbidden "
                f"construct {type(node).__name__}"
            )
        # Allow only ``<name>.shape``, ``<name>.ndim`` chains; reject
        # dunder probes that could pivot into arbitrary state.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(
                f"{op_name}: roofline.vars[{var_name!r}] references "
                f"forbidden dunder attribute {node.attr!r}"
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(
                f"{op_name}: roofline.vars[{var_name!r}] references "
                f"unknown name {node.id!r}; allowed names are "
                f"{sorted(allowed_names)!r}"
            )
        if isinstance(node, ast.Call):
            # Callees must be plain Names resolving to a helper. Reject
            # ``obj.method(...)`` and chained calls; tensors expose data
            # via attribute/subscript, never via method calls in vars.
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"{op_name}: roofline.vars[{var_name!r}] performs a "
                    f"non-helper call (only whitelisted helper names may "
                    f"be invoked)"
                )
            if node.func.id not in _VARS_HELPERS:
                raise ValueError(
                    f"{op_name}: roofline.vars[{var_name!r}] calls "
                    f"non-whitelisted name {node.func.id!r}; vars-layer "
                    f"helpers are {sorted(_VARS_HELPERS)!r}"
                )
    return tree


def _validate_arithmetic_expr(
    op_name: str,
    label: str,
    expr: str,
    allowed_names: set[str],
) -> ast.Expression:
    """Parse and AST-check an arithmetic-layer expression.

    Per ``docs/design/roofline.md`` §4.4.3, the arithmetic layer
    forbids tensor access, shape slicing, comprehensions, attributes,
    and arbitrary calls. Only ``BinOp``, ``UnaryOp``, ``IfExp``,
    ``Compare``, ``BoolOp``, constants, names from *allowed_names*, and
    calls to ``ceil`` / ``floor`` / ``log2`` survive.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{op_name}: roofline.{label} is not a valid Python "
            f"expression ({exc})"
        ) from exc

    forbidden_nodes = (
        ast.Attribute,
        ast.Subscript,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Lambda,
        ast.NamedExpr,
        ast.Starred,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(
                f"{op_name}: roofline.{label} uses forbidden construct "
                f"{type(node).__name__} (arithmetic layer permits only "
                f"BinOp/UnaryOp/IfExp/Compare/BoolOp/constants/names "
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
    for idx, n in enumerate(input_names):
        # Inputs bind from op-instance state via a tensor resolver that
        # accepts either the real tensor or a shape-bearing proxy derived
        # from canonical op attributes (``self.shape``, ``self.N_total``,
        # ``self.num_channels``). The first input is treated as primary
        # so derived shape fallbacks apply.
        is_primary = "True" if idx == 0 else "False"
        src_lines.append(
            f"    {n} = _resolve_tensor_binding(self, {n!r}, {is_primary})"
        )
    for n in param_names:
        # Params may or may not be stored on the op (some are stored at
        # ``__init__`` time, others arrive via forward args). Bind when
        # present so vars expressions can reference them; otherwise let
        # a NameError surface to the caller.
        src_lines.append(
            f"    if hasattr(self, {n!r}): {n} = getattr(self, {n!r})"
        )
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
        "getattr": getattr,
        "hasattr": hasattr,
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
