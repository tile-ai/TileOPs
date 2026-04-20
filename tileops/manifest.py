"""Programmatic access to ops_manifest.yaml.

Provides two core functions:
- load_workloads(op_name) — returns workload dicts for an op
- eval_roofline(op_name, **variables) — evaluates flops/bytes expressions
"""

from __future__ import annotations

import ast
import functools
import importlib
import math
import operator
from importlib import resources
from math import prod as _math_prod
from typing import Any

import yaml

# Load ops_manifest.yaml from package data via importlib.resources.
_MANIFEST_REF = resources.files("tileops").joinpath("ops_manifest.yaml")

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (AST-based)
# ---------------------------------------------------------------------------

_BINOP_MAP: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARYOP_MAP: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _safe_eval(expr: str, variables: dict[str, float]) -> float:
    """Evaluate a simple arithmetic expression safely using AST walking.

    Only allows: numeric literals, variable lookups, binary arithmetic
    (+, -, *, /, //, **, %), unary +/-, and whitelisted function calls
    (log2, ceil, floor).
    """
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree.body, variables))


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float | int:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"Unknown variable: {node.id!r}")

    if isinstance(node, ast.BinOp):
        op_func = _BINOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARYOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand, variables))

    if isinstance(node, ast.Call):
        # Only allow simple name calls (e.g., log2(...)), not attribute access
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"Only simple function calls allowed, got: {ast.dump(node.func)}"
            )
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name!r}")
        if node.keywords:
            raise ValueError("Keyword arguments not allowed in safe eval")
        args = [_eval_node(arg, variables) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    """Load and cache the manifest. Called once per process."""
    text = _MANIFEST_REF.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data["ops"]


def load_workloads(op_name: str) -> list[dict[str, Any]]:
    """Return the workloads list for *op_name*.

    *op_name* must be the canonical PascalCase manifest key
    (e.g. ``RMSNormFwdOp``).

    >>> workloads = load_workloads("RMSNormFwdOp")
    >>> workloads[0]
    {'x_shape': [2048, 4096], 'dtypes': ['float16', 'bfloat16'], 'label': 'llama-3.1-8b-prefill'}
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")
    return ops[op_name]["workloads"]


def eval_roofline(op_name: str, **variables: float) -> tuple[float, float]:
    """Evaluate roofline expressions for *op_name* with given variable bindings.

    Returns (flops, bytes).

    >>> flops, mem_bytes = eval_roofline("RMSNormFwdOp", M=2048, N=4096, elem_bytes=2)
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")

    roofline = ops[op_name]["roofline"]

    if "func" in roofline:
        func_ref = roofline["func"]
        module_path, func_name = func_ref.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"Failed to import roofline module for '{op_name}': {module_path}"
            ) from exc
        fn = getattr(mod, func_name, None)
        if fn is None:
            raise ValueError(
                f"Roofline function '{func_name}' not found in module '{module_path}' "
                f"(op={op_name})"
            )
        result = fn(**variables)
        return float(result["flops"]), float(result["bytes"])

    flops_expr = roofline["flops"]
    bytes_expr = roofline["bytes"]

    try:
        flops = float(_safe_eval(flops_expr, variables))
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate roofline flops for '{op_name}': "
            f"{flops_expr!r} with {variables}"
        ) from exc

    try:
        mem_bytes = float(_safe_eval(bytes_expr, variables))
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate roofline bytes for '{op_name}': "
            f"{bytes_expr!r} with {variables}"
        ) from exc

    return flops, mem_bytes


# ---------------------------------------------------------------------------
# roofline.vars resolver
# ---------------------------------------------------------------------------


def _product(iterable: Any) -> int:
    """Manifest-facing ``product`` builtin.

    Accepts any iterable of numbers and returns their product. Empty iterable
    returns 1 (matches :func:`math.prod` / the convention used in manifest
    expressions such as ``product(x.shape[:-1])`` when ``x.ndim == 1``).
    """
    return _math_prod(iterable)


# Builtins exposed to roofline.vars expressions. Kept small and explicit so
# the evaluation context is auditable and cannot reach arbitrary globals.
_ROOFLINE_VARS_BUILTINS: dict[str, Any] = {
    "product": _product,
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


class _ShapeProxy:
    """Read-only proxy over a tensor shape.

    Exposes only ``__getitem__`` (indexing / slicing), ``__len__``, and
    ``__iter__``. In addition, attribute access is narrowly restricted to
    two names — ``shape`` (returns ``self``) and ``ndim`` (returns
    ``len(self)``) — so that existing manifest expressions using the
    ``x.shape[i]`` / ``x.ndim`` idiom keep working without opening a door
    to arbitrary object-graph traversal.

    All other attribute access (including every dunder) raises
    :class:`AttributeError`, and the AST walker rejects ``Attribute`` nodes
    whose attr name is not in this narrow whitelist.
    """

    __slots__ = ("_shape",)

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = tuple(shape)

    def __getitem__(self, key: Any) -> Any:
        result = self._shape[key]
        if isinstance(result, tuple):
            return _ShapeProxy(result)
        return result

    def __len__(self) -> int:
        return len(self._shape)

    def __iter__(self):
        return iter(self._shape)

    def __repr__(self) -> str:
        return f"_ShapeProxy({self._shape!r})"

    # Narrow attribute whitelist: ``.shape`` returns this proxy (so
    # ``x.shape[-1]`` works), ``.ndim`` returns ``len``. Any other attr
    # (including dunders like ``__class__``) raises.
    def __getattr__(self, name: str) -> Any:
        if name == "shape":
            return self
        if name == "ndim":
            return len(self._shape)
        raise AttributeError(name)


# AST node whitelist for the roofline.vars evaluator. Everything outside
# this set is rejected up-front so new Python releases cannot silently
# expand the attack surface.
_COMPARE_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _check_no_dunder(name: str) -> None:
    """Reject any identifier that begins or ends with a double underscore.

    Blocks escapes via ``__class__``, ``__builtins__``, ``__subclasses__``,
    ``__import__`` and friends at the earliest point in the walker.
    """
    if name.startswith("__") or name.endswith("__"):
        raise ValueError(
            f"disallowed dunder identifier in roofline.vars expression: {name!r}"
        )


def _eval_roofline_expr(expr: str, namespace: dict[str, Any]) -> Any:
    """Evaluate a roofline.vars expression using an AST-walking evaluator.

    This is the sandboxed replacement for :func:`eval`: every AST node type
    is checked against an explicit whitelist and every identifier is
    checked against :func:`_check_no_dunder`. The evaluator never looks at
    builtins, never resolves arbitrary attributes, and never calls anything
    that was not registered via ``namespace``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid syntax: {expr!r}") from exc
    return _walk(tree.body, namespace)


def _walk(node: ast.AST, ns: dict[str, Any]) -> Any:  # noqa: C901 - explicit dispatch
    # Literals.
    if isinstance(node, ast.Constant):
        return node.value

    # Name lookup — reject dunders and unknown names.
    if isinstance(node, ast.Name):
        _check_no_dunder(node.id)
        if node.id in ns:
            return ns[node.id]
        raise ValueError(f"unknown name in roofline.vars expression: {node.id!r}")

    # Arithmetic.
    if isinstance(node, ast.BinOp):
        op_func = _BINOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(
                f"unsupported binary operator: {type(node.op).__name__}"
            )
        return op_func(_walk(node.left, ns), _walk(node.right, ns))

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARYOP_MAP.get(type(node.op))
        if op_func is None:
            # Allow `not` in addition to the arithmetic unary operators.
            if isinstance(node.op, ast.Not):
                return not _walk(node.operand, ns)
            raise ValueError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        return op_func(_walk(node.operand, ns))

    # Boolean / comparison.
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _walk(v, ns)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = _walk(v, ns)
                if result:
                    return result
            return result
        raise ValueError(f"unsupported bool op: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _walk(node.left, ns)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            cmp_func = _COMPARE_OPS.get(type(op))
            if cmp_func is None:
                raise ValueError(
                    f"unsupported comparison op: {type(op).__name__}"
                )
            right = _walk(right_node, ns)
            if not cmp_func(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        if _walk(node.test, ns):
            return _walk(node.body, ns)
        return _walk(node.orelse, ns)

    # Containers.
    if isinstance(node, ast.Tuple):
        return tuple(_walk(e, ns) for e in node.elts)
    if isinstance(node, ast.List):
        return [_walk(e, ns) for e in node.elts]
    if isinstance(node, ast.Set):
        return {_walk(e, ns) for e in node.elts}

    # Subscript / slice.
    if isinstance(node, ast.Subscript):
        value = _walk(node.value, ns)
        key = _walk_slice(node.slice, ns)
        return value[key]

    # Attribute: narrow whitelist — only ``.shape`` and ``.ndim`` are
    # permitted, exclusively to support the legacy ``x.shape[i]`` idiom
    # on :class:`_ShapeProxy` bindings. All dunders are rejected.
    if isinstance(node, ast.Attribute):
        _check_no_dunder(node.attr)
        if node.attr not in ("shape", "ndim"):
            raise ValueError(
                f"attribute access not allowed: .{node.attr}"
            )
        value = _walk(node.value, ns)
        if not isinstance(value, _ShapeProxy):
            raise ValueError(
                f"attribute .{node.attr} is only allowed on shape proxies"
            )
        return getattr(value, node.attr)

    # Calls — only bare-Name callables registered in ``ns``.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "only direct calls to whitelisted functions are allowed"
            )
        _check_no_dunder(node.func.id)
        if node.func.id not in ns:
            raise ValueError(f"unknown function: {node.func.id!r}")
        func = ns[node.func.id]
        if func not in _ROOFLINE_VARS_BUILTINS.values():
            raise ValueError(
                f"call to non-whitelisted callable: {node.func.id!r}"
            )
        if node.keywords:
            raise ValueError(
                "keyword arguments are not allowed in roofline.vars calls"
            )
        args = [_walk(a, ns) for a in node.args]
        return func(*args)

    # Comprehensions.
    if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
        return _run_comp(node, ns)
    if isinstance(node, ast.DictComp):
        return _run_dict_comp(node, ns)

    raise ValueError(
        f"unsupported AST node in roofline.vars expression: {type(node).__name__}"
    )


def _walk_slice(node: ast.AST, ns: dict[str, Any]) -> Any:
    if isinstance(node, ast.Slice):
        lower = _walk(node.lower, ns) if node.lower is not None else None
        upper = _walk(node.upper, ns) if node.upper is not None else None
        step = _walk(node.step, ns) if node.step is not None else None
        return slice(lower, upper, step)
    return _walk(node, ns)


def _run_comp(
    node: ast.GeneratorExp | ast.ListComp | ast.SetComp,
    ns: dict[str, Any],
) -> Any:
    """Evaluate generator / list / set comprehensions in a child namespace."""
    collected: list[Any] = []

    def _recurse(gens: list[ast.comprehension], scope: dict[str, Any]) -> None:
        if not gens:
            collected.append(_walk(node.elt, scope))
            return
        gen = gens[0]
        iterable = _walk(gen.iter, scope)
        for item in iterable:
            bound = dict(scope)
            _bind_target(gen.target, item, bound)
            if all(_walk(cond, bound) for cond in gen.ifs):
                _recurse(gens[1:], bound)

    _recurse(list(node.generators), dict(ns))
    if isinstance(node, ast.ListComp):
        return list(collected)
    if isinstance(node, ast.SetComp):
        return set(collected)
    # GeneratorExp — materialize eagerly to an iterator so callers like
    # ``product(...)`` consume a ready sequence without re-entering the
    # evaluator after namespace teardown.
    return iter(collected)


def _run_dict_comp(node: ast.DictComp, ns: dict[str, Any]) -> dict[Any, Any]:
    result: dict[Any, Any] = {}

    def _recurse(gens: list[ast.comprehension], scope: dict[str, Any]) -> None:
        if not gens:
            result[_walk(node.key, scope)] = _walk(node.value, scope)
            return
        gen = gens[0]
        iterable = _walk(gen.iter, scope)
        for item in iterable:
            bound = dict(scope)
            _bind_target(gen.target, item, bound)
            if all(_walk(cond, bound) for cond in gen.ifs):
                _recurse(gens[1:], bound)

    _recurse(list(node.generators), dict(ns))
    return result


def _bind_target(target: ast.AST, value: Any, scope: dict[str, Any]) -> None:
    """Bind a comprehension target (Name or Tuple/List of Names)."""
    if isinstance(target, ast.Name):
        _check_no_dunder(target.id)
        scope[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        values = list(value)
        if len(values) != len(target.elts):
            raise ValueError("comprehension tuple-unpack arity mismatch")
        for sub, v in zip(target.elts, values, strict=True):
            _bind_target(sub, v, scope)
    else:
        raise ValueError(
            f"unsupported comprehension target: {type(target).__name__}"
        )


def has_roofline_vars(op_name: str) -> bool:
    """Return True iff the manifest declares a non-empty ``roofline.vars``
    mapping for *op_name*.

    Returns False when the op is absent from the manifest, when it has no
    ``roofline`` section, when ``roofline`` is not a mapping, or when
    ``roofline.vars`` is missing / empty / not a mapping. This is the
    precondition callers should use before :func:`resolve_roofline_vars`,
    so they can distinguish "nothing to resolve" (legitimate fallback)
    from "resolution failed" (propagate).
    """
    ops = _load_manifest()
    entry = ops.get(op_name)
    if not isinstance(entry, dict):
        return False
    roofline = entry.get("roofline")
    if not isinstance(roofline, dict):
        return False
    vars_decl = roofline.get("vars")
    return isinstance(vars_decl, dict) and bool(vars_decl)


def resolve_roofline_vars(
    op_name: str,
    tensor_shapes: dict[str, tuple[int, ...]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate ``roofline.vars`` expressions for *op_name* against real inputs.

    Parameters
    ----------
    op_name:
        Manifest key for the op (e.g. ``"SumFwdOp"``).
    tensor_shapes:
        Mapping from tensor name (as declared in ``signature.inputs``) to the
        concrete ``shape`` tuple. Each shape is exposed to the expression as
        ``<name>.shape`` / ``<name>.ndim`` via a ``SimpleNamespace``.
    params:
        Optional mapping of op param names to their concrete values (e.g.
        ``{"dim": 0, "keepdim": False}``). Param defaults declared in the
        manifest ``signature.params`` are filled in for any key not supplied.

    Returns
    -------
    dict
        Mapping from var name (as declared in ``roofline.vars``) to the
        evaluated scalar value.

    Notes
    -----
    Expressions are evaluated by an AST-walking sandbox (see
    :func:`_eval_roofline_expr`). Only an explicit whitelist of AST node
    types is accepted, dunder identifiers are always rejected, and
    attribute access is narrowed to ``.shape`` / ``.ndim`` on
    :class:`_ShapeProxy` bindings. The manifest is project-owned data, but
    the sandbox hardens against accidental mis-authored expressions and
    future supply-chain paths that could let an attacker influence
    ``roofline.vars``.
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")

    entry = ops[op_name]
    roofline = entry.get("roofline")
    vars_decl = roofline.get("vars") if isinstance(roofline, dict) else None
    if not isinstance(vars_decl, dict) or not vars_decl:
        raise ValueError(
            f"op '{op_name}' has no 'roofline.vars' mapping; cannot resolve "
            "variables from manifest"
        )

    # Build locals namespace: read-only shape proxies + params (with
    # defaults). The proxy exposes ``__getitem__`` / ``__len__`` /
    # ``__iter__`` plus a narrow ``.shape`` / ``.ndim`` attribute whitelist
    # so existing manifest expressions such as ``x.shape[-1]`` keep working
    # without opening up arbitrary attribute traversal.
    eval_locals: dict[str, Any] = {}
    for tname, shape in tensor_shapes.items():
        eval_locals[tname] = _ShapeProxy(tuple(shape))

    # Fill param defaults from manifest, then override with caller params.
    sig_params = entry.get("signature", {}).get("params", {})
    if isinstance(sig_params, dict):
        for pname, pspec in sig_params.items():
            if isinstance(pspec, dict) and "default" in pspec:
                eval_locals[pname] = pspec["default"]
    if params:
        for k, v in params.items():
            eval_locals[k] = v

    # Merge tensor bindings + params + whitelisted helpers into the single
    # namespace consumed by the AST walker. No ``__builtins__`` are ever
    # exposed; callers and attributes are both whitelisted in ``_walk``.
    eval_namespace: dict[str, Any] = {
        **_ROOFLINE_VARS_BUILTINS,
        **eval_locals,
    }

    resolved: dict[str, Any] = {}
    for var_name, expr in vars_decl.items():
        if not isinstance(expr, str):
            raise ValueError(
                f"op '{op_name}': roofline.vars[{var_name!r}] must be a "
                f"string expression, got {type(expr).__name__}"
            )
        try:
            value = _eval_roofline_expr(expr, eval_namespace)
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate roofline.vars[{var_name!r}] for "
                f"'{op_name}': {expr!r} with locals={eval_locals}"
            ) from exc
        resolved[var_name] = value

    return resolved
