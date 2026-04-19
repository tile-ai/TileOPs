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
from types import SimpleNamespace
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
    "type": type,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
}


def has_roofline_vars(op_name: str) -> bool:
    """Return True iff the manifest declares a non-empty ``roofline.vars``
    mapping for *op_name*.

    Returns False when the op is absent from the manifest, when it has no
    ``roofline`` section, or when ``roofline.vars`` is missing / empty / not
    a mapping. This is the precondition callers should use before
    :func:`resolve_roofline_vars`, so they can distinguish "nothing to
    resolve" (legitimate fallback) from "resolution failed" (propagate).
    """
    ops = _load_manifest()
    entry = ops.get(op_name)
    if not isinstance(entry, dict):
        return False
    vars_decl = entry.get("roofline", {}).get("vars")
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
    The evaluator uses Python's :func:`eval` with a restricted ``globals``
    dict (no ``__builtins__``; only whitelisted helpers such as ``product``,
    ``isinstance``, ``len``, ``set``, ``range``) and a fresh ``locals`` dict
    populated from ``tensor_shapes`` + ``params``. The manifest is
    project-owned data; expressions cannot originate from untrusted input.
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")

    entry = ops[op_name]
    roofline = entry.get("roofline", {})
    vars_decl = roofline.get("vars")
    if not isinstance(vars_decl, dict) or not vars_decl:
        raise ValueError(
            f"op '{op_name}' has no 'roofline.vars' mapping; cannot resolve "
            "variables from manifest"
        )

    # Build locals namespace: tensor.shape objects + params (with defaults).
    eval_locals: dict[str, Any] = {}
    for tname, shape in tensor_shapes.items():
        shape_tuple = tuple(shape)
        eval_locals[tname] = SimpleNamespace(
            shape=shape_tuple, ndim=len(shape_tuple)
        )

    # Fill param defaults from manifest, then override with caller params.
    sig_params = entry.get("signature", {}).get("params", {})
    if isinstance(sig_params, dict):
        for pname, pspec in sig_params.items():
            if isinstance(pspec, dict) and "default" in pspec:
                eval_locals[pname] = pspec["default"]
    if params:
        for k, v in params.items():
            eval_locals[k] = v

    # Merge tensor bindings + params + whitelisted helpers into a single
    # namespace and pass it as *globals* to ``eval``. Using a unified
    # namespace (rather than a separate locals dict) is required because
    # Python's generator expressions and set/dict comprehensions create an
    # inner function scope that can only see *globals*, not outer locals —
    # and the manifest expressions rely heavily on genexps / comprehensions
    # (e.g. ``product(x.shape[i] for i in range(x.ndim) if ...)``).
    eval_namespace: dict[str, Any] = {
        "__builtins__": {},
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
            value = eval(  # noqa: S307 -- restricted globals, project-owned expr
                expr, eval_namespace
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate roofline.vars[{var_name!r}] for "
                f"'{op_name}': {expr!r} with locals={eval_locals}"
            ) from exc
        resolved[var_name] = value

    return resolved
