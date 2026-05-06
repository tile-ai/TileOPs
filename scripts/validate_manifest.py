#!/usr/bin/env python3
"""Validate the ops manifest.

Checks:
  schema    — YAML structure: required fields, types, nesting
  signature — Op.forward() params match manifest inputs+params
  shape     — shape_rules are parseable Python expressions
  dtype     — dtype strings are valid torch dtype names or references
  bench     — benchmark file uses load_workloads and op-local eval_roofline()

Spec-only ops get schema only. Implemented ops get all checks.

Usage:
    python scripts/validate_manifest.py [--verbose] [--levels schema,shape,dtype,bench] [--check-op NAME]

Exit code 0 = all checks pass; 1 = failures found.

The --levels flag selects which checks to run. When omitted, all are enabled.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import itertools
import re
import sys
import types
import warnings as _warnings
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "tileops" / "manifest"

from tileops.manifest.shape_rules import (  # noqa: E402
    dim_range_validity,
    dim_uniqueness,
    reduced_axes,
)

# Valid torch dtype base names (without same_as references)
_TORCH_DTYPES = {
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64",
    "uint8", "bool",
    "complex64", "complex128",
    "float8_e4m3fn", "float8_e5m2",
    "float8_e4m3", "float8_e5m2fnuz", "float8_e4m3fnuz",
}

_SAME_AS_RE = re.compile(r"^same_as\(\s*(\w+)\s*\)$")

# Required top-level fields per op entry
_REQUIRED_TOP = {"family", "status", "signature", "workloads", "roofline", "source"}
_REQUIRED_SIGNATURE = {"inputs", "outputs"}
_REQUIRED_SOURCE = {"kernel", "op", "test", "bench"}

# Valid tensor layout values (R19)
_VALID_LAYOUTS = {"channels_last"}

# Single-axis reference: `<tensor>.shape[<int_literal_or_identifier>]` (R20)
_STATIC_DIM_EXPR_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\.shape\[(-?\d+|[A-Za-z_][A-Za-z0-9_]*)\]$"
)


def _check_static_dims(op_name: str, sdims: object, sig: dict) -> list[str]:
    """Validate `signature.static_dims` per R20.

    - Must be a mapping of str → str.
    - Each value must be a single-axis reference: `<tensor>.shape[<axis>]`
      where `<tensor>` is a name in `signature.inputs` and `<axis>` is either
      an integer literal or a param name declared in `signature.params`.
    """
    errors: list[str] = []

    if not isinstance(sdims, dict):
        errors.append(
            f"[schema] {op_name}: signature.static_dims must be a mapping"
        )
        return errors

    # Tolerate malformed inputs/params (reported as schema errors elsewhere):
    # treat non-dicts as empty so static_dims checks don't crash the validator.
    inputs = sig.get("inputs")
    params = sig.get("params")
    input_names = set(inputs.keys()) if isinstance(inputs, dict) else set()
    param_names = set(params.keys()) if isinstance(params, dict) else set()

    for dname, expr in sdims.items():
        if not isinstance(expr, str):
            errors.append(
                f"[schema] {op_name}: static_dims.{dname} must be a "
                f"string expression (got {type(expr).__name__})"
            )
            continue
        match = _STATIC_DIM_EXPR_RE.match(expr)
        if match is None:
            errors.append(
                f"[schema] {op_name}: static_dims.{dname} expression "
                f"{expr!r} is not a single-axis reference of the form "
                f"`<tensor>.shape[<const_or_param>]` (R20)"
            )
            continue
        tensor_name, axis_ref = match.groups()
        if tensor_name not in input_names:
            errors.append(
                f"[schema] {op_name}: static_dims.{dname} references tensor "
                f"{tensor_name!r}, which is not in signature.inputs "
                f"(known: {sorted(input_names) or 'none'})"
            )
        # axis_ref is an int literal (possibly negative) or an identifier
        if not (axis_ref.lstrip("-").isdigit() or axis_ref in param_names):
            errors.append(
                f"[schema] {op_name}: static_dims.{dname} axis reference "
                f"{axis_ref!r} is neither an integer literal nor a declared "
                f"param (known params: {sorted(param_names) or 'none'})"
            )
    return errors


# ---------------------------------------------------------------------------
# schema: YAML structure validation
# ---------------------------------------------------------------------------

def check_l0(
    op_name: str, entry: dict, *, warnings: list[str] | None = None,
) -> list[str]:
    """Validate structural schema of a manifest entry. Returns error strings."""
    errors: list[str] = []

    if not isinstance(entry, dict):
        errors.append(f"[schema] {op_name}: entry must be a mapping, got {type(entry).__name__}")
        return errors

    # Top-level required fields
    missing_top = _REQUIRED_TOP - set(entry.keys())
    if missing_top:
        errors.append(f"[schema] {op_name}: missing top-level fields: {missing_top}")

    # Signature structure
    sig = entry.get("signature")
    if isinstance(sig, dict):
        missing_sig = _REQUIRED_SIGNATURE - set(sig.keys())
        if missing_sig:
            errors.append(f"[schema] {op_name}: signature missing: {missing_sig}")

        # Check inputs/outputs are dicts with dtype
        for direction in ("inputs", "outputs"):
            tensors = sig.get(direction)
            if not isinstance(tensors, dict):
                if direction in sig:
                    errors.append(
                        f"[schema] {op_name}: signature.{direction} must be a dict"
                    )
                continue
            for tname, attrs in tensors.items():
                if not isinstance(attrs, dict):
                    errors.append(
                        f"[schema] {op_name}: {direction}.{tname} must be a dict"
                    )
                    continue
                if "dtype" not in attrs:
                    errors.append(
                        f"[schema] {op_name}: {direction}.{tname} missing 'dtype'"
                    )
                # layout validation (R19)
                if "layout" in attrs:
                    layout = attrs["layout"]
                    if not isinstance(layout, str):
                        errors.append(
                            f"[schema] {op_name}: {direction}.{tname}.layout "
                            f"must be a string"
                        )
                    elif layout not in _VALID_LAYOUTS:
                        errors.append(
                            f"[schema] {op_name}: {direction}.{tname}.layout "
                            f"'{layout}' is not recognized "
                            f"(valid: {', '.join(sorted(_VALID_LAYOUTS))})"
                        )

        # Params must be a mapping if present; each entry must have 'type' (R1)
        if "params" in sig:
            params = sig["params"]
            if not isinstance(params, dict):
                errors.append(
                    f"[schema] {op_name}: signature.params must be a mapping"
                )
            else:
                for pname, pattrs in params.items():
                    if not isinstance(pattrs, dict):
                        errors.append(
                            f"[schema] {op_name}: params.{pname} must be a dict"
                        )
                        continue
                    if "type" not in pattrs:
                        errors.append(
                            f"[schema] {op_name}: params.{pname} missing 'type'"
                        )

        # dtype_combos must be a list of dicts if present (R4)
        if "dtype_combos" in sig:
            combos = sig["dtype_combos"]
            if not isinstance(combos, list):
                errors.append(
                    f"[schema] {op_name}: signature.dtype_combos must be a list"
                )
            else:
                tensor_names = set()
                for d in ("inputs", "outputs"):
                    t = sig.get(d)
                    if isinstance(t, dict):
                        tensor_names.update(t.keys())
                for i, combo in enumerate(combos):
                    if not isinstance(combo, dict):
                        errors.append(
                            f"[schema] {op_name}: dtype_combos[{i}] must be a dict"
                        )
                        continue
                    for key in combo:
                        if key not in tensor_names:
                            errors.append(
                                f"[schema] {op_name}: dtype_combos[{i}] key "
                                f"'{key}' is not a declared tensor name"
                            )

        # shape_rules must be list of strings if present
        if "shape_rules" in sig:
            rules = sig["shape_rules"]
            if not isinstance(rules, list):
                errors.append(f"[schema] {op_name}: shape_rules must be a list")
            else:
                for i, rule in enumerate(rules):
                    if not isinstance(rule, str):
                        errors.append(
                            f"[schema] {op_name}: shape_rules[{i}] must be a string"
                        )

        # Reject the deprecated `init_dims` key explicitly (R20 rename).
        # L0 doesn't flag unknown signature keys, so without this check an
        # accidental reintroduction would silently pass and be ignored by L1.
        if "init_dims" in sig:
            errors.append(
                f"[schema] {op_name}: `signature.init_dims` is deprecated — "
                f"use `signature.static_dims` with flat `<name>: \"<tensor>.shape[<axis>]\"` "
                f"entries per R20"
            )

        # static_dims must be a mapping of str -> str expression (R20)
        if "static_dims" in sig:
            errors.extend(
                _check_static_dims(op_name, sig["static_dims"], sig)
            )
    elif "signature" in entry:
        errors.append(f"[schema] {op_name}: signature must be a mapping")

    # Workloads
    workloads = entry.get("workloads")
    if isinstance(workloads, list):
        for i, w in enumerate(workloads):
            if not isinstance(w, dict):
                errors.append(f"[schema] {op_name}: workloads[{i}] must be a dict")
                continue
            if "dtypes" not in w:
                errors.append(
                    f"[schema] {op_name}: workloads[{i}] missing 'dtypes'"
                )
    elif "workloads" in entry:
        errors.append(f"[schema] {op_name}: workloads must be a list")

    # Roofline
    roofline = entry.get("roofline")
    if isinstance(roofline, dict):
        has_inline = "flops" in roofline and "bytes" in roofline
        has_func = "func" in roofline
        if not has_inline and not has_func:
            errors.append(
                f"[schema] {op_name}: roofline must have (flops + bytes) or func"
            )
    elif "roofline" in entry:
        errors.append(f"[schema] {op_name}: roofline must be a mapping")

    # Source
    source = entry.get("source")
    if isinstance(source, dict):
        missing_src = _REQUIRED_SOURCE - set(source.keys())
        if missing_src:
            errors.append(
                f"[schema] {op_name}: source missing fields: {missing_src}"
            )
        # source.kernel: string or list of strings
        kernel = source.get("kernel")
        if kernel is not None:
            if isinstance(kernel, list):
                for i, k in enumerate(kernel):
                    if not isinstance(k, str):
                        errors.append(
                            f"[schema] {op_name}: source.kernel[{i}] "
                            f"must be a string"
                        )
            elif not isinstance(kernel, str):
                errors.append(
                    f"[schema] {op_name}: source.kernel must be a string or list"
                )
        if "bench_manifest_driven" in source and not isinstance(
            source["bench_manifest_driven"], bool,
        ):
            errors.append(
                f"[schema] {op_name}: source.bench_manifest_driven must be a bool"
            )
    elif "source" in entry:
        errors.append(f"[schema] {op_name}: source must be a mapping")

    # variant_of: must be a string if present (R16); cross-entry checks in
    # check_variant_of_consistency()
    if "variant_of" in entry and not isinstance(entry["variant_of"], str):
        errors.append(
            f"[schema] {op_name}: variant_of must be a string"
        )

    # ref_api: required string — fully qualified PyTorch API equivalent or "none"
    if "ref_api" not in entry:
        errors.append(
            f"[schema] {op_name}: missing required field 'ref_api'"
        )
    elif not isinstance(entry["ref_api"], str):
        errors.append(
            f"[schema] {op_name}: ref_api must be a string"
        )

    # status: must be "implemented" or "spec-only"
    # (skip if already caught by missing top-level fields check)
    status = entry.get("status")
    if "status" in entry and not isinstance(status, str):
        errors.append(
            f"[schema] {op_name}: status must be a string, "
            f"got {type(status).__name__}"
        )
    elif isinstance(status, str) and status not in ("implemented", "spec-only"):
        errors.append(
            f"[schema] {op_name}: status must be 'implemented' or 'spec-only', "
            f"got '{status}'"
        )

    # kernel_map: lives under source (source.kernel_map per manifest spec)
    source = entry.get("source", {})
    kernel_map = source.get("kernel_map") if isinstance(source, dict) else None
    if kernel_map is not None:
        if not isinstance(kernel_map, dict):
            errors.append(
                f"[schema] {op_name}: kernel_map must be a mapping, "
                f"got {type(kernel_map).__name__}"
            )
        else:
            for k, v in kernel_map.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    errors.append(
                        f"[schema] {op_name}: kernel_map entries must be "
                        f"str -> str, got {k!r}: {v!r}"
                    )
    elif status == "implemented" and warnings is not None:
        warnings.append(
            f"[schema] {op_name}: status is 'implemented' but "
            f"kernel_map is missing (should be a mapping of str -> str)"
        )

    return errors


# ---------------------------------------------------------------------------
# variant_of: cross-entry consistency (R16)
# ---------------------------------------------------------------------------

def check_variant_of_consistency(
    ops: dict, *, scope: set[str] | None = None
) -> list[str]:
    """Validate variant_of references across all entries.

    Per R16:
    - variant_of must reference an existing op in the manifest.
    - Single-level: the primary (referenced) entry must NOT itself have
      variant_of (no chaining).
    - Variant and primary must share source.kernel and source.op.

    When *scope* is given, only ops whose names are in *scope* are checked;
    lookups into *ops* still use the full dict so reference resolution works.
    """
    errors: list[str] = []

    for op_name, entry in ops.items():
        if scope is not None and op_name not in scope:
            continue
        if not isinstance(entry, dict):
            continue  # malformed entry — check_l0 will report it
        primary_name = entry.get("variant_of")
        if primary_name is None:
            continue

        # Target must exist
        if primary_name not in ops:
            errors.append(
                f"[schema] {op_name}: variant_of '{primary_name}' "
                f"does not exist in the manifest"
            )
            continue

        primary = ops[primary_name]
        if not isinstance(primary, dict):
            continue  # malformed primary — check_l0 will report it

        # Single-level: primary must not be a variant itself
        if "variant_of" in primary:
            errors.append(
                f"[schema] {op_name}: variant_of '{primary_name}' is itself "
                f"a variant (chaining not allowed per R16)"
            )

        # Shared source.kernel and source.op
        src = entry.get("source", {})
        pri_src = primary.get("source", {})
        if src.get("kernel") != pri_src.get("kernel"):
            errors.append(
                f"[schema] {op_name}: source.kernel differs from primary "
                f"'{primary_name}' (must match per R16)"
            )
        if src.get("op") != pri_src.get("op"):
            errors.append(
                f"[schema] {op_name}: source.op differs from primary "
                f"'{primary_name}' (must match per R16)"
            )

    return errors


# ---------------------------------------------------------------------------
# signature: Op.forward() vs manifest consistency
# ---------------------------------------------------------------------------

def check_l1_signature(
    op_name: str,
    manifest_inputs: dict,
    manifest_params: dict,
    forward_params: list[str],
    *,
    init_params: list[str] | None = None,
    manifest_static_dims: dict | None = None,
) -> list[str]:
    """Check that forward() params match manifest inputs + params.

    The strict rule: every manifest-declared param must appear in the union
    of ``__init__()`` and ``forward()`` parameter names. Manifest inputs must
    appear in ``forward()`` in declaration order. Every ``static_dims`` key
    must appear as an ``__init__()`` parameter (per R20).

    Args:
        op_name: Manifest op name.
        manifest_inputs: The signature.inputs dict from manifest.
        manifest_params: The signature.params dict from manifest.
        forward_params: List of parameter names from Op.forward() (excluding 'self').
        init_params: List of parameter names from Op.__init__() (excluding 'self').
            When None, treated as empty (only forward is checked).
        manifest_static_dims: The signature.static_dims dict from manifest (may be None).

    Returns:
        List of error strings (empty if OK).
    """
    errors: list[str] = []

    # Guard: manifest_params must be a dict (schema should catch this, but be safe)
    if not isinstance(manifest_params, dict):
        errors.append(
            f"[signature] {op_name}: signature.params is not a mapping, "
            f"cannot validate forward() consistency"
        )
        return errors

    if init_params is None:
        init_params = []

    # 1. forward() order check: manifest inputs + forward-visible params, in order
    expected = list(manifest_inputs.keys()) + [
        name for name in manifest_params.keys() if name in forward_params
    ]
    if forward_params != expected:
        errors.append(
            f"[signature] {op_name}: forward() params {forward_params} do not match "
            f"manifest order {expected}"
        )

    # 2. Strict subset check: every manifest param must exist in init OR forward
    code_params = set(forward_params) | set(init_params)
    for pname in manifest_params:
        if pname not in code_params:
            errors.append(
                f"[signature] {op_name}: manifest param {pname!r} not found in "
                f"__init__() or forward() parameters"
            )

    # 3. static_dims check (R20): every static_dims key must be an __init__ param
    if manifest_static_dims:
        if not isinstance(manifest_static_dims, dict):
            errors.append(
                f"[signature] {op_name}: signature.static_dims is not a mapping"
            )
        else:
            init_param_set = set(init_params)
            for dim_name in manifest_static_dims:
                if dim_name not in init_param_set:
                    errors.append(
                        f"[signature] {op_name}: static_dims key {dim_name!r} not found in "
                        f"__init__() parameters (R20: static_dims keys are required __init__ params)"
                    )

    return errors


class _ResolveResult:
    """Result of attempting to resolve an Op class from a module path."""

    __slots__ = ("cls", "import_error", "warning")

    def __init__(self, cls=None, import_error: bool = False, warning: str = ""):
        self.cls = cls
        self.import_error = import_error
        self.warning = warning


def _resolve_op_class(op_file: str, op_name: str) -> _ResolveResult:
    """Try to import the Op class from the source.op file.

    Returns a _ResolveResult with:
      - cls set if the Op class was found
      - import_error=True if the module could not be imported due to
        missing dependencies (ImportError / ModuleNotFoundError)
    """
    # Convert file path to module path
    # e.g., "tileops/ops/norm/rms_norm.py" -> "tileops.ops.norm.rms_norm"
    mod_path = op_file.replace("/", ".").replace(".py", "")
    try:
        mod = importlib.import_module(mod_path)
    except (ImportError, ModuleNotFoundError):
        return _ResolveResult(import_error=True)
    except Exception:
        return _ResolveResult()

    # Find Op subclass in the module. We look for classes defined in this module
    # that have a forward() method.
    seen_ids: set[int] = set()
    candidates = []
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if id(obj) in seen_ids:
            continue
        if hasattr(obj, "forward") and callable(obj.forward):
            seen_ids.add(id(obj))
            candidates.append(obj)

    if not candidates:
        return _ResolveResult()

    # Require exact class-name identity: cls.__name__ == manifest key.
    # No single-candidate bypass, no heuristic fallback.
    direct = [c for c in candidates if c.__name__ == op_name]
    if len(direct) == 1:
        return _ResolveResult(cls=direct[0])

    if len(direct) > 1:
        match_names = [c.__name__ for c in direct]
        ambiguity_msg = (
            f"Ambiguous op class resolution for '{op_name}': "
            f"multiple classes named '{op_name}' in '{op_file}': {match_names}. "
            f"Returning unresolved (cls=None)."
        )
        _warnings.warn(ambiguity_msg, UserWarning, stacklevel=2)
        return _ResolveResult(warning=ambiguity_msg)

    # No exact match found among multiple candidates.
    candidate_names = [c.__name__ for c in candidates]
    ambiguity_msg = (
        f"No class named '{op_name}' found in '{op_file}'. "
        f"Candidates: {candidate_names}. "
        f"Manifest key must exactly match cls.__name__."
    )
    _warnings.warn(ambiguity_msg, UserWarning, stacklevel=2)
    return _ResolveResult(warning=ambiguity_msg)


_EXPLICIT_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
}


def _get_forward_params(cls) -> list[str] | None:
    """Get explicit parameter names of cls.forward(), excluding 'self'.

    Only returns explicitly named parameters — *args and **kwargs are
    excluded because manifest params must appear as named arguments.
    """
    try:
        sig = inspect.signature(cls.forward)
        return [
            p for p, v in sig.parameters.items()
            if p != "self" and v.kind in _EXPLICIT_KINDS
        ]
    except (ValueError, TypeError):
        return None


def _get_init_params(cls) -> list[str]:
    """Get explicit parameter names of cls.__init__(), excluding 'self'.

    Only returns explicitly named parameters — *args and **kwargs are
    excluded. Handles monkey-patched ``__init__`` methods: if the live
    signature has no explicit params, walk the MRO to find the first
    concrete ``__init__`` with explicit parameters.
    """
    def _extract(func):
        try:
            sig = inspect.signature(func)
            params = [
                p for p, v in sig.parameters.items()
                if p != "self" and v.kind in _EXPLICIT_KINDS
            ]
            if not params:
                return None  # no explicit params — try next in MRO
            return params
        except (ValueError, TypeError):
            return None

    # Try the live __init__ first
    result = _extract(cls.__init__)
    if result is not None:
        return result

    # Walk MRO for the first concrete __init__
    for base in cls.__mro__[1:]:
        if "__init__" in base.__dict__:
            result = _extract(base.__dict__["__init__"])
            if result is not None:
                return result

    return []


def check_l1(
    op_name: str, entry: dict, *, warnings: list[str] | None = None,
) -> list[str]:
    """Signature check: resolve Op class and compare forward() to manifest.

    Checks both ``__init__()`` and ``forward()`` parameter names against
    the manifest signature.

    Args:
        op_name: Manifest op name.
        entry: The manifest entry dict.
        warnings: Optional list to append warning messages to.

    Returns:
        List of error strings (empty if OK).
    """
    errors: list[str] = []
    sig = entry.get("signature", {})
    source = entry.get("source", {})
    op_file = source.get("op", "")

    result = _resolve_op_class(op_file, op_name)

    if result.warning and warnings is not None:
        warnings.append(f"[signature] {op_name}: {result.warning}")

    if result.import_error:
        errors.append(
            f"[signature] {op_name}: could not import {op_file} "
            f"(missing dependencies)"
        )
        return errors

    if result.cls is None:
        errors.append(f"[signature] {op_name}: could not resolve Op class from {op_file}")
        return errors

    forward_params = _get_forward_params(result.cls)
    if forward_params is None:
        errors.append(
            f"[signature] {op_name}: could not inspect forward() on {result.cls.__name__}"
        )
        return errors

    manifest_inputs = sig.get("inputs", {})
    manifest_params = sig.get("params", {})
    manifest_static_dims = sig.get("static_dims")
    init_params = _get_init_params(result.cls)

    return check_l1_signature(
        op_name, manifest_inputs, manifest_params, forward_params,
        init_params=init_params,
        manifest_static_dims=manifest_static_dims,
    )


# ---------------------------------------------------------------------------
# shape: shape_rules syntax validation
# ---------------------------------------------------------------------------

def check_l2(op_name: str, entry: dict) -> list[str]:
    """Validate shape_rules are parseable Python expressions."""
    errors: list[str] = []
    sig = entry.get("signature", {})
    rules = sig.get("shape_rules", [])

    for i, rule in enumerate(rules):
        if not isinstance(rule, str):
            continue
        try:
            ast.parse(rule, mode="eval")
        except SyntaxError as exc:
            errors.append(
                f"[shape] {op_name}: shape_rules[{i}] invalid syntax: {rule!r} ({exc})"
            )
    return errors


# ---------------------------------------------------------------------------
# dtype: dtype string conformance
# ---------------------------------------------------------------------------

def _parse_dtype_expr(dtype_str: str) -> list[str]:
    """Parse a dtype expression into individual dtype tokens.

    Handles: "float16", "float16 | bfloat16", "same_as(x)".
    Returns list of raw tokens (may include same_as references).
    """
    return [t.strip() for t in dtype_str.split("|")]


def _validate_dtype_token(
    op_name: str, context: str, token: str, tensor_names: set[str],
) -> str | None:
    """Validate a single dtype token. Returns an error string or None."""
    m = _SAME_AS_RE.match(token)
    if m:
        ref = m.group(1)
        if ref not in tensor_names:
            return (
                f"[dtype] {op_name}: {context} dtype same_as({ref}) "
                f"references unknown tensor"
            )
    elif token not in _TORCH_DTYPES:
        return f"[dtype] {op_name}: {context} has unrecognized dtype '{token}'"
    return None


def _build_same_as_map(all_tensors: dict) -> dict[str, str]:
    """Build a mapping from tensor name to its same_as reference target.

    For each tensor whose dtype is ``same_as(ref)``, maps tensor → ref.
    Only pure same_as dtypes are tracked (not ``float16 | same_as(x)``).
    """
    same_as_map: dict[str, str] = {}
    for tname, attrs in all_tensors.items():
        dtype_str = attrs.get("dtype", "")
        tokens = _parse_dtype_expr(dtype_str)
        if len(tokens) == 1:
            m = _SAME_AS_RE.match(tokens[0])
            if m:
                same_as_map[tname] = m.group(1)
    return same_as_map


def _check_dtype_combos_same_as_identity(
    op_name: str, dtype_combos: list, same_as_map: dict[str, str],
) -> list[str]:
    """Enforce same_as identity constraint in dtype_combos entries.

    For each dtype_combos entry, every tensor bound by same_as(ref) must
    have the exact same dtype as its reference tensor (R3 identity constraint).
    """
    errors: list[str] = []
    for i, combo in enumerate(dtype_combos):
        if not isinstance(combo, dict):
            continue
        for tensor, ref in same_as_map.items():
            t_in = tensor in combo
            r_in = ref in combo
            if t_in and r_in and combo[tensor] != combo[ref]:
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}] violates "
                    f"same_as identity constraint — {tensor} "
                    f"({combo[tensor]}) must match {ref} "
                    f"({combo[ref]}) per R3"
                )
            elif t_in and not r_in:
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}] has "
                    f"same_as-bound tensor '{tensor}' without its "
                    f"reference '{ref}' — cannot verify identity"
                )
    return errors


def check_l3(op_name: str, entry: dict) -> list[str]:
    """Validate dtype strings are recognized torch types or same_as references.

    Checks both signature tensor dtypes and workload dtype entries.
    Also enforces same_as identity constraint in dtype_combos (R3).
    """
    errors: list[str] = []
    sig = entry.get("signature", {})
    all_tensors = {}
    all_tensors.update(sig.get("inputs", {}))
    all_tensors.update(sig.get("outputs", {}))

    tensor_names = set(all_tensors.keys())

    # Validate signature tensor dtypes
    for tname, attrs in all_tensors.items():
        dtype_str = attrs.get("dtype", "")
        tokens = _parse_dtype_expr(dtype_str)
        for token in tokens:
            err = _validate_dtype_token(op_name, tname, token, tensor_names)
            if err:
                errors.append(err)

    # Validate same_as identity constraint in dtype_combos (R3)
    dtype_combos = sig.get("dtype_combos", [])
    if isinstance(dtype_combos, list) and dtype_combos:
        same_as_map = _build_same_as_map(all_tensors)
        errors.extend(
            _check_dtype_combos_same_as_identity(op_name, dtype_combos, same_as_map)
        )
        # Hard data-validation for combo values: every combo entry must
        # resolve to a concrete torch dtype (or a ``same_as(ref)`` whose
        # ref resolves to concrete torch dtypes). Runs unconditionally —
        # independent of whether the op overrides ``_validate_dtypes`` —
        # so an un-migrated op carrying invalid combo data still surfaces
        # a hard L3 error rather than only a missing-override warning.
        errors.extend(check_l3_dtype_combos_data(op_name, sig))

    # Validate workload dtypes
    workloads = entry.get("workloads", [])
    if isinstance(workloads, list):
        for i, w in enumerate(workloads):
            if not isinstance(w, dict):
                continue
            dtypes = w.get("dtypes", [])
            if not isinstance(dtypes, list):
                continue
            for j, dt in enumerate(dtypes):
                if not isinstance(dt, str):
                    errors.append(
                        f"[dtype] {op_name}: workloads[{i}].dtypes[{j}] "
                        f"is not a string"
                    )
                    continue
                tokens = _parse_dtype_expr(dt)
                for token in tokens:
                    err = _validate_dtype_token(
                        op_name, f"workloads[{i}].dtypes[{j}]",
                        token, tensor_names,
                    )
                    if err:
                        errors.append(err)

    return errors


def _diagnose_unresolvable_signature(op_name: str, sig: dict) -> list[str]:
    """Emit hard L3 errors describing why a signature failed to resolve.

    Called when ``_resolve_tensor_dtype_options(sig)`` returns None inside
    combo validation. Walks the pure ``same_as`` edges to distinguish:

      * dangling references (``same_as(ref)`` where ``ref`` is not a
        declared tensor) — per-tensor error;
      * ``same_as`` cycles (``x -> y -> ... -> x``) — one error per cycle
        naming every participating tensor;
      * an unknown-token / ``same_as`` in a mixed expression that resolves
        to nothing — generic fallback, so callers are never left guessing.
    """
    errors: list[str] = []
    inputs = sig.get("inputs") or {}
    outputs = sig.get("outputs") or {}
    all_tensors: dict[str, dict] = {}
    if isinstance(inputs, dict):
        all_tensors.update(inputs)
    if isinstance(outputs, dict):
        all_tensors.update(outputs)

    # Pure ``same_as(ref)`` edges only — mixed expressions (``float16 |
    # same_as(x)``) are not part of the cycle graph; a cycle in pure edges
    # is what makes fixpoint resolution stall.
    edges: dict[str, str] = {}
    for tname, attrs in all_tensors.items():
        if not isinstance(attrs, dict):
            continue
        tokens = _parse_dtype_expr(attrs.get("dtype", ""))
        if len(tokens) == 1:
            m = _SAME_AS_RE.match(tokens[0])
            if m:
                edges[tname] = m.group(1)

    # Dangling references: ``same_as(ref)`` where ``ref`` is not declared.
    dangling: set[str] = set()
    for tname, ref in edges.items():
        if ref not in all_tensors:
            errors.append(
                f"[dtype] {op_name}: signature.inputs/outputs — tensor "
                f"{tname!r} declares dtype same_as({ref}) but {ref!r} is "
                f"not a declared tensor (dangling reference; combo "
                f"validation cannot proceed)"
            )
            dangling.add(tname)

    # Cycle detection via DFS over pure same_as edges. Only tensors that
    # have not already been reported as dangling are considered (a chain
    # ending in a dangling ref is not a cycle).
    reported_cycles: set[frozenset[str]] = set()
    visited: set[str] = set()
    for start in edges:
        if start in visited or start in dangling:
            continue
        path: list[str] = []
        seen_in_path: dict[str, int] = {}
        node: str | None = start
        while node is not None and node not in visited:
            if node in seen_in_path:
                cycle_nodes = path[seen_in_path[node]:]
                key = frozenset(cycle_nodes)
                if key not in reported_cycles:
                    reported_cycles.add(key)
                    errors.append(
                        f"[dtype] {op_name}: same_as cycle detected "
                        f"among tensors "
                        f"{sorted(cycle_nodes)} — dtype options cannot "
                        f"be resolved (combo validation skipped)"
                    )
                break
            seen_in_path[node] = len(path)
            path.append(node)
            nxt = edges.get(node)
            if nxt is None or nxt in dangling:
                break
            if nxt not in edges:
                # Chain terminates at a concrete-dtype tensor — not a
                # cycle. Stop walking.
                break
            node = nxt
        visited.update(path)

    if not errors:
        # Fixpoint failed but we cannot pinpoint a cycle or dangling edge
        # (e.g. a mixed expression containing an unknown token that did
        # not trip per-token validation). Emit a generic hard error so
        # combo validation is never silently skipped.
        errors.append(
            f"[dtype] {op_name}: could not resolve signature.inputs/outputs "
            f"dtype options — combo validation cannot proceed. Check "
            f"signature.inputs/outputs dtype declarations for unresolved "
            f"same_as references or malformed expressions."
        )
    return errors


def check_l3_dtype_combos_data(op_name: str, sig: dict) -> list[str]:
    """Validate ``dtype_combos`` entries resolve to concrete torch dtypes.

    Manifest-data check, independent of any op class / ``_validate_dtypes``
    implementation. Every combo value must be either:
      * a concrete dtype name in ``_TORCH_DTYPES``; or
      * a ``same_as(ref)`` expression whose ref resolves transitively to
        concrete dtype names.

    Anything else (e.g. ``"not_a_real_dtype"``, ``same_as(unknown)``) is a
    hard L3 error — callers must not silently proceed with invalid combo
    data.
    """
    errors: list[str] = []
    dtype_combos = sig.get("dtype_combos")
    if not isinstance(dtype_combos, list) or not dtype_combos:
        return errors
    dtype_options = _resolve_tensor_dtype_options(sig)
    if dtype_options is None:
        # Unresolvable signature. Previously this branch returned silently
        # under the assumption that ``check_l3`` had already flagged the
        # culprit, but a pure ``same_as`` cycle (e.g. ``x: same_as(y)`` and
        # ``y: same_as(x)``) satisfies per-token validation *and* the R3
        # identity check, so combo validation would be silently skipped and
        # invalid combo data would pass. Emit a hard L3 error with a
        # specific diagnosis when possible (cycle / dangling reference),
        # falling back to a generic unresolved-signature error otherwise.
        errors.extend(_diagnose_unresolvable_signature(op_name, sig))
        return errors
    inputs = sig.get("inputs") or {}
    declared_input_names: list[str] = (
        list(inputs.keys()) if isinstance(inputs, dict) else []
    )
    for i, combo in enumerate(dtype_combos):
        if not isinstance(combo, dict):
            continue
        # Combo-row completeness: every declared signature.inputs tensor
        # must be assigned a dtype in every combo row. Otherwise a combo
        # row that silently omits an input would pass L3 when no
        # ``_validate_dtypes`` override exists, since ``_combo_accepted``
        # is never exercised for omitted inputs.
        for input_name in declared_input_names:
            if input_name not in combo:
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}] is missing "
                    f"declared input {input_name!r} (every combo row "
                    f"must cover every signature.inputs tensor)"
                )
        for key, val in combo.items():
            if not isinstance(val, str):
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}].{key} = "
                    f"{val!r} is not a string"
                )
                continue
            # Per manifest.md R4, each combo value must be a single
            # concrete dtype token (or a ``same_as(ref)`` naming one
            # sibling in the same combo row). Reject union expressions
            # like ``"float16 | bfloat16"`` outright — a union in a
            # combo row would let an implementation silently widen the
            # accepted-dtype set beyond what the manifest authored.
            if "|" in val:
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}].{key} = "
                    f"{val!r} — combo values must be a single concrete "
                    f"dtype, not a union"
                )
                continue
            opts = _dtype_options_for_tensor(key, val, dtype_options)
            if opts is None:
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}].{key} = "
                    f"{val!r} is not a valid dtype (unresolved "
                    f"same_as reference or not in torch dtype set)"
                )
            elif not all(t in _TORCH_DTYPES for t in opts):
                bad = [t for t in opts if t not in _TORCH_DTYPES]
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}].{key} = "
                    f"{val!r} resolves to unknown dtype(s) {bad!r}"
                )
    return errors


# ---------------------------------------------------------------------------
# shape parity: _infer_output_shapes vs shape_rules (L2 extension)
# ---------------------------------------------------------------------------

# Default mock sizes for symbolic shape dimensions. Chosen small to keep
# evaluation cheap; 4 avoids degenerate cases (e.g. shape[0]==1 matching
# scalar broadcasts) while staying small. Distinct symbolic dims get
# ``_MOCK_DIM_SIZE + counter`` so cross-tensor equality checks remain
# meaningful (see ``_mock_input_shapes``).
_MOCK_DIM_SIZE = 4

# Safety bound for Cartesian-product enumeration in L3 dtype parity. A
# pathological future op with many inputs × wide dtype unions could blow
# CI budgets (each candidate allocates tiny tensors and invokes
# _validate_dtypes). Current manifest maxes out at ~5 inputs × ~4 options
# = 1024 combos, so this cap only fires on genuinely outsized specs; when
# it does we skip the op deterministically with a warning rather than
# sampling, so validation output stays reproducible.
_MAX_DTYPE_COMBOS = 4096

# Sentinel pool used only by the same_as-identity negative probe, where
# the goal is a dtype *different from the ref's baseline* (union membership
# is irrelevant). The out-of-union probes in both branches of
# ``check_l3_validate_dtypes_parity`` derive their candidate pool from
# ``sorted(_TORCH_DTYPES - declared)`` instead — that guarantees a
# non-empty probe whenever declared does not cover the entire torch dtype
# universe, closing a prior engulfment gap where a fixed 8-dtype pool
# could be fully absorbed by a wide union and leave the probe vacuous.
_DTYPE_SENTINELS: tuple[str, ...] = (
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64",
)


def _out_of_union_candidates(declared: set[str]) -> list[str]:
    """Return torch dtypes that are outside ``declared``.

    Deterministic (sorted) so validator output is reproducible; bounded
    because ``_TORCH_DTYPES`` is a fixed small set. Callers should still
    cap the iteration length via ``_MAX_DTYPE_COMBOS`` when combining
    with other enumeration.
    """
    return sorted(_TORCH_DTYPES - declared)


class _MockShape(tuple):
    """Tuple subclass representing a tensor shape, exposed via ``.shape``.

    Used in the shape_rules evaluation context so expressions like
    ``x.shape == (B, S, H, D)`` or ``x.ndim`` resolve correctly without
    constructing real tensors.
    """

    @property
    def shape(self) -> "tuple":  # type: ignore[override]
        return tuple(self)

    @property
    def ndim(self) -> int:
        return len(self)


def _extract_shape_tuple_literals(rules: list) -> dict[str, int]:
    """Parse ``<name>.shape == (<ids>...)`` rules for input-tensor rank hints.

    Returns a mapping tensor-name → rank. Only handles the simple literal
    form; other shape_rules patterns are skipped.
    """
    ranks: dict[str, int] = {}
    shape_eq_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\.shape\s*==\s*\(([^)]*)\)\s*$"
    )
    for rule in rules:
        if not isinstance(rule, str):
            continue
        m = shape_eq_re.match(rule)
        if m is None:
            continue
        name, body = m.group(1), m.group(2)
        parts = [p.strip() for p in body.split(",") if p.strip()]
        # Require all parts to be bare identifiers so we can assign mock
        # sizes by name; skip otherwise.
        if all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
            ranks[name] = len(parts)
    return ranks


_SHAPE_DECL_RE = re.compile(r"^\s*\[([^\]]*)\]\s*$")


def _parse_shape_decl(shape_str: str) -> list[str] | None:
    """Parse a ``signature.inputs[*].shape`` declaration like ``"[N, C, L]"``.

    Returns the list of dimension identifiers if the declaration is a bare
    comma-separated identifier list; returns None otherwise (e.g. contains
    arithmetic, literals, or other expressions that cannot be bound as
    mock dim names by this tool).
    """
    if not isinstance(shape_str, str):
        return None
    m = _SHAPE_DECL_RE.match(shape_str)
    if m is None:
        return None
    body = m.group(1)
    parts = [p.strip() for p in body.split(",") if p.strip()]
    if not parts:
        return None
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
        return None
    return parts


def _input_bound_symbols(sig: dict) -> set[str]:
    """Return symbolic dim names bound by INPUT shapes only.

    A symbol is input-bound when it appears in either:
      - a ``<input>.shape == (...)`` literal in ``signature.shape_rules``
      - a ``signature.inputs[*].shape`` declaration like ``"[N, C, L]"``

    Symbols that appear only in output shape declarations (e.g. ``L_out``
    in ``signature.outputs.y.shape = "[N, C, L_out]"`` where ``L_out`` is
    derived by a ``shape_rules`` entry such as ``L_out = L_in - kW + 1``)
    are **not** included. The L2 parity check uses this set to decide
    whether a declared output-shape symbol carries a concrete mock size
    (input-bound) versus a value derived by ``_infer_output_shapes``
    (output-only): comparing the inferred output against an arbitrary
    mock size for output-only symbols would misreport a correct
    implementation as a parity mismatch.
    """
    bound: set[str] = set()
    ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    # shape_rules: <name>.shape == (<ids>...)
    rules = sig.get("shape_rules") or []
    shape_eq_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\.shape\s*==\s*\(([^)]*)\)\s*$"
    )
    inputs = sig.get("inputs") or {}
    input_names = set(inputs.keys()) if isinstance(inputs, dict) else set()
    for rule in rules:
        if not isinstance(rule, str):
            continue
        m = shape_eq_re.match(rule)
        if m is None:
            continue
        tname, body = m.group(1), m.group(2)
        if tname not in input_names:
            continue
        for p in (q.strip() for q in body.split(",")):
            if p and ident_re.fullmatch(p):
                bound.add(p)
    # Per-tensor shape decl on inputs
    if isinstance(inputs, dict):
        for attrs in inputs.values():
            if not isinstance(attrs, dict):
                continue
            parts = _parse_shape_decl(attrs.get("shape", ""))
            if parts is not None:
                bound.update(parts)
    return bound


def _mock_input_shapes(
    sig: dict,
) -> tuple[dict[str, _MockShape], dict[str, int]] | None:
    """Derive concrete mock input shapes for every declared input.

    Uses rank hints from ``shape_rules`` (literal ``tensor.shape == (...)``
    forms) and from ``signature.inputs[*].shape`` declarations (e.g.
    ``"[N, C_in, L_in]"``). Falls back to a default 2D shape when the
    rank is unknown. Returns (shapes, dim_sizes) where ``dim_sizes`` maps
    each symbolic dimension name (e.g. ``B``, ``S``, ``H``, ``D``, or
    ``N``, ``C_in``, ``L_in`` from shape declarations) to the integer
    size used in the mock shapes, so callers can bind those names into a
    shape_rules evaluation context. Returns None only if
    ``signature.inputs`` is malformed.
    """
    inputs = sig.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return None
    rules = sig.get("shape_rules") or []
    ranks = _extract_shape_tuple_literals(rules)

    # Extra rank hints from per-tensor shape declarations.
    shape_decls: dict[str, list[str]] = {}
    for name, attrs in inputs.items():
        if not isinstance(attrs, dict):
            continue
        parts = _parse_shape_decl(attrs.get("shape", ""))
        if parts is not None:
            shape_decls[name] = parts
            ranks.setdefault(name, len(parts))

    shapes: dict[str, _MockShape] = {}
    # Assign dim-name → size map shared across tensors for consistent rules.
    # Use a global counter keyed on first-seen order so distinct symbolic
    # dims get distinct sizes across rules (e.g. rule1 ``x.shape == (A, B)``
    # and rule2 ``y.shape == (C, D)`` produce A=4, B=5, C=6, D=7 rather than
    # colliding A==C and B==D, which would spuriously satisfy cross-tensor
    # equality checks).
    dim_sizes: dict[str, int] = {}
    shape_eq_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\.shape\s*==\s*\(([^)]*)\)\s*$"
    )
    for rule in rules:
        if not isinstance(rule, str):
            continue
        m = shape_eq_re.match(rule)
        if m is None:
            continue
        body = m.group(2)
        parts = [p.strip() for p in body.split(",") if p.strip()]
        for p in parts:
            if (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p)
                and p not in dim_sizes
            ):
                dim_sizes[p] = _MOCK_DIM_SIZE + len(dim_sizes)

    # Also bind symbolic dim names from per-tensor shape declarations so
    # downstream rule/shape-decl checks resolve them against the same mock
    # sizes used to build the input tensors.
    for parts in shape_decls.values():
        for p in parts:
            if p not in dim_sizes:
                dim_sizes[p] = _MOCK_DIM_SIZE + len(dim_sizes)
    # Also bind symbolic dim names from declared output shapes, so rules
    # referencing output dim names (via signature.outputs[*].shape) can be
    # evaluated when shape_rules are absent.
    outputs_map = sig.get("outputs") or {}
    if isinstance(outputs_map, dict):
        for attrs in outputs_map.values():
            if not isinstance(attrs, dict):
                continue
            out_parts = _parse_shape_decl(attrs.get("shape", ""))
            if out_parts is None:
                continue
            for p in out_parts:
                if p not in dim_sizes:
                    dim_sizes[p] = _MOCK_DIM_SIZE + len(dim_sizes)

    for name in inputs:
        if name in ranks:
            parts = None
            for rule in rules:
                if not isinstance(rule, str):
                    continue
                m = shape_eq_re.match(rule)
                if m is None or m.group(1) != name:
                    continue
                parts = [
                    p.strip() for p in m.group(2).split(",") if p.strip()
                ]
                break
            if parts is not None and all(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts
            ):
                shapes[name] = _MockShape(
                    dim_sizes.get(p, _MOCK_DIM_SIZE) for p in parts
                )
                continue
        # Fallback: per-tensor shape declaration from signature.inputs.
        if name in shape_decls:
            shapes[name] = _MockShape(
                dim_sizes.get(p, _MOCK_DIM_SIZE) for p in shape_decls[name]
            )
            continue
        # Fallback: 2D shape
        shapes[name] = _MockShape(
            (_MOCK_DIM_SIZE, _MOCK_DIM_SIZE)
        )
    return shapes, dim_sizes


def _param_defaults(params: dict) -> dict:
    """Extract ``default`` values from a signature.params dict.

    Parameters without a default are omitted.
    """
    out: dict = {}
    if not isinstance(params, dict):
        return out
    for pname, pattrs in params.items():
        if isinstance(pattrs, dict) and "default" in pattrs:
            out[pname] = pattrs["default"]
    return out


def _static_dim_values(
    sig: dict,
    mock_shapes: dict[str, _MockShape],
    param_defaults: dict,
) -> dict:
    """Resolve ``signature.static_dims`` to concrete integer values.

    Each entry is declared as ``<name>: "<tensor>.shape[<axis>]"`` where
    ``<tensor>`` is an input and ``<axis>`` is either an integer literal
    or a param name. Returns a mapping ``{static_dim_name: int}`` with
    only successfully resolved entries (malformed / out-of-range entries
    are silently skipped — validator's L0 schema check reports those).

    Used by parity mock_self builders so methods that consult
    ``self.<static_dim_name>`` attributes (e.g. ``_infer_output_shapes``
    reading ``self.N``) see the concrete size carried by the synthetic
    inputs, rather than raising ``AttributeError``.
    """
    out: dict = {}
    sdims = sig.get("static_dims")
    if not isinstance(sdims, dict):
        return out
    for dname, expr in sdims.items():
        if not isinstance(expr, str):
            continue
        m = _STATIC_DIM_EXPR_RE.match(expr)
        if m is None:
            continue
        tname, axis_ref = m.groups()
        shape = mock_shapes.get(tname)
        if shape is None:
            continue
        # Resolve axis: integer literal or param-name lookup.
        if axis_ref.lstrip("-").isdigit():
            axis = int(axis_ref)
        elif axis_ref in param_defaults and isinstance(
            param_defaults[axis_ref], int
        ):
            axis = param_defaults[axis_ref]
        else:
            continue
        try:
            out[dname] = int(shape[axis])
        except (IndexError, TypeError, ValueError):
            continue
    return out


def _parity_opted_out(entry: dict, check: str) -> bool:
    """Return True when the manifest entry opts out of *check*.

    Recognises two shapes for the ``parity_opt_out`` field:
      - ``parity_opt_out: true`` — opt out of every parity check.
      - ``parity_opt_out: [shape_parity, dtype_parity]`` — opt out of
        specific checks only.

    Used for documented GPU-only ops where the manifest-derived method
    cannot be invoked from a CPU-only validator context.
    """
    opt = entry.get("parity_opt_out")
    if opt is True:
        return True
    return bool(isinstance(opt, list) and check in opt)


def _class_overrides_method(cls: type, name: str) -> bool:
    """Return True when *cls* (or a non-Op ancestor) defines *name*.

    We walk the MRO skipping the root ``Op`` base class; the goal is to
    detect user-authored overrides, not the base no-op.
    """
    from tileops.ops.op_base import Op as _OpBase  # local to avoid top-level import cost
    for base in cls.__mro__:
        if base is _OpBase or base is object:
            continue
        if name in base.__dict__:
            return True
    return False


def _broadcast_shapes(*shapes: object) -> tuple:
    """Pure-Python equivalent of ``torch.broadcast_shapes``.

    Computes the broadcasted output shape from one or more input shapes
    using NumPy/PyTorch broadcasting rules: shapes are right-aligned,
    each dimension must be equal, or one of them must be 1 (or missing).

    Args:
        *shapes: Iterables of integers (typically tuples or lists)
            representing tensor shapes. May be empty (scalar shape).

    Returns:
        The broadcasted shape as a tuple of ints. Returns ``()`` when
        called with no arguments.

    Raises:
        ValueError: If the shapes are not broadcast-compatible.
    """
    if not shapes:
        return ()
    normalized = [tuple(int(d) for d in s) for s in shapes]
    ndim = max((len(s) for s in normalized), default=0)
    out: list[int] = []
    for axis in range(ndim):
        # Right-align: walk from the trailing dim back.
        dim = 1
        for s in normalized:
            i = len(s) - ndim + axis
            if i < 0:
                # This shape has no entry at this axis (treat as 1).
                continue
            d = s[i]
            if d == 1 or d == dim:
                continue
            if dim == 1:
                dim = d
                continue
            raise ValueError(
                f"shapes {shapes!r} are not broadcast-compatible at axis {axis}",
            )
        out.append(dim)
    return tuple(out)


def _is_broadcastable_to(src: object, dst: object) -> bool:
    """Return True if ``src`` is broadcastable *to* ``dst`` (unidirectional).

    Unlike ``broadcast_shapes`` which is symmetric, this predicate fixes
    the destination shape and asks whether ``src`` can expand into it
    without shrinking ``dst``: each ``src`` dim (right-aligned) must be
    equal to the matching ``dst`` dim or be 1, and ``src`` may not have
    more dimensions than ``dst``.

    Args:
        src: Source shape (iterable of ints).
        dst: Destination shape (iterable of ints).

    Returns:
        True iff ``src`` broadcasts to ``dst``.
    """
    src_t = tuple(int(d) for d in src)
    dst_t = tuple(int(d) for d in dst)
    if len(src_t) > len(dst_t):
        return False
    offset = len(dst_t) - len(src_t)
    for i, s_dim in enumerate(src_t):
        d_dim = dst_t[offset + i]
        if s_dim == d_dim or s_dim == 1:
            continue
        return False
    return True


# Safe builtins allowed in shape_rules eval — matches the R11 / R11a
# documented helper set (see docs/design/ops-design-reference.md). Keep this list
# aligned with manifest spec; widening it changes the rule language.
#
# Three name groups live here together: Python primitives (``len`` etc.),
# broadcasting helpers (``broadcast_shapes`` / ``is_broadcastable_to``,
# pure-Python so the validator does not require ``torch``), and
# reduction-dim helpers from ``tileops.manifest.shape_rules``. All three
# share the same eval-scope contract — callable by bare name from any
# rule body. Group membership is editorial; the eval scope sees one
# flat namespace.
#
# The dict is built from an explicit (name, callable) list so a name
# collision between groups raises at validator import time. Silent
# dict-merge override would let a future helper shadow a Python primitive
# (or an existing broadcasting helper) without surfacing the conflict.
_SHAPE_RULE_BUILTIN_PAIRS = [
    ("len", len),
    ("isinstance", isinstance),
    ("int", int),
    ("tuple", tuple),
    ("list", list),
    ("type", type),
    ("all", all),
    ("any", any),
    ("range", range),
    ("set", set),
    ("abs", abs),
    ("min", min),
    ("max", max),
    ("broadcast_shapes", _broadcast_shapes),
    ("is_broadcastable_to", _is_broadcastable_to),
    ("dim_range_validity", dim_range_validity),
    ("dim_uniqueness", dim_uniqueness),
    ("reduced_axes", reduced_axes),
]
_SHAPE_RULE_BUILTINS: dict = {}
for _entry_name, _entry_fn in _SHAPE_RULE_BUILTIN_PAIRS:
    if _entry_name in _SHAPE_RULE_BUILTINS:
        raise RuntimeError(
            f"shape_rule builtin name collision: {_entry_name!r} is "
            f"registered twice. Two callables cannot share the same "
            f"name in the rule eval scope; rename one or unify them."
        )
    _SHAPE_RULE_BUILTINS[_entry_name] = _entry_fn


def _eval_shape_rule(
    rule: str, ctx: dict,
) -> tuple[bool, str | None]:
    """Evaluate a single shape_rule in *ctx*.

    Returns (ok, failure_reason). ``ok=False`` with reason=None means the
    rule evaluated to a falsy non-exception value; a non-None reason
    indicates the rule could not be evaluated (treated as skipped, not a
    parity error).

    The eval globals expose the ``_SHAPE_RULE_BUILTINS`` helper set
    (``len``, ``isinstance``, ``int``, ``tuple``, ``list``, ``type``,
    ``all``, ``any``, ``range``, ``set``, ``abs``, ``min``, ``max``,
    ``broadcast_shapes``, ``is_broadcastable_to``, ``dim_range_validity``,
    ``dim_uniqueness``, ``reduced_axes``) so R11 / R11a-style rules that
    use these helpers can be evaluated against the mock context instead
    of being silently skipped.

    The context names (inputs / outputs / params) are injected into both
    eval globals and locals. Comprehensions (generator / set / list /
    dict) create their own enclosing scope at compile time that only
    sees the eval globals, not the locals dict; passing ctx as globals
    too lets rules like ``all(d % x.ndim in ... for d in dim)`` resolve
    ``x`` and ``dim`` inside the generator expression.
    """
    # Defense-in-depth: even though manifest content is trusted (PR review
    # gates it), parse the rule first and reject any dunder attribute
    # access. This closes the classic ``().__class__.__mro__[1].
    # __subclasses__()`` sandbox-escape against the restricted builtins.
    try:
        tree = ast.parse(rule, mode="eval")
    except SyntaxError as exc:
        return False, f"eval error: SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr.endswith("__")
        ):
            return False, (
                f"eval error: dunder attribute access not permitted "
                f"({node.attr!r})"
            )

    eval_globals = {"__builtins__": _SHAPE_RULE_BUILTINS}
    # Ctx names must be visible inside comprehensions, which only see
    # globals. Merge ctx into globals while keeping locals=ctx so plain
    # (non-comprehension) lookups behave identically.
    eval_globals.update(ctx)
    # Defense-in-depth: a manifest identifier literally named
    # ``__builtins__`` (via a rule context key) would otherwise overwrite
    # the sandboxed builtins mapping installed above and re-expose the
    # full unrestricted builtins set. Reinstate the sandbox after the
    # update so ctx cannot escape it.
    eval_globals["__builtins__"] = _SHAPE_RULE_BUILTINS
    try:
        result = eval(  # noqa: S307 — manifest-controlled
            rule, eval_globals, ctx,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"eval error: {exc.__class__.__name__}: {exc}"
    try:
        return bool(result), None
    except Exception as exc:  # noqa: BLE001
        return False, f"non-boolean result: {exc}"


def _build_mock_self(
    cls: type,
    param_defaults: dict,
    extra_attrs: dict | None = None,
) -> object:
    """Build a mock ``self`` instance without running ``__init__``.

    Uses ``cls.__new__(cls)`` so that methods bound to ``cls`` (and any
    helpers defined on its MRO) remain accessible as ``self.method(...)``
    calls — a plain :class:`types.SimpleNamespace` cannot satisfy methods
    that read attributes defined on the class or reach for class helpers
    during a parity probe.

    ``param_defaults`` is the manifest-derived params map (from
    ``signature.params``). Each default is installed as an instance
    attribute so ``self.<param>`` lookups resolve without running any
    initialization logic.

    ``extra_attrs`` carries additional manifest-derived attributes to
    install after the params — typically ``static_dims`` values resolved
    from the synthetic mock inputs (so methods reading ``self.<N>``
    where ``N`` is a static dim see a concrete size) and the dtype axis
    (so ``self.dtype`` reflects the candidate combo instead of the
    inherited ``Op.dtype = None`` base-class default). Entries in
    ``extra_attrs`` override same-named entries in ``param_defaults``
    because they are specific to the current parity probe context.

    Falls back to :class:`types.SimpleNamespace` if ``cls.__new__``
    raises (defensive; Python ``type`` subclasses can override ``__new__``
    with required positional arguments).
    """
    merged: dict = dict(param_defaults)
    if extra_attrs:
        merged.update(extra_attrs)
    try:
        instance = cls.__new__(cls)
    except Exception:  # noqa: BLE001
        return types.SimpleNamespace(**merged)
    for k, v in merged.items():
        # __slots__ or read-only descriptors may reject setattr; ignore
        # — parity check will surface any resulting AttributeError as a
        # skip when the target method actually reads ``self.<k>``.
        with contextlib.suppress(AttributeError, TypeError):
            setattr(instance, k, v)
    return instance


def check_l2_infer_parity(
    op_name: str,
    entry: dict,
    cls: type | None,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """L2 extension: ``_infer_output_shapes`` parity with ``shape_rules``.

    Calls the Op class's ``_infer_output_shapes`` with concrete mock input
    shapes (no tensor allocation, no kernel execution). Plugs the result
    into a shape_rules evaluation context and verifies every rule holds.

    Behaviour:
      - For implemented ops whose class does not override
        ``_infer_output_shapes``, emits a warning reporting the missing
        manifest-derived method. The parity check itself is skipped
        because there is no concrete method to compare against, but the
        gap is surfaced (no silent pass).
      - Honors ``parity_opt_out: [shape_parity]`` (or a bare
        ``parity_opt_out: true``) declared in the manifest entry — used
        for documented GPU-only cases where the method cannot be called
        outside a GPU context. Opt-out suppresses the missing-method
        warning entirely.
      - An exception raised from the body of ``_infer_output_shapes``
        (i.e. after argument binding succeeds) is a hard L2 error unless
        the manifest entry declares ``parity_opt_out: [shape_parity]``
        (or a bare ``parity_opt_out: true``). This is a policy change
        from earlier revisions, which downgraded body exceptions to
        warnings and let real bugs pass silently. Opt-out is reserved
        for documented GPU-only methods that cannot be exercised outside
        a GPU context; an introspection-level failure (signature binding
        mismatch) is still reported separately as a signature error.
      - Produces L2 errors for concrete disagreement: the method returns
        shapes that fail one or more ``shape_rules`` or disagree with a
        declared ``signature.outputs[*].shape``.
    """
    errors: list[str] = []
    if cls is None:
        return errors

    sig = entry.get("signature", {})
    rules = sig.get("shape_rules") or []
    if not isinstance(rules, list):
        rules = []
    outputs_map = sig.get("outputs") or {}
    declared_output_shapes: dict[str, list[str]] = {}
    if isinstance(outputs_map, dict):
        for oname, oattrs in outputs_map.items():
            if not isinstance(oattrs, dict):
                continue
            parts = _parse_shape_decl(oattrs.get("shape", ""))
            if parts is not None:
                declared_output_shapes[oname] = parts
    # Nothing to check: neither rules nor declared output shapes.
    if not rules and not declared_output_shapes:
        return errors

    if not _class_overrides_method(cls, "_infer_output_shapes"):
        if not _parity_opted_out(entry, "shape_parity") and warnings is not None:
            warnings.append(
                f"[shape] {op_name}: class does not override "
                f"_infer_output_shapes — manifest-derived method not yet "
                f"generated; parity check skipped. Declare "
                f"'parity_opt_out: [shape_parity]' on the manifest entry "
                f"to suppress this warning for documented GPU-only ops."
            )
        return errors

    infer_fn = getattr(cls, "_infer_output_shapes", None)
    if infer_fn is None:
        return errors

    mock = _mock_input_shapes(sig)
    if mock is None:
        return errors
    mock_shapes, dim_sizes = mock

    params = sig.get("params") or {}
    param_defaults = _param_defaults(params)

    # Build a mock ``self`` via ``cls.__new__(cls)`` so class-defined
    # helpers and attribute descriptors remain reachable, then install
    # manifest-derived params as instance attributes without running
    # __init__. A plain SimpleNamespace would raise AttributeError when
    # _infer_output_shapes consults an unrelated ``self.<attr>`` helper.
    #
    # Generated ``_infer_output_shapes`` implementations commonly consult
    # static_dims via ``self.<dim>`` (e.g. ``self.N`` for
    # ``static_dims: {N: x.shape[-1]}``). Resolve them against the
    # synthetic mock inputs so the parity call does not raise a spurious
    # ``AttributeError`` and skip the check.
    extra_attrs = _static_dim_values(sig, mock_shapes, param_defaults)
    mock_self = _build_mock_self(cls, param_defaults, extra_attrs)

    shape_kwargs = {f"{name}_shape": tuple(shape) for name, shape in mock_shapes.items()}
    # First, validate the callable signature independently of the body: a
    # TypeError from inspect.signature().bind is a genuine signature mismatch
    # between the expected ``<input>_shape=`` kwargs and _infer_output_shapes.
    # TypeErrors raised inside the body (e.g. arithmetic on None) must not be
    # misreported as signature mismatch.
    try:
        inspect.signature(infer_fn).bind(mock_self, **shape_kwargs)
    except TypeError as exc:
        errors.append(
            f"[shape] {op_name}: _infer_output_shapes signature does not match "
            f"manifest inputs (expected kwargs {sorted(shape_kwargs)}): {exc}"
        )
        return errors
    except Exception as exc:  # noqa: BLE001
        # signature() itself failed (e.g. builtin without introspection) —
        # skip parity rather than fabricating a signature error.
        if warnings is not None:
            warnings.append(
                f"[shape] {op_name}: _infer_output_shapes parity skipped — "
                f"inspect.signature raised {exc.__class__.__name__}: {exc}"
            )
        return errors

    try:
        result = infer_fn(mock_self, **shape_kwargs)
    except Exception as exc:  # noqa: BLE001
        # Signature is valid but the body raised. This is a genuine
        # implementation bug: a correct manifest-derived
        # ``_infer_output_shapes`` must succeed on manifest-compatible
        # mock inputs. Surface as a hard L2 parity error unless the
        # entry explicitly opts out via ``parity_opt_out: [shape_parity]``
        # (documented GPU-only cases). Previously downgraded to a
        # warning — that let real bugs pass L2.
        if _parity_opted_out(entry, "shape_parity"):
            if warnings is not None:
                warnings.append(
                    f"[shape] {op_name}: _infer_output_shapes parity "
                    f"skipped (opt-out) — call raised "
                    f"{exc.__class__.__name__}: {exc}"
                )
            return errors
        errors.append(
            f"[shape] {op_name}: _infer_output_shapes raised "
            f"{exc.__class__.__name__} under mock inputs "
            f"{shape_kwargs}: {exc}"
        )
        return errors

    if not isinstance(result, dict):
        errors.append(
            f"[shape] {op_name}: _infer_output_shapes must return a dict "
            f"(output_name -> shape), got {type(result).__name__}"
        )
        return errors

    outputs = sig.get("outputs") or {}
    for out_name in outputs:
        if out_name not in result:
            errors.append(
                f"[shape] {op_name}: _infer_output_shapes missing output "
                f"{out_name!r} (declared in manifest)"
            )

    # Assemble evaluation context: symbolic dims + inputs + outputs +
    # params. Symbolic dim names (e.g. B, S, H, D extracted from literal
    # shape_rules like ``q.shape == (B, S, H, D)``) are bound first so
    # param / tensor names later in the dict take precedence on any
    # accidental collision.
    ctx: dict = {}
    ctx.update(dim_sizes)
    ctx.update(param_defaults)
    for name, shape in mock_shapes.items():
        ctx[name] = _MockShape(shape)
    # Identify output-only symbols (symbols that appear only in declared
    # output shapes, not in any input shape). Their concrete sizes are
    # derived by ``_infer_output_shapes`` (possibly via a ``shape_rules``
    # formula such as ``L_out == L_in - kW + 1``). For classification
    # and rule evaluation, rebind these from the inferred ``result`` so
    # a rule defining them is checked against the actual computed value
    # rather than a synthetic mock size — otherwise a wrong
    # _infer_output_shapes would silently pass, because the synthetic
    # size pre-bound in ``dim_sizes`` would make the rule fail in both
    # ctx and input_only_ctx and be misclassified as an input-only
    # precondition to skip.
    input_bound = _input_bound_symbols(sig)
    output_only_symbols: set[str] = set()
    # Rebind output-only symbols from the inferred ``result`` tuple
    # positions. When the same symbol appears in multiple output
    # positions that yield differing sizes, prefer the first and leave
    # the consistency check below to flag the mismatch.
    output_only_rebindings: dict[str, int] = {}
    for out_name, decl_parts in declared_output_shapes.items():
        for p in decl_parts:
            if p not in input_bound:
                output_only_symbols.add(p)
        if out_name not in result:
            continue
        try:
            inferred_tuple = tuple(result[out_name])
        except TypeError:
            continue
        if len(inferred_tuple) != len(decl_parts):
            continue
        for p, got in zip(decl_parts, inferred_tuple, strict=True):
            if p in input_bound:
                continue
            if not isinstance(got, int):
                continue
            if p not in output_only_rebindings:
                output_only_rebindings[p] = got
    # Apply output-only rebindings before rule evaluation. These replace
    # the synthetic sizes that ``_mock_input_shapes`` seeded via the
    # declared-output-shape pass in ``dim_sizes``.
    for p, v in output_only_rebindings.items():
        ctx[p] = v
    # Input-only context (no inferred outputs, no output-only symbols)
    # is used to detect rules that already fail on the mock inputs
    # themselves — such rules encode input-only preconditions (e.g.
    # ``weight.shape == (x.shape[dim],)``) that mock inputs may violate.
    # A correct ``_infer_output_shapes`` must not be blamed in that
    # case. Strip output-only symbols so a rule like
    # ``L_out == L_in - kW + 1`` is never reachable via this path (it is
    # output-dependent by construction).
    input_only_ctx: dict = {
        k: v for k, v in ctx.items() if k not in output_only_symbols
    }
    for out_name, out_shape in result.items():
        try:
            ctx[out_name] = _MockShape(tuple(out_shape))
        except TypeError:
            errors.append(
                f"[shape] {op_name}: _infer_output_shapes returned "
                f"non-iterable shape for {out_name!r}: {out_shape!r}"
            )

    output_names = set(result.keys()) | set(outputs.keys())
    for i, rule in enumerate(rules):
        if not isinstance(rule, str):
            continue
        ok, reason = _eval_shape_rule(rule, ctx)
        if reason is not None:
            # Could not evaluate this rule under the mock context; do not
            # flag as parity mismatch.
            if warnings is not None:
                warnings.append(
                    f"[shape] {op_name}: shape_rules[{i}] could not be "
                    f"evaluated against mock inputs ({reason}); rule: {rule!r}"
                )
            continue
        if not ok:
            # Distinguish a genuine parity mismatch from a mock-input
            # precondition violation: if the rule already fails with
            # inputs only (and does not reference any declared output
            # tensor name *or* any output-only symbol), the mock input
            # shapes themselves violate the rule — skip with a warning
            # instead of blaming _infer_output_shapes.
            mentions_output = any(
                re.search(rf"\b{re.escape(o)}\b", rule) for o in output_names
            ) or any(
                re.search(rf"\b{re.escape(s)}\b", rule)
                for s in output_only_symbols
            )
            if not mentions_output:
                ok_inputs, reason_inputs = _eval_shape_rule(
                    rule, input_only_ctx,
                )
                if reason_inputs is None and not ok_inputs:
                    if warnings is not None:
                        warnings.append(
                            f"[shape] {op_name}: shape_rules[{i}] "
                            f"{rule!r} not satisfied by synthetic mock "
                            f"inputs {shape_kwargs}; parity check "
                            f"skipped (input-only precondition)"
                        )
                    continue
            errors.append(
                f"[shape] {op_name}: _infer_output_shapes output violates "
                f"shape_rules[{i}] {rule!r} under mock inputs "
                f"{shape_kwargs} -> {result}"
            )

    # Compare inferred outputs against per-tensor declared shapes in
    # signature.outputs[*].shape, independently of shape_rules. This
    # catches ops whose outputs are only specified via declared shape
    # fields (no equivalent shape_rule).
    #
    # Only symbols **bound by input shapes** (input shape_rules literals
    # or signature.inputs[*].shape declarations) carry a concrete mock
    # size that ``_infer_output_shapes`` is expected to echo back.
    # Output-only symbols (e.g. ``L_out`` in a conv output shape that is
    # derived by a ``shape_rules`` entry such as ``L_out = L_in - kW + 1``)
    # cannot be meaningfully compared against an arbitrary
    # ``dim_sizes`` entry — doing so would flag a correct
    # implementation, since ``_infer_output_shapes`` computes the real
    # post-conv length, not the synthetic size assigned to ``L_out``.
    # For such symbols we enforce rank + per-symbol consistency instead
    # (same symbol must map to the same concrete size across every
    # position it appears in any declared output shape).
    # ``input_bound`` is already computed above (reused for the
    # output-only rebinding pass before rule evaluation).
    #
    # Static-dim resolution: symbols declared in ``signature.static_dims``
    # (e.g. ``static_dims: {N: "x.shape[-1]"}``) are resolved to concrete
    # integer sizes against the mock inputs via ``_static_dim_values``
    # (already stored in ``extra_attrs`` above). Treat those as pinned
    # expected sizes for the declared-output-shape comparison — a bad
    # ``_infer_output_shapes`` returning arbitrary integers for a
    # static-dim position must be caught, not mistakenly reclassified as
    # an output-only symbol with only rank/consistency enforcement.
    static_expected: dict[str, int] = {
        name: int(val) for name, val in extra_attrs.items()
        if isinstance(val, int) and not isinstance(val, bool)
    }
    # Params with a concrete integer ``default`` are also compile-time
    # known and should pin declared-output-shape dims with the same
    # authority as ``static_dims``. Without this, a declared output
    # ``shape: "[k]"`` where ``k`` is a param default would be
    # classified as an output-only symbol, and a bad
    # ``_infer_output_shapes`` returning an arbitrary integer for that
    # position would only trip rank/consistency checks rather than exact
    # value comparison. Edge cases: params without a default (supplied
    # at op construction, unknown to the validator) are skipped; params
    # whose default is not a single ``int`` (e.g. ``list[int]``) are
    # skipped so they cannot pin a scalar dim position.
    for pname, pdefault in param_defaults.items():
        if pname in static_expected:
            continue  # static_dims wins — it is the declared source of truth.
        if isinstance(pdefault, bool):
            continue
        if isinstance(pdefault, int):
            static_expected[pname] = int(pdefault)
    output_only_seen: dict[str, int] = {}
    for out_name, decl_parts in declared_output_shapes.items():
        if out_name not in result:
            continue
        try:
            inferred = tuple(result[out_name])
        except TypeError:
            continue  # already reported above
        if len(inferred) != len(decl_parts):
            errors.append(
                f"[shape] {op_name}: _infer_output_shapes output "
                f"{out_name!r} rank {len(inferred)} disagrees with "
                f"declared shape {decl_parts} (rank {len(decl_parts)}) "
                f"under mock inputs {shape_kwargs} -> {inferred}"
            )
            continue
        for idx, (p, got) in enumerate(zip(decl_parts, inferred, strict=True)):
            if p in input_bound or p in static_expected:
                # Input-bound or static-dim symbol: concrete size is
                # pinned by mock inputs (or by the static_dims
                # expression resolved against them) and must match
                # exactly.
                expected = static_expected.get(
                    p, dim_sizes.get(p, _MOCK_DIM_SIZE)
                )
                if got != expected:
                    errors.append(
                        f"[shape] {op_name}: _infer_output_shapes output "
                        f"{out_name!r} dim[{idx}]={got} disagrees with "
                        f"declared {p}={expected} under mock inputs "
                        f"{shape_kwargs} -> {inferred}"
                    )
            else:
                # Output-only symbol: value is derived by
                # _infer_output_shapes (and possibly a shape_rules
                # formula). Only enforce consistency — the same symbol
                # must resolve to the same concrete size everywhere it
                # appears across all declared outputs.
                prev = output_only_seen.get(p)
                if prev is None:
                    output_only_seen[p] = got
                elif prev != got:
                    errors.append(
                        f"[shape] {op_name}: _infer_output_shapes output "
                        f"{out_name!r} binds output-only symbol {p!r} to "
                        f"{got} but earlier output bound it to {prev} "
                        f"(inconsistent under mock inputs {shape_kwargs})"
                    )
    return errors


# ---------------------------------------------------------------------------
# dtype parity: _validate_dtypes vs dtype_combos / dtype unions (L3 extension)
# ---------------------------------------------------------------------------


def _dtype_options_for_tensor(
    tname: str, dtype_str: str, resolved: dict[str, list[str]],
) -> list[str] | None:
    """Expand a dtype expression into concrete torch dtype names.

    ``same_as(ref)`` resolves to whatever *ref* has already been resolved
    to in the *resolved* map. Declaration order is irrelevant: callers
    (``_resolve_tensor_dtype_options``) iterate to a fixpoint, retrying
    tensors whose ``same_as(ref)`` targets unresolved refs until every
    tensor resolves or no progress is made. Returns None when the
    expression cannot be resolved in the current pass (caller decides
    whether that is a temporary state inside the fixpoint loop or a
    permanent failure).
    """
    tokens = _parse_dtype_expr(dtype_str)
    # Pure same_as(ref): inherits ref's options.
    if len(tokens) == 1:
        m = _SAME_AS_RE.match(tokens[0])
        if m:
            ref = m.group(1)
            if ref not in resolved:
                # Unresolved reference — propagate failure per docstring
                # contract. Returning [] here would silently disable parity.
                return None
            return list(resolved[ref])
    out: list[str] = []
    for tok in tokens:
        m = _SAME_AS_RE.match(tok)
        if m:
            ref = m.group(1)
            if ref not in resolved:
                return None
            out.extend(resolved[ref])
        elif tok in _TORCH_DTYPES:
            out.append(tok)
        else:
            return None
    # De-dup preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _resolve_tensor_dtype_options(
    sig: dict,
) -> dict[str, list[str]] | None:
    """Return dtype options for every declared tensor (inputs + outputs).

    Resolves ``same_as`` references to a fixpoint: declaration order is
    irrelevant, so ``x: same_as(y)`` declared before ``y: float16`` still
    resolves. Returns None only if some tensor's expression is genuinely
    unresolvable (unknown token, dangling ``same_as`` reference, or a
    ``same_as`` cycle).
    """
    # Collect every tensor's raw dtype string first, so iteration order
    # cannot affect the result.
    pending: dict[str, str] = {}
    for group in ("inputs", "outputs"):
        tensors = sig.get(group) or {}
        if not isinstance(tensors, dict):
            continue
        for tname, attrs in tensors.items():
            if not isinstance(attrs, dict):
                return None
            pending[tname] = attrs.get("dtype", "")

    resolved: dict[str, list[str]] = {}
    # Iterate to fixpoint: each pass resolves every tensor whose
    # dependencies are already known. Bound the loop by len(pending) + 1
    # — any longer progression implies a cycle (no new resolutions).
    for _ in range(len(pending) + 1):
        made_progress = False
        for tname, dtype_str in list(pending.items()):
            opts = _dtype_options_for_tensor(tname, dtype_str, resolved)
            if opts is None:
                continue
            resolved[tname] = opts
            del pending[tname]
            made_progress = True
        if not pending:
            return resolved
        if not made_progress:
            # Remaining tensors reference something unresolvable (unknown
            # dtype name, dangling ref, or a same_as cycle). Propagate
            # failure per docstring contract.
            return None
    return resolved if not pending else None


def _primary_dtype_input(
    sig: dict, forward_inputs: list[str],
) -> str | None:
    """Return the first input whose dtype is not bound by ``same_as(ref)``.

    The returned input is used to stamp ``self.dtype`` on the mock self
    for dtype parity. Same_as-bound inputs are skipped because their
    dtype is derivative: their dtype follows ``ref``, and
    manifest-derived ``_validate_dtypes`` implementations typically
    compare the op's ``self.dtype`` against the unbound primary input.
    """
    inputs = sig.get("inputs") or {}
    if not isinstance(inputs, dict):
        return None
    for name in forward_inputs:
        attrs = inputs.get(name)
        if not isinstance(attrs, dict):
            continue
        dstr = attrs.get("dtype", "")
        tokens = _parse_dtype_expr(dstr)
        if len(tokens) == 1 and _SAME_AS_RE.match(tokens[0]):
            continue
        return name
    # Fallback: no fully-free input; use the first declared input even
    # if it's same_as-bound, so ``self.dtype`` is at least non-None.
    return forward_inputs[0] if forward_inputs else None


def _make_mock_tensor(dtype_name: str):
    """Build a 0-sized torch tensor of the named dtype (CPU).

    Uses 0 elements so allocation is cheap and no GPU is touched.
    """
    import torch
    torch_dtype = getattr(torch, dtype_name, None)
    if torch_dtype is None:
        return None
    try:
        return torch.empty(0, dtype=torch_dtype, device="cpu")
    except (RuntimeError, TypeError):
        return None


def _combo_accepted(
    cls: type, forward_inputs: list[str], combo: dict[str, str],
    param_defaults: dict, sig: dict | None = None,
    self_dtype_name: str | None = None,
) -> tuple[bool, str | None]:
    """Invoke ``cls._validate_dtypes`` on a mock-self with *combo*.

    Returns (accepted, error_reason). ``accepted=False`` with
    reason=None means the op raised during validation (rejected);
    reason!=None indicates the call could not be performed (skip).

    ``sig`` (optional) is the manifest signature; when provided, the
    mock-self is enriched with static_dims values resolved against
    synthetic mock input shapes and with ``self.dtype`` bound to the
    candidate's dtype axis (see below). Both attributes are commonly
    consulted by generated ``_validate_dtypes`` implementations (e.g.
    ``if x.dtype != self.dtype: raise``); without them the parity
    probe would spuriously reject listed combos.

    ``self_dtype_name`` (optional) pins the dtype installed on
    ``mock_self.dtype``. Used by out-of-union probes to keep the op's
    configured dtype at a valid baseline while mutating the input
    tensor's dtype — otherwise ``self.dtype`` would follow the bad
    candidate and a ``x.dtype != self.dtype`` check would spuriously
    pass. When omitted, defaults to the combo entry for the first
    non-same_as-bound input (the listed-combo convention).
    """
    validate_fn = getattr(cls, "_validate_dtypes", None)
    if validate_fn is None:
        return False, "no _validate_dtypes"

    tensors: dict = {}
    for name in forward_inputs:
        dtype_name = combo.get(name)
        if dtype_name is None:
            return False, f"combo missing input {name!r}"
        t = _make_mock_tensor(dtype_name)
        if t is None:
            return False, f"cannot build mock tensor for dtype {dtype_name!r}"
        tensors[name] = t

    # Build mock self via ``cls.__new__(cls)`` so _validate_dtypes
    # methods that consult other class helpers or instance attributes
    # (beyond manifest params) do not falsely raise AttributeError.
    extra_attrs: dict = {}
    if sig is not None:
        mock = _mock_input_shapes(sig)
        if mock is not None:
            mock_shapes, _ = mock
            extra_attrs.update(
                _static_dim_values(sig, mock_shapes, param_defaults)
            )
        # Install self.dtype mirroring the manifest convention: the op's
        # dtype attribute tracks the candidate's primary dtype (first
        # non-same_as-bound input by default) unless an explicit
        # ``self_dtype_name`` override is supplied (out-of-union probes
        # pin the baseline valid dtype so only the input tensor's dtype
        # deviates). A manifest-derived _validate_dtypes that compares
        # ``x.dtype != self.dtype`` then sees a real torch.dtype instead
        # of the base-class ``None``.
        if self_dtype_name is not None:
            override_t = _make_mock_tensor(self_dtype_name)
            if override_t is not None:
                extra_attrs["dtype"] = override_t.dtype
        else:
            primary = _primary_dtype_input(sig, forward_inputs)
            if primary is not None and primary in tensors:
                extra_attrs["dtype"] = tensors[primary].dtype
    mock_self = _build_mock_self(cls, param_defaults, extra_attrs)
    # Pre-bind the callable signature so only genuine signature mismatches
    # surface as ``TypeError: ...``. TypeError raised from inside the body
    # (e.g. comparing incompatible torch dtypes) is a legitimate rejection
    # and must not be misreported as a signature mismatch.
    try:
        inspect.signature(validate_fn).bind(mock_self, **tensors)
    except TypeError as exc:
        return False, f"TypeError: {exc}"
    except Exception as exc:  # noqa: BLE001
        # inspect.signature itself failed to introspect — treat as a
        # validator-side skip (not an op-side bug). Tagged distinctly
        # from body-level unexpected exceptions so callers can enforce
        # policy differences.
        return False, f"introspect-failed {exc.__class__.__name__}: {exc}"

    try:
        validate_fn(mock_self, **tensors)
    except (ValueError, TypeError):
        # Body-level rejection: either an explicit ValueError or a
        # TypeError arising from dtype comparisons. Both are legitimate
        # rejections once the signature has been validated above.
        return False, None
    except Exception as exc:  # noqa: BLE001
        # Body raised a non-ValueError/TypeError exception. This is a
        # genuine implementation bug (a correct manifest-derived
        # ``_validate_dtypes`` must either accept or raise
        # ValueError/TypeError, never e.g. RuntimeError). Callers
        # enforce this as a hard L3 parity error unless the entry opts
        # out via ``parity_opt_out: [dtype_parity]``.
        return False, f"unexpected {exc.__class__.__name__}: {exc}"
    return True, None


def check_l3_validate_dtypes_parity(
    op_name: str,
    entry: dict,
    cls: type | None,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """L3 extension: ``_validate_dtypes`` parity with manifest dtypes.

    With ``dtype_combos`` declared: iterate all combos and verify the
    method accepts each listed combo and rejects at least one non-listed
    combination drawn from the same input dtype universe.

    Without ``dtype_combos``: verify every combination in the Cartesian
    product of each input's declared dtype union is accepted.

    For ops whose class does not override ``_validate_dtypes``, emits a
    warning reporting the missing manifest-derived method (no silent
    pass). Honors ``parity_opt_out: [dtype_parity]`` (or a bare
    ``parity_opt_out: true``) declared in the manifest entry to suppress
    the warning for documented GPU-only cases.
    """
    errors: list[str] = []
    if cls is None:
        return errors

    if not _class_overrides_method(cls, "_validate_dtypes"):
        if not _parity_opted_out(entry, "dtype_parity") and warnings is not None:
            warnings.append(
                f"[dtype] {op_name}: class does not override "
                f"_validate_dtypes — manifest-derived method not yet "
                f"generated; parity check skipped. Declare "
                f"'parity_opt_out: [dtype_parity]' on the manifest entry "
                f"to suppress this warning for documented GPU-only ops."
            )
        return errors

    sig = entry.get("signature", {})
    inputs = sig.get("inputs") or {}
    if not isinstance(inputs, dict) or not inputs:
        return errors

    # Only pass tensors corresponding to manifest inputs (forward args).
    forward_inputs = list(inputs.keys())
    params = sig.get("params") or {}
    param_defaults = _param_defaults(params)

    dtype_options = _resolve_tensor_dtype_options(sig)
    if dtype_options is None:
        # L3 dtype check will already have reported unresolved tokens.
        return errors

    dtype_combos = sig.get("dtype_combos")
    if isinstance(dtype_combos, list) and dtype_combos:
        # Combo-data validity: surface invalid entries as hard L3 errors
        # so downstream parity probing does not run on junk data (which
        # would otherwise produce a cascade of misleading "rejects" /
        # "skipped" diagnostics). The same check also runs
        # unconditionally in ``check_l3`` — the driver dedupes error
        # strings so users see each message once even when both entry
        # points are invoked in the same run.
        combo_validation_errors = check_l3_dtype_combos_data(op_name, sig)
        if combo_validation_errors:
            errors.extend(combo_validation_errors)
            return errors

        # Expand ``same_as(ref)`` in combo values to a concrete dtype
        # before parity probing: ``_combo_accepted`` / ``_make_mock_tensor``
        # expect literal torch dtype names and would otherwise try to look
        # up ``same_as(x)`` as a torch attribute. Per R3 + R4 we already
        # enforce identity (``_check_dtype_combos_same_as_identity``), so
        # each ``same_as(ref)`` value resolves to the same concrete dtype
        # the ref carries in the same combo row.
        expanded_combos: list[dict[str, str]] = []
        for combo in dtype_combos:
            if not isinstance(combo, dict):
                expanded_combos.append({})
                continue
            expanded: dict[str, str] = {}
            for key, val in combo.items():
                if isinstance(val, str):
                    m = _SAME_AS_RE.match(val.strip())
                    if m:
                        ref = m.group(1)
                        ref_val = combo.get(ref)
                        expanded[key] = ref_val if isinstance(ref_val, str) else val
                        continue
                expanded[key] = val
            expanded_combos.append(expanded)
        dtype_combos = expanded_combos

        # Each listed combo should be accepted.
        for i, combo in enumerate(dtype_combos):
            if not isinstance(combo, dict):
                continue
            accepted, reason = _combo_accepted(
                cls, forward_inputs, combo, param_defaults, sig=sig,
            )
            if reason and reason.startswith("TypeError"):
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes signature does "
                    f"not match manifest inputs (expected kwargs "
                    f"{sorted(forward_inputs)}): {reason}"
                )
                return errors
            if reason and reason.startswith("introspect-failed"):
                # Validator-side introspection failure (inspect.signature
                # could not parse the method). Not an op-side bug —
                # skip with warning.
                if warnings is not None:
                    warnings.append(
                        f"[dtype] {op_name}: _validate_dtypes parity skipped "
                        f"for dtype_combos[{i}] — {reason}"
                    )
                continue
            if reason and reason.startswith("unexpected"):
                # Body-level exception that is not ValueError / TypeError
                # — a real implementation bug. Surface as a hard L3
                # parity error unless the entry opts out. Previously
                # downgraded to warning, which let broken implementations
                # pass.
                if _parity_opted_out(entry, "dtype_parity"):
                    if warnings is not None:
                        warnings.append(
                            f"[dtype] {op_name}: _validate_dtypes parity "
                            f"skipped (opt-out) for dtype_combos[{i}] — "
                            f"{reason}"
                        )
                    continue
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes raised "
                    f"unexpected exception on dtype_combos[{i}] "
                    f"{combo!r} — {reason}"
                )
                continue
            if reason and reason.startswith("cannot build mock tensor"):
                # Validator limitation (no torch dtype for this name) — emit
                # a parity-skip warning rather than reporting a rejection.
                if warnings is not None:
                    warnings.append(
                        f"[dtype] {op_name}: _validate_dtypes parity skipped "
                        f"for dtype_combos[{i}] — {reason}"
                    )
                continue
            if reason and reason.startswith("combo missing input"):
                # Manifest error: combo doesn't specify a dtype for every
                # declared input. Surface as parity error but not a reject.
                errors.append(
                    f"[dtype] {op_name}: dtype_combos[{i}] {combo!r} "
                    f"{reason}"
                )
                continue
            if not accepted:
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes rejects "
                    f"dtype_combos[{i}] {combo!r} listed in manifest"
                )

        # Every non-listed combo drawn from the inputs' union must be
        # rejected. Enumerate the full Cartesian product and report any
        # non-listed combo that ``_validate_dtypes`` accepts. Breaking on
        # the first rejection would miss a later accepted combo.
        input_options: list[list[str]] = [
            dtype_options.get(name, []) for name in forward_inputs
        ]
        product_size = 1
        for opts in input_options:
            product_size *= max(len(opts), 1)
        if product_size > _MAX_DTYPE_COMBOS:
            if warnings is not None:
                warnings.append(
                    f"[dtype] {op_name}: Cartesian product of dtype "
                    f"options ({product_size}) exceeds "
                    f"_MAX_DTYPE_COMBOS={_MAX_DTYPE_COMBOS}; non-listed "
                    f"rejection check skipped "
                    f"({len(forward_inputs)} inputs × options "
                    f"{[len(o) for o in input_options]})"
                )
            return errors
        listed_combo_keys = {
            tuple(combo.get(n) for n in forward_inputs)
            for combo in dtype_combos if isinstance(combo, dict)
        }
        rejected_at_least_one = False
        checked_any = False
        for tup in itertools.product(*input_options):
            if tup in listed_combo_keys:
                continue
            candidate = dict(zip(forward_inputs, tup, strict=True))
            checked_any = True
            accepted, reason = _combo_accepted(
                cls, forward_inputs, candidate, param_defaults, sig=sig,
            )
            if reason and reason.startswith(
                ("introspect-failed", "TypeError")
            ):
                continue
            if reason and reason.startswith("unexpected"):
                # Body-level unexpected exception — hard error unless
                # opt-out (parity policy tightened).
                if _parity_opted_out(entry, "dtype_parity"):
                    continue
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes raised "
                    f"unexpected exception on non-listed combo "
                    f"{candidate!r} — {reason}"
                )
                continue
            if not accepted:
                rejected_at_least_one = True
                continue
            # Accepted non-listed combo — parity violation. Keep scanning
            # so multiple such combos are all surfaced in a single run.
            errors.append(
                f"[dtype] {op_name}: _validate_dtypes accepts non-listed "
                f"combo {candidate!r} (not in dtype_combos)"
            )

        # --- Out-of-union negative probe (AC-3 rejection side) ---------
        # Mirrors the probe in the no-dtype_combos branch. Picks a listed
        # combo as a baseline (known to be accepted) and substitutes an
        # out-of-union sentinel for each non-same_as-bound input in turn.
        # Each resulting candidate must be rejected. Bounded by
        # _MAX_DTYPE_COMBOS to preserve the Cartesian safety bound.
        baseline_combo: dict[str, str] | None = None
        for c in dtype_combos:
            if isinstance(c, dict) and all(
                n in c for n in forward_inputs
            ):
                baseline_combo = dict(c)
                break
        if baseline_combo is not None:
            same_as_refs = _same_as_refs(sig)
            # Baseline's primary dtype pins ``self.dtype`` during the
            # probe so only the input tensor's dtype deviates from the
            # op's configured dtype (otherwise the ``x.dtype !=
            # self.dtype`` check in a generated _validate_dtypes would
            # spuriously pass, since both sides track the bad dtype).
            baseline_primary = _primary_dtype_input(sig, forward_inputs)
            baseline_self_dtype = (
                baseline_combo.get(baseline_primary)
                if baseline_primary is not None else None
            )
            probe_budget = _MAX_DTYPE_COMBOS
            probed = 0
            for target in forward_inputs:
                # Skip same_as(ref)-bound tensors: their dtype is
                # controlled by ``ref``. Mutate the ref instead and let
                # same_as propagation carry the out-of-union dtype.
                if target in same_as_refs:
                    continue
                declared = set(dtype_options.get(target, []))
                out_of_union = _out_of_union_candidates(declared)
                if not out_of_union:
                    # Declared union covers the entire torch dtype
                    # universe for this input — no candidate exists to
                    # probe rejection. Emit a parity-skip warning naming
                    # the op/input so the gap is visible rather than a
                    # vacuous pass. Only possible when declared ==
                    # _TORCH_DTYPES (wildly permissive spec).
                    if warnings is not None:
                        warnings.append(
                            f"[dtype] {op_name}: out-of-union probe "
                            f"skipped for input {target!r} — declared "
                            f"dtype union covers the entire torch dtype "
                            f"set; rejection side cannot be exercised"
                        )
                    continue
                for bad_dtype in out_of_union:
                    if probed >= probe_budget:
                        break
                    probed += 1
                    candidate = dict(baseline_combo)
                    candidate[target] = bad_dtype
                    for tname, ref in same_as_refs.items():
                        if ref == target and tname in candidate:
                            candidate[tname] = bad_dtype
                    accepted, reason = _combo_accepted(
                        cls, forward_inputs, candidate, param_defaults,
                        sig=sig, self_dtype_name=baseline_self_dtype,
                    )
                    if reason and reason.startswith(
                        ("introspect-failed", "TypeError",
                         "cannot build", "combo missing")
                    ):
                        continue
                    if reason and reason.startswith("unexpected"):
                        if _parity_opted_out(entry, "dtype_parity"):
                            continue
                        errors.append(
                            f"[dtype] {op_name}: _validate_dtypes raised "
                            f"unexpected exception on out-of-union probe "
                            f"{candidate!r} — {reason}"
                        )
                        continue
                    if accepted:
                        errors.append(
                            f"[dtype] {op_name}: _validate_dtypes "
                            f"accepts out-of-union dtype "
                            f"{candidate!r} (input {target!r} declared "
                            f"{sorted(declared)})"
                        )
                if probed >= probe_budget:
                    break

        if not errors and warnings is not None:
            if not checked_any:
                # No non-listed combo exists in the Cartesian product —
                # dtype_combos already enumerates every reachable tuple.
                warnings.append(
                    f"[dtype] {op_name}: could not find a non-listed combo "
                    f"to exercise rejection (dtype_combos exhausts the "
                    f"union)"
                )
            elif not rejected_at_least_one:
                # Non-listed combos were tried but none were rejected —
                # either _validate_dtypes is too lax or every non-listed
                # candidate was skipped (unexpected/TypeError).
                warnings.append(
                    f"[dtype] {op_name}: no non-listed dtype combo was "
                    f"rejected by _validate_dtypes; parity coverage may be "
                    f"incomplete"
                )
    else:
        # No dtype_combos — verify every Cartesian combination is accepted.
        input_options = [
            dtype_options.get(name, []) for name in forward_inputs
        ]
        if not all(input_options):
            return errors
        product_size = 1
        for opts in input_options:
            product_size *= len(opts)
        if product_size > _MAX_DTYPE_COMBOS:
            if warnings is not None:
                warnings.append(
                    f"[dtype] {op_name}: Cartesian product of dtype "
                    f"options ({product_size}) exceeds "
                    f"_MAX_DTYPE_COMBOS={_MAX_DTYPE_COMBOS}; parity check "
                    f"skipped ({len(forward_inputs)} inputs × options "
                    f"{[len(o) for o in input_options]})"
                )
            return errors
        for tup in itertools.product(*input_options):
            # Only keep combos that honour same_as identity constraints:
            # when tensor T has dtype same_as(R), T and R must match.
            candidate = dict(zip(forward_inputs, tup, strict=True))
            if not _honours_same_as(sig, candidate):
                continue
            accepted, reason = _combo_accepted(
                cls, forward_inputs, candidate, param_defaults, sig=sig,
            )
            if reason and reason.startswith("TypeError"):
                # Signature mismatch between manifest inputs and the op's
                # _validate_dtypes — surface as a parity error (analogous
                # to the L2 _infer_output_shapes signature check).
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes signature does "
                    f"not match manifest inputs (expected kwargs "
                    f"{sorted(forward_inputs)}): {reason}"
                )
                return errors
            if reason and reason.startswith("introspect-failed"):
                if warnings is not None:
                    warnings.append(
                        f"[dtype] {op_name}: _validate_dtypes parity skipped "
                        f"for combo {candidate!r} — {reason}"
                    )
                continue
            if reason and reason.startswith("unexpected"):
                # Body-level unexpected exception — hard error unless
                # opt-out. See ``_combo_accepted`` docstring.
                if _parity_opted_out(entry, "dtype_parity"):
                    if warnings is not None:
                        warnings.append(
                            f"[dtype] {op_name}: _validate_dtypes parity "
                            f"skipped (opt-out) for combo {candidate!r} — "
                            f"{reason}"
                        )
                    continue
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes raised "
                    f"unexpected exception on combo {candidate!r} — "
                    f"{reason}"
                )
                continue
            if not accepted:
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes rejects valid "
                    f"combo {candidate!r} drawn from manifest dtype unions"
                )

        # --- Out-of-union negative probe (AC-3 rejection side) ---------
        # For each input, substitute one dtype outside its declared union
        # and assert _validate_dtypes rejects it. Candidates are built on
        # a same_as-honouring baseline so the only deviation is the
        # out-of-union dtype. Bounded by _MAX_DTYPE_COMBOS.
        baseline: dict[str, str] | None = None
        for tup in itertools.product(*input_options):
            cand = dict(zip(forward_inputs, tup, strict=True))
            if _honours_same_as(sig, cand):
                baseline = cand
                break
        if baseline is not None:
            same_as_refs = _same_as_refs(sig)
            # Keep ``self.dtype`` pinned to the baseline's primary valid
            # dtype during out-of-union probes — see the dtype_combos
            # branch above for rationale.
            baseline_primary = _primary_dtype_input(sig, forward_inputs)
            baseline_self_dtype = (
                baseline.get(baseline_primary)
                if baseline_primary is not None else None
            )
            probe_budget = _MAX_DTYPE_COMBOS
            probed = 0
            for target in forward_inputs:
                # Don't directly mutate tensors that are bound by
                # same_as(ref) — their dtype is controlled by ``ref``.
                # Instead mutate the ref (or a free tensor) and let
                # same_as propagation carry the out-of-union dtype.
                if target in same_as_refs:
                    continue
                declared = set(dtype_options.get(target, []))
                out_of_union = _out_of_union_candidates(declared)
                if not out_of_union:
                    # Declared union covers every torch dtype — cannot
                    # produce a rejection candidate. Skip with a warning
                    # rather than vacuously pass.
                    if warnings is not None:
                        warnings.append(
                            f"[dtype] {op_name}: out-of-union probe "
                            f"skipped for input {target!r} — declared "
                            f"dtype union covers the entire torch dtype "
                            f"set; rejection side cannot be exercised"
                        )
                    continue
                for bad_dtype in out_of_union:
                    if probed >= probe_budget:
                        break
                    probed += 1
                    candidate = dict(baseline)
                    candidate[target] = bad_dtype
                    # Propagate to all same_as(target) tensors so the
                    # only manifest violation is the out-of-union dtype.
                    for tname, ref in same_as_refs.items():
                        if ref == target and tname in candidate:
                            candidate[tname] = bad_dtype
                    accepted, reason = _combo_accepted(
                        cls, forward_inputs, candidate, param_defaults,
                        sig=sig, self_dtype_name=baseline_self_dtype,
                    )
                    if reason and reason.startswith(
                        ("introspect-failed", "TypeError")
                    ):
                        continue
                    if reason and reason.startswith("unexpected"):
                        if _parity_opted_out(entry, "dtype_parity"):
                            continue
                        errors.append(
                            f"[dtype] {op_name}: _validate_dtypes raised "
                            f"unexpected exception on out-of-union probe "
                            f"{candidate!r} — {reason}"
                        )
                        continue
                    if accepted:
                        errors.append(
                            f"[dtype] {op_name}: _validate_dtypes "
                            f"accepts out-of-union dtype "
                            f"{candidate!r} (input {target!r} declared "
                            f"{sorted(declared)})"
                        )
                if probed >= probe_budget:
                    break

        # --- same_as identity negative probe (R3 rejection side) -------
        # For each same_as(ref) input, build a candidate where that
        # tensor's dtype differs from its ref and assert rejection.
        # Complements (does not replace) the ``_honours_same_as`` skip
        # in the union-iteration loop above.
        if baseline is not None:
            same_as_refs = _same_as_refs(sig)
            probed_same_as = 0
            for tname, ref in same_as_refs.items():
                if probed_same_as >= _MAX_DTYPE_COMBOS:
                    break
                if tname not in baseline or ref not in baseline:
                    continue
                ref_dtype = baseline[ref]
                # Pick any dtype different from the ref. Prefer values in
                # the tensor's own declared options (so a pure same_as
                # check is the only violation); fall back to sentinels.
                own_opts = dtype_options.get(tname, [])
                alt_dtypes = [d for d in own_opts if d != ref_dtype]
                if not alt_dtypes:
                    alt_dtypes = [
                        d for d in _DTYPE_SENTINELS if d != ref_dtype
                    ]
                for alt in alt_dtypes[:1]:  # one probe per same_as edge
                    probed_same_as += 1
                    candidate = dict(baseline)
                    candidate[tname] = alt
                    accepted, reason = _combo_accepted(
                        cls, forward_inputs, candidate, param_defaults, sig=sig,
                    )
                    if reason and reason.startswith(
                        ("introspect-failed", "TypeError")
                    ):
                        continue
                    if reason and reason.startswith("unexpected"):
                        if _parity_opted_out(entry, "dtype_parity"):
                            continue
                        errors.append(
                            f"[dtype] {op_name}: _validate_dtypes raised "
                            f"unexpected exception on same_as probe "
                            f"{candidate!r} — {reason}"
                        )
                        continue
                    if accepted:
                        errors.append(
                            f"[dtype] {op_name}: _validate_dtypes "
                            f"accepts same_as violation {candidate!r} "
                            f"(input {tname!r} declared same_as({ref}))"
                        )
    return errors


def _same_as_refs(sig: dict) -> dict[str, str]:
    """Return ``{tensor: ref}`` for every pure ``same_as(ref)`` input.

    Used by the negative-probe pass in
    :func:`check_l3_validate_dtypes_parity` to identify edges that must
    be exercised against a mismatched dtype and to propagate out-of-union
    substitutions to dependent tensors.
    """
    refs: dict[str, str] = {}
    inputs = sig.get("inputs") or {}
    if not isinstance(inputs, dict):
        return refs
    for tname, attrs in inputs.items():
        if not isinstance(attrs, dict):
            continue
        dstr = attrs.get("dtype", "")
        tokens = _parse_dtype_expr(dstr)
        if len(tokens) == 1:
            m = _SAME_AS_RE.match(tokens[0])
            if m:
                refs[tname] = m.group(1)
    return refs


def _honours_same_as(sig: dict, candidate: dict[str, str]) -> bool:
    """Return True when *candidate* satisfies same_as identity (R3)."""
    inputs = sig.get("inputs") or {}
    if not isinstance(inputs, dict):
        return True
    for tname, attrs in inputs.items():
        if not isinstance(attrs, dict):
            continue
        dstr = attrs.get("dtype", "")
        tokens = _parse_dtype_expr(dstr)
        if len(tokens) == 1:
            m = _SAME_AS_RE.match(tokens[0])
            if m:
                ref = m.group(1)
                if ref in candidate and candidate.get(tname) != candidate[ref]:
                    return False
    return True


# ---------------------------------------------------------------------------
# bench: benchmark file uses manifest workloads
# ---------------------------------------------------------------------------

def _resolve_constant_str_bindings(tree: ast.Module) -> dict[str, str]:
    """Collect simple module-level string constants: NAME = 'value'."""
    bindings: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            bindings[target.id] = node.value.value
    return bindings


def _call_uses_expected_op_name(
    call: ast.Call, expected_op_name: str, bindings: dict[str, str],
) -> bool:
    """Return True when call(arg0, ...) uses the expected op name."""
    if not call.args:
        return False
    first_arg = call.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value == expected_op_name
    if isinstance(first_arg, ast.Name):
        return bindings.get(first_arg.id) == expected_op_name
    return False


def _ast_manifest_call_usage(
    tree: ast.Module,
    op_name: str,
    target_names: set[str],
) -> dict[str, bool]:
    """Check whether target functions are imported and called with this op name.

    Recognises three patterns:

    1. **Direct** — ``from tileops.manifest import load_workloads`` called
       with the op name and ``op.eval_roofline()`` called on an Op instance.
    2. **Indirect via benchmarks.benchmark_base** — ``workloads_to_params``
       (wraps ``load_workloads``) and ``ManifestBenchmark`` (wraps
       op-local ``eval_roofline``) imported from ``benchmarks.benchmark_base``
       and called with the op name as the first argument.
    """
    # Maps from the indirect helper name → the direct target it satisfies.
    _INDIRECT_EQUIV: dict[str, str] = {
        "workloads_to_params": "load_workloads",
        "ManifestBenchmark": "eval_roofline",
    }

    imported: set[str] = set()
    matched_calls: set[str] = set()
    bindings = _resolve_constant_str_bindings(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "tileops.manifest" and node.names:
                for alias in node.names:
                    if alias.name in target_names:
                        imported.add(alias.name)
            # Indirect helpers live in benchmarks.benchmark_base.
            if node.module == "benchmarks.benchmark_base" and node.names:
                for alias in node.names:
                    equiv = _INDIRECT_EQUIV.get(alias.name)
                    if equiv and equiv in target_names:
                        imported.add(equiv)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Direct call (load_workloads).
            if func_name in target_names and _call_uses_expected_op_name(
                node, op_name, bindings,
            ):
                matched_calls.add(func_name)
            # Indirect call (workloads_to_params / ManifestBenchmark).
            equiv = _INDIRECT_EQUIV.get(func_name)
            if equiv and equiv in target_names and _call_uses_expected_op_name(
                node, op_name, bindings,
            ):
                matched_calls.add(equiv)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "eval_roofline"
            and "eval_roofline" in target_names
        ):
            imported.add("eval_roofline")
            matched_calls.add("eval_roofline")
    return {name: (name in imported and name in matched_calls) for name in target_names}


def check_l4_benchmark(
    op_name: str, bench_path: str, repo_root: Path,
) -> list[str]:
    """Check that the benchmark file uses manifest workloads and op roofline.

    Uses Python AST parsing (no execution) to verify actual import and usage,
    rather than raw substring matching which can be fooled by comments.

    Returns a list of hard validation errors.
    """
    errors: list[str] = []
    full_path = Path(bench_path)
    if not full_path.is_absolute():
        full_path = repo_root / bench_path

    if not full_path.is_file():
        errors.append(f"[bench] {op_name}: bench file not found: {bench_path}")
        return errors

    content = full_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(content, filename=bench_path)
    except SyntaxError as exc:
        errors.append(
            f"[bench] {op_name}: bench file {bench_path} has syntax error: {exc}"
        )
        return errors

    targets = {"load_workloads", "eval_roofline"}
    usage = _ast_manifest_call_usage(tree, op_name, targets)

    if not usage["load_workloads"]:
        errors.append(
            f"[bench] {op_name}: bench file {bench_path} must import "
            f"load_workloads from tileops.manifest and call it with op name {op_name!r}"
        )
    if not usage["eval_roofline"]:
        errors.append(
            f"[bench] {op_name}: bench file {bench_path} must call "
            "eval_roofline() on an Op instance or use ManifestBenchmark "
            f"with op name {op_name!r}"
        )
    return errors


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _is_spec_only(entry: dict) -> bool:
    """Check if the entry is spec-only.

    Returns True for missing or non-string status (safe default).
    """
    status = entry.get("status")
    if not isinstance(status, str):
        # Missing or non-string status — treat as spec-only (safe default).
        # Schema validation catches this; defensive here for --levels bypass.
        return True
    return status == "spec-only"


def _is_bench_manifest_driven(entry: dict) -> bool:
    """Bench strictness is opt-in until all legacy benchmarks are migrated."""
    return bool(entry.get("source", {}).get("bench_manifest_driven", False))


ALL_LEVELS = frozenset({"schema", "signature", "shape", "dtype", "bench"})


def validate_manifest(
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
    verbose: bool = False,
    levels: frozenset[str] | None = None,
    check_op: str | None = None,
) -> tuple[list[str], list[str]]:
    """Run applicable validation levels on the manifest.

    Args:
        manifest_path: Optional path to a single manifest YAML file. When None,
            the merged manifest is loaded from the ``tileops.manifest`` package
            (one file per family). Tests pass a temp file to validate synthetic
            single-file manifests.
        repo_root: Repository root directory.
        verbose: If True, print progress.
        levels: Set of check names to run (e.g. {"schema", "shape", "dtype", "bench"}).
                When None, all checks are enabled.
        check_op: When set, force all validation levels (L0-L4) on this op and
                  its variants, ignoring the ``status`` field. Only this variant
                  family is validated; all other ops are skipped.

    Returns:
        A tuple of (errors, warnings). Errors are hard failures; warnings
        are informational messages (e.g. signature skipped due to missing deps).
    """
    if repo_root is None:
        repo_root = REPO_ROOT
    if levels is None:
        levels = ALL_LEVELS

    if manifest_path is None:
        from tileops.manifest import load_manifest

        ops = load_manifest()
    else:
        with open(manifest_path) as f:
            ops = yaml.safe_load(f) or {}
        if not isinstance(ops, dict):
            return [
                f"--manifest-path: {manifest_path} must contain a top-level "
                f"mapping of op name -> entry, got {type(ops).__name__}"
            ], []

    # Fail fast: --check-op with a name not in the manifest
    if check_op is not None and check_op not in ops:
        return [f"--check-op: op '{check_op}' not found in manifest"], []

    # When --check-op is set, compute the "variant family" scope: the named
    # op plus all ops where variant_of == check_op.  This ensures that
    # modifications to a variant are caught when validating the primary.
    variant_family: set[str] | None = None
    if check_op is not None:
        variant_family = {check_op} | {
            name for name, ent in ops.items()
            if isinstance(ent, dict) and ent.get("variant_of") == check_op
        }

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Cross-entry checks (must run before per-entry checks).
    # When --check-op is set, scope cross-entry checks to the variant family
    # so that unrelated ops with invalid variant_of references don't cause
    # failures for the selected op.
    if "schema" in levels:
        all_errors.extend(
            check_variant_of_consistency(ops, scope=variant_family)
        )

    for op_name, entry in ops.items():
        # --check-op scopes validation to the variant family; skip all others.
        if variant_family is not None and op_name not in variant_family:
            continue

        if verbose:
            print(f"  Checking {op_name}...")

        # schema: YAML structure validation
        if "schema" in levels:
            schema_errors = check_l0(op_name, entry, warnings=all_warnings)
            all_errors.extend(schema_errors)
            if schema_errors:
                continue

        spec_only = _is_spec_only(entry)
        if spec_only and check_op is None:
            if verbose:
                print(f"    {op_name}: spec-only, skipping signature/shape/dtype/bench")
            continue

        # Resolve Op class once per entry so parity checks can reuse it.
        # Resolution is lightweight (import + getattr); skipped silently
        # when unnecessary.
        source = entry.get("source", {})
        op_file = source.get("op", "")
        resolve_result = _resolve_op_class(op_file, op_name) if op_file else None
        op_cls = resolve_result.cls if resolve_result is not None else None

        # signature: Op.forward() consistency
        if "signature" in levels:
            all_errors.extend(check_l1(op_name, entry, warnings=all_warnings))

        # shape: shape_rules syntax + _infer_output_shapes parity (L2 extension)
        if "shape" in levels:
            all_errors.extend(check_l2(op_name, entry))
            all_errors.extend(
                check_l2_infer_parity(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )

        # dtype: dtype string conformance + _validate_dtypes parity (L3 extension)
        if "dtype" in levels:
            all_errors.extend(check_l3(op_name, entry))
            all_errors.extend(
                check_l3_validate_dtypes_parity(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )

        # bench: benchmark uses manifest workloads
        if "bench" in levels:
            bench_path = entry.get("source", {}).get("bench", "")
            if bench_path:
                bench_errors = check_l4_benchmark(op_name, bench_path, repo_root)
                if _is_bench_manifest_driven(entry):
                    all_errors.extend(bench_errors)
                else:
                    all_warnings.extend(bench_errors)

    # Deduplicate aggregate error/warning strings while preserving order.
    # ``check_l3`` and ``check_l3_validate_dtypes_parity`` both surface
    # ``dtype_combos`` data errors (each is a valid standalone entry
    # point); deduping at the driver keeps user-visible reports crisp.
    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    return _dedup(all_errors), _dedup(all_warnings)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_levels(argv: list[str]) -> frozenset[str] | None:
    """Parse ``--levels schema,shape,dtype`` from argv. Returns None when flag absent."""
    for i, arg in enumerate(argv):
        if arg == "--levels" and i + 1 < len(argv):
            raw_str = argv[i + 1]
        elif arg.startswith("--levels="):
            raw_str = arg.split("=", 1)[1]
        else:
            continue
        parsed = frozenset(t.strip().lower() for t in raw_str.split(","))
        unknown = parsed - ALL_LEVELS
        if unknown:
            print(f"ERROR: unknown levels: {unknown}")
            print(f"  Valid levels: {', '.join(sorted(ALL_LEVELS))}")
            sys.exit(2)
        return parsed
    return None


def _parse_check_op(argv: list[str]) -> str | None:
    """Parse ``--check-op <name>`` from argv.

    Returns the op name, ``None`` when the flag is absent, or calls
    ``sys.exit(2)`` when the value is missing or looks like another flag.
    """
    for i, arg in enumerate(argv):
        if arg == "--check-op":
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                print("ERROR: --check-op requires an op name argument")
                sys.exit(2)
            return argv[i + 1]
        if arg.startswith("--check-op="):
            value = arg.split("=", 1)[1]
            if not value or value.startswith("-"):
                print("ERROR: --check-op requires an op name argument")
                sys.exit(2)
            return value
    return None


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    levels = _parse_levels(sys.argv)
    check_op = _parse_check_op(sys.argv)

    level_label = ",".join(sorted(levels)) if levels else "all"
    check_op_label = f", check-op: {check_op}" if check_op else ""
    print(
        f"Validating {MANIFEST_DIR.relative_to(REPO_ROOT)}/*.yaml "
        f"(levels: {level_label}{check_op_label})..."
    )

    errors, warnings = validate_manifest(
        verbose=verbose, levels=levels, check_op=check_op,
    )

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARNING: {w}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s) found:\n")
        for e in errors:
            print(f"  {e}")
        return 1

    print("All manifest checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
