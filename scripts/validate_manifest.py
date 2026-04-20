#!/usr/bin/env python3
"""Validate ops_manifest.yaml.

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
MANIFEST_PATH = REPO_ROOT / "tileops" / "ops_manifest.yaml"

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

# Sentinel pool used to probe out-of-union dtype rejection in the
# no-dtype_combos branch of ``check_l3_validate_dtypes_parity``. Chosen to
# be common, cheap-to-allocate torch dtypes that are unlikely to appear
# in every manifest declared union simultaneously — giving the probe at
# least one out-of-union candidate per input on realistic specs. Probes
# are bounded by ``_MAX_DTYPE_COMBOS`` so wide unions still stay within
# CI budget.
_DTYPE_SENTINELS: tuple[str, ...] = (
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64",
)


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


def _mock_input_shapes(
    sig: dict,
) -> tuple[dict[str, _MockShape], dict[str, int]] | None:
    """Derive concrete mock input shapes for every declared input.

    Uses rank hints from ``shape_rules`` (literal ``tensor.shape == (...)``
    forms). Falls back to a default 2D shape when the rank is unknown.
    Returns (shapes, dim_sizes) where ``dim_sizes`` maps each symbolic
    dimension name (e.g. ``B``, ``S``, ``H``, ``D``) to the integer size
    used in the mock shapes, so callers can bind those names into a
    shape_rules evaluation context. Returns None only if
    ``signature.inputs`` is malformed.
    """
    inputs = sig.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return None
    rules = sig.get("shape_rules") or []
    ranks = _extract_shape_tuple_literals(rules)

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
                shapes[name] = _MockShape(dim_sizes.get(p, _MOCK_DIM_SIZE)
                                           for p in parts)
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


# Safe builtins allowed in shape_rules eval — matches the R11 / R11a
# documented helper set (see docs/ops-design-reference.md). Keep this list
# aligned with manifest spec; widening it changes the rule language.
_SHAPE_RULE_BUILTINS: dict = {
    "len": len,
    "isinstance": isinstance,
    "int": int,
    "tuple": tuple,
    "list": list,
    "type": type,
    "all": all,
    "any": any,
    "range": range,
    "set": set,
    "abs": abs,
    "min": min,
    "max": max,
}


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
    ``all``, ``any``, ``range``, ``set``, ``abs``, ``min``, ``max``) so
    R11 / R11a-style rules that use these helpers can be evaluated
    against the mock context instead of being silently skipped.

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
      - Skips ops whose ``_infer_output_shapes`` raises during the call
        (e.g. method expects GPU-only state); emits a warning instead.
      - Produces L2 errors only for concrete disagreement: the method
        returns shapes that fail one or more ``shape_rules``.
    """
    errors: list[str] = []
    if cls is None:
        return errors

    sig = entry.get("signature", {})
    rules = sig.get("shape_rules") or []
    if not isinstance(rules, list) or not rules:
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

    # Build a mock ``self`` carrying param attributes; no __init__ invoked.
    mock_self = types.SimpleNamespace(**param_defaults)

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
        # Signature is valid but the body raised. Treat as an
        # implementation issue surfaced via warning (parity skipped) rather
        # than a signature mismatch.
        if warnings is not None:
            warnings.append(
                f"[shape] {op_name}: _infer_output_shapes parity skipped — "
                f"call raised {exc.__class__.__name__}: {exc}"
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
    # Input-only context (no inferred outputs) is used to detect rules
    # that already fail on the mock inputs themselves — such rules
    # encode input-only preconditions (e.g. ``weight.shape ==
    # (x.shape[dim],)``) that mock inputs may violate. A correct
    # ``_infer_output_shapes`` must not be blamed in that case.
    input_only_ctx: dict = dict(ctx)
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
            # inputs only (and does not reference any declared output),
            # the mock input shapes themselves violate the rule — skip
            # with a warning instead of blaming _infer_output_shapes.
            mentions_output = any(
                re.search(rf"\b{re.escape(o)}\b", rule) for o in output_names
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
    return errors


# ---------------------------------------------------------------------------
# dtype parity: _validate_dtypes vs dtype_combos / dtype unions (L3 extension)
# ---------------------------------------------------------------------------


def _dtype_options_for_tensor(
    tname: str, dtype_str: str, resolved: dict[str, list[str]],
) -> list[str] | None:
    """Expand a dtype expression into concrete torch dtype names.

    ``same_as(ref)`` resolves to whatever *ref* was resolved to (must
    appear earlier). Returns None if the expression cannot be resolved.
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

    Resolves ``same_as`` references in declaration order; returns None if
    any tensor's expression cannot be resolved.
    """
    resolved: dict[str, list[str]] = {}
    for group in ("inputs", "outputs"):
        tensors = sig.get(group) or {}
        if not isinstance(tensors, dict):
            continue
        for tname, attrs in tensors.items():
            if not isinstance(attrs, dict):
                return None
            opts = _dtype_options_for_tensor(
                tname, attrs.get("dtype", ""), resolved,
            )
            if opts is None:
                return None
            resolved[tname] = opts
    return resolved


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
    param_defaults: dict,
) -> tuple[bool, str | None]:
    """Invoke ``cls._validate_dtypes`` on a mock-self with *combo*.

    Returns (accepted, error_reason). ``accepted=False`` with
    reason=None means the op raised during validation (rejected);
    reason!=None indicates the call could not be performed (skip).
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

    mock_self = types.SimpleNamespace(**param_defaults)
    # Pre-bind the callable signature so only genuine signature mismatches
    # surface as ``TypeError: ...``. TypeError raised from inside the body
    # (e.g. comparing incompatible torch dtypes) is a legitimate rejection
    # and must not be misreported as a signature mismatch.
    try:
        inspect.signature(validate_fn).bind(mock_self, **tensors)
    except TypeError as exc:
        return False, f"TypeError: {exc}"
    except Exception as exc:  # noqa: BLE001
        # inspect.signature failed to introspect — treat as a skip.
        return False, f"unexpected {exc.__class__.__name__}: {exc}"

    try:
        validate_fn(mock_self, **tensors)
    except (ValueError, TypeError):
        # Body-level rejection: either an explicit ValueError or a
        # TypeError arising from dtype comparisons. Both are legitimate
        # rejections once the signature has been validated above.
        return False, None
    except Exception as exc:  # noqa: BLE001
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
        # Each listed combo should be accepted.
        for i, combo in enumerate(dtype_combos):
            if not isinstance(combo, dict):
                continue
            accepted, reason = _combo_accepted(
                cls, forward_inputs, combo, param_defaults,
            )
            if reason and reason.startswith("TypeError"):
                errors.append(
                    f"[dtype] {op_name}: _validate_dtypes signature does "
                    f"not match manifest inputs (expected kwargs "
                    f"{sorted(forward_inputs)}): {reason}"
                )
                return errors
            if reason and reason.startswith("unexpected"):
                if warnings is not None:
                    warnings.append(
                        f"[dtype] {op_name}: _validate_dtypes parity skipped "
                        f"for dtype_combos[{i}] — {reason}"
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
                cls, forward_inputs, candidate, param_defaults,
            )
            if reason and reason.startswith(("unexpected", "TypeError")):
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
                cls, forward_inputs, candidate, param_defaults,
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
            if reason and reason.startswith("unexpected"):
                if warnings is not None:
                    warnings.append(
                        f"[dtype] {op_name}: _validate_dtypes parity skipped "
                        f"for combo {candidate!r} — {reason}"
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
                out_of_union = [
                    d for d in _DTYPE_SENTINELS if d not in declared
                ]
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
                    )
                    if reason and reason.startswith(
                        ("unexpected", "TypeError")
                    ):
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
                        cls, forward_inputs, candidate, param_defaults,
                    )
                    if reason and reason.startswith(
                        ("unexpected", "TypeError")
                    ):
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
        manifest_path: Path to ops_manifest.yaml.
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
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    if repo_root is None:
        repo_root = REPO_ROOT
    if levels is None:
        levels = ALL_LEVELS

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    ops = data.get("ops", {})

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

    return all_errors, all_warnings


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
        f"Validating {MANIFEST_PATH.relative_to(REPO_ROOT)} "
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
