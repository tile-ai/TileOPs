#!/usr/bin/env python3
"""Validate ops_manifest.yaml.

Checks:
  schema    — YAML structure: required fields, types, nesting
  signature — Op.forward() params match manifest inputs+params
  shape     — shape_rules are parseable Python expressions
  dtype     — dtype strings are valid torch dtype names or references
  bench     — benchmark file uses load_workloads/eval_roofline with this op name

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
import re
import sys
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

        # static_dims must be a mapping of str -> str expression (R20)
        if "static_dims" in sig:
            sdims = sig["static_dims"]
            if not isinstance(sdims, dict):
                errors.append(
                    f"[schema] {op_name}: signature.static_dims must be a mapping"
                )
            else:
                for dname, expr in sdims.items():
                    if not isinstance(expr, str):
                        errors.append(
                            f"[schema] {op_name}: static_dims.{dname} must be a "
                            f"string expression (got {type(expr).__name__})"
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
# variant_of: cross-entry consistency (R16-R18)
# ---------------------------------------------------------------------------

def check_variant_of_consistency(
    ops: dict, *, scope: set[str] | None = None
) -> list[str]:
    """Validate variant_of references across all entries.

    Rules (R16-R18):
    - variant_of must reference an existing op in the manifest.
    - The primary (referenced) entry must NOT itself have variant_of (no chaining).
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

        # R16: target must exist
        if primary_name not in ops:
            errors.append(
                f"[schema] {op_name}: variant_of '{primary_name}' "
                f"does not exist in the manifest"
            )
            continue

        primary = ops[primary_name]
        if not isinstance(primary, dict):
            continue  # malformed primary — check_l0 will report it

        # R17: no chaining — primary must not be a variant itself
        if "variant_of" in primary:
            errors.append(
                f"[schema] {op_name}: variant_of '{primary_name}' is itself "
                f"a variant (chaining not allowed, R17)"
            )

        # R18: shared source.kernel and source.op
        src = entry.get("source", {})
        pri_src = primary.get("source", {})
        if src.get("kernel") != pri_src.get("kernel"):
            errors.append(
                f"[schema] {op_name}: source.kernel differs from primary "
                f"'{primary_name}' (must match per R18)"
            )
        if src.get("op") != pri_src.get("op"):
            errors.append(
                f"[schema] {op_name}: source.op differs from primary "
                f"'{primary_name}' (must match per R18)"
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

    1. **Direct** — ``from tileops.manifest import load_workloads`` /
       ``eval_roofline`` called with the op name.
    2. **Indirect via benchmarks.benchmark_base** — ``workloads_to_params``
       (wraps ``load_workloads``) and ``ManifestBenchmark`` (wraps
       ``eval_roofline``) imported from ``benchmarks.benchmark_base`` and
       called with the op name as the first argument.
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
            # Direct call (load_workloads / eval_roofline).
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
    return {name: (name in imported and name in matched_calls) for name in target_names}


def check_l4_benchmark(
    op_name: str, bench_path: str, repo_root: Path,
) -> list[str]:
    """Check that the benchmark file imports and calls load_workloads/eval_roofline.

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
            f"[bench] {op_name}: bench file {bench_path} must import "
            f"eval_roofline from tileops.manifest and call it with op name {op_name!r}"
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

        # signature: Op.forward() consistency
        if "signature" in levels:
            all_errors.extend(check_l1(op_name, entry, warnings=all_warnings))

        # shape: shape_rules syntax
        if "shape" in levels:
            all_errors.extend(check_l2(op_name, entry))

        # dtype: dtype string conformance
        if "dtype" in levels:
            all_errors.extend(check_l3(op_name, entry))

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
