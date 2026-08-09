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
import textwrap
import types
import warnings as _warnings
from collections.abc import Collection
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Documented and skill-driven callers run this script directly, without
# installing the package, so the source tree has to be reachable.
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import tileops.manifest as manifest_pkg  # noqa: E402
from tileops.manifest.dtype_rules import PROMOTE_INT_TO_FLOAT_RE, SAME_AS_RE  # noqa: E402
from tileops.manifest.shape_rules import (  # noqa: E402
    dim_range_validity,
    dim_uniqueness,
    reduced_axes,
)

PACKAGE_ROOT = "src"
DISTRIBUTION_RELATIVE_KEYS = frozenset({"kernel", "op"})

MANIFEST_DIR = REPO_ROOT / PACKAGE_ROOT / "tileops" / "manifest"

# Valid torch dtype base names (without same_as references)
_TORCH_DTYPES = {
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64",
    "uint8", "bool",
    "complex64", "complex128",
    "float8_e4m3fn", "float8_e5m2",
    "float8_e4m3", "float8_e5m2fnuz", "float8_e4m3fnuz",
}

# ``promote_int_to_float(ref)``: ``float32`` for integral ``ref``, else
# ``same_as(ref)``. Models PyTorch int-input promotion (``torch.reciprocal``).
_SAME_AS_RE = SAME_AS_RE
_PROMOTE_INT_TO_FLOAT_RE = PROMOTE_INT_TO_FLOAT_RE

# Integral torch dtypes that ``promote_int_to_float`` rewrites to ``float32``.
# Restricted to the dtypes PyTorch's int-input promotion treats as integral
# (bool is excluded — it is not part of the integral promotion contract).
_PROMOTE_INT_DTYPES: frozenset[str] = frozenset({
    "uint8", "int8", "int16", "int32", "int64",
})

# Target dtype for integral inputs under ``promote_int_to_float``. Matches
# PyTorch's default scalar type.
_PROMOTE_TARGET_DTYPE: str = "float32"

# Required top-level fields per op entry
_REQUIRED_TOP = {"family", "status", "signature", "workloads", "roofline", "source"}
_VALID_TOP_KEYS = _REQUIRED_TOP | {"ref_api", "variant_of", "torch_compile_fullgraph"}
_REQUIRED_SIGNATURE = {"inputs", "outputs"}
_VALID_SIGNATURE_KEYS = {
    "inputs", "outputs", "params", "shape_rules", "dtype_combos",
    "static_dims",
}
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

def _check_shape_rule_callables(
    op_name: str, index: int, rule_str: str,
) -> list[str]:
    """Validate that bare-name calls in a shape_rule reference known helpers.

    Walks the parsed rule AST; every ``ast.Call`` whose ``func`` is a
    bare ``ast.Name`` must be registered in ``_SHAPE_RULE_BUILTINS``.
    Method / subscript calls are skipped — only direct name lookups are
    validated. A ``SyntaxError`` surfaces as a single ``[schema]`` error
    so malformed rules are rejected at L0 without the L2 eval context.
    Returns ``[schema]``-prefixed error strings (empty when clean).
    """
    errors: list[str] = []
    try:
        tree = ast.parse(rule_str, mode="eval")
    except SyntaxError as exc:
        errors.append(
            f"[schema] {op_name}: shape_rules[{index}] invalid syntax: "
            f"{rule_str!r} ({exc})"
        )
        return errors
    # _SHAPE_RULE_BUILTINS is defined later in the module; the forward
    # reference is intentional. The dict resolves at call time (validation
    # runs after import), and keeping its single source of truth alongside
    # the helper callables it maps to avoids splitting the registry.
    seen_unknown: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name):
            continue
        if func.id in _SHAPE_RULE_BUILTINS or func.id in seen_unknown:
            continue
        seen_unknown.add(func.id)
        errors.append(
            f"[schema] {op_name}: shape_rules[{index}] calls unknown "
            f"helper {func.id!r}; allowed callables are "
            f"{', '.join(sorted(_SHAPE_RULE_BUILTINS))}"
        )
    return errors


def _check_single_input_workload_keys(
    op_name: str, sig: dict, workloads: list,
) -> list[str]:
    """Check R21: workload keys must derive from the signature.

    Out of scope: multi-input signatures and workloads with no ``*_shape``
    key.
    """
    contract = manifest_pkg.single_input_workload_contract(sig)
    if contract is None:
        return []
    if not any(
        isinstance(w, dict)
        and any(isinstance(k, str) and k.endswith("_shape") for k in w)
        for w in workloads
    ):
        return []
    shape_key, allowed = contract
    params = sig.get("params")
    param_names = set(params) if isinstance(params, dict) else set()
    reserved = manifest_pkg.WORKLOAD_RESERVED_KEYS | {shape_key}
    errors: list[str] = []
    collisions = sorted(param_names & reserved)
    if collisions:
        errors.append(
            f"[schema] {op_name}: signature params {collisions} collide "
            "with reserved workload keys"
        )
    for i, w in enumerate(workloads):
        if not isinstance(w, dict):
            continue
        if shape_key not in w:
            errors.append(
                f"[schema] {op_name}: workloads[{i}] missing {shape_key!r} "
                "(shape key is derived from the signature's tensor input "
                "name)"
            )
        unknown = sorted(
            k for k in w
            if isinstance(k, str) and k not in allowed and not k.startswith("__")
        )
        if unknown:
            errors.append(
                f"[schema] {op_name}: workloads[{i}] has unknown keys "
                f"{unknown}; allowed are {shape_key!r}, 'dtypes', 'label', "
                "and declared signature params"
            )
    return errors


def _l0_key_format(
    op_name: str, all_op_names: Collection[str],
) -> list[str]:
    """Key format: variant words precede the direction suffix.

    The direction suffix itself is required only when the manifest
    carries a direction sibling of the same op.
    """
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)
    key_match = re.match(r"^(.*)(Fwd|Bwd)Op(.+)$", op_name)
    if key_match:
        stem, direction, trailing = key_match.groups()
        err(
            f"variant word '{trailing}' follows '{direction}Op'; variant "
            f"words must precede the direction suffix "
            f"(expected '{stem}{trailing}{direction}Op')"
        )
    elif op_name.endswith("Op") and not op_name.endswith(("FwdOp", "BwdOp")):
        stem = op_name[:-2]
        siblings = [
            s for s in (f"{stem}FwdOp", f"{stem}BwdOp") if s in all_op_names
        ]
        if siblings:
            err(
                f"missing direction suffix; direction sibling "
                f"'{siblings[0]}' exists in the manifest"
            )
    return errors


def _l0_signature(op_name: str, entry: dict, sig: dict) -> list[str]:
    """Signature sub-schema: tensors, params, dtype_combos, shape_rules."""
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)

    missing_sig = _REQUIRED_SIGNATURE - set(sig.keys())
    if missing_sig:
        err(f"signature missing: {missing_sig}")

    # inputs/outputs/params names must be strings.
    for field in ("inputs", "outputs", "params"):
        names = sig.get(field)
        if not isinstance(names, dict):
            continue
        non_str = sorted(repr(k) for k in names if not isinstance(k, str))
        if non_str:
            err(
                f"signature.{field} has non-string names "
                f"[{', '.join(non_str)}]"
            )

    # inputs/outputs are dicts of tensor-attr dicts carrying a dtype.
    for direction in ("inputs", "outputs"):
        tensors = sig.get(direction)
        if not isinstance(tensors, dict):
            if direction in sig:
                err(f"signature.{direction} must be a dict")
            continue
        for tname, attrs in tensors.items():
            if not isinstance(attrs, dict):
                err(f"{direction}.{tname} must be a dict")
                continue
            if "dtype" not in attrs:
                err(f"{direction}.{tname} missing 'dtype'")
            # constraints keys must name dims of the declared shape
            if "constraints" in attrs:
                constraints = attrs["constraints"]
                if not isinstance(constraints, dict):
                    err(f"{direction}.{tname}.constraints must be a mapping")
                elif not isinstance(attrs.get("shape"), str):
                    err(f"{direction}.{tname} has constraints but no shape")
                else:
                    dims = {
                        d.strip()
                        for d in attrs["shape"].strip("[]").split(",")
                        if d.strip()
                    }
                    for ckey in constraints:
                        if ckey not in dims:
                            err(
                                f"{direction}.{tname} constraints key "
                                f"'{ckey}' is not in shape dims "
                                f"{sorted(dims)}"
                            )
            # layout validation (R19)
            if "layout" in attrs:
                layout = attrs["layout"]
                if not isinstance(layout, str):
                    err(f"{direction}.{tname}.layout must be a string")
                elif layout not in _VALID_LAYOUTS:
                    err(
                        f"{direction}.{tname}.layout '{layout}' is not "
                        f"recognized "
                        f"(valid: {', '.join(sorted(_VALID_LAYOUTS))})"
                    )

    # Params must be a mapping if present; each entry needs 'type' (R1).
    if "params" in sig:
        params = sig["params"]
        if not isinstance(params, dict):
            err("signature.params must be a mapping")
        else:
            for pname, pattrs in params.items():
                if not isinstance(pattrs, dict):
                    err(f"params.{pname} must be a dict")
                    continue
                if "type" not in pattrs:
                    err(f"params.{pname} missing 'type'")

    # Surface invariant: every op produces at least one output, and has
    # at least one construction handle — a tensor input or a declared
    # param. ``inputs: {}`` is permitted (generative ops synthesize the
    # output from params alone).
    raw_inputs = sig.get("inputs")
    raw_outputs = sig.get("outputs")
    raw_params = sig.get("params")
    inputs_count = len(raw_inputs) if isinstance(raw_inputs, dict) else 0
    outputs_count = len(raw_outputs) if isinstance(raw_outputs, dict) else 0
    params_count = len(raw_params) if isinstance(raw_params, dict) else 0
    if outputs_count < 1:
        err("signature.outputs must declare at least one tensor")
    if inputs_count < 1 and params_count < 1:
        err(
            "signature must declare at least one input tensor or one "
            "param (both are empty)"
        )

    # Every output declares a shape, or shape_rules pin the output shapes.
    raw_rules = sig.get("shape_rules")
    has_shape_rules = isinstance(raw_rules, list) and len(raw_rules) > 0
    if isinstance(raw_outputs, dict) and not has_shape_rules:
        for tname, attrs in raw_outputs.items():
            if isinstance(attrs, dict) and "shape" not in attrs:
                err(
                    f"output '{tname}' must declare 'shape' or the "
                    f"signature must have shape_rules"
                )

    # dtype_combos must be a list of dicts if present (R4).
    if "dtype_combos" in sig:
        combos = sig["dtype_combos"]
        if not isinstance(combos, list):
            err("signature.dtype_combos must be a list")
        else:
            tensor_names = set()
            for d in ("inputs", "outputs"):
                t = sig.get(d)
                if isinstance(t, dict):
                    tensor_names.update(t.keys())
            for i, combo in enumerate(combos):
                if not isinstance(combo, dict):
                    err(f"dtype_combos[{i}] must be a dict")
                    continue
                for key in combo:
                    if key not in tensor_names:
                        err(
                            f"dtype_combos[{i}] key '{key}' is not a "
                            f"declared tensor name"
                        )

    # shape_rules must be a list of strings if present.
    if "shape_rules" in sig:
        rules = sig["shape_rules"]
        if not isinstance(rules, list):
            err("shape_rules must be a list")
        else:
            for i, rule in enumerate(rules):
                if not isinstance(rule, str):
                    err(f"shape_rules[{i}] must be a string")
                    continue
                errors.extend(_check_shape_rule_callables(op_name, i, rule))

    # Unknown signature keys are silently ignored by L1+; reject here.
    unknown_sig = sorted(repr(k) for k in set(sig) - _VALID_SIGNATURE_KEYS)
    if unknown_sig:
        err(
            f"unknown signature keys [{', '.join(unknown_sig)}]; valid "
            f"keys are {sorted(_VALID_SIGNATURE_KEYS)}"
        )

    # static_dims must be a mapping of str -> str expression (R20).
    if "static_dims" in sig:
        errors.extend(_check_static_dims(op_name, sig["static_dims"], sig))
    return errors


def _l0_workloads(op_name: str, entry: dict, workloads: list) -> list[str]:
    """Workload policy: count, dtypes, required-param pinning, R21 keys."""
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)
    # Implemented ops need benchmarkable coverage: at least 2 workloads.
    if entry.get("status") == "implemented" and len(workloads) < 2:
        err(
            f"implemented op must have at least 2 workloads, "
            f"got {len(workloads)}"
        )
    # Params without a default must be pinned by every workload.
    sig_params = entry.get("signature", {})
    sig_params = sig_params.get("params") if isinstance(sig_params, dict) else None
    required_params = {
        pname
        for pname, pattrs in (sig_params or {}).items()
        if isinstance(pname, str)
        and isinstance(pattrs, dict)
        and "default" not in pattrs
    } if isinstance(sig_params, dict) else set()
    for i, w in enumerate(workloads):
        if not isinstance(w, dict):
            err(f"workloads[{i}] must be a dict")
            continue
        if "dtypes" not in w:
            err(f"workloads[{i}] missing 'dtypes'")
        non_str = [k for k in w if not isinstance(k, str)]
        if non_str:
            err(f"workloads[{i}] has non-string keys {non_str}")
        missing_params = required_params - set(w.keys())
        if missing_params:
            err(
                f"workloads[{i}] missing required param(s): "
                f"{sorted(missing_params)}"
            )
    if isinstance(entry.get("signature"), dict):
        errors.extend(
            _check_single_input_workload_keys(
                op_name, entry["signature"], workloads
            )
        )
    return errors


def _l0_roofline(op_name: str, entry: dict, roofline: dict) -> list[str]:
    """Roofline structural rules per docs/design/roofline.md §4.1."""
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)
    has_inline = "flops" in roofline and "bytes" in roofline
    has_func = "func" in roofline
    if not has_inline and not has_func:
        err("roofline must have (flops + bytes) or func")
    if has_func and ({"flops", "bytes", "vars"} & set(roofline)):
        err(
            "roofline modes are exclusive — func must not coexist with "
            "flops/bytes/vars"
        )
    for field in ("flops", "bytes", "func"):
        if field in roofline and not (
            isinstance(roofline[field], str) and roofline[field].strip()
        ):
            err(f"roofline.{field} must be a non-empty string")
    rl_vars = roofline.get("vars")
    if rl_vars is not None:
        if not isinstance(rl_vars, dict):
            err("roofline.vars must be a mapping")
        else:
            for k, v in rl_vars.items():
                if not isinstance(k, str):
                    err(f"roofline.vars key {k!r} must be a string")
                if not (isinstance(v, str) and v.strip()):
                    err(f"roofline.vars[{k!r}] must be a non-empty string")
    if has_func and isinstance(roofline.get("func"), str):
        mod, _, attr = roofline["func"].rpartition(".")
        try:
            target = importlib.import_module(mod) if mod else None
        except ImportError:
            target = None
        if target is None or not callable(getattr(target, attr, None)):
            err(
                f"roofline.func {roofline['func']!r} does not resolve "
                f"to a callable"
            )
    return errors


def _l0_source(op_name: str, entry: dict, source: dict) -> list[str]:
    """Source block: required path fields; kernel string-or-list."""
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)
    missing_src = _REQUIRED_SOURCE - set(source.keys())
    if missing_src:
        err(f"source missing fields: {missing_src}")
    # source.kernel: string or list of strings
    kernel = source.get("kernel")
    if kernel is not None:
        if isinstance(kernel, list):
            for i, k in enumerate(kernel):
                if not isinstance(k, str):
                    err(f"source.kernel[{i}] must be a string")
        elif not isinstance(kernel, str):
            err("source.kernel must be a string or list")
    if "bench_manifest_driven" in source and not isinstance(
        source["bench_manifest_driven"], bool,
    ):
        err("source.bench_manifest_driven must be a bool")
    return errors


def _l0_kernel_map(
    op_name: str, entry: dict, warnings: list[str] | None,
) -> list[str]:
    """kernel_map (under source): mapping of str -> str, required when implemented.

    An implemented op dispatches through ``default_kernel_map``; omitting the
    declaration hides that dispatch table from the spec.
    """
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)
    source = entry.get("source", {})
    kernel_map = source.get("kernel_map") if isinstance(source, dict) else None
    if kernel_map is not None:
        if not isinstance(kernel_map, dict):
            err(
                f"kernel_map must be a mapping, "
                f"got {type(kernel_map).__name__}"
            )
        else:
            for k, v in kernel_map.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    err(
                        f"kernel_map entries must be str -> str, "
                        f"got {k!r}: {v!r}"
                    )
    elif entry.get("status") == "implemented":
        err(
            "status is 'implemented' but kernel_map is missing "
            "(must be a mapping of str -> str)"
        )
    return errors


# Table-driven L0 sections, in emission order. Each row: (field, expected
# container type, type-error phrase, section validator run on type match).
# Genuinely custom rules (key format, scalar fields, kernel_map) stay as
# dedicated small validators around the table loop in ``check_l0``.
_L0_SECTIONS = (
    ("signature", dict, "a mapping", _l0_signature),
    ("workloads", list, "a list", _l0_workloads),
    ("roofline", dict, "a mapping", _l0_roofline),
    ("source", dict, "a mapping", _l0_source),
)


def check_l0(
    op_name: str, entry: dict, *, warnings: list[str] | None = None,
    all_op_names: Collection[str] = (),
) -> list[str]:
    """Validate structural schema of a manifest entry. Returns error strings."""
    if not isinstance(entry, dict):
        return [
            f"[schema] {op_name}: entry must be a mapping, "
            f"got {type(entry).__name__}"
        ]
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)

    errors.extend(_l0_key_format(op_name, all_op_names))

    # Top-level required fields
    missing_top = _REQUIRED_TOP - set(entry.keys())
    if missing_top:
        err(f"missing top-level fields: {missing_top}")

    for field, container, desc, section in _L0_SECTIONS:
        value = entry.get(field)
        if isinstance(value, container):
            errors.extend(section(op_name, entry, value))
        elif field in entry:
            err(f"{field} must be {desc}")

    # Unknown top-level keys are ignored by every later level, so reject
    # them here (covers removed fields like parity_opt_out).
    unknown_top = sorted(repr(k) for k in set(entry) - _VALID_TOP_KEYS)
    if unknown_top:
        err(
            f"unknown entry keys [{', '.join(unknown_top)}]; "
            f"valid keys are {sorted(_VALID_TOP_KEYS)}"
        )

    # variant_of: must be a string if present (R16); cross-entry checks
    # in check_variant_of_consistency().
    if "variant_of" in entry and not isinstance(entry["variant_of"], str):
        err("variant_of must be a string")

    # ref_api: required string — fully qualified PyTorch API equivalent
    # or "none".
    if "ref_api" not in entry:
        err("missing required field 'ref_api'")
    elif not isinstance(entry["ref_api"], str):
        err("ref_api must be a string")

    # status: must be "implemented" or "spec-only"
    # (missing status is already caught by the required-fields check).
    status = entry.get("status")
    if "status" in entry and not isinstance(status, str):
        err(f"status must be a string, got {type(status).__name__}")
    elif isinstance(status, str) and status not in ("implemented", "spec-only"):
        err(
            f"status must be 'implemented' or 'spec-only', "
            f"got '{status}'"
        )

    # Only literal `true` is accepted; absence is the only spelling of "no
    # promise". Invalid on spec-only — no implementation to capture.
    if "torch_compile_fullgraph" in entry:
        tcf = entry["torch_compile_fullgraph"]
        if tcf is not True:
            err(
                f"torch_compile_fullgraph must be literal true when "
                f"present (omit the field to make no promise), got {tcf!r}"
            )
        elif status == "spec-only":
            err(
                "torch_compile_fullgraph is invalid on 'status: "
                "spec-only' entries — the promise requires an "
                "implementation"
            )

    errors.extend(_l0_kernel_map(op_name, entry, warnings))
    return errors


def check_source_paths(op_name: str, entry: dict, repo_root: Path) -> list[str]:
    """Check that string ``source`` values of non-spec-only ops are real files.

    ``kernel`` and ``op`` are written as they appear inside the distribution,
    because the manifest ships in the wheel and a path it names must exist
    there; on disk they resolve under ``src/``. ``test`` and ``bench`` name
    trees that never ship, so they are repo-relative and resolve as written.

    Spec-only entries are skipped — their source paths are placeholders
    until implementation. Non-string values (e.g. ``kernel_map`` mappings,
    ``source.kernel`` lists, nulls) are out of scope here; their structure
    is validated by :func:`check_l0`.
    """
    if not isinstance(entry, dict) or _is_spec_only(entry):
        return []
    source = entry.get("source")
    if not isinstance(source, dict):
        return []
    errors: list[str] = []
    for key, rel_path in source.items():
        if not isinstance(rel_path, str):
            continue
        base = repo_root / PACKAGE_ROOT if key in DISTRIBUTION_RELATIVE_KEYS else repo_root
        if not (base / rel_path).is_file():
            errors.append(
                f"[schema] {op_name}: source.{key} is not a file: {rel_path}"
            )
    return errors


# ---------------------------------------------------------------------------
# variant_of: cross-entry consistency (R16)
# ---------------------------------------------------------------------------

def check_variant_of_consistency(
    ops: dict, *, scope: set[str] | None = None
) -> list[str]:
    """Validate variant_of references across all entries.

    Per R16: variant_of must reference an existing op; the primary must
    not itself be a variant (no chaining); variant and primary must
    share source.kernel and source.op. When *scope* is given, only ops
    named in *scope* are checked; lookups still use the full dict so
    reference resolution works.
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

    ``init_params=None`` is treated as empty (only forward is checked).
    """
    errors: list[str] = []
    err = _emit_to(errors, "signature", op_name)

    # Guard: manifest_params must be a dict (schema should catch this, but be safe)
    if not isinstance(manifest_params, dict):
        err(
            "signature.params is not a mapping, cannot validate "
            "forward() consistency"
        )
        return errors

    if init_params is None:
        init_params = []

    # forward() order check: manifest inputs + forward-visible params, in
    # order. Empty manifest inputs collapses to a forward-visible-params
    # equality check (no inputs to align).
    expected = list(manifest_inputs.keys()) + [
        name for name in manifest_params.keys() if name in forward_params
    ]
    if forward_params != expected:
        err(
            f"forward() params {forward_params} do not match "
            f"manifest order {expected}"
        )

    # Strict subset check: every manifest param must exist in init OR forward
    code_params = set(forward_params) | set(init_params)
    for pname in manifest_params:
        if pname not in code_params:
            err(
                f"manifest param {pname!r} not found in __init__() or "
                f"forward() parameters"
            )

    # static_dims check (R20): every static_dims key must be an __init__ param
    if manifest_static_dims:
        if not isinstance(manifest_static_dims, dict):
            err("signature.static_dims is not a mapping")
        else:
            init_param_set = set(init_params)
            for dim_name in manifest_static_dims:
                if dim_name not in init_param_set:
                    err(
                        f"static_dims key {dim_name!r} not found in "
                        f"__init__() parameters (R20: static_dims keys "
                        f"are required __init__ params)"
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

    Returns a _ResolveResult: ``cls`` set when found; ``import_error``
    True when the module could not be imported (missing dependencies).
    """
    # "tileops/ops/norm/rms_norm.py" -> "tileops.ops.norm.rms_norm"
    mod_path = op_file.replace("/", ".").replace(".py", "")
    try:
        mod = importlib.import_module(mod_path)
    except (ImportError, ModuleNotFoundError):
        return _ResolveResult(import_error=True)
    except Exception:
        return _ResolveResult()

    # Candidates: classes defined in this module with a forward() method.
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


_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _forward_positional_params(cls) -> list[str] | None:
    """Get positional parameter names of cls.forward(), excluding 'self'.

    Only POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD count. KEYWORD_ONLY
    params (those after ``*``) are not part of the positional tuple
    that manifest ``signature.inputs`` aligns against. Shared by check_l1
    and the C4 forward-signature parity check so they stay in lockstep.
    """
    try:
        sig = inspect.signature(cls.forward)
        return [
            p for p, v in sig.parameters.items()
            if p != "self" and v.kind in _POSITIONAL_KINDS
        ]
    except (ValueError, TypeError) as exc:
        # Stash exception text so callers that surface diagnostics can
        # report ``exc.__class__.__name__: exc`` without changing the
        # ``None`` return contract for "not inspectable".
        _forward_positional_params._last_error = (  # type: ignore[attr-defined]
            f"{exc.__class__.__name__}: {exc}"
        )
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
    the manifest signature. Returns error strings (empty if OK).
    """
    errors: list[str] = []
    sig = entry.get("signature", {})
    source = entry.get("source", {})
    op_file = source.get("op", "")
    if not op_file:
        if entry.get("status") == "spec-only":
            if warnings is not None:
                warnings.append(
                    f"[signature] {op_name}: skipped because status is spec-only "
                    "and source.op is null"
                )
            return []
        return [f"[signature] {op_name}: missing source.op"]

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

    Empty tokens are kept, unlike ``manifest.dtype_rules.parse_tokens``: a
    trailing or doubled ``|`` must surface downstream as an unknown-dtype
    error rather than being silently dropped.
    """
    return [t.strip() for t in dtype_str.split("|")]


def _validate_dtype_token(
    op_name: str,
    context: str,
    token: str,
    tensor_names: set[str],
    *,
    allow_promote_int_to_float: bool = True,
    input_tensor_names: set[str] | None = None,
) -> str | None:
    """Validate a single dtype token. Returns an error string or None.

    ``promote_int_to_float(ref)`` is an output-side-only construct per
    docs/design/manifest.md R3a. Callers validating input tensors set
    ``allow_promote_int_to_float=False`` to reject it on the input side.
    When ``allow_promote_int_to_float`` is True, ``input_tensor_names``
    must be supplied: ``ref`` must name a signature input tensor — not
    an output, and not the tensor itself.
    """
    m = _SAME_AS_RE.match(token)
    if m:
        ref = m.group(1)
        if ref not in tensor_names:
            return (
                f"[dtype] {op_name}: {context} dtype same_as({ref}) "
                f"references unknown tensor"
            )
        return None
    m = _PROMOTE_INT_TO_FLOAT_RE.match(token)
    if m:
        if not allow_promote_int_to_float:
            return (
                f"[dtype] {op_name}: {context} uses promote_int_to_float "
                f"— this construct is output-side only"
            )
        ref = m.group(1)
        if input_tensor_names is None or ref not in input_tensor_names:
            return (
                f"[dtype] {op_name}: {context} dtype "
                f"promote_int_to_float({ref}) must reference a signature "
                f"input tensor"
            )
        return None
    if token not in _TORCH_DTYPES:
        return f"[dtype] {op_name}: {context} has unrecognized dtype '{token}'"
    return None


def _build_same_as_map(all_tensors: dict) -> dict[str, str]:
    """Map tensor name → same_as reference target for pure same_as dtypes.

    Mixed expressions (``float16 | same_as(x)``) are not tracked.
    """
    same_as_map: dict[str, str] = {}
    for tname, attrs in all_tensors.items():
        tokens = _parse_dtype_expr(attrs.get("dtype", ""))
        if len(tokens) == 1:
            m = _SAME_AS_RE.match(tokens[0])
            if m:
                same_as_map[tname] = m.group(1)
    return same_as_map


def _check_dtype_combos_same_as_identity(
    op_name: str, dtype_combos: list, same_as_map: dict[str, str],
) -> list[str]:
    """Enforce same_as identity in dtype_combos entries (R3).

    Every tensor bound by same_as(ref) must have the exact same dtype as
    its reference tensor in every combo row.
    """
    errors: list[str] = []
    err = _emit_to(errors, "dtype", op_name)
    for i, combo in enumerate(dtype_combos):
        if not isinstance(combo, dict):
            continue
        for tensor, ref in same_as_map.items():
            t_in = tensor in combo
            r_in = ref in combo
            if t_in and r_in and combo[tensor] != combo[ref]:
                err(
                    f"dtype_combos[{i}] violates same_as identity "
                    f"constraint — {tensor} ({combo[tensor]}) must match "
                    f"{ref} ({combo[ref]}) per R3"
                )
            elif t_in and not r_in:
                err(
                    f"dtype_combos[{i}] has same_as-bound tensor "
                    f"'{tensor}' without its reference '{ref}' — cannot "
                    f"verify identity"
                )
    return errors


def check_l3(op_name: str, entry: dict) -> list[str]:
    """Validate dtype strings are recognized torch types or same_as references.

    Checks both signature tensor dtypes and workload dtype entries.
    Also enforces same_as identity constraint in dtype_combos (R3).
    """
    errors: list[str] = []
    err = _emit_to(errors, "dtype", op_name)
    sig = entry.get("signature", {})
    raw_inputs = sig.get("inputs")
    raw_outputs = sig.get("outputs")
    inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
    outputs = raw_outputs if isinstance(raw_outputs, dict) else {}
    all_tensors = {}
    all_tensors.update(inputs)
    all_tensors.update(outputs)

    tensor_names = set(all_tensors.keys())
    input_names = set(inputs.keys())

    # Signature tensor dtypes. ``promote_int_to_float`` is output-side
    # only (R3a) — reject it on input tensors.
    for tname, attrs in all_tensors.items():
        if not isinstance(attrs, dict):
            continue
        for token in _parse_dtype_expr(attrs.get("dtype", "")):
            token_err = _validate_dtype_token(
                op_name, tname, token, tensor_names,
                allow_promote_int_to_float=tname not in input_names,
                input_tensor_names=input_names,
            )
            if token_err:
                errors.append(token_err)

    # same_as identity constraint in dtype_combos (R3)
    dtype_combos = sig.get("dtype_combos", [])
    if isinstance(dtype_combos, list) and dtype_combos:
        same_as_map = _build_same_as_map(all_tensors)
        errors.extend(
            _check_dtype_combos_same_as_identity(op_name, dtype_combos, same_as_map)
        )
        # Hard data-validation for combo values, run unconditionally —
        # independent of whether the op overrides ``_validate_dtypes`` —
        # so an un-migrated op carrying invalid combo data still surfaces
        # a hard L3 error rather than only a missing-override warning.
        errors.extend(check_l3_dtype_combos_data(op_name, sig))

    # Workload dtypes
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
                    err(f"workloads[{i}].dtypes[{j}] is not a string")
                    continue
                for token in _parse_dtype_expr(dt):
                    token_err = _validate_dtype_token(
                        op_name, f"workloads[{i}].dtypes[{j}]",
                        token, tensor_names,
                        allow_promote_int_to_float=False,
                    )
                    if token_err:
                        errors.append(token_err)

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
    err = _emit_to(errors, "dtype", op_name)
    inputs = sig.get("inputs") or {}
    outputs = sig.get("outputs") or {}
    all_tensors: dict[str, dict] = {}
    if isinstance(inputs, dict):
        all_tensors.update(inputs)
    if isinstance(outputs, dict):
        all_tensors.update(outputs)

    # Pure ``same_as(ref)`` edges only — mixed expressions are not part
    # of the cycle graph; a cycle in pure edges is what stalls fixpoint
    # resolution.
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
            err(
                f"signature.inputs/outputs — tensor {tname!r} declares "
                f"dtype same_as({ref}) but {ref!r} is not a declared "
                f"tensor (dangling reference; combo validation cannot "
                f"proceed)"
            )
            dangling.add(tname)

    # Cycle detection via DFS over pure same_as edges. A chain ending in
    # a dangling ref is not a cycle.
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
                    err(
                        f"same_as cycle detected among tensors "
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
        # Fixpoint failed but no cycle or dangling edge was found (e.g. a
        # mixed expression containing an unknown token that did not trip
        # per-token validation). Emit a generic hard error so combo
        # validation is never silently skipped.
        err(
            "could not resolve signature.inputs/outputs dtype options — "
            "combo validation cannot proceed. Check "
            "signature.inputs/outputs dtype declarations for unresolved "
            "same_as references or malformed expressions."
        )
    return errors


def check_l3_dtype_combos_data(op_name: str, sig: dict) -> list[str]:
    """Validate ``dtype_combos`` entries resolve to concrete torch dtypes.

    Manifest-data check, independent of any op class / ``_validate_dtypes``
    implementation. Every combo value must be either a concrete dtype
    name in ``_TORCH_DTYPES`` or a ``same_as(ref)`` expression whose ref
    resolves transitively to concrete dtype names. Anything else is a
    hard L3 error — callers must not silently proceed with invalid combo
    data.
    """
    errors: list[str] = []
    err = _emit_to(errors, "dtype", op_name)
    dtype_combos = sig.get("dtype_combos")
    if not isinstance(dtype_combos, list) or not dtype_combos:
        return errors
    dtype_options = _resolve_tensor_dtype_options(sig)
    if dtype_options is None:
        # A pure ``same_as`` cycle passes per-token validation and the R3
        # identity check, so returning silently would let it through.
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
        # must be assigned a dtype in every combo row; otherwise a row
        # omitting an input would pass L3 when no ``_validate_dtypes``
        # override exists (``_combo_accepted`` never runs for it).
        for input_name in declared_input_names:
            if input_name not in combo:
                err(
                    f"dtype_combos[{i}] is missing declared input "
                    f"{input_name!r} (every combo row must cover every "
                    f"signature.inputs tensor)"
                )
        for key, val in combo.items():
            if not isinstance(val, str):
                err(f"dtype_combos[{i}].{key} = {val!r} is not a string")
                continue
            # Per manifest.md R4, each combo value pins a single concrete
            # dtype token (or a ``same_as(ref)`` naming a sibling in the
            # same row). A union would let an implementation silently
            # widen the accepted-dtype set beyond what was authored.
            if "|" in val:
                err(
                    f"dtype_combos[{i}].{key} = {val!r} — combo values "
                    f"must be a single concrete dtype, not a union"
                )
                continue
            # promote_int_to_float(ref) may expand to multiple concrete
            # dtypes, so it cannot pin a combo row either; authors expand
            # the rows manually or use same_as(ref).
            if _PROMOTE_INT_TO_FLOAT_RE.match(val):
                err(
                    f"dtype_combos[{i}].{key} = {val!r} — combo values "
                    f"must be a single concrete dtype; "
                    f"promote_int_to_float(...) is allowed only on "
                    f"signature.outputs"
                )
                continue
            opts = _dtype_options_for_tensor(key, val, dtype_options)
            if opts is None:
                err(
                    f"dtype_combos[{i}].{key} = {val!r} is not a valid "
                    f"dtype (unresolved same_as reference or not in "
                    f"torch dtype set)"
                )
            elif not all(t in _TORCH_DTYPES for t in opts):
                bad = [t for t in opts if t not in _TORCH_DTYPES]
                err(
                    f"dtype_combos[{i}].{key} = {val!r} resolves to "
                    f"unknown dtype(s) {bad!r}"
                )
    return errors


# ---------------------------------------------------------------------------
# shape parity: _infer_output_shapes vs shape_rules (L2 extension)
# ---------------------------------------------------------------------------

# Default mock size for symbolic shape dims: small for cheap evaluation,
# 4 avoids degenerate cases (e.g. shape[0]==1 matching scalar broadcasts).
# Distinct symbolic dims get ``_MOCK_DIM_SIZE + counter`` so cross-tensor
# equality checks remain meaningful (see ``_mock_input_shapes``).
_MOCK_DIM_SIZE = 4

# Safety bound for Cartesian-product enumeration in L3 dtype parity: an op
# with many inputs × wide dtype unions could blow CI budgets. Over-bound
# ops are skipped deterministically with a warning (no sampling), so
# validation output stays reproducible.
_MAX_DTYPE_COMBOS = 4096

# Used only by the same_as-identity negative probe, which needs a dtype
# differing from the ref's baseline. Out-of-union probes derive their pool
# from ``_TORCH_DTYPES - declared`` instead.
_DTYPE_SENTINELS: tuple[str, ...] = (
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64",
)


def _out_of_union_candidates(declared: set[str]) -> list[str]:
    """Return torch dtypes outside ``declared``, sorted for reproducibility.

    Bounded because ``_TORCH_DTYPES`` is a fixed small set; callers still
    cap iteration via ``_MAX_DTYPE_COMBOS`` when combining with other
    enumeration.
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


_SHAPE_EQ_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\.shape\s*==\s*\(([^)]*)\)\s*$"
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _shape_eq_literals(rules: list) -> list[tuple[str, list[str]]]:
    """Extract ``<name>.shape == (<parts>...)`` literals from shape_rules.

    Returns ``(tensor_name, parts)`` pairs in rule order; empty parts are
    dropped. Only this simple literal form is recognized — consumers
    decide whether the parts must be bare identifiers.
    """
    out: list[tuple[str, list[str]]] = []
    for rule in rules:
        if not isinstance(rule, str):
            continue
        m = _SHAPE_EQ_RE.match(rule)
        if m is None:
            continue
        parts = [p.strip() for p in m.group(2).split(",") if p.strip()]
        out.append((m.group(1), parts))
    return out


def _extract_shape_tuple_literals(rules: list) -> dict[str, int]:
    """Parse ``<name>.shape == (<ids>...)`` rules for input-tensor rank hints.

    Only the all-bare-identifier form contributes a rank; other
    shape_rules patterns are skipped.
    """
    ranks: dict[str, int] = {}
    for name, parts in _shape_eq_literals(rules):
        if all(_IDENT_RE.fullmatch(p) for p in parts):
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
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    if not parts:
        return None
    if not all(_IDENT_RE.fullmatch(p) for p in parts):
        return None
    return parts


def _input_bound_symbols(sig: dict) -> set[str]:
    """Return symbolic dim names bound by INPUT shapes only.

    A symbol is input-bound when it appears in either a
    ``<input>.shape == (...)`` literal in ``signature.shape_rules`` or a
    ``signature.inputs[*].shape`` declaration like ``"[N, C, L]"``.
    Symbols that appear only in output shape declarations (e.g. a conv
    ``L_out`` derived by a shape_rules formula) are **not** included:
    the L2 parity check compares input-bound symbols against concrete
    mock sizes, while output-only symbols carry values derived by
    ``_infer_output_shapes`` — comparing those against arbitrary mock
    sizes would misreport a correct implementation.
    """
    bound: set[str] = set()
    rules = sig.get("shape_rules") or []
    inputs_raw = sig.get("inputs")
    inputs = inputs_raw if isinstance(inputs_raw, dict) else {}
    input_names = set(inputs.keys())
    for tname, parts in _shape_eq_literals(rules):
        if tname not in input_names:
            continue
        for p in parts:
            if _IDENT_RE.fullmatch(p):
                bound.add(p)
    # Per-tensor shape decl on inputs
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
    forms) and from ``signature.inputs[*].shape`` declarations, falling
    back to a default 2D shape when the rank is unknown. Returns
    ``(shapes, dim_sizes)`` where ``dim_sizes`` maps each symbolic dim
    name to the integer size used in the mock shapes, so callers can bind
    those names into a shape_rules evaluation context. Returns None only
    if ``signature.inputs`` is malformed.
    """
    inputs = sig.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return None
    rules = sig.get("shape_rules") or []
    rule_literals = _shape_eq_literals(rules)
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
    # Dim-name → size map shared across tensors. Distinct symbolic dims
    # get distinct sizes (first-seen order) so cross-tensor equality
    # rules do not spuriously pass on colliding mock sizes.
    dim_sizes: dict[str, int] = {}
    for _tname, parts in rule_literals:
        for p in parts:
            if _IDENT_RE.fullmatch(p) and p not in dim_sizes:
                dim_sizes[p] = _MOCK_DIM_SIZE + len(dim_sizes)

    # Also bind symbolic dims from input shape declarations, then from
    # declared output shapes, so downstream rule / shape-decl checks
    # resolve them against the same mock sizes.
    for parts in shape_decls.values():
        for p in parts:
            if p not in dim_sizes:
                dim_sizes[p] = _MOCK_DIM_SIZE + len(dim_sizes)
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
            # First matching rule literal for this tensor, if all parts
            # are bare identifiers.
            parts = next(
                (ps for tn, ps in rule_literals if tn == name), None,
            )
            if parts is not None and all(
                _IDENT_RE.fullmatch(p) for p in parts
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
    or a param name. Returns only successfully resolved entries
    (malformed / out-of-range entries are silently skipped — the L0
    schema check reports those). Used by parity mock_self builders so
    methods consulting ``self.<static_dim_name>`` see the concrete size
    carried by the synthetic inputs instead of raising AttributeError.
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

    Shapes are right-aligned; each dimension must be equal, or one of
    them must be 1 (or missing). Returns ``()`` when called with no
    arguments.

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

    Unlike the symmetric ``broadcast_shapes``, this predicate fixes the
    destination shape: each ``src`` dim (right-aligned) must equal the
    matching ``dst`` dim or be 1, and ``src`` may not have more
    dimensions than ``dst``.
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


# Safe builtins for shape_rules eval — the R11 / R11a helper set. Widening
# it widens the rule language, so keep it aligned with the manifest spec.
# An explicit pair list (not a dict merge) makes a name collision raise at
# import time instead of silently shadowing a primitive.
_SHAPE_RULE_BUILTIN_PAIRS = [
    ("len", len),
    ("isinstance", isinstance),
    ("int", int),
    # ``float`` lets manifest rules spell sentinels like
    # ``ord == float('inf')``. Add new callables only when an existing
    # manifest rule needs them and the semantics are obviously bounded.
    ("float", float),
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

    The eval globals expose the ``_SHAPE_RULE_BUILTINS`` helper set so
    R11 / R11a-style rules can be evaluated against the mock context
    instead of being silently skipped. Context names (inputs / outputs /
    params) are injected into both eval globals and locals: comprehension
    scopes only see globals, so rules like
    ``all(d % x.ndim in ... for d in dim)`` still resolve ``x`` / ``dim``.
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
    eval_globals.update(ctx)
    # A ctx key literally named ``__builtins__`` would overwrite the
    # sandboxed mapping installed above and re-expose the unrestricted
    # builtins; reinstate the sandbox after the update.
    eval_globals["__builtins__"] = _SHAPE_RULE_BUILTINS
    try:
        result = eval(
            rule, eval_globals, ctx,
        )
    except Exception as exc:
        return False, f"eval error: {exc.__class__.__name__}: {exc}"
    try:
        return bool(result), None
    except Exception as exc:
        return False, f"non-boolean result: {exc}"


def _build_mock_self(
    cls: type,
    param_defaults: dict,
    extra_attrs: dict | None = None,
) -> object:
    """Build a mock ``self`` instance without running ``__init__``.

    Uses ``cls.__new__(cls)`` so methods and helpers on the MRO remain
    reachable — a plain :class:`types.SimpleNamespace` cannot satisfy
    methods that read class attributes during a parity probe. Each
    ``param_defaults`` entry (from ``signature.params``) is installed as
    an instance attribute so ``self.<param>`` lookups resolve.
    ``extra_attrs`` carries probe-specific attributes installed last
    (overriding same-named params): static_dims values resolved from the
    synthetic mock inputs, and the dtype axis so ``self.dtype`` reflects
    the candidate combo instead of the ``Op.dtype = None`` base default.
    Falls back to SimpleNamespace if ``cls.__new__`` raises.
    """
    merged: dict = dict(param_defaults)
    if extra_attrs:
        merged.update(extra_attrs)
    try:
        instance = cls.__new__(cls)
    except Exception:
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

    A missing override is surfaced as a warning (no silent pass); a
    body-level exception after successful argument binding is a hard L2
    error (binding mismatches report separately as signature errors);
    concrete disagreement — rule violations or mismatches against a
    declared ``signature.outputs[*].shape`` — produces L2 errors.
    """
    errors: list[str] = []
    if cls is None:
        return errors
    err = _emit_to(errors, "shape", op_name)
    warn = _emit_to(warnings, "shape", op_name)

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
        warn(
            "class does not override _infer_output_shapes — "
            "manifest-derived method not yet generated; parity check "
            "skipped. Demote the op to 'status: spec-only' if the method "
            "genuinely cannot be exercised from the CPU validator."
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

    # Resolve static_dims against the mock inputs so implementations reading
    # ``self.<dim>`` do not AttributeError and silently skip the check.
    extra_attrs = _static_dim_values(sig, mock_shapes, param_defaults)
    mock_self = _build_mock_self(cls, param_defaults, extra_attrs)

    shape_kwargs = {f"{name}_shape": tuple(shape) for name, shape in mock_shapes.items()}
    # Bind before calling: only a TypeError from ``bind`` is a signature
    # mismatch. TypeErrors from the body must not be reported as one.
    try:
        inspect.signature(infer_fn).bind(mock_self, **shape_kwargs)
    except TypeError as exc:
        err(
            f"_infer_output_shapes signature does not match manifest "
            f"inputs (expected kwargs {sorted(shape_kwargs)}): {exc}"
        )
        return errors
    except Exception as exc:
        # signature() itself failed (e.g. builtin without introspection) —
        # skip parity rather than fabricating a signature error.
        warn(
            f"_infer_output_shapes parity skipped — inspect.signature "
            f"raised {exc.__class__.__name__}: {exc}"
        )
        return errors

    try:
        result = infer_fn(mock_self, **shape_kwargs)
    except Exception as exc:
        # Signature is valid but the body raised. A correct manifest-
        # derived ``_infer_output_shapes`` must succeed on manifest-
        # compatible mock inputs; treat any body-level exception as a
        # hard L2 parity error.
        err(
            f"_infer_output_shapes raised {exc.__class__.__name__} "
            f"under mock inputs {shape_kwargs}: {exc}"
        )
        return errors

    if not isinstance(result, dict):
        err(
            f"_infer_output_shapes must return a dict "
            f"(output_name -> shape), got {type(result).__name__}"
        )
        return errors

    outputs = sig.get("outputs") or {}
    for out_name in outputs:
        if out_name not in result:
            err(
                f"_infer_output_shapes missing output {out_name!r} "
                f"(declared in manifest)"
            )

    # Assemble evaluation context: symbolic dims + inputs + outputs +
    # params. Symbolic dim names are bound first so param / tensor names
    # later in the dict take precedence on any accidental collision.
    ctx: dict = {}
    ctx.update(dim_sizes)
    ctx.update(param_defaults)
    for name, shape in mock_shapes.items():
        ctx[name] = _MockShape(shape)
    # Rebind output-only symbols from the inferred ``result`` so their rules
    # check the computed value, not a synthetic mock size — otherwise a wrong
    # implementation looks like an input-only precondition and gets skipped.
    # First binding wins; the consistency check below flags conflicts.
    input_bound = _input_bound_symbols(sig)
    output_only_symbols: set[str] = set()
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
    for p, v in output_only_rebindings.items():
        ctx[p] = v
    # Rules failing on the mock inputs alone encode input-only preconditions
    # that mock shapes may violate; ``_infer_output_shapes`` is not to blame.
    # Output-only symbols are stripped so output-dependent rules can't land here.
    input_only_ctx: dict = {
        k: v for k, v in ctx.items() if k not in output_only_symbols
    }
    for out_name, out_shape in result.items():
        try:
            ctx[out_name] = _MockShape(tuple(out_shape))
        except TypeError:
            err(
                f"_infer_output_shapes returned non-iterable shape "
                f"for {out_name!r}: {out_shape!r}"
            )

    output_names = set(result.keys()) | set(outputs.keys())
    for i, rule in enumerate(rules):
        if not isinstance(rule, str):
            continue
        ok, reason = _eval_shape_rule(rule, ctx)
        if reason is not None:
            # Could not evaluate this rule under the mock context; do not
            # flag as parity mismatch.
            warn(
                f"shape_rules[{i}] could not be evaluated against mock "
                f"inputs ({reason}); rule: {rule!r}"
            )
            continue
        if not ok:
            # Fails with inputs only and references no output → the mock
            # shapes violate a precondition, not a parity mismatch.
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
                    warn(
                        f"shape_rules[{i}] {rule!r} not satisfied by "
                        f"synthetic mock inputs {shape_kwargs}; parity "
                        f"check skipped (input-only precondition)"
                    )
                    continue
            err(
                f"_infer_output_shapes output violates shape_rules[{i}] "
                f"{rule!r} under mock inputs {shape_kwargs} -> {result}"
            )

    # Independent of shape_rules, so ops specified only via declared shapes
    # are still covered. Input-bound and static-dim symbols pin exact sizes;
    # output-only symbols get rank plus per-symbol consistency instead.
    static_expected: dict[str, int] = {
        name: int(val) for name, val in extra_attrs.items()
        if isinstance(val, int) and not isinstance(val, bool)
    }
    # An int ``default`` is compile-time known, so it pins a dim with the same
    # authority as ``static_dims``. No default or non-int → cannot pin.
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
            err(
                f"_infer_output_shapes output {out_name!r} rank "
                f"{len(inferred)} disagrees with declared shape "
                f"{decl_parts} (rank {len(decl_parts)}) under mock "
                f"inputs {shape_kwargs} -> {inferred}"
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
                    err(
                        f"_infer_output_shapes output {out_name!r} "
                        f"dim[{idx}]={got} disagrees with declared "
                        f"{p}={expected} under mock inputs "
                        f"{shape_kwargs} -> {inferred}"
                    )
            else:
                # Derived by _infer_output_shapes — only enforce that the
                # symbol resolves to one size across all declared outputs.
                prev = output_only_seen.get(p)
                if prev is None:
                    output_only_seen[p] = got
                elif prev != got:
                    err(
                        f"_infer_output_shapes output {out_name!r} binds "
                        f"output-only symbol {p!r} to {got} but earlier "
                        f"output bound it to {prev} (inconsistent under "
                        f"mock inputs {shape_kwargs})"
                    )
    return errors


# ---------------------------------------------------------------------------
# dtype parity: _validate_dtypes vs dtype_combos / dtype unions (L3 extension)
# ---------------------------------------------------------------------------


def _expand_promote_int_to_float(ref_options: list[str]) -> list[str]:
    """Resolve ``promote_int_to_float(ref)`` against ``ref``'s dtype options.

    Each integral token in ``ref_options`` (uint8 / int8 / int16 / int32 /
    int64) maps to ``float32``; non-integral tokens pass through unchanged.
    Result is de-duplicated, preserving first-seen order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for opt in ref_options:
        target = _PROMOTE_TARGET_DTYPE if opt in _PROMOTE_INT_DTYPES else opt
        if target not in seen:
            seen.add(target)
            out.append(target)
    return out


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
    out: list[str] = []
    for tok in _parse_dtype_expr(dtype_str):
        m = _SAME_AS_RE.match(tok)
        if m:
            ref = m.group(1)
            # Unresolved reference — propagate failure per docstring
            # contract. Returning [] here would silently disable parity.
            if ref not in resolved:
                return None
            out.extend(resolved[ref])
            continue
        m = _PROMOTE_INT_TO_FLOAT_RE.match(tok)
        if m:
            ref = m.group(1)
            if ref not in resolved:
                return None
            out.extend(_expand_promote_int_to_float(resolved[ref]))
            continue
        if tok in _TORCH_DTYPES:
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

    When ``sig`` is provided, the mock-self is enriched with static_dims
    values resolved against synthetic mock inputs and with ``self.dtype``
    bound to the candidate's dtype axis — both commonly consulted by
    generated ``_validate_dtypes`` implementations (``if x.dtype !=
    self.dtype: raise``); without them the probe would spuriously reject
    listed combos. ``self_dtype_name`` pins ``mock_self.dtype``
    explicitly (out-of-union probes keep the op's configured dtype at a
    valid baseline while mutating the input tensor's dtype); when
    omitted it follows the combo entry for the first non-same_as-bound
    input.
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
        # self.dtype tracks the candidate's primary dtype (first non-same_as
        # input, or ``self_dtype_name``) so a derived _validate_dtypes comparing
        # ``x.dtype != self.dtype`` sees a real dtype, not the base-class None.
        # Out-of-union probes override it so only the input tensor deviates.
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
    except Exception as exc:
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
    except Exception as exc:
        # A correct ``_validate_dtypes`` either accepts or raises
        # ValueError/TypeError; anything else is an implementation bug.
        return False, f"unexpected {exc.__class__.__name__}: {exc}"
    return True, None


def _emit_to(sink, tag: str, op_name: str):
    """Return an emitter appending ``[tag] op_name: msg`` strings to *sink*.

    A ``None`` sink yields a no-op emitter so callers can bind a warning
    emitter without guarding every call site.
    """
    if sink is None:
        return lambda msg: None
    prefix = f"[{tag}] {op_name}: "
    return lambda msg: sink.append(prefix + msg)


# ``_combo_accepted`` reason prefixes → dispatch kinds. Callers branch on
# the kind instead of re-matching prefixes at every probe site.
_REASON_KINDS = (
    ("TypeError", "signature"),
    ("introspect-failed", "introspect"),
    ("unexpected", "unexpected"),
    ("cannot build mock tensor", "no-mock"),
    ("combo missing input", "missing-input"),
)


def _probe_reason_kind(reason: str | None) -> str | None:
    """Classify a ``_combo_accepted`` reason string by its prefix.

    Returns None for a clean probe (reason is None); ``"other"`` for an
    unrecognized reason (no caller acts on it, matching the previous
    per-site prefix cascades).
    """
    if reason is None:
        return None
    for prefix, kind in _REASON_KINDS:
        if reason.startswith(prefix):
            return kind
    return "other"


def _probe_out_of_union(
    op_name: str,
    cls: type,
    sig: dict,
    forward_inputs: list[str],
    baseline: dict[str, str],
    dtype_options: dict[str, list[str]],
    param_defaults: dict,
    errors: list[str],
    warnings: list[str] | None,
) -> None:
    """Out-of-union negative probe (rejection side), shared by both branches.

    Starting from *baseline* (a combination known to be accepted),
    substitutes an out-of-union sentinel for each non-same_as-bound input
    in turn; every candidate must be rejected. ``self.dtype`` stays
    pinned to the baseline's primary dtype so only the input tensor's
    dtype deviates (a generated ``x.dtype != self.dtype`` check would
    otherwise spuriously pass). same_as-bound tensors follow their ref
    via propagation. Bounded by ``_MAX_DTYPE_COMBOS``.
    """
    err = _emit_to(errors, "dtype", op_name)
    warn = _emit_to(warnings, "dtype", op_name)
    same_as_refs = _same_as_refs(sig)
    baseline_primary = _primary_dtype_input(sig, forward_inputs)
    baseline_self_dtype = (
        baseline.get(baseline_primary)
        if baseline_primary is not None else None
    )
    probed = 0
    for target in forward_inputs:
        if target in same_as_refs:
            continue
        declared = set(dtype_options.get(target, []))
        out_of_union = _out_of_union_candidates(declared)
        if not out_of_union:
            # Declared union covers the entire torch dtype set (wildly
            # permissive spec) — no rejection candidate exists. Warn
            # instead of vacuously passing.
            warn(
                f"out-of-union probe skipped for input {target!r} — "
                f"declared dtype union covers the entire torch dtype "
                f"set; rejection side cannot be exercised"
            )
            continue
        for bad_dtype in out_of_union:
            if probed >= _MAX_DTYPE_COMBOS:
                break
            probed += 1
            candidate = dict(baseline)
            candidate[target] = bad_dtype
            for tname, ref in same_as_refs.items():
                if ref == target and tname in candidate:
                    candidate[tname] = bad_dtype
            accepted, reason = _combo_accepted(
                cls, forward_inputs, candidate, param_defaults,
                sig=sig, self_dtype_name=baseline_self_dtype,
            )
            if _probe_reason_kind(reason) == "unexpected":
                err(
                    f"_validate_dtypes raised unexpected exception on "
                    f"out-of-union probe {candidate!r} — {reason}"
                )
                continue
            if accepted:
                err(
                    f"_validate_dtypes accepts out-of-union dtype "
                    f"{candidate!r} (input {target!r} declared "
                    f"{sorted(declared)})"
                )
        if probed >= _MAX_DTYPE_COMBOS:
            break


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
    pass).
    """
    errors: list[str] = []
    if cls is None:
        return errors
    err = _emit_to(errors, "dtype", op_name)
    warn = _emit_to(warnings, "dtype", op_name)

    if not _class_overrides_method(cls, "_validate_dtypes"):
        warn(
            "class does not override _validate_dtypes — manifest-derived "
            "method not yet generated; parity check skipped. Demote the "
            "op to 'status: spec-only' if the method genuinely cannot be "
            "exercised from the CPU validator."
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
        # so downstream parity probing does not run on junk data. The
        # same check also runs unconditionally in ``check_l3`` — the
        # driver dedupes error strings so users see each message once.
        combo_validation_errors = check_l3_dtype_combos_data(op_name, sig)
        if combo_validation_errors:
            errors.extend(combo_validation_errors)
            return errors

        # ``_combo_accepted`` expects literal dtype names. R3 + R4 identity is
        # already enforced, so ``same_as(ref)`` is the ref's dtype in this row.
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
            kind = _probe_reason_kind(reason)
            if kind == "signature":
                err(
                    f"_validate_dtypes signature does not match manifest "
                    f"inputs (expected kwargs {sorted(forward_inputs)}): "
                    f"{reason}"
                )
                return errors
            if kind in ("introspect", "no-mock"):
                # Validator-side limitation (inspect.signature failed or
                # the local torch build lacks the dtype) — skip with a
                # parity-skip warning, not an op-side error.
                warn(
                    f"_validate_dtypes parity skipped for "
                    f"dtype_combos[{i}] — {reason}"
                )
                continue
            if kind == "unexpected":
                # Body-level exception that is not ValueError / TypeError
                # — a real implementation bug. Hard L3 parity error.
                err(
                    f"_validate_dtypes raised unexpected exception on "
                    f"dtype_combos[{i}] {combo!r} — {reason}"
                )
                continue
            if kind == "missing-input":
                # Manifest error: combo doesn't specify a dtype for every
                # declared input. Parity error, not a rejection.
                err(f"dtype_combos[{i}] {combo!r} {reason}")
                continue
            if not accepted:
                err(
                    f"_validate_dtypes rejects dtype_combos[{i}] "
                    f"{combo!r} listed in manifest"
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
            warn(
                f"Cartesian product of dtype options ({product_size}) "
                f"exceeds _MAX_DTYPE_COMBOS={_MAX_DTYPE_COMBOS}; "
                f"non-listed rejection check skipped "
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
            kind = _probe_reason_kind(reason)
            if kind in ("introspect", "signature"):
                continue
            if kind == "unexpected":
                err(
                    f"_validate_dtypes raised unexpected exception on "
                    f"non-listed combo {candidate!r} — {reason}"
                )
                continue
            if not accepted:
                rejected_at_least_one = True
                continue
            # Accepted non-listed combo — parity violation. Keep scanning
            # so multiple such combos are all surfaced in a single run.
            err(
                f"_validate_dtypes accepts non-listed combo "
                f"{candidate!r} (not in dtype_combos)"
            )

        # Out-of-union negative probe: baseline is the first listed combo
        # covering every input (known to be accepted).
        baseline_combo: dict[str, str] | None = None
        for c in dtype_combos:
            if isinstance(c, dict) and all(
                n in c for n in forward_inputs
            ):
                baseline_combo = dict(c)
                break
        if baseline_combo is not None:
            _probe_out_of_union(
                op_name, cls, sig, forward_inputs, baseline_combo,
                dtype_options, param_defaults, errors, warnings,
            )

        if not errors:
            if not checked_any:
                # No non-listed combo exists in the Cartesian product —
                # dtype_combos already enumerates every reachable tuple.
                warn(
                    "could not find a non-listed combo to exercise "
                    "rejection (dtype_combos exhausts the union)"
                )
            elif not rejected_at_least_one:
                # Non-listed combos were tried but none were rejected —
                # either _validate_dtypes is too lax or every non-listed
                # candidate was skipped (unexpected/TypeError).
                warn(
                    "no non-listed dtype combo was rejected by "
                    "_validate_dtypes; parity coverage may be incomplete"
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
            warn(
                f"Cartesian product of dtype options ({product_size}) "
                f"exceeds _MAX_DTYPE_COMBOS={_MAX_DTYPE_COMBOS}; parity "
                f"check skipped ({len(forward_inputs)} inputs × options "
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
            kind = _probe_reason_kind(reason)
            if kind == "signature":
                # Signature mismatch between manifest inputs and the op's
                # _validate_dtypes — surface as a parity error (analogous
                # to the L2 _infer_output_shapes signature check).
                err(
                    f"_validate_dtypes signature does not match manifest "
                    f"inputs (expected kwargs {sorted(forward_inputs)}): "
                    f"{reason}"
                )
                return errors
            if kind == "introspect":
                warn(
                    f"_validate_dtypes parity skipped for combo "
                    f"{candidate!r} — {reason}"
                )
                continue
            if kind == "unexpected":
                # Body-level unexpected exception — hard error.
                # See ``_combo_accepted`` docstring.
                err(
                    f"_validate_dtypes raised unexpected exception on "
                    f"combo {candidate!r} — {reason}"
                )
                continue
            if not accepted:
                err(
                    f"_validate_dtypes rejects valid combo {candidate!r} "
                    f"drawn from manifest dtype unions"
                )

        # Out-of-union negative probe: baseline is the first
        # same_as-honouring candidate from the union.
        baseline: dict[str, str] | None = None
        for tup in itertools.product(*input_options):
            cand = dict(zip(forward_inputs, tup, strict=True))
            if _honours_same_as(sig, cand):
                baseline = cand
                break
        if baseline is not None:
            _probe_out_of_union(
                op_name, cls, sig, forward_inputs, baseline,
                dtype_options, param_defaults, errors, warnings,
            )

        # same_as identity negative probe (R3 rejection side): each
        # same_as(ref) input gets a candidate deviating from its ref, which
        # must be rejected. Complements the ``_honours_same_as`` skip above.
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
                    kind = _probe_reason_kind(reason)
                    if kind in ("introspect", "signature"):
                        continue
                    if kind == "unexpected":
                        err(
                            f"_validate_dtypes raised unexpected "
                            f"exception on same_as probe {candidate!r} "
                            f"— {reason}"
                        )
                        continue
                    if accepted:
                        err(
                            f"_validate_dtypes accepts same_as violation "
                            f"{candidate!r} (input {tname!r} declared "
                            f"same_as({ref}))"
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

    Recognises the direct pattern (``load_workloads`` from
    ``tileops.manifest`` called with the op name + ``op.eval_roofline()``)
    and the indirect one (``workloads_to_params`` / ``ManifestBenchmark``
    from ``benchmarks.benchmark_base``, op name as first argument).
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
# Strict parity checks for status: implemented ops
# ---------------------------------------------------------------------------
# C1 shape parity and C2 dtype parity live in ``check_l2_infer_parity`` /
# ``check_l3_validate_dtypes_parity``. This block adds:
#   C3 ctor signature parity (defaults + kw-only beyond L1 names)
#   C4 forward positional names match ``signature.inputs`` order
#   C5 ``dispatch_kernel`` sentinel pass-through
#   C6 / C7 ``_validate_dtypes`` / ``eval_roofline`` are not the base stubs

# Infrastructure params that the validator filters out of ctor parity:
# they never appear in manifest ``signature.params`` but are part of the
# Op interface contract.
_CTOR_INFRA_PARAMS = frozenset({"self", "kernel_map", "tune"})

# Sentinel for "manifest did not declare this attribute" — distinct from
# any legitimate manifest value (including the string "REQUIRED" used to
# explicitly mark a parameter as required).
_MISSING = object()


def _init_calls_dispatch_kernel(cls: type) -> "bool | None":
    """Return True if ``cls.__init__`` body either calls
    ``self.dispatch_kernel(...)`` directly or delegates via
    ``super().__init__(...)``.

    ``None`` means the source could not be parsed (built-in /
    dynamically generated ``__init__``); callers treat that as
    inconclusive. Pure-AST inspection per the Slot S13 contract in
    ``docs/design/ops-design-reference.md``; ``super().__init__(...)``
    satisfies S13 transitively. No runtime construction; no GPU.
    """
    try:
        lines, start_lineno = inspect.getsourcelines(cls.__init__)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except SyntaxError:
        return None
    code = getattr(cls.__init__, "__code__", None)
    target_lineno = getattr(code, "co_firstlineno", None)
    init_node = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        if target_lineno is None or start_lineno + node.lineno - 1 == target_lineno:
            init_node = node
            break
    if init_node is None:
        return None

    for node in ast.walk(init_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        # self.dispatch_kernel(...)
        if (
            node.func.attr == "dispatch_kernel"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            return True
        # super().__init__(...): delegates to the parent's body which is
        # expected to honor S13 itself.
        if (
            node.func.attr == "__init__"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            return True
    return False


def check_c3_ctor_signature_parity(
    op_name: str,
    entry: dict,
    cls: type | None,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """C3: ctor parameters match manifest ``signature.params``.

    Compares the names, defaults, and keyword-only flag of every
    ``__init__`` parameter (after stripping ``_CTOR_INFRA_PARAMS``)
    against ``signature.params``. L1 already covers presence; this
    check adds the default-value and kw-only contracts.
    """
    errors: list[str] = []
    if cls is None:
        return errors
    err = _emit_to(errors, "ctor", op_name)

    sig = entry.get("signature", {})
    manifest_params = sig.get("params") or {}
    if not isinstance(manifest_params, dict):
        return errors

    try:
        py_sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError) as exc:
        if warnings is not None:
            warnings.append(
                f"[ctor] {op_name}: inspect.signature(__init__) raised "
                f"{exc.__class__.__name__}: {exc}"
            )
        return errors

    code_params: dict[str, inspect.Parameter] = {}
    for pname, p in py_sig.parameters.items():
        if pname in _CTOR_INFRA_PARAMS:
            continue
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        code_params[pname] = p

    for pname, pattrs in manifest_params.items():
        if pname not in code_params:
            # L1 already reports missing params; do not double-fire.
            continue
        if not isinstance(pattrs, dict):
            continue
        code_p = code_params[pname]

        # A declared default must match the ctor default value-for-value;
        # ``REQUIRED`` or absent means no manifest default.
        manifest_default = pattrs.get("default", _MISSING)
        manifest_has_default = (
            manifest_default is not _MISSING and manifest_default != "REQUIRED"
        )
        # A ``torch.dtype`` default is spelled by name in YAML ("float16"),
        # so compare the resolved dtype rather than the spelling. Without this
        # no entry can declare a dtype default at all.
        if (
            manifest_has_default
            and pattrs.get("type") == "torch.dtype"
            and isinstance(manifest_default, str)
        ):
            import torch

            resolved = getattr(torch, manifest_default, None)
            if isinstance(resolved, torch.dtype):
                manifest_default = resolved
        code_has_default = code_p.default is not inspect.Parameter.empty
        if manifest_has_default and not code_has_default:
            err(
                f"param {pname!r} has manifest default "
                f"{manifest_default!r} but no default on __init__"
            )
        elif (not manifest_has_default) and code_has_default:
            err(
                f"param {pname!r} has __init__ default "
                f"{code_p.default!r} but no manifest default"
            )
        elif (
            manifest_has_default
            and code_has_default
            and code_p.default != manifest_default
        ):
            err(
                f"param {pname!r} default mismatch — "
                f"manifest={manifest_default!r}, code={code_p.default!r}"
            )

        # Keyword-only parity: manifest may declare ``kw_only: true``.
        manifest_kw_only = bool(pattrs.get("kw_only", False))
        code_kw_only = code_p.kind is inspect.Parameter.KEYWORD_ONLY
        if manifest_kw_only != code_kw_only:
            err(
                f"param {pname!r} kw_only mismatch — "
                f"manifest={manifest_kw_only}, code={code_kw_only}"
            )

    # Code-only extras detection deferred: a faithful "kwarg not declared
    # anywhere in the manifest" rule needs the protocol-derived allowed
    # set (params + static_dims + per-family shape/dtype variables),
    # which is out of scope for the C3 helper. Strict-parity follow-up.

    return errors


def check_c4_forward_signature_parity(
    op_name: str,
    entry: dict,
    cls: type | None,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """C4: forward positional names match ``signature.inputs`` order.

    L1 already enforces this for the legacy signature path; the C4
    function exists so the strict gate can run independently of the
    older ``check_l1`` and report under the ``[forward]`` tag for
    follow-up triage.
    """
    errors: list[str] = []
    if cls is None:
        return errors

    sig = entry.get("signature", {})
    manifest_inputs = sig.get("inputs") or {}
    if not isinstance(manifest_inputs, dict):
        return errors
    expected = list(manifest_inputs.keys())

    positional = _forward_positional_params(cls)
    if positional is None:
        if warnings is not None:
            detail = getattr(
                _forward_positional_params, "_last_error", None
            )
            if detail:
                warnings.append(
                    f"[forward] {op_name}: inspect.signature(forward) "
                    f"raised {detail}"
                )
                # Clear so a later call site sees only its own failure.
                _forward_positional_params._last_error = None  # type: ignore[attr-defined]
            else:
                warnings.append(
                    f"[forward] {op_name}: inspect.signature(forward) failed"
                )
        return errors

    actual_prefix = positional[: len(expected)]
    if actual_prefix != expected:
        errors.append(
            f"[forward] {op_name}: forward() positional names "
            f"{positional!r} do not start with manifest inputs "
            f"{expected!r}"
        )
    return errors


def check_c5_dispatch_kernel_invariant(
    op_name: str,
    entry: dict,
    cls: type | None,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """C5: ``__init__`` complies with the dispatch-kernel slot contract.

    Two static checks per ``docs/design/ops-design-reference.md``:

    - **S12** — ``__init__`` accepts a ``kernel_map`` keyword (or
      ``**kwargs`` that absorbs it).
    - **S13** — ``__init__`` body contains a call ``self.dispatch_kernel(...)``.

    Pure inspection: ``inspect.signature`` for S12, AST walk for S13.
    No runtime construction, no GPU, no JIT — both contracts are
    declarative properties of the source code.

    Source-unavailable ``__init__`` (built-in / dynamically generated /
    decorator-wrapped without ``__wrapped__``) degrades S13 to advisory;
    S12 always runs.
    """
    errors: list[str] = []
    if cls is None:
        return errors

    # S12: signature carries kernel_map (or **kwargs).
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError) as exc:
        if warnings is not None:
            warnings.append(
                f"[dispatch] {op_name}: inspect.signature(__init__) raised "
                f"{exc.__class__.__name__}: {exc}; advisory"
            )
        return errors
    has_kernel_map_kw = "kernel_map" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_kernel_map_kw:
        errors.append(
            f"[dispatch] {op_name}: __init__ does not accept a "
            f"'kernel_map' parameter (Slot S12) — kernel-map override is "
            f"unreachable"
        )
        return errors

    # S13: body calls self.dispatch_kernel(...).
    body_calls = _init_calls_dispatch_kernel(cls)
    if body_calls is None:
        if warnings is not None:
            warnings.append(
                f"[dispatch] {op_name}: __init__ source unavailable "
                f"(built-in or dynamically generated); S13 advisory"
            )
        return errors
    if not body_calls:
        errors.append(
            f"[dispatch] {op_name}: __init__ body does not call "
            f"self.dispatch_kernel(...) (Slot S13) — kernel_map override "
            f"is silently dropped"
        )
    return errors


def check_c6_validate_dtypes_not_stub(
    op_name: str, entry: dict, cls: type | None,
) -> list[str]:
    """C6: ``_validate_dtypes`` is not the base ``Op`` stub."""
    if cls is None:
        return []
    from tileops.ops.op_base import Op as _OpBase
    if cls._validate_dtypes is _OpBase._validate_dtypes:
        return [
            f"[stub] {op_name}: _validate_dtypes is the Op base stub "
            f"(not implemented by the concrete class)"
        ]
    return []


def check_c7_eval_roofline_not_stub(
    op_name: str, entry: dict, cls: type | None,
) -> list[str]:
    """C7: ``eval_roofline`` is not the base ``Op`` stub."""
    if cls is None:
        return []
    from tileops.ops.op_base import Op as _OpBase
    if cls.eval_roofline is _OpBase.eval_roofline:
        return [
            f"[stub] {op_name}: eval_roofline is the Op base stub "
            f"(not implemented by the concrete class)"
        ]
    return []


# Triage aids only — routing is structural (the orchestrator extends
# ``strict_errors`` with each check's return). ``[shape]`` / ``[dtype]`` also
# come from non-strict L2 / L3, so assert leakage via ``STRICT_ONLY_TAGS``.
STRICT_TAGS: tuple[str, ...] = (
    "[shape]", "[dtype]", "[ctor]", "[forward]", "[dispatch]", "[stub]",
)

# Subset of ``STRICT_TAGS`` that only strict-parity checks emit.
STRICT_ONLY_TAGS: tuple[str, ...] = (
    "[ctor]", "[forward]", "[dispatch]", "[stub]",
)


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
    """Whether the entry claims its benchmark reads manifest workloads."""
    return bool(entry.get("source", {}).get("bench_manifest_driven", False))


def check_bench_declaration(op_name: str, entry: dict) -> list[str]:
    """Require every implemented op with a bench to declare the L4 contract.

    Omitting ``source.bench_manifest_driven`` downgrades the L4 AST check to a
    warning, so leaving it unset is an opt-out from the benchmark contract
    rather than a neutral default. Implemented ops must declare it.
    """
    if entry.get("status") != "implemented":
        return []
    source = entry.get("source") or {}
    if not source.get("bench"):
        return []
    if _is_bench_manifest_driven(entry):
        return []
    return [
        f"[bench] {op_name}: source.bench_manifest_driven must be declared "
        f"true — implemented ops may not opt out of the manifest-driven "
        f"benchmark contract"
    ]


ALL_LEVELS = frozenset({"schema", "signature", "shape", "dtype", "bench"})


def validate_manifest(
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
    verbose: bool = False,
    levels: frozenset[str] | None = None,
    check_op: str | None = None,
    strict_parity: bool = False,
) -> tuple[list[str], list[str]]:
    """Run applicable validation levels on the manifest.

    Returns ``(errors, warnings)``: errors are hard failures; warnings
    are informational. ``manifest_path=None`` loads the merged manifest
    from the ``tileops.manifest`` package (tests pass a temp file for
    synthetic single-file manifests). ``levels=None`` enables all
    checks. ``check_op`` forces all levels (L0-L4) on the named op and
    its variants, ignoring ``status``; all other ops are skipped.
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

    # --check-op scope: the named op plus its immediate variants, so a
    # variant edit is caught when validating the primary.
    variant_family: set[str] | None = None
    if check_op is not None:
        variant_family = {check_op} | {
            name for name, ent in ops.items()
            if isinstance(ent, dict) and ent.get("variant_of") == check_op
        }

    all_errors: list[str] = []
    all_warnings: list[str] = []
    # Strict-parity (C1-C7) failures: collected separately so the
    # orchestrator can route them to either errors (strict mode) or
    # warnings (advisory mode) once all per-op checks have run.
    strict_errors: list[str] = []

    # Cross-entry checks (must run before per-entry checks), scoped to
    # the variant family under --check-op so unrelated ops with invalid
    # variant_of references don't fail the selected op.
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
            schema_errors = check_l0(
                op_name, entry, warnings=all_warnings, all_op_names=ops.keys(),
            )
            schema_errors.extend(check_source_paths(op_name, entry, repo_root))
            all_errors.extend(schema_errors)
            if schema_errors:
                continue

        spec_only = _is_spec_only(entry)
        if spec_only and check_op is None:
            if verbose:
                print(f"    {op_name}: spec-only, skipping signature/shape/dtype/bench")
            continue

        # Resolve Op class once per entry so parity checks can reuse it.
        source = entry.get("source", {})
        op_file = source.get("op", "")
        resolve_result = _resolve_op_class(op_file, op_name) if op_file else None
        op_cls = resolve_result.cls if resolve_result is not None else None

        # signature: Op.forward() consistency
        if "signature" in levels:
            all_errors.extend(check_l1(op_name, entry, warnings=all_warnings))

        # shape: shape_rules syntax + _infer_output_shapes parity (C1)
        if "shape" in levels:
            all_errors.extend(check_l2(op_name, entry))
            strict_errors.extend(
                check_l2_infer_parity(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )

        # dtype: dtype string conformance + _validate_dtypes parity (C2)
        if "dtype" in levels:
            all_errors.extend(check_l3(op_name, entry))
            strict_errors.extend(
                check_l3_validate_dtypes_parity(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )

        # C3-C7: strict parity gates for status: implemented ops, each
        # gated by the level whose contract it enforces (``--levels
        # schema`` triggers no strict-parity work). Routed to
        # strict_errors so advisory mode can downgrade them.
        if "signature" in levels:
            strict_errors.extend(
                check_c3_ctor_signature_parity(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )
            strict_errors.extend(
                check_c4_forward_signature_parity(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )
            strict_errors.extend(
                check_c5_dispatch_kernel_invariant(
                    op_name, entry, op_cls, warnings=all_warnings,
                )
            )
        if "dtype" in levels:
            strict_errors.extend(
                check_c6_validate_dtypes_not_stub(op_name, entry, op_cls)
            )
        if "bench" in levels:
            strict_errors.extend(
                check_c7_eval_roofline_not_stub(op_name, entry, op_cls)
            )

        # bench: benchmark uses manifest workloads
        if "bench" in levels:
            all_errors.extend(check_bench_declaration(op_name, entry))
            bench_path = entry.get("source", {}).get("bench", "")
            if bench_path:
                bench_errors = check_l4_benchmark(op_name, bench_path, repo_root)
                if _is_bench_manifest_driven(entry):
                    all_errors.extend(bench_errors)
                else:
                    all_warnings.extend(bench_errors)

    # Deduplicate while preserving order: ``check_l3`` and
    # ``check_l3_validate_dtypes_parity`` both surface ``dtype_combos``
    # data errors (each is a valid standalone entry point).
    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    # Route strict-parity (C1-C7) failures: in strict mode they are
    # blocking errors; in advisory mode they downgrade to warnings so
    # the gate can land before all current main violations are fixed.
    if strict_parity:
        all_errors.extend(strict_errors)
    else:
        for s in strict_errors:
            all_warnings.append(f"STRICT-PARITY (advisory): {s}")

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
    import os

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    levels = _parse_levels(sys.argv)
    check_op = _parse_check_op(sys.argv)
    strict_parity = (
        "--strict" in sys.argv
        or os.environ.get("MANIFEST_STRICT_BLOCKING", "") == "1"
    )

    level_label = ",".join(sorted(levels)) if levels else "all"
    check_op_label = f", check-op: {check_op}" if check_op else ""
    mode_label = "STRICT" if strict_parity else "ADVISORY"
    print(
        f"Validating {MANIFEST_DIR.relative_to(REPO_ROOT)}/*.yaml "
        f"(levels: {level_label}{check_op_label}, parity-mode: {mode_label})..."
    )
    if not strict_parity:
        print(
            "ADVISORY MODE — strict-parity (C1-C7) failures are reported "
            "as warnings and do NOT block. Pass --strict (or set "
            "MANIFEST_STRICT_BLOCKING=1) to make them blocking."
        )

    errors, warnings = validate_manifest(
        verbose=verbose, levels=levels, check_op=check_op,
        strict_parity=strict_parity,
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
