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
    python scripts/validate_manifest.py [--verbose] [--levels schema,shape,dtype,bench]

Exit code 0 = all checks pass; 1 = failures found.

The --levels flag selects which checks to run. When omitted, all are enabled.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
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
_REQUIRED_TOP = {"family", "signature", "workloads", "roofline", "source"}
_REQUIRED_SIGNATURE = {"inputs", "outputs"}
_REQUIRED_SOURCE = {"kernel", "op", "test", "bench"}


# ---------------------------------------------------------------------------
# schema: YAML structure validation
# ---------------------------------------------------------------------------

def check_l0(op_name: str, entry: dict) -> list[str]:
    """Validate structural schema of a manifest entry. Returns error strings."""
    errors: list[str] = []

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

        # Params must be a mapping if present
        if "params" in sig and not isinstance(sig["params"], dict):
            errors.append(
                f"[schema] {op_name}: signature.params must be a mapping"
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
        if "bench_manifest_driven" in source and not isinstance(
            source["bench_manifest_driven"], bool,
        ):
            errors.append(
                f"[schema] {op_name}: source.bench_manifest_driven must be a bool"
            )
    elif "source" in entry:
        errors.append(f"[schema] {op_name}: source must be a mapping")

    return errors


# ---------------------------------------------------------------------------
# signature: Op.forward() vs manifest consistency
# ---------------------------------------------------------------------------

def check_l1_signature(
    op_name: str,
    manifest_inputs: dict,
    manifest_params: dict,
    forward_params: list[str],
) -> list[str]:
    """Check that forward() params match manifest inputs + params.

    Args:
        op_name: Manifest op name.
        manifest_inputs: The signature.inputs dict from manifest.
        manifest_params: The signature.params dict from manifest.
        forward_params: List of parameter names from Op.forward() (excluding 'self').

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

    expected = list(manifest_inputs.keys()) + [
        name for name in manifest_params.keys() if name in forward_params
    ]
    if forward_params != expected:
        errors.append(
            f"[signature] {op_name}: forward() params {forward_params} do not match "
            f"manifest order {expected}"
        )

    return errors


class _ResolveResult:
    """Result of attempting to resolve an Op class from a module path."""

    __slots__ = ("cls", "import_error")

    def __init__(self, cls=None, import_error: bool = False):
        self.cls = cls
        self.import_error = import_error


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
    candidates = []
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if hasattr(obj, "forward") and callable(obj.forward):
            candidates.append(obj)

    if not candidates:
        return _ResolveResult()

    # If the op_name gives a hint (e.g., "batchnorm_fwd" -> "Fwd", "batchnorm_bwd" -> "Bwd")
    # try to match more precisely
    if len(candidates) == 1:
        return _ResolveResult(cls=candidates[0])

    # Multiple candidates — try to match by name convention
    # e.g., batchnorm_fwd -> BatchNormFwdOp, batchnorm_bwd -> BatchNormBwdOp
    for cls in candidates:
        cls_lower = cls.__name__.lower()
        # Check if the op_name parts appear in the class name
        parts = op_name.split("_")
        if all(p in cls_lower for p in parts):
            return _ResolveResult(cls=cls)

    # Fallback: use suffix matching for fwd/bwd
    if op_name.endswith("_fwd"):
        for cls in candidates:
            if "fwd" in cls.__name__.lower():
                return _ResolveResult(cls=cls)
    if op_name.endswith("_bwd"):
        for cls in candidates:
            if "bwd" in cls.__name__.lower():
                return _ResolveResult(cls=cls)

    return _ResolveResult(cls=candidates[0]) if candidates else _ResolveResult()


def _get_forward_params(cls) -> list[str] | None:
    """Get parameter names of cls.forward(), excluding 'self'."""
    try:
        sig = inspect.signature(cls.forward)
        return [p for p in sig.parameters if p != "self"]
    except (ValueError, TypeError):
        return None


def check_l1(
    op_name: str, entry: dict, *, warnings: list[str] | None = None,
) -> list[str]:
    """Signature check: resolve Op class and compare forward() to manifest.

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

    return check_l1_signature(op_name, manifest_inputs, manifest_params, forward_params)


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


def check_l3(op_name: str, entry: dict) -> list[str]:
    """Validate dtype strings are recognized torch types or same_as references.

    Checks both signature tensor dtypes and workload dtype entries.
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
    """Check whether target functions are imported and called with this op name."""
    imported: set[str] = set()
    matched_calls: set[str] = set()
    bindings = _resolve_constant_str_bindings(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "tileops.manifest" and node.names:
                for alias in node.names:
                    if alias.name in target_names:
                        imported.add(alias.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in target_names and _call_uses_expected_op_name(
                node, op_name, bindings,
            ):
                matched_calls.add(func_name)

    return {name: (name in imported and name in matched_calls) for name in target_names}


def check_l4_benchmark(
    op_name: str, bench_path: str, repo_root: Path,
) -> tuple[list[str], list[str]]:
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
    """Spec-only behavior must be explicit in the manifest status."""
    return entry.get("status") == "spec-only"


def _is_bench_manifest_driven(entry: dict) -> bool:
    """Bench strictness is opt-in until all legacy benchmarks are migrated."""
    return bool(entry.get("source", {}).get("bench_manifest_driven", False))


ALL_LEVELS = frozenset({"schema", "signature", "shape", "dtype", "bench"})


def validate_manifest(
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
    verbose: bool = False,
    levels: frozenset[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Run applicable validation levels on the manifest.

    Args:
        manifest_path: Path to ops_manifest.yaml.
        repo_root: Repository root directory.
        verbose: If True, print progress.
        levels: Set of check names to run (e.g. {"schema", "shape", "dtype", "bench"}).
                When None, all checks are enabled.

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
    all_errors: list[str] = []
    all_warnings: list[str] = []

    for op_name, entry in ops.items():
        if verbose:
            print(f"  Checking {op_name}...")

        # schema: YAML structure validation
        if "schema" in levels:
            schema_errors = check_l0(op_name, entry)
            all_errors.extend(schema_errors)
            if schema_errors:
                continue

        spec_only = _is_spec_only(entry)
        if spec_only:
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


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    levels = _parse_levels(sys.argv)

    level_label = ",".join(sorted(levels)) if levels else "all"
    print(
        f"Validating {MANIFEST_PATH.relative_to(REPO_ROOT)} "
        f"(levels: {level_label})..."
    )

    errors, warnings = validate_manifest(verbose=verbose, levels=levels)

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
