#!/usr/bin/env python3
"""Validate ops_manifest.yaml with L0-L4 checks.

Levels:
  L0 — YAML schema: required fields, types, structure
  L1 — Signature consistency: Op.forward() params match manifest inputs+params
  L2 — Shape rules: shape_rules are parseable Python expressions
  L3 — Dtype conformance: dtype strings are valid torch dtype names or references
  L4 — Benchmark file: bench file imports load_workloads from tileops.manifest

Spec-only ops (source files don't exist) get L0 only.
Implemented ops get all checks.

Usage:
    python scripts/validate_manifest.py [--verbose] [--levels L0,L2,L3,L4]

Exit code 0 = all checks pass; 1 = failures found.

The --levels flag selects which validation levels to run. When omitted, all
levels (L0-L4) are enabled. This allows lightweight CI environments to
explicitly exclude levels that require heavy dependencies (e.g. L1 needs
the full tileops runtime to import Op modules).
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
# L0: YAML schema validation
# ---------------------------------------------------------------------------

def check_l0(op_name: str, entry: dict) -> list[str]:
    """Validate structural schema of a manifest entry. Returns error strings."""
    errors: list[str] = []

    # Top-level required fields
    missing_top = _REQUIRED_TOP - set(entry.keys())
    if missing_top:
        errors.append(f"[L0] {op_name}: missing top-level fields: {missing_top}")

    # Signature structure
    sig = entry.get("signature")
    if isinstance(sig, dict):
        missing_sig = _REQUIRED_SIGNATURE - set(sig.keys())
        if missing_sig:
            errors.append(f"[L0] {op_name}: signature missing: {missing_sig}")

        # Check inputs/outputs are dicts with dtype
        for direction in ("inputs", "outputs"):
            tensors = sig.get(direction)
            if not isinstance(tensors, dict):
                if direction in sig:
                    errors.append(
                        f"[L0] {op_name}: signature.{direction} must be a dict"
                    )
                continue
            for tname, attrs in tensors.items():
                if not isinstance(attrs, dict):
                    errors.append(
                        f"[L0] {op_name}: {direction}.{tname} must be a dict"
                    )
                    continue
                if "dtype" not in attrs:
                    errors.append(
                        f"[L0] {op_name}: {direction}.{tname} missing 'dtype'"
                    )

        # Params must be a mapping if present
        if "params" in sig and not isinstance(sig["params"], dict):
            errors.append(
                f"[L0] {op_name}: signature.params must be a mapping"
            )

        # shape_rules must be list of strings if present
        if "shape_rules" in sig:
            rules = sig["shape_rules"]
            if not isinstance(rules, list):
                errors.append(f"[L0] {op_name}: shape_rules must be a list")
            else:
                for i, rule in enumerate(rules):
                    if not isinstance(rule, str):
                        errors.append(
                            f"[L0] {op_name}: shape_rules[{i}] must be a string"
                        )
    elif "signature" in entry:
        errors.append(f"[L0] {op_name}: signature must be a mapping")

    # Workloads
    workloads = entry.get("workloads")
    if isinstance(workloads, list):
        for i, w in enumerate(workloads):
            if not isinstance(w, dict):
                errors.append(f"[L0] {op_name}: workloads[{i}] must be a dict")
                continue
            if "dtypes" not in w:
                errors.append(
                    f"[L0] {op_name}: workloads[{i}] missing 'dtypes'"
                )
    elif "workloads" in entry:
        errors.append(f"[L0] {op_name}: workloads must be a list")

    # Roofline
    roofline = entry.get("roofline")
    if isinstance(roofline, dict):
        has_inline = "flops" in roofline and "bytes" in roofline
        has_func = "func" in roofline
        if not has_inline and not has_func:
            errors.append(
                f"[L0] {op_name}: roofline must have (flops + bytes) or func"
            )
    elif "roofline" in entry:
        errors.append(f"[L0] {op_name}: roofline must be a mapping")

    # Source
    source = entry.get("source")
    if isinstance(source, dict):
        missing_src = _REQUIRED_SOURCE - set(source.keys())
        if missing_src:
            errors.append(
                f"[L0] {op_name}: source missing fields: {missing_src}"
            )
    elif "source" in entry:
        errors.append(f"[L0] {op_name}: source must be a mapping")

    return errors


# ---------------------------------------------------------------------------
# L1: Signature consistency (Op.forward() vs manifest)
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

    # Guard: manifest_params must be a dict (L0 should catch this, but be safe)
    if not isinstance(manifest_params, dict):
        errors.append(
            f"[L1] {op_name}: signature.params is not a mapping, "
            f"cannot validate forward() consistency"
        )
        return errors

    # Expected forward params = manifest inputs keys + manifest params keys
    expected = set(manifest_inputs.keys()) | set(manifest_params.keys())
    actual = set(forward_params)

    extra_in_forward = actual - expected

    # Inputs must appear in forward()
    missing_inputs = set(manifest_inputs.keys()) - actual
    if missing_inputs:
        errors.append(
            f"[L1] {op_name}: manifest inputs not in forward(): {missing_inputs}"
        )

    if extra_in_forward:
        errors.append(
            f"[L1] {op_name}: forward() has params not in manifest: {extra_in_forward}"
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
    """Full L1 check: resolve Op class and compare forward() to manifest.

    If the Op module cannot be imported (e.g. missing runtime dependencies),
    L1 is skipped with a warning instead of failing. This allows the
    validator to run in lightweight CI environments (PyYAML-only) where
    L0/L2/L3/L4 still execute, while L1 runs when full deps are available.

    Args:
        op_name: Manifest op name.
        entry: The manifest entry dict.
        warnings: Optional list to append warning messages to.

    Returns:
        List of error strings (empty if OK or skipped).
    """
    errors: list[str] = []
    sig = entry.get("signature", {})
    source = entry.get("source", {})
    op_file = source.get("op", "")

    result = _resolve_op_class(op_file, op_name)

    if result.import_error:
        # Missing dependencies — skip L1 gracefully
        msg = (
            f"[L1] {op_name}: skipped — could not import {op_file} "
            f"(missing dependencies)"
        )
        if warnings is not None:
            warnings.append(msg)
        return errors

    if result.cls is None:
        errors.append(f"[L1] {op_name}: could not resolve Op class from {op_file}")
        return errors

    forward_params = _get_forward_params(result.cls)
    if forward_params is None:
        errors.append(
            f"[L1] {op_name}: could not inspect forward() on {result.cls.__name__}"
        )
        return errors

    manifest_inputs = sig.get("inputs", {})
    manifest_params = sig.get("params", {})

    return check_l1_signature(op_name, manifest_inputs, manifest_params, forward_params)


# ---------------------------------------------------------------------------
# L2: Shape rules evaluation
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
                f"[L2] {op_name}: shape_rules[{i}] invalid syntax: {rule!r} ({exc})"
            )
    return errors


# ---------------------------------------------------------------------------
# L3: Dtype conformance
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
                f"[L3] {op_name}: {context} dtype same_as({ref}) "
                f"references unknown tensor"
            )
    elif token not in _TORCH_DTYPES:
        return f"[L3] {op_name}: {context} has unrecognized dtype '{token}'"
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
                        f"[L3] {op_name}: workloads[{i}].dtypes[{j}] "
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
# L4: Benchmark file uses manifest workloads
# ---------------------------------------------------------------------------

def _ast_has_import_and_usage(
    tree: ast.Module,
    target_names: set[str],
) -> dict[str, bool]:
    """Check whether *target_names* are imported AND referenced in *tree*.

    Returns a dict mapping each target name to True/False.
    A name counts as "used" if it appears as:
      - An ``ImportFrom`` name (``from tileops.manifest import name``),  AND
      - A bare ``Name`` node elsewhere in the AST (e.g., ``load_workloads(...)``).

    Attribute-style calls like ``fake.load_workloads(...)`` are NOT accepted,
    because the required import form is ``from tileops.manifest import name``
    which binds ``name`` as a bare identifier, not an attribute on some object.
    """
    imported: set[str] = set()
    called: set[str] = set()

    for node in ast.walk(tree):
        # from tileops.manifest import load_workloads, eval_roofline
        if isinstance(node, ast.ImportFrom):
            if node.module == "tileops.manifest" and node.names:
                for alias in node.names:
                    if alias.name in target_names:
                        imported.add(alias.name)
        # Direct bare-name usage only: load_workloads('op')
        elif isinstance(node, ast.Name) and node.id in target_names:
            called.add(node.id)

    return {name: (name in imported and name in called) for name in target_names}


def check_l4_benchmark(
    op_name: str, bench_path: str, repo_root: Path,
) -> list[str]:
    """Check that the benchmark file imports and calls load_workloads/eval_roofline.

    Uses Python AST parsing (no execution) to verify actual import and usage,
    rather than raw substring matching which can be fooled by comments.
    """
    errors: list[str] = []
    full_path = Path(bench_path)
    if not full_path.is_absolute():
        full_path = repo_root / bench_path

    if not full_path.is_file():
        errors.append(f"[L4] {op_name}: bench file not found: {bench_path}")
        return errors

    content = full_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(content, filename=bench_path)
    except SyntaxError as exc:
        errors.append(
            f"[L4] {op_name}: bench file {bench_path} has syntax error: {exc}"
        )
        return errors

    targets = {"load_workloads", "eval_roofline"}
    usage = _ast_has_import_and_usage(tree, targets)

    if not usage["load_workloads"]:
        errors.append(
            f"[L4] {op_name}: bench file {bench_path} does not import and use "
            f"load_workloads from tileops.manifest"
        )
    if not usage["eval_roofline"]:
        errors.append(
            f"[L4] {op_name}: bench file {bench_path} does not import and use "
            f"eval_roofline from tileops.manifest"
        )
    return errors


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _is_spec_only(entry: dict, repo_root: Path) -> bool:
    """An op is spec-only if status is 'spec-only' or its source.op file does not exist."""
    if entry.get("status") == "spec-only":
        return True
    source = entry.get("source", {})
    op_file = source.get("op", "")
    return not (repo_root / op_file).is_file()


ALL_LEVELS = frozenset({"L0", "L1", "L2", "L3", "L4"})


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
        levels: Set of level names to run (e.g. {"L0", "L2", "L3", "L4"}).
                When None, all levels are enabled.

    Returns:
        A tuple of (errors, warnings). Errors are hard failures; warnings
        are informational messages (e.g. L1 skipped due to missing deps).
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

        # L0: always run if selected
        if "L0" in levels:
            l0_errors = check_l0(op_name, entry)
            all_errors.extend(l0_errors)
            # If L0 fails, skip higher levels for this op
            if l0_errors:
                continue

        spec_only = _is_spec_only(entry, repo_root)
        if spec_only:
            if verbose:
                print(f"    {op_name}: spec-only, skipping L1-L4")
            continue

        # L1: signature consistency
        if "L1" in levels:
            all_errors.extend(check_l1(op_name, entry, warnings=all_warnings))

        # L2: shape rules
        if "L2" in levels:
            all_errors.extend(check_l2(op_name, entry))

        # L3: dtype conformance
        if "L3" in levels:
            all_errors.extend(check_l3(op_name, entry))

        # L4: benchmark uses manifest workloads
        if "L4" in levels:
            bench_path = entry.get("source", {}).get("bench", "")
            if bench_path:
                all_errors.extend(
                    check_l4_benchmark(op_name, bench_path, repo_root)
                )

    return all_errors, all_warnings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_levels(argv: list[str]) -> frozenset[str] | None:
    """Parse ``--levels L0,L2,L3`` from argv. Returns None when flag absent."""
    for i, arg in enumerate(argv):
        if arg == "--levels" and i + 1 < len(argv):
            raw = argv[i + 1].upper().split(",")
            parsed = frozenset(t.strip() for t in raw)
            unknown = parsed - ALL_LEVELS
            if unknown:
                print(f"ERROR: unknown levels: {unknown}")
                sys.exit(2)
            return parsed
        if arg.startswith("--levels="):
            raw = arg.split("=", 1)[1].upper().split(",")
            parsed = frozenset(t.strip() for t in raw)
            unknown = parsed - ALL_LEVELS
            if unknown:
                print(f"ERROR: unknown levels: {unknown}")
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
