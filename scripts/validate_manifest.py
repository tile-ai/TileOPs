#!/usr/bin/env python3
"""Validate the ops manifest (docs/design/manifest.md § Validation).

Levels:
  schema    — top-level fields, `family`, `ref_api`, `composition`, `roofline`, `types.yaml`
  signature — the signature, its workload rows and effects; for an implemented entry, the
              class's `__init__`, `forward` and sub-op and kernel declarations
  bench     — every benchmark file takes its calls from the manifest and its roofline off the op

Usage:
    python scripts/validate_manifest.py [--verbose] [--levels schema,signature,bench] [--check-op NAME]

Exit code 0 = all checks pass; 1 = failures found.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Collection
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Documented and skill-driven callers run this script directly, without
# installing the package, so the source tree has to be reachable.
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tileops.manifest import types_document  # noqa: E402
from tileops.manifest.plan import check_adts as _check_adts  # noqa: E402
from tileops.manifest.plan import check_entry as _check_signature  # noqa: E402
from tileops.manifest.plan import (  # noqa: E402
    effect_errors,
    roofline_plan,
)
from tileops.manifest.plan import (  # noqa: E402
    signature_schema_errors as _signature_schema_errors,
)
from tileops.manifest.registry import op_class  # noqa: E402
from tileops.manifest.signature import parse_signature  # noqa: E402
from tileops.manifest.workload import check_workloads as _check_workloads  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "src" / "tileops" / "manifest" / "spec"

ALL_LEVELS = frozenset({"schema", "signature", "bench"})

_VALID_COMPOSITION_KINDS = {"composite"}
_COMPOSITION_KEYS = {"kind", "stages"}
_STAGE_KEYS = {"name", "op", "kernel", "optional"}


def _key_format_errors(
    op_name: str,
    all_op_names: Collection[str],
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
        siblings = [s for s in (f"{stem}FwdOp", f"{stem}BwdOp") if s in all_op_names]
        if siblings:
            err(
                f"missing direction suffix; direction sibling "
                f"'{siblings[0]}' exists in the manifest"
            )
    return errors


def _emit_to(sink, tag: str, op_name: str):
    """Return an emitter appending ``[tag] op_name: msg`` strings to *sink*."""
    prefix = f"[{tag}] {op_name}: "
    return lambda msg: sink.append(prefix + msg)


def _composition_errors(
    op_name: str, composition: dict, all_op_names: Collection[str]
) -> list[str]:
    """`composition` (docs/design/manifest.md § Composition): a kind and a non-empty list of
    uniquely named stages, each naming a manifest entry (`op`) or a kernel role (`kernel`).

    Whether the stages are the class's `delegate_types` and `kernel_types` is the parity check's.
    """
    errors: list[str] = []
    err = _emit_to(errors, "schema", op_name)
    unknown = sorted(repr(k) for k in composition if k not in _COMPOSITION_KEYS)
    if unknown:
        err(f"composition has unknown keys [{', '.join(unknown)}]")
    kind = composition.get("kind")
    if kind not in _VALID_COMPOSITION_KINDS:
        err(f"composition.kind must be one of {sorted(_VALID_COMPOSITION_KINDS)}, got {kind!r}")
    stages = composition.get("stages")
    if not isinstance(stages, list) or not stages:
        err("composition.stages must be a non-empty list")
        return errors
    seen: set[str] = set()
    for i, stage in enumerate(stages):
        where = f"composition.stages[{i}]"
        if not isinstance(stage, dict):
            err(f"{where} must be a mapping, got {type(stage).__name__}")
            continue
        unknown = sorted(repr(k) for k in stage if k not in _STAGE_KEYS)
        if unknown:
            err(f"{where} has unknown keys [{', '.join(unknown)}]")
        name = stage.get("name")
        if not isinstance(name, str) or not name.strip():
            err(f"{where} must have a non-empty string 'name'")
        elif name in seen:
            err(f"{where}.name {name!r} is declared twice")
        else:
            seen.add(name)
        if ("op" in stage) == ("kernel" in stage):
            err(f"{where} must have exactly one of 'op' or 'kernel'")
        elif "op" in stage and not (isinstance(stage["op"], str) and stage["op"] in all_op_names):
            err(f"{where}.op {stage['op']!r} is not a manifest entry")
        elif "kernel" in stage and not (isinstance(stage["kernel"], str) and stage["kernel"]):
            err(f"{where}.kernel must be a non-empty string")
        if "optional" in stage and not isinstance(stage["optional"], bool):
            err(f"{where}.optional must be a bool")
    return errors


def _reads_manifest_calls(tree: ast.Module) -> bool:
    """Whether the file imports and calls ``manifest_calls`` from ``benchmarks.benchmark_base``.

    Which op it names is a run-time fact, checked against a benchmark run by
    ``scripts/check_bench_coverage.py``, never against the source: a bench file may reach its
    op through a loop, a factory or a helper, and none of those shapes is worse than a literal.
    """
    imported = False
    called = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "benchmarks.benchmark_base":
            if any(alias.name == "manifest_calls" for alias in node.names):
                imported = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "manifest_calls"
        ):
            called = True
    return imported and called


def _reads_op_roofline(tree: ast.Module) -> bool:
    """Whether the file takes its roofline off an Op, not off its own arithmetic.

    ``<expr>.eval_roofline()`` directly, or a ``ManifestBenchmark`` — which
    reads the roofline off the op it wraps — imported and constructed.
    """
    imported = False
    called = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "benchmarks.benchmark_base":
            if any(alias.name == "ManifestBenchmark" for alias in node.names):
                imported = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "ManifestBenchmark":
                called = True
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "eval_roofline":
                return True
    return imported and called


def check_benchmark(op_name: str, bench_path: str, repo_root: Path) -> list[str]:
    """Check that the benchmark file obeys the benchmark contract.

    Uses Python AST parsing (no execution) to verify actual import and usage,
    rather than raw substring matching which can be fooled by comments.

    This is a property of the file: its workloads come from the manifest and
    its roofline comes from the op. Which manifest entry the file benchmarks is
    a separate question, answered from a run's report by
    ``scripts/check_bench_coverage.py``.

    Returns a list of hard validation errors.
    """
    errors: list[str] = []
    full_path = Path(bench_path)
    if not full_path.is_absolute():
        full_path = repo_root / bench_path

    if not full_path.is_file():
        errors.append(f"[bench] {op_name}: bench file not found: {bench_path}")
        return errors

    try:
        tree = ast.parse(full_path.read_text(encoding="utf-8"), filename=bench_path)
    except SyntaxError as exc:
        errors.append(f"[bench] {op_name}: bench file {bench_path} has syntax error: {exc}")
        return errors

    if not _reads_manifest_calls(tree):
        errors.append(
            f"[bench] {op_name}: bench file {bench_path} must import and call "
            "manifest_calls from benchmarks.benchmark_base"
        )
    if not _reads_op_roofline(tree):
        errors.append(
            f"[bench] {op_name}: bench file {bench_path} must take its roofline "
            "off the op — eval_roofline() on an Op instance, or a "
            "ManifestBenchmark construction"
        )
    return errors


def _check_bench_files(repo_root: Path) -> list[str]:
    """The benchmark contract on every ``benchmarks/ops/**/bench_*.py``."""
    errors = []
    for path in sorted((repo_root / "benchmarks" / "ops").rglob("bench_*.py")):
        relative = path.relative_to(repo_root).as_posix()
        errors += check_benchmark(relative, relative, repo_root)
    return errors


_ENTRY_KEYS = {
    "family": str,
    "status": str,
    "signature": dict,
    "workloads": list,
    "roofline": dict,
    "ref_api": str,
    "composition": dict,
}
_REQUIRED = ("family", "status", "signature", "workloads", "roofline")


def _schema_errors(op_name: str, entry: dict, all_op_names) -> list[str]:
    """Top-level fields of an entry (docs/design/manifest.md § Top-Level Fields)."""
    if not isinstance(op_name, str):
        return [f"[schema] {op_name!r}: the key is not an op class name `<Name>FwdOp`"]
    errors = _key_format_errors(op_name, all_op_names)
    if not _OP_KEY.fullmatch(op_name):
        errors.append(f"[schema] {op_name}: the key is not an op class name `<Name>FwdOp`")
    errors += _family_errors(op_name, entry)
    if isinstance(entry.get("signature"), dict):
        errors += [f"[schema] {op_name}: {e}" for e in _signature_schema_errors(entry["signature"])]
    if isinstance(entry.get("composition"), dict):
        errors += _composition_errors(op_name, entry["composition"], all_op_names)
    ref = entry.get("ref_api")
    if isinstance(ref, str):
        errors += _ref_api_errors(op_name, ref)
    errors += [
        f"[schema] {op_name}: missing required field '{k}'" for k in _REQUIRED if k not in entry
    ]
    for key, value in entry.items():
        expected = _ENTRY_KEYS.get(key)
        if expected is None:
            errors.append(f"[schema] {op_name}: unknown field '{key}'")
        elif not isinstance(value, expected):
            errors.append(f"[schema] {op_name}: '{key}' must be a {expected.__name__}")
    if entry.get("status") not in (None, "implemented", "spec-only"):
        errors.append(f"[schema] {op_name}: status must be 'implemented' or 'spec-only'")
    return errors


_OP_KEY = re.compile(r"[A-Z][A-Za-z0-9]*(Fwd|Bwd)Op")


def _family_errors(op_name: str, entry: dict) -> list[str]:
    """`family` names a public module; an implemented op is exported from it by its key."""
    family = entry.get("family")
    where = f"[schema] {op_name}: family {family!r}"
    if not isinstance(family, str):
        return []  # reported as a missing or mistyped field
    if not family.isidentifier():
        return [f"{where} is not a module name"]
    try:
        found = importlib.util.find_spec(f"tileops.{family}") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return [f"{where} is not a tileops module"]
    if entry.get("status") != "implemented":
        return []
    module = importlib.import_module(f"tileops.{family}")
    exported = getattr(module, op_name, None) if op_name in getattr(module, "__all__", ()) else None
    if not inspect.isclass(exported) or exported.__name__ != op_name:
        return [f"{where} does not export the class {op_name} in its __all__"]
    return []


def _ref_api_errors(op_name: str, ref: str) -> list[str]:
    """`ref_api` is a qualified name that resolves once its module imports."""
    parts = ref.split(".")
    if len(parts) < 2 or not all(p.isidentifier() for p in parts):
        return [f"[schema] {op_name}: ref_api {ref!r} is not a qualified name"]
    for i in range(len(parts) - 1, 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            if not hasattr(target, attr):
                return [f"[schema] {op_name}: ref_api {ref!r} does not resolve"]
            target = getattr(target, attr)
        return []
    return [f"[schema] {op_name}: ref_api {ref!r}: no prefix of it is an importable module"]


# Execution-policy parameters every op takes, in order with their defaults, and the reserved one
# it may take (docs/design/manifest.md § Signature).
_POLICY_PARAMETERS = {"target": None, "kernel_map": None, "tune": False}
_RESERVED_POLICY = "config"


def _normal_default(value):
    """A default as the manifest writes it: a dtype by name, a tuple as a list."""
    if isinstance(value, tuple):
        return list(value)
    return str(value).removeprefix("torch.") if type(value).__module__ == "torch" else value


def _parity_errors(op_name: str, entry: dict) -> list[str]:
    """`__init__`, `forward` and the class's sub-op and kernel declarations against the entry
    (docs/design/manifest.md § Signature, § Composition).

    `__init__` takes `signature.params` in order with their defaults, a `kw_only` one after
    `*`, then keyword-only `target`, `kernel_map` and `tune`, and only the injected objects the
    class lists in `execution_parameters` or the reserved `config`. `forward` begins with the
    call-time inputs in order, positional, the optional ones defaulting to `None` and the
    others to nothing, then `out` when an output is a buffer. The `op` stages of `composition`
    are `delegate_types`, and an entry with a composition lists `kernel_types` as its `kernel`
    stages, each in order.
    """
    where = f"[signature] {op_name}"
    if not isinstance(entry.get("family"), str):
        return []  # reported as a missing or mistyped field
    try:
        cls = op_class(op_name, entry)
    except (ImportError, AttributeError) as exc:
        return [f"{where}: cannot import tileops.{entry['family']}.{op_name}: {exc}"]
    sig = entry["signature"]
    errors = []
    empty = inspect.Parameter.empty
    keyword = inspect.Parameter.KEYWORD_ONLY
    init = list(inspect.signature(cls.__init__).parameters.values())[1:]
    params = list(sig.get("params") or {})
    for i, name in enumerate(params):
        decl = sig["params"][name]
        if "shape" in decl and decl.get("optional", False) is not False:
            decl = {**decl, "default": None}  # an optional construction-time tensor
        got = init[i] if i < len(init) else None
        kind = keyword if decl.get("kw_only") else inspect.Parameter.POSITIONAL_OR_KEYWORD
        if got is None or got.name != name or got.kind is not kind:
            errors.append(f"{where}: __init__ parameter {i + 1} must be {name!r} ({kind.name})")
        elif ("default" in decl) != (got.default is not empty) or (
            "default" in decl and _normal_default(got.default) != decl["default"]
        ):
            errors.append(
                f"{where}: __init__ {name!r} defaults to {got.default!r}, "
                f"not {decl.get('default', 'nothing')!r}"
            )
    rest = {p.name: p for p in init[len(params) :]}
    policy = [n for n in rest if n in _POLICY_PARAMETERS]
    if policy != list(_POLICY_PARAMETERS) or any(
        rest[n].kind is not keyword or rest[n].default is not d
        for n, d in _POLICY_PARAMETERS.items()
    ):
        suffix = ", ".join(f"{n}={d}" for n, d in _POLICY_PARAMETERS.items())
        errors.append(f"{where}: __init__ must end its policy parameters with *, {suffix}")
    allowed = {*_POLICY_PARAMETERS, _RESERVED_POLICY, *getattr(cls, "execution_parameters", ())}
    errors += [
        f"{where}: __init__ parameter {p.name!r} is not a signature or execution-policy parameter"
        for p in rest.values()
        if p.kind is not keyword or p.name not in allowed
    ]
    forward = list(inspect.signature(cls.forward).parameters.values())[1:]
    inputs = sig.get("inputs") or {}
    buffered = any(o.get("buffer") == "out" for o in (sig.get("outputs") or {}).values())
    prefix = list(inputs) + (["out"] if buffered else [])
    for i, name in enumerate(prefix):
        got = forward[i] if i < len(forward) else None
        optional = name == "out" or inputs[name].get("optional", False) is not False
        positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
        if got is None or got.name != name or got.kind is not positional:
            errors.append(f"{where}: forward parameter {i + 1} must be {name!r}, positional")
        elif got.default is not (None if optional else empty):
            want = "default to None" if optional else "have no default"
            errors.append(f"{where}: forward {name!r} must {want}")
    stages = [
        st for st in (entry.get("composition") or {}).get("stages") or [] if isinstance(st, dict)
    ]
    ops = [(st.get("name"), st["op"]) for st in stages if "op" in st]
    delegates = [(stage, c.__name__) for stage, c in getattr(cls, "delegate_types", {}).items()]
    if ops != delegates:
        errors.append(f"{where}: composition op stages {ops} are not delegate_types {delegates}")
    kernels = [st["kernel"] for st in stages if "kernel" in st]
    kernel_types = list(getattr(cls, "kernel_types", {}))
    if stages and kernels != kernel_types:
        errors.append(
            f"{where}: composition kernel stages {kernels} are not kernel_types {kernel_types}"
        )
    return errors


def validate_manifest(
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
    verbose: bool = False,
    levels: frozenset[str] | None = None,
    check_op: str | None = None,
) -> tuple[list[str], list[str]]:
    """Run the selected levels on the manifest.

    Returns ``(errors, warnings)``: errors are failures; warnings are advisory diagnostics.
    ``manifest_path=None`` loads the merged manifest from the ``tileops.manifest`` package
    (tests pass a temp file for synthetic single-file manifests). ``levels=None`` enables
    every level. ``check_op`` scopes the entry checks to the named op.
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

    if check_op is not None and check_op not in ops:
        return [f"--check-op: op '{check_op}' not found in manifest"], []

    all_errors: list[str] = []
    all_warnings: list[str] = []
    # Entries see only the ADTs `check_adts` accepts; the others are reported once, here.
    document = types_document()
    adts, adt_errors = {}, []
    if document is not None:
        well_formed = isinstance(document, dict) and set(document) == {"adts"}
        adts, adt_errors = _check_adts(document["adts"] if well_formed else None)
    if "schema" in levels:
        all_errors.extend(f"[schema] types.yaml: {e}" for e in adt_errors)

    for op_name, entry in ops.items():
        if check_op is not None and op_name != check_op:
            continue
        if verbose:
            print(f"  Checking {op_name}...")
        if not isinstance(entry, dict):
            all_errors.append(f"[schema] {op_name}: entry must be a mapping")
            continue
        if "schema" in levels:
            all_errors.extend(_schema_errors(op_name, entry, ops))
        signature_errors, signature_warnings = _check_signature(op_name, entry, adts)
        sig = None if signature_errors else parse_signature(op_name, entry, adts)
        implemented = entry.get("status") == "implemented"
        if "schema" in levels:
            all_errors.extend(
                f"[schema] {op_name}: {e}"
                for e in roofline_plan(sig, entry.get("roofline"), resolve=implemented)[0]
            )
        if "signature" in levels:
            all_errors.extend(f"[signature] {e}" for e in signature_errors)
            all_warnings.extend(f"[signature] {w}" for w in signature_warnings)
            if sig is not None:
                all_errors.extend(
                    f"[signature] {e}" for e in _check_workloads(op_name, entry, adts)
                )
                all_errors.extend(f"[signature] {op_name}: {e}" for e in effect_errors(sig))
                if implemented:
                    all_errors.extend(_parity_errors(op_name, entry))

    if "bench" in levels and check_op is None:
        all_errors.extend(_check_bench_files(repo_root))
    return all_errors, all_warnings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_levels(argv: list[str]) -> frozenset[str] | None:
    """Parse ``--levels schema,signature`` from argv. Returns None when flag absent."""
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
            print(f"ERROR: unknown levels: {sorted(unknown)}")
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
    errors, warnings = validate_manifest(verbose=verbose, levels=levels, check_op=check_op)

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
