#!/usr/bin/env python3
"""Assert the built sdist and wheel ship the files the package needs.

Wheel: every tracked non-``.py`` file under ``tileops/`` must be present.
Those files — kernel headers and YAML data — are opened at run time by path
relative to ``__file__``, so one missing from the wheel is an install that
imports cleanly and then fails on first use. The expected set is derived
from ``git ls-files`` at run time rather than listed here, so a resource
added later is covered the day it lands.

Sdist: the manifest YAMLs, ``LICENSE``, and ``README.md`` must be present,
and the development-only trees (``tests/``, ``benchmarks/``, ``docs/``,
``assets/``, ``workloads/``, and tooling directories) must be absent.

Stdlib only (``zipfile``/``tarfile``), so it runs identically in CI and on a
developer machine: ``python scripts/ci/check_dist_contents.py`` after
``python -m build``.
"""

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PACKAGE_DIR = "tileops"
SDIST_EXTRA_REQUIRED = ("LICENSE", "README.md")
SDIST_FORBIDDEN_PREFIXES = (
    "tests/",
    "benchmarks/",
    "docs/",
    "assets/",
    "workloads/",
    ".github/",
    ".claude/",
    ".foundry/",
)
# Build residue that lives beside the sources but is not a shipped resource.
_IGNORED_DIRS = frozenset({"__pycache__", ".egg-info"})


def _tracked_resources_from_git(repo_root: Path) -> list[str] | None:
    """Return tracked non-``.py`` paths under the package, or None without git."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--", PACKAGE_DIR],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    paths = [line for line in completed.stdout.splitlines() if line and not line.endswith(".py")]
    return sorted(paths) if paths else None


def _resources_from_source_tree(repo_root: Path) -> list[str]:
    """Return non-``.py`` paths under the package by walking the source tree."""
    package = repo_root / PACKAGE_DIR
    paths = []
    for path in package.rglob("*"):
        if not path.is_file() or path.suffix == ".py":
            continue
        rel = path.relative_to(repo_root)
        if any(part in _IGNORED_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        paths.append(rel.as_posix())
    return sorted(paths)


def expected_resources(repo_root: Path) -> tuple[list[str], str]:
    """Return the resources the wheel must ship, and the source they came from."""
    tracked = _tracked_resources_from_git(repo_root)
    if tracked is not None:
        return tracked, "git ls-files"
    return _resources_from_source_tree(repo_root), "source-tree walk (git unavailable)"


def check_wheel(wheel_path: Path, required: list[str]) -> list[str]:
    """Return one error string per required resource missing from the wheel."""
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    return [f"wheel {wheel_path.name}: missing {entry}" for entry in required if entry not in names]


def check_sdist(
    sdist_path: Path, required: list[str], forbidden_prefixes: tuple[str, ...]
) -> list[str]:
    """Return errors for required files missing from, or pruned trees present in, the sdist."""
    with tarfile.open(sdist_path) as tf:
        # Strip the `<name>-<version>/` top-level directory from every member.
        members = {name.split("/", 1)[1] for name in tf.getnames() if "/" in name}
    errors = [
        f"sdist {sdist_path.name}: missing {entry}" for entry in required if entry not in members
    ]
    for prefix in forbidden_prefixes:
        leaked = sorted(m for m in members if m.startswith(prefix))
        if leaked:
            errors.append(
                f"sdist {sdist_path.name}: contains pruned tree {prefix} "
                f"({len(leaked)} entries, e.g. {leaked[0]})"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    errors: list[str] = []

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if not wheels:
        errors.append(f"no wheel (*.whl) found in {args.dist_dir}")
    if not sdists:
        errors.append(f"no sdist (*.tar.gz) found in {args.dist_dir}")

    resources, source = expected_resources(args.repo_root)
    print(f"expected {len(resources)} shipped resources under {PACKAGE_DIR}/ (from {source})")
    if not resources:
        errors.append(f"no non-.py resources found under {args.repo_root}/{PACKAGE_DIR}")

    manifest_yamls = [
        r for r in resources if r.startswith(f"{PACKAGE_DIR}/manifest/") and r.endswith(".yaml")
    ]
    sdist_required = manifest_yamls + list(SDIST_EXTRA_REQUIRED)
    for wheel in wheels:
        errors.extend(check_wheel(wheel, resources))
    for sdist in sdists:
        errors.extend(check_sdist(sdist, sdist_required, SDIST_FORBIDDEN_PREFIXES))

    for error in errors:
        print(error)
    if errors:
        print(f"{len(errors)} problem(s) found")
        return 1
    checked = [p.name for p in wheels + sdists]
    print(f"dist contents OK: {', '.join(checked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
