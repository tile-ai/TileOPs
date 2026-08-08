#!/usr/bin/env python3
"""Assert the built sdist and wheel ship the files the package needs.

Wheel: every ``tileops/manifest/*.yaml`` present in the repo, plus
``tileops/kernels/moe/_atomic_helper.h``. A manifest YAML missing from the
wheel means an installed package silently loses ops.

Sdist: the manifest YAMLs, ``LICENSE``, and ``README.md`` — and none of the
pruned trees (``tests/``, ``benchmarks/``, ``docs/``, ``assets/``,
``workloads/``, ``.github/``).

Stdlib only (``zipfile``/``tarfile``), so it runs identically in CI and on a
developer machine: ``python scripts/ci/check_dist_contents.py`` after
``python -m build``.
"""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

WHEEL_EXTRA_REQUIRED = ("tileops/kernels/moe/_atomic_helper.h",)
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


def repo_manifest_yamls(repo_root: Path) -> list[str]:
    """Return repo-relative paths of every manifest YAML, sorted."""
    manifest_dir = repo_root / "tileops" / "manifest"
    return sorted(f"tileops/manifest/{p.name}" for p in manifest_dir.glob("*.yaml"))


def check_wheel(wheel_path: Path, required: list[str]) -> list[str]:
    """Return one error string per required file missing from the wheel."""
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

    manifest_yamls = repo_manifest_yamls(args.repo_root)
    if not manifest_yamls:
        errors.append(f"no manifest YAMLs found under {args.repo_root}/tileops/manifest")

    wheel_required = manifest_yamls + list(WHEEL_EXTRA_REQUIRED)
    sdist_required = manifest_yamls + list(SDIST_EXTRA_REQUIRED)
    for wheel in wheels:
        errors.extend(check_wheel(wheel, wheel_required))
    for sdist in sdists:
        errors.extend(check_sdist(sdist, sdist_required, SDIST_FORBIDDEN_PREFIXES))

    for error in errors:
        print(error)
    if errors:
        return 1
    checked = [p.name for p in wheels + sdists]
    print(f"dist contents OK: {', '.join(checked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
