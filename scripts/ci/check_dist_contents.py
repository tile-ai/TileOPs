#!/usr/bin/env python3
"""Assert the built sdist and wheel ship exactly the files the package needs.

Both directions are checked, because each fails silently on its own.

Nothing missing. Every tracked non-``.py`` file under ``src/tileops/`` must be
in the wheel. Those files — kernel headers and YAML data — are opened at run
time by path relative to ``__file__``, so one missing from the wheel is an
install that imports cleanly and then fails on first use. The expected set is
derived from ``git ls-files`` at run time rather than listed here, so a
resource added later is covered the day it lands.

Nothing extra. The top level of each artifact is checked against a whitelist:
the wheel may contain only the package and its ``.dist-info``; the sdist may
contain only the ``src/`` tree plus root files. A whitelist is what makes a
newly added top-level directory an error; a blacklist only catches the trees
someone remembered to name.

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

# Import name, and the wheel's only permitted top-level entry.
PACKAGE_DIR = "tileops"
SOURCE_PACKAGE_DIR = f"src/{PACKAGE_DIR}"
SDIST_EXTRA_REQUIRED = ("LICENSE", "README.md")
SDIST_ALLOWED_DIRS = frozenset({"src"})
# `PKG-INFO` and `setup.cfg` are written by the build; the rest a build reads.
SDIST_ALLOWED_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "CONTRIBUTING.md",
        "MANIFEST.in",
        "pyproject.toml",
        "PKG-INFO",
        "setup.cfg",
    }
)
# Build residue that lives beside the sources but is not a shipped resource.
_IGNORED_DIRS = frozenset({"__pycache__", ".egg-info"})


def _tracked_resources_from_git(repo_root: Path) -> list[str] | None:
    """Return tracked non-``.py`` paths under the package, or None without git."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--", SOURCE_PACKAGE_DIR],
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
    package = repo_root / SOURCE_PACKAGE_DIR
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
    """Return errors for resources missing from, or extra trees present in, the wheel."""
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    # A wheel is unpacked into site-packages, so its paths carry no `src/`.
    errors = [
        f"wheel {wheel_path.name}: missing {wheel_entry}"
        for wheel_entry in (_strip_src(entry) for entry in required)
        if wheel_entry not in names
    ]
    tops = {name.split("/", 1)[0] for name in names if name}
    # Not any `.dist-info`: a second one is another distribution's metadata.
    own_dist_info = f"{PACKAGE_DIR}-"
    extra = sorted(
        t
        for t in tops
        if t != PACKAGE_DIR and not (t.startswith(own_dist_info) and t.endswith(".dist-info"))
    )
    if extra:
        errors.append(
            f"wheel {wheel_path.name}: unexpected top-level {', '.join(extra)} "
            f"(only {PACKAGE_DIR}/ and the .dist-info may ship)"
        )
    return errors


def check_sdist(sdist_path: Path, required: list[str]) -> list[str]:
    """Return errors for required files missing from, or extra trees present in, the sdist."""
    with tarfile.open(sdist_path) as tf:
        # Strip the `<name>-<version>/` top-level directory from every member.
        members = {name.split("/", 1)[1] for name in tf.getnames() if "/" in name}
    errors = [
        f"sdist {sdist_path.name}: missing {entry}" for entry in required if entry not in members
    ]
    dirs = {m.split("/", 1)[0] for m in members if "/" in m}
    # A directory is also a member in its own right; keep it out of the files.
    files = {m for m in members if "/" not in m} - dirs
    extra = sorted((dirs - SDIST_ALLOWED_DIRS) | (files - SDIST_ALLOWED_FILES))
    if extra:
        errors.append(
            f"sdist {sdist_path.name}: unexpected top-level {', '.join(extra)} "
            f"(prune it in MANIFEST.in, or add it to the sdist whitelist)"
        )
    return errors


def _strip_src(path: str) -> str:
    """Map a repo-relative package path to its position inside the wheel."""
    prefix = f"{SOURCE_PACKAGE_DIR.split('/', 1)[0]}/"
    return path[len(prefix) :] if path.startswith(prefix) else path


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
    print(
        f"expected {len(resources)} shipped resources under {SOURCE_PACKAGE_DIR}/ (from {source})"
    )
    if not resources:
        errors.append(f"no non-.py resources found under {args.repo_root}/{SOURCE_PACKAGE_DIR}")

    manifest_yamls = [
        r
        for r in resources
        if r.startswith(f"{SOURCE_PACKAGE_DIR}/manifest/") and r.endswith(".yaml")
    ]
    sdist_required = manifest_yamls + list(SDIST_EXTRA_REQUIRED)
    for wheel in wheels:
        errors.extend(check_wheel(wheel, resources))
    for sdist in sdists:
        errors.extend(check_sdist(sdist, sdist_required))

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
