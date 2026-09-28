#!/usr/bin/env python3
"""Compare test-node counts between the current branch and main.

Usage
-----
    python scripts/test_node_delta.py          # auto-detect changed test files
    python scripts/test_node_delta.py tests/ops/test_foo.py tests/ops/test_bar.py

Run it with a Python that has the dev dependencies (torch, tilelang, pytest). Each side is
collected against its own tree: the working tree for HEAD, and a ``git archive`` of the merge
base with ``--base`` for the base, each with its own ``src/`` first on ``PYTHONPATH``. An
installed ``tileops`` therefore never serves either side, and the script works from a git
worktree.

The script always exits 0 (non-blocking); a problem it cannot count past is printed to
stderr. Output is a human-readable table showing per-file and total node deltas, suitable
for pasting into a PR description. A file that fails to collect shows ``error`` and stays
out of the totals.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_BRANCH = "main"


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], capture_output=True, cwd=cwd)


def _collect(root: Path, files: list[str]) -> tuple[dict[str, int], dict[str, str]]:
    """Collect *files* under *root*; return node counts and collection errors by file.

    A file missing under *root* is absent from both dicts.
    """
    present = [f for f in files if (root / f).is_file()]
    if not present:
        return {}, {}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), *filter(None, [env.get("PYTHONPATH")])]
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *present,
        "--collect-only",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "--continue-on-collection-errors",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=root, env=env, timeout=600
        )
    except subprocess.TimeoutExpired:
        return {}, dict.fromkeys(present, "collection timed out after 600 s")
    counts = dict.fromkeys(present, 0)
    errors: dict[str, str] = {}
    for line in result.stdout.splitlines():
        path, sep, _ = line.partition("::")
        if sep and path in counts:
            counts[path] += 1
        elif line.startswith("ERROR ") and line.split()[1] in counts:
            errors[line.split()[1]] = "collection error"
    # Exit 5 is "no tests collected". Any other non-zero exit that no ERROR line names is a
    # failure of the whole session (a missing dependency, a conftest UsageError), so no count
    # from it holds.
    if result.returncode not in (0, 5) and not errors:
        tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
        errors = dict.fromkeys(present, "\n    ".join(tail) or f"exit {result.returncode}")
    for path in errors:
        counts.pop(path, None)
    return counts, errors


def _base_tree(root: Path, base: str, dest: Path) -> str:
    """Extract the merge base of *base* and HEAD into *dest*; return its commit.

    Raises:
        RuntimeError: git or tar failed; the message says which step.
    """
    merge_base = _git("merge-base", base, "HEAD", cwd=root)
    if merge_base.returncode != 0:
        raise RuntimeError(f"no merge base of {base} and HEAD: {merge_base.stderr.decode()}")
    commit = merge_base.stdout.decode().strip()
    archive = _git("archive", "--format=tar", commit, cwd=root)
    if archive.returncode != 0:
        raise RuntimeError(f"git archive {commit} failed: {archive.stderr.decode()}")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True)
    if tar.returncode != 0:
        raise RuntimeError(f"extracting {commit} failed: {tar.stderr.decode()}")
    return commit


def _changed_test_files(root: Path, commit: str) -> list[str]:
    """Return test modules under tests/ that differ between *commit* and the working tree."""
    result = _git("diff", "--name-only", "--diff-filter=ACMR", commit, "--", "tests/", cwd=root)
    if result.returncode != 0:
        return []
    return [
        f
        for f in result.stdout.decode().strip().splitlines()
        if Path(f).name.startswith("test_") and f.endswith(".py")
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="Test files to check (default: auto-detect from git diff against the base)",
    )
    parser.add_argument(
        "--base",
        default=BASE_BRANCH,
        help=f"Base branch for comparison (default: {BASE_BRANCH})",
    )
    args = parser.parse_args(argv)

    top = _git("rev-parse", "--show-toplevel")
    if top.returncode != 0:
        print(f"not inside a git work tree: {top.stderr.decode()}", file=sys.stderr)
        return
    root = Path(top.stdout.decode().strip())
    files = []
    for f in args.files:
        path = (Path.cwd() / f).resolve()
        if not path.is_relative_to(root):
            print(f"  warning: {f} is outside {root}; skipped", file=sys.stderr)
            continue
        files.append(path.relative_to(root).as_posix())

    with tempfile.TemporaryDirectory(prefix="node_delta_") as tmp:
        base_root = Path(tmp)
        try:
            commit = _base_tree(root, args.base, base_root)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return
        if not args.files:
            files = _changed_test_files(root, commit)
        if not files:
            print("No changed test files detected.")
            return
        base_counts, base_errors = _collect(base_root, files)
    head_counts, head_errors = _collect(root, files)

    for side, errors in (("base", base_errors), ("HEAD", head_errors)):
        for path, message in sorted(errors.items()):
            print(f"  warning: {side} collection failed for {path}: {message}", file=sys.stderr)

    rows: list[tuple[str, str, str, str]] = []
    total_base = 0
    total_head = 0
    for f in sorted(files):
        if f not in head_counts and f not in head_errors:
            continue  # file does not exist on HEAD (deleted)
        base_str = (
            "error" if f in base_errors else "new" if f not in base_counts else str(base_counts[f])
        )
        head_str = "error" if f in head_errors else str(head_counts[f])
        if f in base_errors or f in head_errors:
            delta_str = "?"
        elif f not in base_counts:
            delta_str = "(new)"
            total_head += head_counts[f]
        else:
            delta = head_counts[f] - base_counts[f]
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            total_base += base_counts[f]
            total_head += head_counts[f]
        rows.append((f, base_str, head_str, delta_str))

    if not rows:
        print("No test files to report.")
        return

    print(f"Base: {args.base} (merge base {commit[:9]})\n")
    col_file = max(4, *(len(r[0]) for r in rows))
    header = f"{'File':<{col_file}}  {'Base':>6}  {'HEAD':>6}  {'Delta':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for path, base_str, head_str, delta_str in rows:
        print(f"{path:<{col_file}}  {base_str:>6}  {head_str:>6}  {delta_str:>7}")

    total_delta = total_head - total_base
    print(sep)
    sign = "+" if total_delta > 0 else ""
    print(f"{'TOTAL':<{col_file}}  {total_base:>6}  {total_head:>6}  {sign + str(total_delta):>7}")

    if total_base > 0:
        pct = (total_delta / total_base) * 100
        print(f"\nGrowth: {pct:+.1f}%")
    elif total_head > 0:
        print(f"\nAll {total_head} nodes are from new files.")
    if base_errors or head_errors:
        print("\nFiles marked error failed to collect and are not in the totals.")


if __name__ == "__main__":
    main()
