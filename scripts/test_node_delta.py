#!/usr/bin/env python3
"""Compare test-node counts between the current branch and main.

Usage
-----
    python scripts/test_node_delta.py          # auto-detect changed test files
    python scripts/test_node_delta.py tests/ops/test_foo.py tests/ops/test_bar.py

The script always exits 0 (non-blocking). Output is a human-readable table
showing per-file and total node deltas, suitable for pasting into a PR
description.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_BRANCH = "main"


def _collect_node_count(test_file: str, *, ref: str | None = None) -> int | None:
    """Return the number of pytest nodes in *test_file*.

    When *ref* is given the file content is read from that git ref and
    written to a temporary file so ``pytest --collect-only`` can process it.

    Returns ``None`` when the file does not exist at the given ref.
    """
    if ref is not None:
        blob = f"{ref}:{test_file}"
        result = subprocess.run(
            ["git", "show", blob],
            capture_output=True,
        )
        if result.returncode != 0:
            return None  # file does not exist at ref
        content = result.stdout

        suffix = Path(test_file).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            target = tmp.name
    else:
        if not Path(test_file).exists():
            return None
        target = test_file

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", target, "--collect-only", "-q", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        # Clean up temp file regardless of success/failure
        if ref is not None:
            Path(target).unlink(missing_ok=True)

    if result.returncode != 0:
        # Collection failed (syntax error, missing dep, etc.)
        print(f"  warning: pytest collection failed for {test_file}", file=sys.stderr)
        return 0

    # Parse the summary line: "N tests collected" or "no tests collected"
    for line in result.stdout.splitlines()[::-1]:
        line = line.strip()
        if "no tests" in line and "collected" in line:
            return 0
        if "collected" in line:
            parts = line.split()
            for i, token in enumerate(parts):
                if token in ("test", "tests") and i > 0:
                    try:
                        return int(parts[i - 1])
                    except ValueError:
                        continue
                # Also match "N tests collected in Xs"
                if token == "collected" and i > 0:
                    try:
                        return int(parts[i - 1])
                    except ValueError:
                        continue
    return 0


def _changed_test_files(base: str) -> list[str]:
    """Return test files changed between HEAD and *base*."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "--", "tests/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().splitlines() if f.endswith(".py")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="Test files to check (default: auto-detect from git diff against main)",
    )
    parser.add_argument(
        "--base",
        default=BASE_BRANCH,
        help=f"Base branch for comparison (default: {BASE_BRANCH})",
    )
    args = parser.parse_args(argv)

    files = args.files or _changed_test_files(args.base)
    if not files:
        print("No changed test files detected.")
        return

    rows: list[tuple[str, int | None, int, int | None]] = []
    total_base = 0
    total_head = 0

    for f in sorted(files):
        base_count = _collect_node_count(f, ref=args.base)
        head_count = _collect_node_count(f)
        if head_count is None:
            continue  # file does not exist on HEAD (deleted)

        delta: int | None
        if base_count is None:
            delta = None  # new file — no baseline
        else:
            delta = head_count - base_count
            total_base += base_count

        total_head += head_count
        rows.append((f, base_count, head_count, delta))

    if not rows:
        print("No test files to report.")
        return

    # Print table
    col_file = max(len(r[0]) for r in rows)
    col_file = max(col_file, 4)  # "File"
    header = f"{'File':<{col_file}}  {'Base':>6}  {'HEAD':>6}  {'Delta':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for path, base_count, head_count, delta in rows:
        base_str = "new" if base_count is None else str(base_count)
        if delta is None:
            delta_str = "(new)"
        elif delta > 0:
            delta_str = f"+{delta}"
        elif delta == 0:
            delta_str = "0"
        else:
            delta_str = str(delta)
        print(f"{path:<{col_file}}  {base_str:>6}  {head_count:>6}  {delta_str:>7}")

    # Totals
    total_delta = total_head - total_base
    print(sep)
    sign = "+" if total_delta > 0 else ""
    print(f"{'TOTAL':<{col_file}}  {total_base:>6}  {total_head:>6}  {sign + str(total_delta):>7}")

    if total_base > 0:
        pct = (total_delta / total_base) * 100
        print(f"\nGrowth: {pct:+.1f}%")
    elif total_head > 0:
        print(f"\nAll {total_head} nodes are from new files.")


if __name__ == "__main__":
    main()
