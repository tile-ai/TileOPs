#!/usr/bin/env python3
"""select_nightly_benchmarks.py — Pick which benchmark files to run tonight.

Strategy:
  1. Discover all benchmarks/ops/bench_*.py files on disk.
  2. Load run history (JSON mapping filename → last-run ISO date).
  3. Select up to N files:
     a. Never-run files first (sorted alphabetically for determinism).
     b. If still < N, fill with oldest-last-run files.
  4. Print selected file paths (one per line) and write rotation_meta.json.

Usage:
    python benchmarks/select_nightly_benchmarks.py \
        --history  bench_run_history.json \
        --n        15 \
        --output   rotation_meta.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def discover_bench_files(bench_dir: str = "benchmarks/ops") -> list[str]:
    """Return sorted list of bench_*.py filenames (basename only)."""
    pattern = os.path.join(bench_dir, "bench_*.py")
    return sorted(os.path.basename(f) for f in glob.glob(pattern))


def load_history(path: str | None) -> dict[str, str]:
    """Load {filename: last_run_date} from JSON. Returns empty dict on failure."""
    if not path or not os.path.exists(path):
        return {}
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def select_benchmarks(
    all_files: list[str],
    history: dict[str, str],
    n: int,
) -> list[str]:
    """Select up to n benchmark files to run.

    Priority:
      1. Files not in history (never run) — alphabetical order.
      2. Files with oldest last_run date — ascending by date.
    """
    never_run = [f for f in all_files if f not in history]
    previously_run = [f for f in all_files if f in history]

    # Sort previously-run by last_run date ascending (oldest first)
    previously_run.sort(key=lambda f: history[f])

    selected: list[str] = []
    # Phase 1: never-run files
    selected.extend(never_run[:n])
    # Phase 2: fill with oldest-run files
    remaining = n - len(selected)
    if remaining > 0:
        selected.extend(previously_run[:remaining])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select nightly benchmark files")
    parser.add_argument(
        "--history", default=None,
        help="Path to bench_run_history.json (optional)",
    )
    parser.add_argument(
        "--n", type=int, default=15,
        help="Max number of benchmark files to run (default: 15)",
    )
    parser.add_argument(
        "--bench-dir", default="benchmarks/ops",
        help="Directory containing bench_*.py files",
    )
    parser.add_argument(
        "--output", default="rotation_meta.json",
        help="Path to write rotation metadata JSON",
    )
    args = parser.parse_args()

    all_files = discover_bench_files(args.bench_dir)
    if not all_files:
        print("::error::No bench_*.py files found", file=sys.stderr)
        sys.exit(1)

    history = load_history(args.history)
    selected = select_benchmarks(all_files, history, args.n)

    # Build full paths
    selected_paths = [os.path.join(args.bench_dir, f) for f in selected]

    # Count stats
    never_run_count = sum(1 for f in selected if f not in history)
    rerun_count = len(selected) - never_run_count

    # Write rotation metadata
    meta = {
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "total_bench_files": len(all_files),
        "selected_count": len(selected),
        "never_run_count": never_run_count,
        "rerun_count": rerun_count,
        "selected_files": selected,
        "n": args.n,
    }
    Path(args.output).write_text(json.dumps(meta, indent=2))

    # Print paths for consumption by the workflow
    print(" ".join(selected_paths))


if __name__ == "__main__":
    main()
