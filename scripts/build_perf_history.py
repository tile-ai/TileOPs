#!/usr/bin/env python3
"""Build the perf history window from the nightly snapshot repository.

One snapshot commit is one nightly run, so the window is every commit inside
the report's retention period, rebuilt on each use and never written back.

    python scripts/build_perf_history.py \
        --url https://github.com/tile-ai/TileOPs-nightly.git \
        --out perf_history.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nightly_report as report  # noqa: E402


def _git(repo: Path, *args: str) -> str | None:
    """Stdout of a git command in ``repo``, or None where it failed."""
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def _clone(url: str, ref: str, dest: Path) -> bool:
    # One commit per run, so the clone is bounded by date and not by count.
    since = datetime.now() - timedelta(days=report.HISTORY_RETENTION_DAYS + 1)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--single-branch",
            "--branch",
            ref,
            f"--shallow-since={since:%Y-%m-%d}",
            url,
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"::warning::cannot clone {url}: {result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0


def _entry(repo: Path, sha: str, workdir: Path) -> dict | None:
    """One history entry from a snapshot commit, or None where it carries none."""
    meta_text = _git(repo, "show", f"{sha}:meta.json")
    if meta_text is None:
        return None
    try:
        meta = json.loads(meta_text)
    except json.JSONDecodeError:
        return None
    if not all(meta.get(key) for key in ("date", "commit", "gpu")):
        return None

    bench_text = _git(repo, "show", f"{sha}:bench_results.xml")
    if bench_text is None:
        return None
    bench_path = workdir / "bench_results.xml"
    bench_path.write_text(bench_text)
    try:
        bench_ops = report.aggregate_bench_results(report.parse_bench_xml(str(bench_path)))
    except ET.ParseError:
        return None
    if not bench_ops:
        return None

    profile = report.load_gpu_profile(meta["gpu"])
    if profile:
        report.annotate_sol(bench_ops, profile)

    coverage = None
    coverage_text = _git(repo, "show", f"{sha}:coverage.xml")
    if coverage_text is not None:
        coverage_path = workdir / "coverage.xml"
        coverage_path.write_text(coverage_text)
        coverage = report.parse_coverage_xml(str(coverage_path))

    return report.build_history_entry(bench_ops, meta, coverage)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the nightly perf history window")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Clone the snapshot repository from here")
    source.add_argument("--repo", help="Read an existing clone instead of cloning")
    parser.add_argument("--ref", default="snapshots", help="Branch holding the snapshots")
    parser.add_argument("--out", required=True, help="Path to write perf_history.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        if args.url:
            repo = workdir / "snapshots"
            if not _clone(args.url, args.ref, repo):
                return 1
        else:
            repo = Path(args.repo)

        listing = _git(repo, "rev-list", args.ref)
        if listing is None:
            print(f"::warning::cannot read {args.ref} in {repo}", file=sys.stderr)
            return 1

        entries = []
        for sha in reversed(listing.split()):
            entry = _entry(repo, sha, workdir)
            if entry is not None:
                entries.append(entry)

    runs = report.history_window(entries)
    Path(args.out).write_text(json.dumps({"runs": runs}, indent=2))
    if runs:
        print(f"History window: {len(runs)} runs, {runs[0]['date']} to {runs[-1]['date']}")
    else:
        print("History window: empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
