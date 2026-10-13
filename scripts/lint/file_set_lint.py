#!/usr/bin/env python3
"""Lint the file set for files that record the work rather than ship its result.

A research note, a translated working note, a log, a SASS dump, a CUDA source generated
outside ``src/tileops/csrc/``, a test whose name marks it as scratch or as one stage of a
change, and the output directories of local runs are artifacts of how a change was made.
They are read once, by the person who made it, and then mislead everyone after. ``_REJECTED``
lists the path forms, each with what it marks.

A file that matches a form and still belongs is allowed in one of two ways: ``ALLOWED`` below
maps its path to the reason, or the file carries the comment ``lint: file-set allow``
(after ``#``, ``//`` or ``<!--``) with the reason beside it.

Every path is checked, not only new ones: the tree holds no unallowed match, so a new file
is the only way to add one.

Usage: ``file_set_lint.py [FILE ...]``. With no arguments, checks every file git tracks.
Exits 1 on a finding.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_REJECTED = (
    ("research note", re.compile(r"^docs/(?:.*/)?[^/]*research[^/]*$", re.IGNORECASE)),
    ("translated working note", re.compile(r"^docs/(?:.*/)?[^/]*\.zh[^/]*\.md$")),
    ("note", re.compile(r"^notes/")),
    ("log", re.compile(r"\.log$")),
    ("SASS dump", re.compile(r"\.sass$")),
    ("CUDA source outside src/tileops/csrc/", re.compile(r"^(?!src/tileops/csrc/).*\.cu$")),
    (
        "scratch or staged test",
        re.compile(r"^tests/(?:.*/)?test_[^/]*(?:scratch|staged|phase|tmp)[^/]*$"),
    ),
    ("benchmark run output", re.compile(r"^benchmark_runs/")),
    ("agent scratch file", re.compile(r"^codex_file/")),
)

# A path a form matches that belongs anyway, with why. Adding one is a review decision.
ALLOWED = {
    "benchmarks/hardware/compute/fma_saturation.cu": (
        "hand-written hardware probe that fma_throughput.py compiles and runs; not generated"
    ),
    "benchmarks/hardware/memory/hbm_saturation.cu": (
        "hand-written hardware probe that hbm_bandwidth.py compiles and runs; not generated"
    ),
    "tests/ops/test_moe_staged.py": (
        "tests the staged MoE ops in tileops.ops.moe.staged; staged names the op, not a stage"
    ),
}

_MARKER = re.compile(rb"(?:#|//|<!--)\s*lint: file-set allow")


def _marked(path: Path) -> bool:
    try:
        return bool(_MARKER.search(path.read_bytes()))
    except OSError:
        return False


def findings(path: str, root: Path = REPO_ROOT) -> list[str]:
    """Why *path*, relative to *root*, may not be in the tree; empty when it may."""
    if path in ALLOWED:
        return []
    for label, form in _REJECTED:
        if form.search(path) and not _marked(root / path):
            return [
                f"{path}: {label} — keep it out of the repository, or allow it with a reason "
                "in file_set_lint.ALLOWED or a `lint: file-set allow` comment"
            ]
    return []


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.splitlines()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)
    paths = [Path(p).as_posix() for p in args.files] or _tracked()
    errors = [e for p in paths for e in findings(p)]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
