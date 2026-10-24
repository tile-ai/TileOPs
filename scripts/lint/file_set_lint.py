#!/usr/bin/env python3
"""Flag tracked files that record how a change was made rather than ship its result.

``_REJECTED`` lists the path forms: research and translated notes, logs, SASS dumps, CUDA
sources outside ``src/tileops/csrc/``, scratch or staged tests, ``notes/``, agent scratch
files and local run output.

Exempt: a path ``ALLOWED`` lists with its reason, or a file carrying
``lint: file-set allow: <reason>`` after ``#``, ``//`` or ``<!--``, outside any Markdown
code fence. Default scan: every file git tracks.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, allow_marker, report

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

# The marker counts only as a comment, so a log or a note that quotes it exempts nothing.
_MARKER = re.compile(r"(?:#|//|<!--)\s*" + allow_marker("file-set").pattern, re.IGNORECASE)
_FENCE = re.compile(r"[ \t]*(`{3,}|~{3,})")


def _unfenced(text: str) -> str:
    """The lines of Markdown *text* outside its fenced code blocks."""
    kept, fence = [], None
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None:
            closes = match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence)
            if closes and not line[match.end() :].strip():
                fence = None
        else:
            kept.append(line)
    return "\n".join(kept)


def _marked(path: Path) -> bool:
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return False
    if path.suffix.lower() in {".md", ".markdown"}:
        text = _unfenced(text)
    return bool(_MARKER.search(text))


def findings(path: str, root: Path = REPO_ROOT) -> list[str]:
    """Why *path*, relative to *root*, may not be in the tree; empty when it may."""
    if path in ALLOWED:
        return []
    for label, form in _REJECTED:
        if form.search(path) and not _marked(root / path):
            return [
                f"{path}:1: {label} — keep it out of the repository, or allow it with a reason "
                'in file_set_lint.ALLOWED or a "lint: file-set allow: <reason>" comment in the '
                "file"
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
    return report(e for p in paths for e in findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
