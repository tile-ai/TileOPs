#!/usr/bin/env python3
"""Flag a ``torch.cuda`` call in the code every target and device runs through.

The op layer, the backend protocol and the benchmark harness serve CUDA and every other
device alike, so they reach the call's device through ``tileops.utils.device`` and
``torch.accelerator``. A ``torch.cuda`` call there makes the path CUDA's again.

Exempt: ``# lint: device-api allow: <reason>`` on the line, for code that runs on CUDA only,
such as CUPTI timing. Default scan: the files and trees in ``SCANNED``.
"""

import ast
import sys
from pathlib import Path

from common import allow_marker, files, relative, report

SCANNED = (
    "src/tileops/backend",
    "src/tileops/kernels/call_spec.py",
    "src/tileops/kernels/kernel_base.py",
    "src/tileops/ops/_signature_codegen.py",
    "src/tileops/ops/compile_boundary.py",
    "src/tileops/ops/op_base.py",
    "src/tileops/utils/device.py",
    "benchmarks/api.py",
    "benchmarks/baselines.py",
    "benchmarks/conftest.py",
    "benchmarks/report.py",
    "benchmarks/timing.py",
)

_ALLOW = allow_marker("device-api")


def _names_cuda(node: ast.AST) -> bool:
    """Whether *node* reaches ``torch.cuda``: an attribute of it, or an import of it."""
    if isinstance(node, ast.Attribute):
        return ast.unparse(node.value) == "torch.cuda"
    if isinstance(node, ast.Import):
        return any(a.name == "torch.cuda" or a.name.startswith("torch.cuda.") for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return (
            node.module == "torch.cuda"
            or node.module.startswith("torch.cuda.")
            or (node.module == "torch" and any(a.name == "cuda" for a in node.names))
        )
    return False


def findings(source: str) -> list[int]:
    """Lines reaching ``torch.cuda`` in *source* that carry no allow marker.

    An import of it is flagged as well as an attribute access, so an alias cannot hide one.
    """
    lines = source.splitlines()
    used = {node.lineno for node in ast.walk(ast.parse(source)) if _names_cuda(node)}
    return sorted(n for n in used if not _ALLOW.search(lines[n - 1]))


def _scanned(rel: str) -> bool:
    return any(rel == s or rel.startswith(f"{s}/") for s in SCANNED)


def file_findings(path: Path) -> list[str]:
    rel = relative(path)
    if not _scanned(rel):
        return []
    return [
        f"{rel}:{line}: torch.cuda in a path every device runs through — use "
        "tileops.utils.device or torch.accelerator, or mark CUDA-only code "
        "'# lint: device-api allow: <reason>'"
        for line in findings(path.read_text())
    ]


def main(argv: list[str]) -> int:
    trees = [s for s in SCANNED if not s.endswith(".py")]
    named = argv or [*files([], trees, (".py",)), *(s for s in SCANNED if s.endswith(".py"))]
    return report(e for p in named for e in file_findings(Path(p)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
