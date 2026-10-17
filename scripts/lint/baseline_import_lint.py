#!/usr/bin/env python3
"""Flag a benchmark that imports a baseline library itself.

``benchmarks/baselines.py`` decides whether a library can run on the run device: absent
on CUDA fails the row, absent or CUDA-only elsewhere skips it or drops its tag. An import
written in a bench file skips that decision. Resolve the library through ``baseline_module``,
``optional_baseline`` or a ``*_op`` resolver instead.

Flags ``import X`` and ``from X import ...`` of a package in ``LIBRARIES``, at any depth.
Exempt: ``benchmarks/baselines.py``. Default scan: ``benchmarks/``.
"""

import ast
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("benchmarks",)
EXEMPT = {"benchmarks/baselines.py"}

# The top-level packages of the libraries benchmarks compare against. torch, triton and
# the profiler bindings are tools the harness itself uses, not baselines.
LIBRARIES = frozenset(
    {
        "deep_gemm",
        "deepspeed",
        "fla",
        "flag_gems",
        "flash_attn",
        "flash_attn_interface",
        "flashinfer",
        "mamba_ssm",
        "nvidia",
        "nvmath",
        "quack",
        "sgl_kernel",
        "vllm",
    }
)


def findings(source: str) -> list[tuple[int, str]]:
    """``(line, package)`` of every import of a baseline library in *source*."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        out += [(node.lineno, m.split(".")[0]) for m in modules if m.split(".")[0] in LIBRARIES]
    return sorted(out)


def file_findings(path: Path) -> list[str]:
    rel = relative(path)
    if not rel.startswith(tuple(f"{d}/" for d in SCANNED)) or rel in EXEMPT:
        return []
    return [
        f"{rel}:{line}: imports baseline library {package} — resolve it through "
        "benchmarks.baselines (baseline_module, optional_baseline or a *_op resolver)"
        for line, package in findings(path.read_text())
    ]


def main(argv: list[str]) -> int:
    return report(e for p in files(argv, SCANNED, (".py",)) for e in file_findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
