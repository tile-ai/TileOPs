#!/usr/bin/env python3
"""Lint kernel interface class names against the naming rule.

A class that inherits ``KernelInterface`` directly is named ``{Name}{Fwd|Bwd}Interface``: the
direction suffix is mandatory and every variant word precedes it, as in an op key
(``docs/design/ops-design-reference.md`` § Naming Conventions). ``BatchNormTrainFwdInterface``
passes; ``BatchNormFwdTrainInterface`` and ``W4A16RepackInterface`` do not.

Usage: ``interface_names_lint.py [FILE ...]``. With no arguments, scans ``src/tileops/kernels/``.
Exits 1 on a finding.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = "src/tileops/kernels"
NAME = re.compile(r"[A-Z][A-Za-z0-9]*(Fwd|Bwd)Interface")


def _inherits_interface(node: ast.ClassDef) -> bool:
    return any(
        (isinstance(b, ast.Name) and b.id == "KernelInterface")
        or (isinstance(b, ast.Attribute) and b.attr == "KernelInterface")
        for b in node.bases
    )


def findings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{path}:{node.lineno}: kernel interface {node.name} is not "
        f"{{Name}}{{Fwd|Bwd}}Interface with its variant words before the direction"
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and _inherits_interface(node)
        and not NAME.fullmatch(node.name)
    ]


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted((REPO_ROOT / SCANNED).rglob("*.py"))
    errors = [e for p in paths if p.suffix == ".py" for e in findings(p)]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
