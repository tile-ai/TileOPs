#!/usr/bin/env python3
"""Flag a direct ``KernelInterface`` subclass not named ``{Name}{Fwd|Bwd}Interface``.

The direction comes after every variant word (``BatchNormTrainFwdInterface``), as in an op
key (``docs/design/ops-design-reference.md`` § Naming Conventions).

Default scan: ``src/tileops/kernels/``.
"""

import ast
import re
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("src/tileops/kernels",)
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
        f"{relative(path)}:{node.lineno}: kernel interface {node.name} is not "
        f"{{Name}}{{Fwd|Bwd}}Interface with its variant words before the direction"
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and _inherits_interface(node)
        and not NAME.fullmatch(node.name)
    ]


def main(argv: list[str]) -> int:
    return report(e for p in files(argv, SCANNED, (".py",)) for e in findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
