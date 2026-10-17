#!/usr/bin/env python3
"""Lint manifest workload labels: rows of one entry that share a label differ only in dtype.

A row's case id is its label followed by its dtype values, and the case id keys nightly
history (``docs/design/manifest.md`` § Rows). Rows that share a label therefore differ only
in dtype: in ``dtype_cases`` or in a parameter whose type is a dtype. Any other difference
files two workloads under one history.

A label shared across entries is not checked: it names the scenario a row models, and
entries of one family share scenarios on purpose. The label's own form and length are the
validator's (``tileops.manifest.workload``). An exceptional row carries
``# lint: manifest-label allow: <reason>`` on the line the row starts on.

Usage: ``manifest_label_lint.py``. Reads every manifest file
``tileops.manifest.manifest_files`` lists. Exits 1 on a finding.
"""

import argparse
import sys
from pathlib import Path

import yaml
from common import allow_marker, relative, report, use_source_tree

use_source_tree()

from tileops.manifest import load_adts, manifest_files  # noqa: E402
from tileops.manifest.signature import param_kind  # noqa: E402

_ALLOW = allow_marker("manifest-label")


def _row_lines(text: str) -> dict[tuple[str, int], int]:
    """The line each workload row starts on, by ``(op, row index)``."""
    root = yaml.compose(text)
    lines = {}
    if not isinstance(root, yaml.MappingNode):
        return lines
    for op, entry in root.value:
        if not isinstance(entry, yaml.MappingNode):
            continue
        for key, rows in entry.value:
            if key.value == "workloads" and isinstance(rows, yaml.SequenceNode):
                for i, row in enumerate(rows.value):
                    lines[(op.value, i)] = row.start_mark.line + 1
    return lines


def _dtype_keys(entry: dict, adts: dict) -> set[str]:
    """The row keys that only choose a dtype: ``dtype_cases`` and dtype parameters."""
    params = (entry.get("signature") or {}).get("params") or {}
    return {"dtype_cases"} | {
        name
        for name, decl in params.items()
        if isinstance(decl, dict) and param_kind(decl.get("type"), adts).payload().tag == "DType"
    }


def _ops(text: str) -> dict:
    data = yaml.safe_load(text) or {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def findings(name: str, text: str, adts: dict | None = None) -> list[str]:
    """Label findings for one manifest file *name* holding *text*."""
    adts = load_adts() if adts is None else adts
    ops, lines = _ops(text), _row_lines(text)
    physical = text.splitlines()
    out = []
    for op, entry in ops.items():
        dropped = _dtype_keys(entry, adts) | {"label"}
        first_by_label: dict[str, dict] = {}
        for i, row in enumerate(entry.get("workloads") or []):
            if not isinstance(row, dict) or not isinstance(row.get("label"), str):
                continue
            label, line = row["label"], lines.get((op, i), 0)
            workload = {k: v for k, v in row.items() if k not in dropped}
            first = first_by_label.setdefault(label, workload)
            if first == workload or (line and _ALLOW.search(physical[line - 1])):
                continue
            keys = sorted(
                k for k in first.keys() | workload.keys() if first.get(k) != workload.get(k)
            )
            out.append(
                f"{name}:{line}: {op}: rows labelled {label!r} differ in {', '.join(keys)} — rows "
                "that share a label differ only in dtype; give this row its own label, or add "
                '"# lint: manifest-label allow: <reason>" on the line'
            )
    return out


def main(argv: list[str]) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    return report(
        e
        for spec in manifest_files()
        for e in findings(relative(Path(str(spec))), spec.read_text(encoding="utf-8"))
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
