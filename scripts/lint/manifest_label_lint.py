#!/usr/bin/env python3
"""Lint manifest workload labels against the workloads they name.

A row's case id is its label followed by its dtype values, and the case id keys nightly
history (``docs/design/manifest.md`` § Rows). A label therefore names one workload of its
entry, its dtypes aside:

- Rows of one entry that share a label differ only in dtype: in ``dtype_cases`` or in a
  parameter whose type is a dtype.
- Two rows of one entry that are the same call carry one label, not two.
- Against the merge base, a label the entry already had names the workload it named there.
  A row whose parameters change takes a new label, and a label is not reused for another
  row, since either would file the new call's numbers under the old call's history.

A label shared across entries is not checked: it names the scenario a row models, and
entries of one family share scenarios on purpose. The label's own form and length are the
validator's (``tileops.manifest.workload``).

Usage: ``manifest_label_lint.py [--base REF]``. Reads every manifest file
``tileops.manifest.manifest_files`` lists. With ``--base``, also compares each with its
content at the merge base of REF and HEAD (``lint_diff.py``); the hook passes
``--base auto`` to find the TileOPs upstream, overridable with ``TILEOPS_LINT_BASE``.
Exits 1 on a finding and 2 when an explicit base has no merge base with HEAD.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from lint_diff import BASE_ENV, NoMergeBaseError, merge_base, resolve_base

REPO_ROOT = Path(__file__).resolve().parents[2]
# The hook runs this script in an environment without the package installed, so the
# manifest loader is read from the source tree.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tileops.manifest import load_adts, manifest_files  # noqa: E402
from tileops.manifest.signature import param_kind  # noqa: E402


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


def _canonical(row: dict, dropped: set[str]) -> str:
    return json.dumps(
        {k: v for k, v in row.items() if k not in dropped}, sort_keys=True, default=str
    )


def _workloads(entry: dict, adts: dict) -> dict[str, set[str]]:
    """Each label's workloads, dtypes aside."""
    dropped = _dtype_keys(entry, adts) | {"label"}
    out = defaultdict(set)
    for row in entry.get("workloads") or []:
        if isinstance(row, dict) and isinstance(row.get("label"), str):
            out[row["label"]].add(_canonical(row, dropped))
    return out


def _changed(before: set[str], after: set[str]) -> list[str]:
    """The keys whose value differs between some workload of *before* and of *after*."""
    keys = set()
    for a in before:
        for b in after:
            old, new = json.loads(a), json.loads(b)
            keys |= {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    return sorted(keys)


def _ops(text: str) -> dict:
    data = yaml.safe_load(text) or {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def findings(
    name: str, text: str, base_text: str | None = None, adts: dict | None = None
) -> list[str]:
    """Label findings for one manifest file *name* holding *text*.

    *base_text* is the file at the merge base, or None to skip the comparison with it.
    """
    adts = load_adts() if adts is None else adts
    ops, lines = _ops(text), _row_lines(text)
    out = []
    for op, entry in ops.items():
        dtype_keys = _dtype_keys(entry, adts)
        first_by_label: dict[str, str] = {}
        first_by_row: dict[str, str] = {}
        for i, row in enumerate(entry.get("workloads") or []):
            if not isinstance(row, dict) or not isinstance(row.get("label"), str):
                continue
            label, where = row["label"], f"{name}:{lines.get((op, i), 0)}: {op}"
            workload = _canonical(row, dtype_keys | {"label"})
            if label in first_by_label and first_by_label[label] != workload:
                keys = _changed({first_by_label[label]}, {workload})
                out.append(
                    f"{where}: rows labelled {label!r} differ in {', '.join(keys)} — rows that "
                    "share a label differ only in dtype; give this row its own label"
                )
            first_by_label.setdefault(label, workload)
            whole = _canonical(row, {"label"})
            if whole in first_by_row and first_by_row[whole] != label:
                out.append(
                    f"{where}: row {label!r} is the same call as row {first_by_row[whole]!r} — "
                    "one workload carries one label"
                )
            first_by_row.setdefault(whole, label)
    if base_text is None:
        return out
    base_ops = _ops(base_text)
    for op, entry in ops.items():
        if op not in base_ops:
            continue
        before, after = _workloads(base_ops[op], adts), _workloads(entry, adts)
        rows = entry.get("workloads") or []
        for label in sorted(before.keys() & after.keys()):
            if before[label] == after[label]:
                continue
            index = next(
                i for i, r in enumerate(rows) if isinstance(r, dict) and r.get("label") == label
            )
            keys = _changed(before[label], after[label])
            out.append(
                f"{name}:{lines.get((op, index), 0)}: {op}: row {label!r} changed "
                f"{', '.join(keys)} since the merge base — a changed row takes a new label, "
                f"since {label!r} keys the old row's nightly history"
            )
    return out


def _base_text(commit: str, rel: str) -> str:
    """*rel* at *commit*; empty when the commit does not hold it."""
    done = subprocess.run(
        ["git", "show", f"{commit}:{rel}"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return done.stdout if done.returncode == 0 else ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="also compare with the merge base of this ref and HEAD")
    args = parser.parse_args(argv)
    commit = None
    if args.base:
        try:
            base = resolve_base(args.base, REPO_ROOT)
            if base is None:
                print(
                    "manifest_label_lint: skipping base comparison: no fetched "
                    f"tile-ai/TileOPs main remote; set {BASE_ENV} to a base ref",
                    file=sys.stderr,
                )
            else:
                commit = merge_base(base, REPO_ROOT)
        except NoMergeBaseError as exc:
            print(f"manifest_label_lint: {exc}", file=sys.stderr)
            return 2
    errors = []
    for spec in manifest_files():
        path = Path(str(spec))
        rel = (
            path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.name
        )
        base = _base_text(commit, rel) if commit else None
        errors += findings(rel, spec.read_text(encoding="utf-8"), base)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
