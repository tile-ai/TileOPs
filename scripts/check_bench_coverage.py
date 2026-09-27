#!/usr/bin/env python3
"""Check that every implemented op's manifest calls were each benchmarked once.

Which op a benchmark measures is a run-time fact: the op the benchmark wraps is the class it
constructs, and ``benchmarks/conftest.py`` records that class's name as the ``op`` property of
every benchmark testcase. This script reads those properties out of a benchmark run's JUnit
report and compares each op's recorded case ids with the case ids its workload rows produce.

``scripts/validate_manifest.py`` covers the other half — that a bench file takes its calls from
the manifest and its roofline off the op — from the source, without naming an op.

A case id recorded twice for one op fails this check: it keys the op's history, so two rows
sharing one would write one record. A case recorded nothing is NOT RUN, reported and left to
the benchmark job's own exit code.

Usage:
    python scripts/check_bench_coverage.py --bench-xml bench_results.xml \\
        [--output bench_coverage.md]

Exit code 0 = no case id was recorded twice; 1 = one was; 2 = the report is missing or
unusable, which is not a pass.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tileops.manifest import load_adts, load_manifest  # noqa: E402
from tileops.manifest.plan import entry_plan  # noqa: E402
from tileops.manifest.workload import instantiate  # noqa: E402

EXIT_OK = 0
EXIT_GAP = 1
EXIT_NO_REPORT = 2

FAIL = "FAIL"
NOT_RUN = "NOT RUN"
OK = "OK"

# Listing order: what needs acting on first.
_ORDER = {FAIL: 0, NOT_RUN: 1, OK: 2}


def _properties(testcase: ET.Element) -> dict[str, str]:
    return {
        p.attrib["name"]: p.attrib.get("value", "")
        for props in testcase.iter("properties")
        for p in props.iter("property")
    }


def parse_run(xml_path: Path) -> dict[str, list[str]]:
    """The case ids each op recorded in a benchmark report, from its passing testcases."""
    recorded: dict[str, list[str]] = {}
    for testcase in ET.parse(xml_path).iter("testcase"):
        if any(testcase.find(tag) is not None for tag in ("skipped", "failure", "error")):
            continue
        # Op names reach the report only from recorded tileops rows: a case timing a baseline
        # alone records none. ``ops`` lists every op the case benchmarked; ``op`` is the first.
        props = _properties(testcase)
        ops = props.get("ops") or props.get("op") or ""
        case = _case_id(testcase.attrib.get("name", ""))
        for name in (n for n in ops.split(",") if n):
            recorded.setdefault(name, []).append(case)
    return recorded


def _case_id(name: str) -> str:
    """The parametrized part of a testcase name, which is the case's own id."""
    if name.endswith("]") and "[" in name:
        return name[name.index("[") + 1 : -1]
    return name


def _declared(op_name: str, entry: dict) -> set[str]:
    """The case ids the entry's workload rows produce."""
    plan = entry_plan(op_name, entry, load_adts(), resolve=False)
    return {
        instantiate(plan, row, case).case_id
        for row in entry.get("workloads") or ()
        for case in row.get("dtype_cases") or [{}]
    }


def verdicts(recorded: dict[str, list[str]], manifest: dict) -> list[tuple[str, str, str]]:
    """One ``(op, verdict, detail)`` row per implemented entry."""
    rows = []
    for op_name, entry in sorted(manifest.items()):
        if entry.get("status") != "implemented":
            continue
        cases = recorded.get(op_name, [])
        repeated = sorted({c for c in cases if cases.count(c) > 1})
        declared = _declared(op_name, entry)
        missing = sorted(declared - set(cases))
        if repeated:
            rows.append((op_name, FAIL, f"case ids recorded twice: {repeated}"))
        elif missing:
            detail = f"{len(missing)} of {len(declared)} cases recorded nothing: {missing[0]!r}"
            rows.append((op_name, NOT_RUN, detail))
        else:
            rows.append((op_name, OK, f"{len(declared)} cases recorded"))
    return rows


def render(rows: list[tuple[str, str, str]]) -> str:
    """A markdown summary listing every row that is not OK."""
    counts = {v: sum(1 for r in rows if r[1] == v) for v in _ORDER}
    lines = [
        "# Benchmark coverage",
        "",
        f"{len(rows)} implemented ops: " + ", ".join(f"{n} {v}" for v, n in counts.items() if n),
        "",
    ]
    listed = sorted((r for r in rows if r[1] != OK), key=lambda r: (_ORDER[r[1]], r[0]))
    if listed:
        lines += ["| Op | Verdict | Detail |", "| --- | --- | --- |"]
        lines += [f"| `{op}` | {verdict} | {detail} |" for op, verdict, detail in listed]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-xml", required=True, help="benchmark run's JUnit report")
    parser.add_argument("--output", help="write the markdown summary here as well")
    args = parser.parse_args(argv)

    xml_path = Path(args.bench_xml)
    if not xml_path.is_file():
        print(f"[coverage] benchmark report not found: {xml_path}", file=sys.stderr)
        return EXIT_NO_REPORT
    try:
        recorded = parse_run(xml_path)
    except ET.ParseError as exc:
        print(f"[coverage] benchmark report {xml_path} is unusable: {exc}", file=sys.stderr)
        return EXIT_NO_REPORT

    rows = verdicts(recorded, load_manifest())
    summary = render(rows)
    print(summary)
    if args.output:
        Path(args.output).write_text(summary, encoding="utf-8")

    failures = [r for r in rows if r[1] == FAIL]
    for op_name, _, detail in failures:
        print(f"[coverage] {op_name}: {detail}", file=sys.stderr)
    return EXIT_GAP if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
