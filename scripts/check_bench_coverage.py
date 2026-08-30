#!/usr/bin/env python3
"""Check that every implemented op's benchmark actually benchmarked it.

Which op a bench file measures is a run-time fact: the op the benchmark wraps is
the class it constructs, and ``benchmarks/conftest.py`` records that class's name
as the ``op`` property of every benchmark testcase. This script reads those
properties out of a benchmark run's JUnit report and compares them with the ops
the manifest declares a benchmark for.

``scripts/validate_manifest.py`` covers the other half — that a bench file takes
its workloads from the manifest and its roofline off the op — from the source,
without naming an op.

Only a file whose testcases all passed and still recorded nothing for its op
fails this check. A file that failed, errored, was skipped or never ran is
reported and left to the benchmark job's own exit code, which already fails on
it; failing twice for one cause buries the row this check exists to surface.

Usage:
    python scripts/check_bench_coverage.py --bench-xml bench_results.xml \\
        [--output bench_coverage.md]

Exit code 0 = every declared op was benchmarked, or its run says why not;
1 = a benchmark passed without benchmarking the op it is declared for;
2 = the report is missing or unusable, which is not a pass.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tileops.manifest import load_manifest  # noqa: E402

EXIT_OK = 0
EXIT_GAP = 1
EXIT_NO_REPORT = 2

FAIL = "FAIL"
NO_VERDICT = "NO VERDICT"
NOT_RUN = "NOT RUN"
SKIPPED = "SKIPPED"
OK = "OK"

# Listing order: what needs acting on first.
_ORDER = {FAIL: 0, NO_VERDICT: 1, NOT_RUN: 2, SKIPPED: 3, OK: 4}


class FileRun:
    """What one bench file's testcases did in a run."""

    __slots__ = ("broken_reasons", "passed", "recorded", "skip_reasons", "testcases")

    def __init__(self) -> None:
        self.testcases = 0
        self.passed = 0
        self.recorded: dict[str, list[str]] = {}
        self.skip_reasons: list[str] = []
        self.broken_reasons: list[str] = []

    def absorb(self, other: "FileRun") -> None:
        self.testcases += other.testcases
        self.passed += other.passed
        for op, cases in other.recorded.items():
            self.recorded.setdefault(op, []).extend(cases)
        self.skip_reasons.extend(other.skip_reasons)
        self.broken_reasons.extend(other.broken_reasons)


def _properties(testcase: ET.Element) -> dict[str, str]:
    return {
        p.attrib["name"]: p.attrib.get("value", "")
        for props in testcase.iter("properties")
        for p in props.iter("property")
    }


def _reason(element: ET.Element) -> str:
    return (element.attrib.get("message") or element.attrib.get("type") or "").strip()


def _module_of(bench_path: str) -> str:
    """``benchmarks/ops/bench_x.py`` -> ``benchmarks.ops.bench_x``."""
    return bench_path.removesuffix(".py").replace("/", ".")


def parse_run(xml_path: Path) -> dict[str, FileRun]:
    """Group a benchmark report's testcases by the class name that holds them."""
    runs: dict[str, FileRun] = {}
    for testcase in ET.parse(xml_path).iter("testcase"):
        run = runs.setdefault(testcase.attrib.get("classname", ""), FileRun())
        run.testcases += 1
        skipped = testcase.find("skipped")
        if skipped is not None:
            run.skip_reasons.append(_reason(skipped))
            continue
        broken = testcase.find("failure")
        if broken is None:
            broken = testcase.find("error")
        if broken is not None:
            run.broken_reasons.append(_reason(broken))
            continue
        run.passed += 1
        # Op names reach the report only from recorded tileops rows: a case
        # timing a baseline alone records none. ``ops`` lists every op the case
        # benchmarked; ``op`` is the first, kept for older reports.
        props = _properties(testcase)
        recorded = props.get("ops") or props.get("op") or ""
        case = testcase.attrib.get("name", "")
        for name in (n for n in recorded.split(",") if n):
            run.recorded.setdefault(name, []).append(case)
    return runs


def _run_of(module: str, runs: dict[str, FileRun]) -> FileRun:
    """Fold every testcase of *module*, including those a class in it holds."""
    folded = FileRun()
    for classname, run in runs.items():
        if classname == module or classname.startswith(f"{module}."):
            folded.absorb(run)
    return folded


def declared_case_ids(entry: dict) -> set[str]:
    """The case ids the entry's own workloads produce: the label, then the dtype.

    A row whose id is none of them measured a shape this op does not declare,
    which is what L4 stopped asserting when it stopped matching op names in the
    source. Ids are compared whole, so ``llama-8b-short`` does not answer for a
    row of ``llama-8b-short-w256``.

    It catches a file reading workloads declared under other names, not one
    reading a sibling's: ops declaring the same labels and dtypes — ``AddFwdOp``
    and ``SubFwdOp``, say — produce the same ids, and rows on them sit on shapes
    this op declares anyway.
    """
    ids = set()
    for workload in entry.get("workloads") or ():
        label = workload.get("label")
        if not label:
            for key, value in workload.items():
                if key.endswith("_shape") and isinstance(value, list):
                    label = "x".join(str(v) for v in value)
                    break
        if not label:
            continue
        ids.add(label)
        for dtype in workload.get("dtypes") or ():
            ids.add(f"{label}-{dtype}")
    return ids


def _case_id(name: str) -> str:
    """The parametrized part of a testcase name, which is the case's own id."""
    if name.endswith("]") and "[" in name:
        return name[name.index("[") + 1 : -1]
    return name


def _verdict(op_name: str, run: FileRun, declared: set[str]) -> tuple[str, str]:
    """Judge one op against its bench file's run.

    A testcase that failed or was skipped carries no op name, so which op it
    belonged to is unknown; a file holding one answers for none of the ops it
    did not record. Only a file whose every testcase passed accuses an
    unrecorded op — there, nothing is left that could have benchmarked it.
    """
    cases = run.recorded.get(op_name)
    if cases:
        if declared and not any(_case_id(case) in declared for case in cases):
            return FAIL, (
                f"{len(cases)} rows, none on a workload the manifest declares "
                f"(e.g. {_case_id(cases[0])!r})"
            )
        return OK, f"{run.passed} testcases passed"
    if run.broken_reasons:
        return NO_VERDICT, f"{len(run.broken_reasons)} failed: {run.broken_reasons[0]}"
    if run.skip_reasons:
        return SKIPPED, f"{len(run.skip_reasons)} skipped: {run.skip_reasons[0]}"
    if run.passed:
        return FAIL, f"{run.passed} testcases passed, recording {sorted(run.recorded) or 'no op'}"
    return NOT_RUN, "no testcases in the report"


def verdicts(runs: dict[str, FileRun], manifest: dict) -> list[tuple[str, str, str, str]]:
    """One ``(op, bench, verdict, detail)`` row per op declaring a benchmark."""
    rows = []
    for op_name, entry in sorted(manifest.items()):
        if entry.get("status") != "implemented":
            continue
        bench = (entry.get("source") or {}).get("bench")
        if not bench:
            continue
        verdict, detail = _verdict(
            op_name, _run_of(_module_of(bench), runs), declared_case_ids(entry)
        )
        rows.append((op_name, bench, verdict, detail))
    return rows


def render(rows: list[tuple[str, str, str, str]]) -> str:
    """A markdown summary listing every row that is not OK."""
    counts = {v: sum(1 for r in rows if r[2] == v) for v in _ORDER}
    lines = [
        "# Benchmark coverage",
        "",
        f"{len(rows)} implemented ops declare a benchmark: "
        + ", ".join(f"{n} {v}" for v, n in counts.items() if n),
        "",
    ]
    listed = sorted((r for r in rows if r[2] != OK), key=lambda r: (_ORDER[r[2]], r[0]))
    if listed:
        lines += ["| Op | Verdict | Bench | Detail |", "| --- | --- | --- | --- |"]
        lines += [
            f"| `{op}` | {verdict} | `{bench}` | {detail} |"
            for op, bench, verdict, detail in listed
        ]
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
        runs = parse_run(xml_path)
    except ET.ParseError as exc:
        print(f"[coverage] benchmark report {xml_path} is unusable: {exc}", file=sys.stderr)
        return EXIT_NO_REPORT

    rows = verdicts(runs, load_manifest())
    summary = render(rows)
    print(summary)
    if args.output:
        Path(args.output).write_text(summary, encoding="utf-8")

    failures = [r for r in rows if r[2] == FAIL]
    for op_name, bench, _, detail in failures:
        print(
            f"[coverage] {op_name}: {bench} — {detail}; it passed without "
            "benchmarking the op it is declared for",
            file=sys.stderr,
        )
    return EXIT_GAP if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
