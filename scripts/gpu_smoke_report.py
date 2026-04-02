#!/usr/bin/env python3
"""Generate a GPU smoke markdown report from pytest JUnit XML."""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

_PASS = "\u2705"
_FAIL = "\u274c"


def _get_properties(testcase: ET.Element) -> dict[str, str]:
    props: dict[str, str] = {}
    for ps in testcase.iter("properties"):
        for prop in ps.iter("property"):
            props[prop.attrib["name"]] = prop.attrib.get("value", "")
    return props


def _get_failure_reason(testcase: ET.Element) -> str:
    failure = testcase.find("failure")
    error = testcase.find("error")
    node = failure if failure is not None else error
    if node is None:
        return ""

    message = (node.attrib.get("message") or "").strip()
    if message:
        return message

    text = (node.text or "").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _escape_markdown_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def parse_test_xml(path: str) -> list[dict[str, str]]:
    tree = ET.parse(path)
    results: list[dict[str, str]] = []
    for testcase in tree.iter("testcase"):
        props = _get_properties(testcase)
        skipped = testcase.find("skipped")
        failure = testcase.find("failure")
        error = testcase.find("error")
        if skipped is not None:
            outcome = "skipped"
        elif failure is not None or error is not None:
            outcome = "failed"
        else:
            outcome = "passed"

        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        results.append(
            {
                "nodeid": f"{classname}::{name}",
                "name": name,
                "outcome": outcome,
                "op": props.get("op", ""),
                "op_module": props.get("op_module", ""),
                "failure_reason": _get_failure_reason(testcase),
            }
        )
    return results


def generate_report(results: list[dict[str, str]], target: str) -> str:
    total_cases = len(results)
    failed_cases = [result for result in results if result["outcome"] == "failed"]
    passed_count = sum(1 for result in results if result["outcome"] == "passed")
    skipped_count = sum(1 for result in results if result["outcome"] == "skipped")
    failed_ops = list(
        OrderedDict.fromkeys(result["op"] for result in failed_cases if result["op"])
    )

    correctness = f"{_PASS} Pass" if not failed_cases else f"{_FAIL} {len(failed_cases)} failed"
    failures_ops_str = ", ".join(failed_ops) if failed_ops else "-"
    health = _PASS if not failed_cases else _FAIL

    lines = [
        f"# {health} TileOPs GPU Smoke Report",
        "",
        f"> `{_get_git_commit()}`",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| **Correctness** | {correctness} |",
        f"| **gpu-smoke target** | `{target}` |",
        (
            f"| **Gpu-smoke ops number** | {total_cases} "
            f"({passed_count} passed, {skipped_count} skipped) |"
        ),
        f"| **Gpu-smoke Failures** | {len(failed_cases)} |",
        f"| **Failures ops** | {failures_ops_str} |",
        "",
    ]

    if failed_cases:
        lines.extend(
            [
                "## Failure Details",
                "",
                "| Op | Testcase | Failure Reason |",
                "|:---|:---------|:---------------|",
            ]
        )
        for result in failed_cases:
            op = _escape_markdown_cell(result["op"] or "-")
            testcase = _escape_markdown_cell(result["name"] or result["nodeid"])
            reason = _truncate(result["failure_reason"] or "No failure message available.")
            reason = _escape_markdown_cell(reason)
            lines.append(f"| {op} | {testcase} | {reason} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TileOPs GPU smoke report")
    parser.add_argument("--test-xml", required=True, help="Path to pytest JUnit XML")
    parser.add_argument("--target", required=True, help="Displayed gpu-smoke target label")
    parser.add_argument("--output", required=True, help="Output markdown report path")
    args = parser.parse_args()

    results = parse_test_xml(args.test_xml)
    report = generate_report(results, args.target)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
