#!/usr/bin/env python3
"""Write the nightly's perf gap queue to the wiki's Perf Tracker section.

Reads the TileOPs-nightly ``snapshots`` branch, newest commit first, and writes
``Perf-Tracker-<date>.md`` for the newest nightly, the ``Perf-Tracker.md`` index, and
the Perf Tracker block of ``_Sidebar.md``. A row is behind by
``nightly_report.row_behind``, the README badge's rule.
"""

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nightly_report import (
    BADGE_NOISE_MS,
    BASELINE_RATIO_ALERT,
    case_reading,
    fastest_comparator,
    row_behind,
)

RUN_URL = "https://github.com/tile-ai/TileOPs/actions/runs/{}"
COMMIT_URL = "https://github.com/tile-ai/TileOPs/commit/{}"
PAGE_PREFIX = "Perf-Tracker"
# The sidebar lines between these markers are this script's; the rest is edited by hand.
SIDEBAR_BEGIN = "<!-- perf-tracker:begin -->"
SIDEBAR_END = "<!-- perf-tracker:end -->"
# An op takes the label of its worst row: the label of the first bound above its ratio.
LABELS = ((0.50, "P0"), (0.80, "P1"), (0.90, "P2"), (0.95, "P3"), (BASELINE_RATIO_ALERT, "P4"))
# Row properties that name how a row was timed.
METHOD_SUFFIXES = ("_timing", "_l2_flush")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def nightlies(repo: Path, count: int) -> list[dict]:
    """The newest *count* nightlies on the snapshots branch, newest first, one per run."""
    seen, runs = set(), []
    for commit in _git(repo, "rev-list", "HEAD").split():
        files = _git(repo, "ls-tree", "--name-only", commit).split()
        if "meta.json" not in files or "bench_results.xml" not in files:
            continue
        meta = json.loads(_git(repo, "show", f"{commit}:meta.json"))
        if meta["run_id"] in seen:
            continue
        seen.add(meta["run_id"])
        runs.append({"commit": commit, "meta": meta})
        if len(runs) == count:
            break
    return runs


def read_rows(repo: Path, commit: str) -> dict:
    """``{(classname, name): row}`` of one nightly's benchmark XML.

    A failed testcase without an ``op``, such as the error the benchmark runner records
    for a crashed file, is kept with ``op`` ``None``; other testcases without one are not
    benchmark rows.
    """
    rows = {}
    for case in ET.fromstring(_git(repo, "show", f"{commit}:bench_results.xml")).iter("testcase"):
        props, reading = case_reading(case)
        failure = next((c for c in case if c.tag in ("failure", "error")), None)
        if not props.get("op") and failure is None:
            continue
        message = "" if failure is None else (failure.get("message") or "")
        workload = re.search(r"\[(.*)\]$", case.get("name") or "")
        fastest = fastest_comparator(props)
        rows[(case.get("classname"), case.get("name"))] = {
            "op": props.get("op"),
            # tileops._ops.<category>.<module>
            "category": ((props.get("op_module") or "").split(".") + ["", "", ""])[2],
            "benchmark": case.get("classname"),
            "workload": workload.group(1) if workload else case.get("name"),
            "failed": failure is not None,
            "message": message.split("\n", 1)[0].replace("|", "\\|"),
            "reading": reading,
            "comparator": fastest[0] if fastest else None,
            "method": {k: v for k, v in props.items() if k.endswith(METHOD_SUFFIXES)},
        }
    return rows


def load(repo: Path, run_id: str, nights: int) -> tuple[list[dict], list[dict]]:
    """The nightlies judged for *nights* nights and their rows; exits unless *run_id* is newest."""
    # One more nightly than judged: the oldest judged night needs its predecessor.
    runs = nightlies(repo, nights + 1)
    if not runs or runs[0]["meta"]["run_id"] != run_id:
        newest = runs[0]["meta"]["run_id"] if runs else "none"
        sys.exit(f"newest snapshot is run {newest}, not run {run_id}")
    return runs, [read_rows(repo, run["commit"]) for run in runs]


def behind_keys(rows: dict, previous: dict | None) -> set:
    """Keys of the rows behind, each judged against the same row in *previous*."""
    earlier = previous or {}
    return {
        key
        for key, row in rows.items()
        if row["reading"] is not None
        and row_behind(row["reading"], (earlier.get(key) or {}).get("reading"))
    }


def label(ratio: float) -> str:
    return next(name for bound, name in LABELS if ratio < bound)


def _ratio(row: dict) -> float:
    tileops_ms, fastest_ms = row["reading"]
    return fastest_ms / tileops_ms


def build_queue(rows_by_run: list[dict]) -> tuple[list[dict], list[set], int]:
    """The ops behind in ``rows_by_run[0]``, by label then worst ratio.

    Each night is judged against the night after it in the list, so the last night is
    judged only when it is the only one. Returns the queue, the behind keys of each
    judged night, and how many nights were judged.
    """
    nights = max(len(rows_by_run) - 1, 1)
    behind = [
        behind_keys(rows_by_run[i], rows_by_run[i + 1] if i + 1 < len(rows_by_run) else None)
        for i in range(nights)
    ]
    rows = rows_by_run[0]
    by_op: dict[str, list] = {}
    for key in behind[0]:
        by_op.setdefault(rows[key]["op"], []).append(rows[key])
    queue = []
    for op, op_rows in by_op.items():
        op_rows.sort(key=lambda r: (_ratio(r), r["workload"]))
        streak = 0
        while streak < nights and op in {rows_by_run[streak][k]["op"] for k in behind[streak]}:
            streak += 1
        worst = _ratio(op_rows[0])
        queue.append(
            {
                "op": op,
                "label": label(worst),
                "worst": worst,
                "rows": op_rows,
                "total": sum(1 for r in rows.values() if r["op"] == op and r["reading"]),
                "streak": streak,
            }
        )
    queue.sort(key=lambda item: (item["label"], item["worst"], item["op"]))
    return queue, behind, nights


def caught_up(rows_by_run: list[dict], behind: list[set]) -> set:
    """The ops caught up in ``rows_by_run[0]``, from ``build_queue``'s *behind*.

    An op is caught up when it has a row measured, none behind and none failed, and every
    row it had behind the night before is measured.
    """
    rows = rows_by_run[0]
    ops = {r["op"] for r in rows.values() if r["reading"]}
    ops -= {rows[k]["op"] for k in behind[0]} | {r["op"] for r in rows.values() if r["failed"]}
    if len(behind) > 1:
        previous = rows_by_run[1]
        ops -= {previous[k]["op"] for k in behind[1] if not (rows.get(k) or {}).get("reading")}
    return ops


def measurement_changes(tonight: dict, previous: dict, rows: dict, previous_rows: dict) -> list:
    """``(what, before, now)`` for what changed in how rows were measured or compared."""
    changes = []
    for section in ("environment", "packages"):
        now, before = tonight["meta"].get(section, {}), previous["meta"].get(section, {})
        # The TileOPs version moves every night by construction.
        for key in sorted((now.keys() | before.keys()) - {"tileops"}):
            if now.get(key) != before.get(key):
                changes.append((key, before.get(key), now.get(key)))
    moved = Counter()
    for key in rows.keys() & previous_rows.keys():
        now, before = rows[key]["method"], previous_rows[key]["method"]
        for prop in now.keys() | before.keys():
            if now.get(prop) != before.get(prop):
                moved[(prop, before.get(prop), now.get(prop))] += 1
    for (prop, before, now), n in sorted(moved.items(), key=lambda item: str(item[0])):
        changes.append((f"{prop}, {n} rows", before, now))
    return changes


def _us(ms: float) -> str:
    return f"{ms * 1000:.3f}"


def _op_list(ops) -> str:
    return ", ".join(f"`{op}`" for op in sorted(ops)) or "none"


def render(date: str, runs: list[dict], rows_by_run: list[dict]) -> str:
    """The page for *date*, showing ``runs[0]``."""
    rows = rows_by_run[0]
    queue, behind, nights = build_queue(rows_by_run)
    meta, env = runs[0]["meta"], runs[0]["meta"].get("environment", {})
    bounds = ", ".join(f"{name} < {bound:.2f}" for bound, name in LABELS)
    lines = [
        f"# Perf Tracker {date}",
        "",
        f"Nightly run [{meta['run_id']}]({RUN_URL.format(meta['run_id'])}), commit "
        f"[`{meta['commit'][:9]}`]({COMMIT_URL.format(meta['commit'])}), {meta.get('gpu', '')}, "
        f"CUDA {env.get('cuda', '?')}, driver {env.get('driver', '?')}, "
        f"torch {env.get('torch', '?')}, TileLang {env.get('tilelang', '?')}, "
        f"image `{env.get('image', '?')}`. Times are CUPTI `device_busy_ms`.",
        "",
        f"A row is behind when its fastest comparator's time over TileOPs' time is below "
        f"{BASELINE_RATIO_ALERT}. A row whose fastest comparator takes under "
        f"{BADGE_NOISE_MS * 1000:g} us is behind only if it was also below that ratio the "
        f"night before. An op is labelled by its worst row: {bounds}.",
        "",
    ]
    if len(runs) > 1:
        changes = measurement_changes(runs[0], runs[1], rows, rows_by_run[1])
        if changes:
            lines += ["**What was measured or compared changed since the previous nightly:**", ""]
            lines += [f"- `{what}`: `{before}` -> `{now}`" for what, before, now in changes]
            lines.append("")

    rated = [r for r in rows.values() if r["reading"]]
    counts = Counter(item["label"] for item in queue)
    lines += [
        "## Summary",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Ops behind | {len(queue)} of {len({r['op'] for r in rated})} |",
        f"| Rows behind | {len(behind[0])} of {len(rated)} |",
        "| By label | " + ", ".join(f"{name} {counts[name]}" for _, name in LABELS) + " |",
        f"| Failed rows | {sum(1 for r in rows.values() if r['failed'])} |",
        "",
    ]
    if nights > 1:
        previous_rows = rows_by_run[1]
        now_ops = {item["op"] for item in queue}
        before_ops = {previous_rows[k]["op"] for k in behind[1]}
        recovered = before_ops & caught_up(rows_by_run, behind)
        lines += [
            f"Since nightly run [{runs[1]['meta']['run_id']}]"
            f"({RUN_URL.format(runs[1]['meta']['run_id'])}):",
            "",
            f"- Newly behind: {_op_list(now_ops - before_ops)}",
            f"- Caught up: {_op_list(recovered)}",
            f"- Not measured tonight: {_op_list(before_ops - now_ops - recovered)}",
            "",
        ]

    lines += [
        "## Queue",
        "",
        f"Nightlies behind counts the consecutive nightlies, up to {nights}, in which the op "
        "had a row behind.",
        "",
        "| # | Op | Category | Label | Worst ratio | Rows behind | Nightlies behind |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for i, item in enumerate(queue, 1):
        op = item["op"]
        lines.append(
            f"| {i} | [{op}](#{op.lower()}) | {item['rows'][0]['category']} | {item['label']} | "
            f"{item['worst']:.4f} | {len(item['rows'])} of {item['total']} | {item['streak']} |"
        )
    lines += ["", "## Rows behind", ""]
    for item in queue:
        lines += [
            f"### {item['op']}",
            "",
            f"Benchmark `{item['rows'][0]['benchmark']}`.",
            "",
            "| Workload | TileOPs (us) | Fastest comparator | Comparator (us) | Ratio |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
        lines += [
            f"| `{r['workload']}` | {_us(r['reading'][0])} | {r['comparator']} | "
            f"{_us(r['reading'][1])} | {_ratio(r):.4f} |"
            for r in item["rows"]
        ]
        lines.append("")

    failed = sorted(
        (r["op"] or "-", r["benchmark"], r["workload"], r["message"])
        for r in rows.values()
        if r["failed"]
    )
    if failed:
        lines += ["## Failed rows", "", "| Op | Benchmark | Workload | Message |"]
        lines.append("| --- | --- | --- | --- |")
        lines += [f"| {op} | `{bench}` | `{wl}` | {msg} |" for op, bench, wl, msg in failed]
        lines.append("")
    return "\n".join(lines)


def _dates(wiki: Path) -> list[str]:
    """The dates of the daily pages, newest first."""
    return sorted(
        (p.stem[len(PAGE_PREFIX) + 1 :] for p in wiki.glob(f"{PAGE_PREFIX}-*.md")), reverse=True
    )


def render_index(wiki: Path) -> str:
    """The section's index: every daily page, newest first."""
    lines = [
        "# Perf Tracker",
        "",
        "One page per date: the ops slower than their fastest comparator in that date's newest "
        "nightly run, written by the nightly workflow. A later run on the same date replaces "
        "the page. Do not edit these pages.",
        "",
    ]
    lines += [f"- [{date}]({PAGE_PREFIX}-{date})" for date in _dates(wiki)]
    return "\n".join(lines) + "\n"


def render_sidebar(wiki: Path) -> str:
    """``_Sidebar.md`` with its Perf Tracker block listing every daily page, newest first."""
    block = [SIDEBAR_BEGIN, f"**[Perf Tracker]({PAGE_PREFIX})**", ""]
    block += [f"- [{date}]({PAGE_PREFIX}-{date})" for date in _dates(wiki)]
    block.append(SIDEBAR_END)
    path = wiki / "_Sidebar.md"
    text = path.read_text() if path.exists() else "**[Home](Home)**\n"
    begins, ends = text.count(SIDEBAR_BEGIN), text.count(SIDEBAR_END)
    if begins == ends == 0:
        return text.rstrip("\n") + "\n\n" + "\n".join(block) + "\n"
    if begins != 1 or ends != 1 or text.index(SIDEBAR_BEGIN) > text.index(SIDEBAR_END):
        raise ValueError(f"_Sidebar.md needs one {SIDEBAR_BEGIN} line before one {SIDEBAR_END}")
    head, rest = text.split(SIDEBAR_BEGIN)
    return head + "\n".join(block) + rest.split(SIDEBAR_END)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, type=Path, help="TileOPs-nightly clone")
    parser.add_argument("--run-id", required=True, help="the nightly run that must be newest")
    parser.add_argument("--wiki", required=True, type=Path, help="wiki clone to write into")
    parser.add_argument("--date", required=True, help="the page's date, YYYY-MM-DD")
    parser.add_argument("--nights", type=int, default=30, help="nightlies judged for the streaks")
    args = parser.parse_args()

    runs, rows_by_run = load(args.snapshots, args.run_id, args.nights)
    page = args.wiki / f"{PAGE_PREFIX}-{args.date}.md"
    page.write_text(render(args.date, runs, rows_by_run) + "\n")
    (args.wiki / f"{PAGE_PREFIX}.md").write_text(render_index(args.wiki))
    (args.wiki / "_Sidebar.md").write_text(render_sidebar(args.wiki))
    print(f"wrote {page.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
