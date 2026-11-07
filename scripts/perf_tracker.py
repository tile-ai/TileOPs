#!/usr/bin/env python3
"""Write the nightly's perf gap queue to the wiki's Perf Tracker section.

Reads the TileOPs-nightly ``snapshots`` branch, newest commit first, and writes
``Perf-Tracker-<date>.md`` for the newest nightly, the ``Perf-Tracker.md`` index, and
the Perf Tracker block of ``_Sidebar.md``.
A row is behind by ``nightly_report.row_behind``, the README badge's rule.
"""

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nightly_report import BADGE_NOISE_MS, BASELINE_RATIO_ALERT, fastest_comparator, row_behind

RUN_URL = "https://github.com/tile-ai/TileOPs/actions/runs/{}"
COMMIT_URL = "https://github.com/tile-ai/TileOPs/commit/{}"
PAGE_PREFIX = "Perf-Tracker"
# The sidebar lines between these markers are this script's; the rest is edited by hand.
SIDEBAR_BEGIN = "<!-- perf-tracker:begin -->"
SIDEBAR_END = "<!-- perf-tracker:end -->"
# An op takes the label of its worst row: the label of the first bound above its ratio.
LABELS = ((0.50, "P0"), (0.80, "P1"), (0.90, "P2"), (0.95, "P3"), (BASELINE_RATIO_ALERT, "P4"))
# Row properties that name how a row was timed; a change in any of them moves every row.
METHOD_SUFFIXES = ("_timing", "_l2_flush")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def nightlies(repo: Path, depth: int) -> list[dict]:
    """The newest *depth* nightlies on the snapshots branch, newest first, one per run."""
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
        if len(runs) == depth:
            break
    return runs


def read_rows(repo: Path, commit: str) -> dict:
    """``{(classname, name): row}`` of one nightly's benchmark XML."""
    root = ET.fromstring(_git(repo, "show", f"{commit}:bench_results.xml"))
    rows = {}
    for case in root.iter("testcase"):
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        if not props.get("op"):
            continue
        fastest = fastest_comparator(props)
        tileops = float(props.get("tileops_device_busy_ms") or 0)
        failed = any(child.tag in ("failure", "error") for child in case)
        workload = re.search(r"\[(.*)\]$", case.get("name") or "")
        rows[(case.get("classname"), case.get("name"))] = {
            "op": props["op"],
            # tileops._ops.<category>.<module>
            "category": ((props.get("op_module") or "").split(".") + ["", "", ""])[2],
            "benchmark": case.get("classname"),
            "workload": workload.group(1) if workload else case.get("name"),
            "failed": failed,
            "reading": None if failed or fastest is None or tileops == 0 else (tileops, fastest[1]),
            "comparator": fastest[0] if fastest else None,
            "method": {k: v for k, v in props.items() if k.endswith(METHOD_SUFFIXES)},
        }
    return rows


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


def measurement_changes(tonight: dict, previous: dict, rows: dict, previous_rows: dict) -> list:
    """``(what, before, now)`` for each environment key or timing method that changed."""
    changes = []
    env_now = tonight["meta"].get("environment", {})
    env_before = previous["meta"].get("environment", {})
    # The TileOPs version moves every night by construction.
    for key in sorted((set(env_now) | set(env_before)) - {"tileops"}):
        if env_now.get(key) != env_before.get(key):
            changes.append((key, env_before.get(key), env_now.get(key)))
    for suffix in METHOD_SUFFIXES:
        now = {v for r in rows.values() for k, v in r["method"].items() if k.endswith(suffix)}
        before = {
            v for r in previous_rows.values() for k, v in r["method"].items() if k.endswith(suffix)
        }
        if now != before:
            changes.append((suffix[1:], ", ".join(sorted(before)), ", ".join(sorted(now))))
    return changes


def _ops(rows: dict, keys: set) -> set:
    return {rows[key]["op"] for key in keys}


def _us(ms: float) -> str:
    return f"{ms * 1000:.3f}"


def render(runs: list[dict], rows_by_run: list[dict]) -> str:
    """The daily page for ``runs[0]``."""
    tonight, rows = runs[0], rows_by_run[0]
    previous_rows = rows_by_run[1] if len(rows_by_run) > 1 else None
    behind = [behind_keys(rows_by_run[i], rows_by_run[i + 1]) for i in range(len(runs) - 1)]
    behind.append(behind_keys(rows_by_run[-1], None))
    tonight_behind = behind[0]

    by_op: dict[str, list] = {}
    for key in tonight_behind:
        by_op.setdefault(rows[key]["op"], []).append(rows[key])
    queue = []
    for op, op_rows in by_op.items():
        op_rows.sort(key=lambda r: r["reading"][1] / r["reading"][0])
        worst = op_rows[0]["reading"][1] / op_rows[0]["reading"][0]
        streak = 0
        while streak < len(behind) and op in _ops(rows_by_run[streak], behind[streak]):
            streak += 1
        total = sum(1 for r in rows.values() if r["op"] == op and r["reading"] is not None)
        queue.append(
            {
                "label": label(worst),
                "worst": worst,
                "op": op,
                "rows": op_rows,
                "total": total,
                "streak": streak,
            }
        )
    queue.sort(key=lambda item: (item["label"], item["worst"]))

    meta = tonight["meta"]
    env = meta.get("environment", {})
    rated = [r for r in rows.values() if r["reading"] is not None]
    lines = [
        f"# Perf Tracker {meta['date']}",
        "",
        f"Nightly run [{meta['run_id']}]({RUN_URL.format(meta['run_id'])}), commit "
        f"[`{meta['commit'][:9]}`]({COMMIT_URL.format(meta['commit'])}), {meta.get('gpu', '')}, "
        f"CUDA {env.get('cuda', '?')}, driver {env.get('driver', '?')}, "
        f"torch {env.get('torch', '?')}, TileLang {env.get('tilelang', '?')}, "
        f"image `{env.get('image', '?')}`. Times are CUPTI `device_busy_ms`.",
        "",
        f"A row is behind when its fastest comparator's time over TileOPs' time is below "
        f"{BASELINE_RATIO_ALERT}. A row whose fastest comparator takes under "
        f"{BADGE_NOISE_MS * 1000:g} us is behind only if it was also behind the night before. "
        "An op is labelled by its worst row: P0 < 0.50, P1 < 0.80, P2 < 0.90, P3 < 0.95, "
        f"P4 < {BASELINE_RATIO_ALERT}.",
        "",
    ]
    if previous_rows is not None:
        changes = measurement_changes(tonight, runs[1], rows, previous_rows)
        if changes:
            lines += [
                "**The measurement changed since the previous nightly; rows may move together.**",
                "",
            ]
            lines += [f"- `{what}`: `{before}` -> `{now}`" for what, before, now in changes]
            lines.append("")

    counts = {name: sum(1 for item in queue if item["label"] == name) for _, name in LABELS}
    lines += [
        "## Summary",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Ops behind | {len(queue)} of {len({r['op'] for r in rated})} |",
        f"| Rows behind | {len(tonight_behind)} of {len(rated)} |",
        "| By label | " + ", ".join(f"{name} {n}" for name, n in counts.items()) + " |",
        f"| Failed rows | {sum(1 for r in rows.values() if r['failed'])} |",
        "",
    ]
    if previous_rows is not None:
        now_ops, before_ops = _ops(rows, tonight_behind), _ops(previous_rows, behind[1])
        newly = sorted(now_ops - before_ops)
        caught_up = sorted(before_ops - now_ops)
        lines += [
            f"Since nightly run [{runs[1]['meta']['run_id']}]"
            f"({RUN_URL.format(runs[1]['meta']['run_id'])}) ({runs[1]['meta']['date']}):",
            "",
            "- Newly behind: " + (", ".join(f"`{op}`" for op in newly) or "none"),
            "- Caught up: " + (", ".join(f"`{op}`" for op in caught_up) or "none"),
            "",
        ]

    lines += [
        "## Queue",
        "",
        "Ordered by label, then by worst ratio. Nightlies behind counts the consecutive "
        f"nightlies, up to {len(runs)}, in which the op had a row behind.",
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
    lines.append("")

    lines += ["## Rows behind", ""]
    for item in queue:
        lines += [
            f"### {item['op']}",
            "",
            f"Benchmark `{item['rows'][0]['benchmark']}`.",
            "",
            "| Workload | TileOPs (us) | Fastest comparator | Comparator (us) | Ratio |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
        for r in item["rows"]:
            tileops_ms, fastest_ms = r["reading"]
            lines.append(
                f"| `{r['workload']}` | {_us(tileops_ms)} | {r['comparator']} | "
                f"{_us(fastest_ms)} | {fastest_ms / tileops_ms:.4f} |"
            )
        lines.append("")

    failed = sorted((r["op"], r["benchmark"], r["workload"]) for r in rows.values() if r["failed"])
    if failed:
        lines += ["## Failed rows", "", "| Op | Benchmark | Workload |", "| --- | --- | --- |"]
        lines += [f"| {op} | `{bench}` | `{wl}` |" for op, bench, wl in failed]
        lines.append("")
    return "\n".join(lines)


def _dates(wiki: Path) -> list[str]:
    """The dates of the daily pages, newest first."""
    return sorted(
        (p.stem[len(PAGE_PREFIX) + 1 :] for p in wiki.glob(f"{PAGE_PREFIX}-*.md")), reverse=True
    )


def render_index(wiki: Path) -> str:
    """The section's index: every daily page, newest first."""
    dates = _dates(wiki)
    lines = [
        "# Perf Tracker",
        "",
        "One page per nightly: the ops slower than their fastest comparator, written by the "
        "nightly workflow. Do not edit these pages; the next nightly overwrites them.",
        "",
    ]
    lines += [f"- [{date}]({PAGE_PREFIX}-{date})" for date in dates]
    return "\n".join(lines) + "\n"


def render_sidebar(wiki: Path) -> str:
    """``_Sidebar.md`` with its Perf Tracker block listing every daily page, newest first."""
    block = [SIDEBAR_BEGIN, f"**[Perf Tracker]({PAGE_PREFIX})**", ""]
    block += [f"- [{date}]({PAGE_PREFIX}-{date})" for date in _dates(wiki)]
    block.append(SIDEBAR_END)
    path = wiki / "_Sidebar.md"
    text = path.read_text() if path.exists() else "**[Home](Home)**\n"
    if SIDEBAR_BEGIN in text and SIDEBAR_END in text:
        head, rest = text.split(SIDEBAR_BEGIN, 1)
        tail = rest.split(SIDEBAR_END, 1)[1]
        return head + "\n".join(block) + tail
    return text.rstrip("\n") + "\n\n" + "\n".join(block) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, type=Path, help="TileOPs-nightly clone")
    parser.add_argument("--run-id", required=True, help="the nightly run that must be newest")
    parser.add_argument("--wiki", required=True, type=Path, help="wiki clone to write into")
    parser.add_argument("--depth", type=int, default=30, help="nightlies read for the streaks")
    args = parser.parse_args()

    runs = nightlies(args.snapshots, args.depth)
    if not runs or runs[0]["meta"]["run_id"] != args.run_id:
        newest = runs[0]["meta"]["run_id"] if runs else "none"
        print(f"newest snapshot is run {newest}, not run {args.run_id}", file=sys.stderr)
        return 1
    rows_by_run = [read_rows(args.snapshots, run["commit"]) for run in runs]
    date = runs[0]["meta"]["date"]
    (args.wiki / f"{PAGE_PREFIX}-{date}.md").write_text(render(runs, rows_by_run) + "\n")
    (args.wiki / f"{PAGE_PREFIX}.md").write_text(render_index(args.wiki))
    (args.wiki / "_Sidebar.md").write_text(render_sidebar(args.wiki))
    print(f"wrote {PAGE_PREFIX}-{date}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
