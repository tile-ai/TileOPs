"""Tests for scripts/perf_tracker.py: the wiki's daily Perf Tracker page."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "perf_tracker.py"


@pytest.fixture(scope="module")
def tracker():
    spec = importlib.util.spec_from_file_location("perf_tracker", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _xml(cases, flush="read"):
    """Each case is ``(op, workload, tileops_ms, {tag: ms})``."""
    body = ""
    for op, workload, tileops_ms, comparators in cases:
        props = {
            "op": op,
            "op_module": f"tileops._ops.cat.{op.lower()}",
            "tileops_device_busy_ms": tileops_ms,
            "tileops_l2_flush": flush,
        }
        props.update({f"{tag}_device_busy_ms": ms for tag, ms in comparators.items()})
        rendered = "".join(f'<property name="{k}" value="{v}"/>' for k, v in props.items())
        body += (
            f'<testcase classname="benchmarks.ops.bench_x" name="test_x[{workload}]">'
            f"<properties>{rendered}</properties></testcase>"
        )
    return f"<testsuites><testsuite>{body}</testsuite></testsuites>"


def _snapshots(tmp_path, nights):
    """A snapshots repository with one commit per night, oldest first."""
    repo = tmp_path / "snapshots"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    for i, (cases, flush) in enumerate(nights):
        meta = {
            "run_id": str(100 + i),
            "commit": f"{i:040d}",
            "date": f"2026-11-0{i + 1}",
            "gpu": "NVIDIA H200",
            "environment": {"driver": "1"},
        }
        (repo / "meta.json").write_text(json.dumps(meta))
        (repo / "bench_results.xml").write_text(_xml(cases, flush))
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", f"night {i}"], check=True)
    return repo


def _run(repo, wiki, run_id):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshots",
            str(repo),
            "--run-id",
            run_id,
            "--wiki",
            str(wiki),
        ],
        capture_output=True,
        text=True,
    )


def test_page_orders_the_queue_and_counts_the_nights_behind(tmp_path):
    level = ("AOp", "a", 1.0, {"torch": 1.0})
    big_behind = ("AOp", "a", 1.0, {"torch": 0.85})  # P2 on the last two nights
    small = ("BOp", "b", 0.005, {"torch": 0.0049})  # under 10 us: behind from its second night
    mild = ("COp", "c", 1.0, {"torch": 2.0, "vllm": 0.97})  # P4 against the fastest comparator
    nights = [[level], [big_behind, small], [big_behind, small, mild]]
    repo = _snapshots(tmp_path, [(cases, "read") for cases in nights])
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    assert _run(repo, wiki, "102").returncode == 0
    page = (wiki / "Perf-Tracker-2026-11-03.md").read_text()
    queue = [line for line in page.splitlines() if line[:4] in ("| 1 ", "| 2 ", "| 3 ")]
    # By label, then by worst ratio: COp's 0.97 before BOp's 0.98.
    assert [line.split("[")[1].split("]")[0] for line in queue] == ["AOp", "COp", "BOp"]
    assert queue[0].endswith("| P2 | 0.8500 | 1 of 1 | 2 |")
    assert queue[2].endswith("| P4 | 0.9800 | 1 of 1 | 1 |")
    assert "| `c` | 1000.000 | vllm | 970.000 | 0.9700 |" in page
    assert "- Newly behind: `BOp`, `COp`" in page
    assert "The measurement changed" not in page
    index = (wiki / "Perf-Tracker.md").read_text().splitlines()
    assert index[-1] == "- [2026-11-03](Perf-Tracker-2026-11-03)"


def test_page_flags_a_change_in_how_rows_are_timed(tmp_path):
    row = ("AOp", "a", 1.0, {"torch": 1.0})
    repo = _snapshots(tmp_path, [([row], "zero"), ([row], "read")])
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    assert _run(repo, wiki, "101").returncode == 0
    page = (wiki / "Perf-Tracker-2026-11-02.md").read_text()
    assert "- `l2_flush`: `zero` -> `read`" in page


def test_a_stale_snapshot_writes_nothing(tmp_path):
    repo = _snapshots(tmp_path, [([("AOp", "a", 1.0, {"torch": 0.5})], "read")])
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    result = _run(repo, wiki, "999")
    assert result.returncode == 1
    assert "not run 999" in result.stderr
    assert not list(wiki.iterdir())


def test_index_lists_every_day_newest_first(tracker, tmp_path):
    for date in ("2026-11-01", "2026-11-03", "2026-11-02"):
        (tmp_path / f"Perf-Tracker-{date}.md").write_text("")
    links = [line for line in tracker.render_index(tmp_path).splitlines() if line.startswith("- ")]
    assert links == [
        f"- [{d}](Perf-Tracker-{d})" for d in ("2026-11-03", "2026-11-02", "2026-11-01")
    ]
