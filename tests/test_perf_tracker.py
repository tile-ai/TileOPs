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


def _xml(cases):
    """Each case is ``(op, workload, tileops_ms, {tag: ms}, extra)``, or a raw testcase
    string; ``extra`` holds more properties, and ``{"failed": True}`` marks it failed."""
    body = ""
    for case in cases:
        if isinstance(case, str):
            body += case
            continue
        op, workload, tileops_ms, comparators, extra = case
        extra = dict(extra)
        failed = extra.pop("failed", False)
        props = {"op": op, "op_module": f"tileops._ops.cat.{op.lower()}"}
        props["tileops_device_busy_ms"] = tileops_ms
        props.update({f"{tag}_device_busy_ms": ms for tag, ms in comparators.items()})
        props.update(extra)
        rendered = "".join(f'<property name="{k}" value="{v}"/>' for k, v in props.items())
        failure = "<failure/>" if failed else ""
        body += (
            f'<testcase classname="benchmarks.ops.bench_x" name="test_x[{workload}]">'
            f"<properties>{rendered}</properties>{failure}</testcase>"
        )
    return f"<testsuites><testsuite>{body}</testsuite></testsuites>"


def _snapshots(tmp_path, nights):
    """A snapshots repository with one commit per night, oldest first."""
    repo = tmp_path / "snapshots"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    for i, cases in enumerate(nights):
        meta = {"run_id": str(100 + i), "commit": f"{i:040d}", "date": f"2026-11-0{i + 1}"}
        (repo / "meta.json").write_text(json.dumps(meta))
        (repo / "bench_results.xml").write_text(_xml(cases))
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", f"night {i}"], check=True)
    return repo


def _run(repo, wiki, run_id, *more):
    wiki.mkdir(exist_ok=True)
    command = [sys.executable, str(SCRIPT), "--snapshots", str(repo), "--run-id", run_id]
    return subprocess.run([*command, "--wiki", str(wiki), *more], capture_output=True, text=True)


def test_page_orders_the_queue_counts_the_nights_and_sorts_out_the_ops_that_left(tmp_path):
    level = ("AOp", "a", 1.0, {"torch": 1.0}, {})
    big = ("AOp", "a", 1.0, {"torch": 0.85}, {})  # P2 on the last three nights
    small = ("BOp", "b", 0.005, {"torch": 0.0049}, {})  # under 10 us: behind from its 2nd night
    mild = ("COp", "c", 1.0, {"torch": 2.0, "vllm": 0.97}, {})  # P4 against the fastest
    recovering = [("DOp", "d", 1.0, {"torch": 0.9}, {}), ("DOp", "d", 1.0, {"torch": 1.0}, {})]
    failing = [
        ("EOp", "e", 1.0, {"torch": 0.9}, {}),
        ("EOp", "e", 1.0, {"torch": 0.9}, {"failed": True}),
    ]
    # What the benchmark runner records for a file whose process died: no properties.
    crash = (
        '<testcase classname="benchmarks.ops.bench_y" name="whole_file">'
        '<error message="bench_y.py: killed"/></testcase>'
    )
    nights = [
        [level],
        [big],
        [big, small, recovering[0], failing[0]],
        [big, small, mild, recovering[1], failing[1], crash],
    ]
    repo = _snapshots(tmp_path, nights)
    # Two judged nights: the third-newest serves only as a predecessor.
    assert (
        _run(repo, tmp_path / "wiki", "103", "--date", "2026-11-05", "--nights", "2").returncode
        == 0
    )
    page = (tmp_path / "wiki" / "Perf-Tracker-2026-11-05.md").read_text()
    assert page.startswith("# Perf Tracker 2026-11-05\n")
    queue = [line for line in page.splitlines() if line[:4] in ("| 1 ", "| 2 ", "| 3 ")]
    # By label, then by worst ratio: COp's 0.97 before BOp's 0.98.
    assert [line.split("[")[1].split("]")[0] for line in queue] == ["AOp", "COp", "BOp"]
    assert queue[0].endswith("| P2 | 0.8500 | 1 of 1 | 2 |")
    assert queue[2].endswith("| P4 | 0.9800 | 1 of 1 | 1 |")
    assert "| `c` | 1000.000 | vllm | 970.000 | 0.9700 |" in page
    assert "- Newly behind: `BOp`, `COp`" in page
    assert "- Caught up: `DOp`" in page
    assert "- Not measured tonight: `EOp`" in page
    assert "| EOp | `benchmarks.ops.bench_x` | `e` |  |" in page
    assert "| - | `benchmarks.ops.bench_y` | `whole_file` | bench_y.py: killed |" in page
    assert "changed since the previous nightly" not in page
    index = (tmp_path / "wiki" / "Perf-Tracker.md").read_text().splitlines()
    assert index[-1] == "- [2026-11-05](Perf-Tracker-2026-11-05)"


def test_page_flags_a_timing_method_that_moved_between_rows(tmp_path):
    # The set of methods is the same both nights; which row uses which is not.
    def night(a, b):
        return [
            ("AOp", "a", 1.0, {"torch": 1.0}, {"tileops_l2_flush": a}),
            ("AOp", "b", 1.0, {"torch": 1.0}, {"tileops_l2_flush": b}),
        ]

    repo = _snapshots(tmp_path, [night("read", "zero"), night("zero", "read")])
    assert _run(repo, tmp_path / "wiki", "101", "--date", "2026-11-02").returncode == 0
    page = (tmp_path / "wiki" / "Perf-Tracker-2026-11-02.md").read_text()
    assert "- `tileops_l2_flush, 1 rows`: `read` -> `zero`" in page
    assert "- `tileops_l2_flush, 1 rows`: `zero` -> `read`" in page


def test_nightlies_takes_one_complete_commit_per_run(tracker, tmp_path):
    repo = _snapshots(tmp_path, [[], [], []])
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    # A second commit for the newest run, then one without a benchmark XML.
    (repo / "badge.json").write_text("{}")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "badge"], check=True)
    badge = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    subprocess.run([*git, "rm", "-q", "bench_results.xml"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "incomplete"], check=True)
    runs = tracker.nightlies(repo, 2)
    assert [run["meta"]["run_id"] for run in runs] == ["102", "101"]
    assert runs[0]["commit"] == badge


def test_a_stale_snapshot_writes_nothing(tmp_path):
    repo = _snapshots(tmp_path, [[("AOp", "a", 1.0, {"torch": 0.5}, {})]])
    result = _run(repo, tmp_path / "wiki", "999", "--date", "2026-11-01")
    assert result.returncode == 1
    assert "not run 999" in result.stderr
    assert not list((tmp_path / "wiki").iterdir())


def test_sidebar_rewrites_only_its_own_block(tracker, tmp_path):
    for date in ("2026-11-01", "2026-11-02"):
        (tmp_path / f"Perf-Tracker-{date}.md").write_text("")
    first = tracker.render_sidebar(tmp_path)
    assert first.startswith("**[Home](Home)**")
    sidebar = tmp_path / "_Sidebar.md"
    sidebar.write_text("**[Guide](Guide)**\n\n" + first + "\n**[Notes](Notes)**\n")
    (tmp_path / "Perf-Tracker-2026-11-03.md").write_text("")
    again = tracker.render_sidebar(tmp_path)
    assert again.startswith("**[Guide](Guide)**")
    assert again.rstrip().endswith("**[Notes](Notes)**")
    links = [line for line in again.splitlines() if line.startswith("- [")]
    assert links == [
        f"- [{d}](Perf-Tracker-{d})" for d in ("2026-11-03", "2026-11-02", "2026-11-01")
    ]
    sidebar.write_text(f"{tracker.SIDEBAR_END}\n{tracker.SIDEBAR_BEGIN}\n")
    with pytest.raises(ValueError, match="needs one"):
        tracker.render_sidebar(tmp_path)


@pytest.fixture(scope="module")
def board():
    path = SCRIPT.with_name("perf_board.py")
    spec = importlib.util.spec_from_file_location("perf_board", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(op, worst, behind):
    rows = [{"op": op, "category": "cat"}] * behind
    return {"op": op, "label": "P2", "worst": worst, "rows": rows, "streak": 1}


def test_board_plan_follows_the_sync_rules(board):
    date = "2026-11-07"
    measured = {"Priority": "P2", "Worst ratio": 0.85, "Rows behind": 1, "Nightlies behind": 1}
    measured.update({"Category": "cat", "Measured": date})
    current = {
        "Back": {"Status": "Done"},
        "Held": {"Status": "Needs decision", **measured},
        "Fixed": {"Status": "PR open"},
        "Accepted": {"Status": "Accepted gap"},
        "Failing": {"Status": "In progress"},
        "Absent": {"Status": "Unclaimed"},
    }
    queue = [_entry(op, 0.850001, 1) for op in ("New", "Back", "Held")]
    rows = {
        (op, i): {"op": op, "reading": (1.0, 1.0), "failed": False}
        for i, op in enumerate(("New", "Back", "Held", "Fixed", "Accepted", "Failing"))
    }
    rows["fail"] = {"op": "Failing", "reading": None, "failed": True}
    changes = board.plan(current, queue, rows, date)
    assert changes == {
        "New": {**measured, "Status": "Unclaimed"},
        "Back": {**measured, "Status": "Unclaimed"},
        "Fixed": {"Status": "Done"},
    }
    for op, values in changes.items():
        current[op] = {**current.get(op, {}), **values}
    assert board.plan(current, queue, rows, date) == {}
