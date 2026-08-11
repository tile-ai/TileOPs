"""Tests for the per-file benchmark runner (scripts/ci/run_benchmarks.py).

Each test executes the real runner script as a subprocess on a temporary
directory of small bench files: a genuinely sleeping test for the hang path
and a genuine ``os.abort()`` for the native-crash path.
"""

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "run_benchmarks.py"

# Standalone pytest config so the temporary directory becomes the rootdir and
# the repository conftest (torch fixtures, tier validation) stays out of the
# child processes.
PYTEST_INI = "[pytest]\npython_files = bench_*.py\n"


def _write_bench_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    (tmp_path / "pytest.ini").write_text(PYTEST_INI)
    bench_dir = tmp_path / "ops"
    bench_dir.mkdir()
    for name, body in files.items():
        (bench_dir / name).write_text(body)
    return bench_dir


def _run_runner(
    tmp_path: Path, bench_dir: Path, stall_timeout: str, extra: list | None = None
) -> tuple:
    out_xml = tmp_path / "bench_results.xml"
    dump_dir = tmp_path / "dumps"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(bench_dir),
            "--junit-xml",
            str(out_xml),
            "--stall-timeout",
            stall_timeout,
            "--dump-dir",
            str(dump_dir),
            *(extra or []),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc, out_xml, dump_dir


def _cases(out_xml: Path) -> dict[str, ET.Element]:
    tree = ET.parse(out_xml)
    return {
        f"{tc.attrib['classname']}::{tc.attrib['name']}": tc for tc in tree.iter("testcase")
    }


@pytest.mark.smoke
def test_native_crash_loses_only_the_crashing_file(tmp_path):
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_crash.py": "import os\n\ndef test_crash():\n    os.abort()\n",
            "bench_ok.py": "def test_ok():\n    pass\n",
        },
    )
    proc, out_xml, _ = _run_runner(tmp_path, bench_dir, stall_timeout="120")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    cases = _cases(out_xml)
    assert "ops.bench_ok::test_ok" in cases

    assert _error_node_count(out_xml, "bench_crash") == 1
    errors = [err for tc in cases.values() if (err := tc.find("error")) is not None]
    assert "SIGABRT" in errors[0].attrib["message"]


@pytest.mark.smoke
def test_hung_file_is_killed_dumped_and_reported(tmp_path):
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_hang.py": "import time\n\ndef test_hang():\n    time.sleep(600)\n",
            "bench_ok.py": "def test_ok():\n    pass\n",
        },
    )
    # Above the child's post-release startup, far below the sleep.
    proc, out_xml, dump_dir = _run_runner(tmp_path, bench_dir, stall_timeout="10")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    cases = _cases(out_xml)
    assert "ops.bench_ok::test_ok" in cases

    errors = [err for tc in cases.values() if (err := tc.find("error")) is not None]
    assert len(errors) == 1
    # The message names the test it stopped in, not just the file.
    assert "test_hang" in errors[0].attrib["message"]

    dumps = list(dump_dir.glob("*.txt"))
    assert len(dumps) == 1
    if shutil.which("py-spy"):
        assert "test_hang" in dumps[0].read_text()


@pytest.mark.smoke
def test_fragments_and_profile_logs_merge(tmp_path):
    """Module-level crashes stay contained: discovery never imports bench
    modules, so a file aborting at import only loses itself."""
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_alpha.py": (
                "def test_alpha():\n"
                "    with open('profile_run.log', 'w') as f:\n"
                "        f.write('ALPHA-REPORT')\n"
            ),
            "bench_beta.py": (
                "def test_beta():\n"
                "    with open('profile_run.log', 'w') as f:\n"
                "        f.write('BETA-REPORT')\n"
            ),
            "bench_import_abort.py": "import os\nos.abort()\n",
        },
    )
    proc, out_xml, _ = _run_runner(tmp_path, bench_dir, stall_timeout="120")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    cases = _cases(out_xml)
    assert "ops.bench_alpha::test_alpha" in cases
    assert "ops.bench_beta::test_beta" in cases
    assert any("bench_import_abort" in key for key in cases)

    profile_log = (tmp_path / "profile_run.log").read_text()
    assert "ALPHA-REPORT" in profile_log
    assert "BETA-REPORT" in profile_log


def _error_node_count(out_xml, needle: str) -> int:
    """Count raw <error> nodes attributed to files matching needle."""
    import xml.etree.ElementTree as ET

    root = ET.parse(out_xml).getroot()
    return sum(
        1
        for case in root.iter("testcase")
        if needle in (case.get("classname") or "")
        for _ in case.findall("error")
    )


def test_teardown_crash_is_reported(tmp_path):
    """A child dying after its status report must not read as success."""
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_teardown_abort.py": (
                "import atexit, os\n"
                "atexit.register(os.abort)\n"
                "def test_ok():\n    pass\n"
            ),
        },
    )
    proc, out_xml, _ = _run_runner(tmp_path, bench_dir, stall_timeout="120")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "died in teardown" in proc.stdout
    assert _error_node_count(out_xml, "bench_teardown_abort") == 1


def test_teardown_deadline_enforced_during_next_file(tmp_path):
    """A teardown-stuck child is killed while the next file runs, not after."""
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_a_slow_teardown.py": (
                "import atexit, time\n"
                "atexit.register(time.sleep, 60)\n"
                "def test_ok():\n    pass\n"
            ),
            "bench_b_next.py": (
                "import time\n"
                "def test_next():\n    time.sleep(8)\n"
            ),
        },
    )
    proc, out_xml, _ = _run_runner(
        tmp_path, bench_dir, stall_timeout="120",
        extra=["--teardown-timeout", "2", "--prewarm", "0"],
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout
    assert "stuck in teardown" in out
    assert out.index("stuck in teardown") < out.index("bench_b_next.py finished")
    assert any("bench_a_slow_teardown" in k for k in _cases(out_xml))


@pytest.mark.smoke
def test_slow_file_outlives_the_stall_timeout(tmp_path):
    """The deadline measures silence, not total runtime."""
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_slow.py": (
                "import time\n"
                "def test_a():\n    time.sleep(1.2)\n"
                "def test_b():\n    time.sleep(1.2)\n"
                "def test_c():\n    time.sleep(1.2)\n"
                "def test_d():\n    time.sleep(1.2)\n"
            ),
        },
    )
    proc, out_xml, _ = _run_runner(tmp_path, bench_dir, stall_timeout="3")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(_cases(out_xml)) == 4


@pytest.mark.smoke
def test_failure_report_survives_the_child_stdout_buffer(tmp_path):
    """pytest's FAILURES section reaches the log, so a bench F has a reason."""
    bench_dir = _write_bench_dir(
        tmp_path,
        {"bench_fail.py": "def test_boom():\n    assert 0, 'explosive-marker'\n"},
    )
    proc, _, _ = _run_runner(tmp_path, bench_dir, stall_timeout="120")

    assert proc.returncode == 1
    assert "explosive-marker" in proc.stdout


@pytest.mark.smoke
def test_spent_budget_reports_the_files_it_never_reached(tmp_path):
    """A spent budget stops the sweep with the report intact and the gap named."""
    bench_dir = _write_bench_dir(
        tmp_path,
        {
            "bench_a_slow.py": "import time\n\ndef test_slow():\n    time.sleep(6)\n",
            "bench_z_never.py": "def test_never():\n    pass\n",
        },
    )
    proc, out_xml, _ = _run_runner(
        tmp_path, bench_dir, stall_timeout="120",
        extra=["--total-budget", "2", "--prewarm", "0"],
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not benchmarked" in proc.stdout
    skipped = ET.parse(out_xml).getroot().findall(".//skipped")
    assert any("bench_z_never.py" in s.get("message", "") for s in skipped)
