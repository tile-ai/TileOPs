"""The scheduler a parallel run takes, which files asserting over process state depend on."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE_PLUGIN = """\
import json
import os


def pytest_configure(config):
    if hasattr(config, "workerinput"):
        return
    with open(os.environ["SCHEDULER_PROBE"], "w") as handle:
        json.dump(config.getoption("dist"), handle)
"""


def test_the_predicate_takes_only_a_parallel_run_that_names_no_scheduler():
    """``-n`` alone is the one case the default applies to."""
    from conftest import _wants_loadfile

    settled = dict(has_xdist=True, numprocesses=4, dist="no", distload=False)
    assert _wants_loadfile(**settled)
    for name, value in (
        ("has_xdist", False),  # a pytest-only environment
        ("numprocesses", 0),  # -n 0, and no -n at all
        ("dist", "load"),  # --dist load, --dist=load, or a config that sets it
        ("distload", True),  # -d
    ):
        assert not _wants_loadfile(**{**settled, name: value}), name


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["-n", "2"], "loadfile", id="parallel"),
        pytest.param(["-n", "2", "-d"], "load", id="dash-d"),
        pytest.param(["-n", "2", "--dist", "load"], "load", id="named-scheduler"),
    ],
)
def test_a_run_takes_the_scheduler_the_caller_asked_for(tmp_path, argv, expected):
    """What the run settles on, which the predicate alone cannot show.

    An ordinary hook implementation, or one that assigns after yielding, reads the
    value xdist already derived and leaves every case below as ``load``.
    """
    pytest.importorskip("xdist")
    shutil.copy(_REPO_ROOT / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "probe_plugin.py").write_text(_PROBE_PLUGIN)
    (tmp_path / "test_one.py").write_text("def test_one():\n    pass\n")
    probe = tmp_path / "dist.json"

    environment = dict(os.environ, SCHEDULER_PROBE=str(probe))
    # The copied conftest imports `workloads`, which lives beside it in the repository.
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(_REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "probe_plugin", *argv, "test_one.py"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(probe.read_text()) == expected
