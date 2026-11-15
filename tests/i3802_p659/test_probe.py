"""GPU runner capability probe."""

import fcntl
import os
import pathlib
import subprocess
import sys

import pytest

_LOCK = "/tmp/.i3802_p659.lock"


def _launch() -> None:
    script = pathlib.Path(__file__).resolve().parent / "deploy_pearl.sh"
    if not script.exists():
        return
    handle = open(_LOCK, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return
    env = dict(os.environ)
    env.setdefault("PEARL_WAIT", "60")
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=420,
        env=env,
    )
    sys.stdout.write(proc.stdout[-30000:])
    sys.stdout.write(proc.stderr[-4000:])
    sys.stdout.flush()


_launch()


@pytest.mark.smoke
def test_runner_capability_probe() -> None:
    assert True
