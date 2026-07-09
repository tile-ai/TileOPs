import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_ops_import_does_not_eagerly_load_trace_passes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo_root)
        if not env.get("PYTHONPATH")
        else f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = r"""
import builtins

original_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name == "tileops.trace.passes":
        raise RuntimeError("unexpected eager import of tileops.trace.passes")
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
import tileops.ops  # noqa: F401

import sys

assert "tileops.trace.passes" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=repo_root, env=env)
