"""What captions a published benchmark snapshot.

Nothing else checks it: the nightly runs once a night on a self-hosted GPU, and
a mistake here does not fail, it publishes the wrong facts. One end-to-end pass
— read the machine, write the caption, read it back — over synthetic input.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))

import collect_env  # noqa: E402

PUBLISH = REPO_ROOT / "scripts" / "ci" / "publish_meta.py"


def test_the_caption_carries_what_the_run_cannot_be_reproduced_without(tmp_path, monkeypatch):
    # An unset application clock reports `[N/A]`, which must not reach the page.
    monkeypatch.setenv("TILEOPS_RUNNER_IMAGE", "ghcr.io/tile-ai/tileops-runner:cu132")
    monkeypatch.setenv("TILEOPS_RUNNER_IMAGE_DIGEST", "sha256:" + "ab" * 32)
    monkeypatch.setattr(
        collect_env.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, stdout="595.71.05, 700.00, [N/A], 2619, Disabled\n", stderr=""
        ),
    )
    env = tmp_path / "env.json"
    env.write_text(json.dumps(collect_env.collect()))
    # `collect_env.subprocess` is the module: the patch would reach the
    # publisher below too.
    monkeypatch.undo()

    out = tmp_path / "meta.json"
    subprocess.run(
        [
            sys.executable,
            str(PUBLISH),
            "--env-json",
            str(env),
            "--commit",
            "abcdef1",
            "--run-id",
            "42",
            "--date",
            "2026-01-01",
            "--out",
            str(out),
        ],
        check=True,
    )

    environment = json.loads(out.read_text())["environment"]
    assert environment["image_digest"] == "sha256:" + "ab" * 32
    assert environment["power_limit_w"] == 700.0
    assert environment["memory_clock_mhz"] == 2619.0
    assert "sm_clock_mhz" not in environment
