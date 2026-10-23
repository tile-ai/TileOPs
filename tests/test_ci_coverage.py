"""Collection failures and inherited compiler caches must not produce release evidence."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import check_cuda_graph_coverage as graph
from scripts.ci import check_packaging_coverage as packaging

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("gate", [packaging, graph], ids=["packaging", "cuda-graph"])
@pytest.mark.parametrize(
    "status", [pytest.ExitCode.INTERRUPTED, pytest.ExitCode.NO_TESTS_COLLECTED]
)
def test_collection_failure_overrides_complete_coverage(monkeypatch, gate, status):
    import tileops

    def collect(args, plugins):
        plugins[0].covered = dict.fromkeys(
            tileops._FAMILIES if gate is packaging else graph.QUALIFIED_OPS, ["test_evidence"]
        )
        return status

    monkeypatch.setattr(gate.pytest, "main", collect)
    assert gate.main() == status


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "missing"])
def test_graph_qualification_requires_collected_evidence(monkeypatch, missing):
    def collect(args, plugins):
        plugins[0].covered = {op: [f"test_{op}"] for op in graph.QUALIFIED_OPS}
        if missing:
            plugins[0].covered.pop(next(iter(graph.QUALIFIED_OPS)))
        return pytest.ExitCode.OK

    monkeypatch.setattr(graph.pytest, "main", collect)
    assert graph.main() == int(missing)


@pytest.mark.parametrize("fault", ["unknown-op", "no-sm90", "xfail"])
def test_graph_evidence_rejects_invalid_markers(fault):
    marks = {
        "cuda_graph": pytest.mark.cuda_graph(op="KDAFwdOp").mark,
        "sm90": pytest.mark.sm90.mark,
    }
    if fault == "unknown-op":
        marks["cuda_graph"] = pytest.mark.cuda_graph(op="UnknownOp").mark
    elif fault == "no-sm90":
        del marks["sm90"]
    else:
        marks["xfail"] = pytest.mark.xfail.mark
    item = SimpleNamespace(
        nodeid="test_graph",
        iter_markers=lambda name: [marks[name]] if name in marks else [],
        get_closest_marker=marks.get,
    )
    recorder = graph._Evidence()
    recorder.pytest_collection_finish(SimpleNamespace(items=[item]))
    assert recorder.errors
    assert not recorder.covered


def test_wheel_verification_starts_with_fresh_compiler_caches(tmp_path):
    """Repeated verification in one work directory cannot reuse the runner's cached kernels."""
    script = Path(__file__).resolve().parents[1] / "scripts/ci/verify_installed_wheel.sh"
    wheel = tmp_path / "tileops.whl"
    wheel.touch()
    probe = tmp_path / "python-probe"
    probe.write_text('#!/bin/sh\nenv > "$CACHE_PROBE"\nexit 17\n')
    probe.chmod(0o755)
    env_file = tmp_path / "env"
    work = tmp_path / "work"
    cache_vars = (
        "TILELANG_CACHE_DIR",
        "TILELANG_TMP_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
    )
    inherited = tmp_path / "runner-cache"
    inherited.mkdir()
    sentinel = inherited / "compiled-kernel"
    sentinel.write_text("existing cache")
    env = dict(os.environ, CACHE_PROBE=str(env_file), **dict.fromkeys(cache_vars, str(inherited)))
    previous = set()
    for _ in range(2):
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--wheel",
                str(wheel),
                "--deps",
                "image",
                "--python",
                str(probe),
                "--work-dir",
                str(work),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 17, result.stdout + result.stderr
        actual = dict(
            line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
        )
        paths = {Path(actual[name]) for name in cache_vars}
        assert len(paths) == len(cache_vars)
        assert paths.isdisjoint(previous)
        for path in paths:
            assert path.is_relative_to(work)
            assert not path.exists()
            path.mkdir(parents=True)
            (path / "compiled-kernel").touch()
        previous = paths
    assert sentinel.read_text() == "existing cache"
