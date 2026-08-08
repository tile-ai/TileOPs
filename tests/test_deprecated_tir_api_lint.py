"""Unit tests for ``scripts/lint/deprecated_tir_api_lint.py``.

Each case runs the linter as a subprocess on a tmp file built from an inline
fixture string — never on the live repo. The deprecated spellings are
assembled by concatenation so this test file itself stays clean under the
hook that checks it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "deprecated_tir_api_lint.py"

# Concatenated so the deprecated spellings never appear verbatim here.
BUFFER = "T." + "Buffer"
REINTERPRET = "T." + "reinterpret"


def run_lint(tmp_path: Path, content: str) -> subprocess.CompletedProcess:
    target = tmp_path / "fixture.py"
    target.write_text(content)
    return subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)],
        capture_output=True,
        text=True,
    )


def test_deprecated_buffer_annotation_fails(tmp_path):
    content = f'def kernel(out: {BUFFER}((16,), "float32")):\n    pass\n'
    result = run_lint(tmp_path, content)
    assert result.returncode == 1
    assert "T.Tensor" in result.stdout


def test_tensor_annotation_passes(tmp_path):
    content = 'def kernel(out: T.Tensor((16,), "float32")):\n    pass\n'
    result = run_lint(tmp_path, content)
    assert result.returncode == 0, result.stdout


def test_dtype_first_reinterpret_fails(tmp_path):
    content = f'bits = {REINTERPRET}("float16", x)\n'
    result = run_lint(tmp_path, content)
    assert result.returncode == 1
    assert "T.reinterpret(value, dtype)" in result.stdout


def test_value_first_reinterpret_passes(tmp_path):
    content = (
        'bits = T.reinterpret(x, "float16")\n'
        'handle = T.reinterpret(T.uint64(0), dtype="handle")\n'
        "packed = T.reinterpret(hval, T.uint16)\n"
    )
    result = run_lint(tmp_path, content)
    assert result.returncode == 0, result.stdout


def test_every_violation_is_reported_with_its_line(tmp_path):
    content = f'a: {BUFFER}((8,), "float32")\nb = 1\nc = {REINTERPRET}("uint32", b)\n'
    result = run_lint(tmp_path, content)
    assert result.returncode == 1
    assert "fixture.py:1:" in result.stdout
    assert "fixture.py:3:" in result.stdout
