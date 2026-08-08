"""Unit tests for ``scripts/lint/deprecated_tir_api_lint.py``.

Each case runs the linter as a subprocess on a tmp file built from an inline
fixture string — never on the live repo. The deprecated spellings appear
verbatim here because the linter skips string literals, which is itself one of
the behaviours under test.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "deprecated_tir_api_lint.py"


def run_lint(tmp_path: Path, content: str) -> subprocess.CompletedProcess:
    target = tmp_path / "fixture.py"
    target.write_text(content)
    return subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)],
        capture_output=True,
        text=True,
    )


def test_deprecated_buffer_annotation_fails(tmp_path):
    result = run_lint(tmp_path, 'def kernel(out: T.Buffer((16,), "float32")):\n    pass\n')
    assert result.returncode == 1
    assert "T.Tensor" in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        'x: T . Buffer((8,), "float32") = None\n',
        'x: T\t.\tBuffer((8,), "float32") = None\n',
        'x: (T\n .\n Buffer((8,), "float32")) = None\n',
    ],
    ids=["spaces", "tabs", "newlines"],
)
def test_layout_does_not_hide_a_violation(tmp_path, source):
    """All of these parse to the same attribute access, so all must be flagged."""
    result = run_lint(tmp_path, source)
    assert result.returncode == 1


def test_space_before_call_paren_fails(tmp_path):
    result = run_lint(tmp_path, 'bits = T.reinterpret ("float16", x)\n')
    assert result.returncode == 1


def test_tensor_annotation_passes(tmp_path):
    result = run_lint(tmp_path, 'def kernel(out: T.Tensor((16,), "float32")):\n    pass\n')
    assert result.returncode == 0, result.stdout


def test_dtype_first_reinterpret_fails(tmp_path):
    result = run_lint(tmp_path, 'bits = T.reinterpret("float16", x)\n')
    assert result.returncode == 1
    assert "T.reinterpret(value, dtype)" in result.stdout


def test_value_first_reinterpret_passes(tmp_path):
    content = (
        'bits = T.reinterpret(x, "float16")\n'
        'handle = T.reinterpret(T.uint64(0), dtype="handle")\n'
        "packed = T.reinterpret(hval, T.uint16)\n"
        # A value that is merely string-producing is still a value.
        'joined = T.reinterpret("abc" + str(x), "float16")\n'
    )
    result = run_lint(tmp_path, content)
    assert result.returncode == 0, result.stdout


def test_prose_mentioning_a_deprecated_form_passes(tmp_path):
    """Naming a deprecated spelling in a comment or string is not usage."""
    content = (
        '"""Prefer T.Tensor over T.Buffer; pass the value first, not '
        'T.reinterpret("f16", x)."""\n'
        "# never write T.Buffer here\n"
        'NOTE = "T.Buffer is gone"\n'
        "my_T_Buffer = 1\n"
    )
    result = run_lint(tmp_path, content)
    assert result.returncode == 0, result.stdout


def test_fstring_literal_text_passes(tmp_path):
    """Python 3.12 tokenizes f-string text apart from ordinary strings."""
    content = 'msg = f"never use T.Buffer here"\nalt = f"T.reinterpret(\'f16\', x) is gone {name}"\n'
    result = run_lint(tmp_path, content)
    assert result.returncode == 0, result.stdout


def test_fstring_replacement_field_is_code(tmp_path):
    """Text inside ``{}`` is evaluated, so a deprecated name there is usage."""
    result = run_lint(tmp_path, 'label = f"{T.Buffer}"\n')
    assert result.returncode == 1


def test_dtype_first_reinterpret_with_string_prefix_fails(tmp_path):
    result = run_lint(tmp_path, 'bits = T.reinterpret(f"{dt}", q)\n')
    assert result.returncode == 1


def test_untokenizable_file_still_scanned(tmp_path):
    """A syntax error must not let a violation through unchecked."""
    result = run_lint(tmp_path, 'def broken(:\n    x: T.Buffer((4,), "f32")\n')
    assert result.returncode == 1


def test_every_violation_is_reported_with_its_line(tmp_path):
    content = 'a: T.Buffer((8,), "float32")\nb = 1\nc = T.reinterpret("uint32", b)\n'
    result = run_lint(tmp_path, content)
    assert result.returncode == 1
    assert "fixture.py:1:" in result.stdout
    assert "fixture.py:3:" in result.stdout
