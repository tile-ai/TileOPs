"""Tests for scripts/gpu_smoke_report.py."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gpu_smoke_report.py"


@pytest.fixture(scope="module")
def smoke_report():
    spec = importlib.util.spec_from_file_location("gpu_smoke_report", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_teardown_error_after_a_skip_is_a_failure(smoke_report, tmp_path):
    """JUnit writes both elements for such a case; reading it as skipped hides the error."""
    xml = tmp_path / "smoke.xml"
    xml.write_text(
        '<testsuite><testcase classname="tests.ops.test_x" name="test_a">'
        '<skipped message="no sm90"/><error message="teardown leaked"/>'
        "</testcase></testsuite>"
    )
    (result,) = smoke_report.parse_test_xml(str(xml))
    assert result["outcome"] == "failed"
    assert "| - | test_a | teardown leaked |" in smoke_report.generate_report([result], "smoke")
