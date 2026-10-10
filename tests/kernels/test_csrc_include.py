"""Unit tests for ``tileops._csrc.csrc_include``."""

from pathlib import Path

import pytest

from tileops import _csrc


@pytest.mark.smoke
def test_header_edit_changes_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: TileLang keys its caches on the flags, so an edited header must change them."""
    header = tmp_path / "helper.h"
    header.write_text("constexpr int kHelper = 1;\n")
    monkeypatch.setattr(_csrc, "_CSRC", tmp_path)
    before = _csrc.csrc_include("helper.h")

    header.write_text("constexpr int kHelper = 2;\n")
    after = _csrc.csrc_include("helper.h")

    assert str(header) in before
    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]
