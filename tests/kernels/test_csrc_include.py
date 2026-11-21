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


@pytest.mark.smoke
def test_unrelated_header_edit_keeps_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A header the kernel does not include must not change its flags, or every nightly
    recompiles and re-autotunes kernels whose source did not change."""
    (tmp_path / "helper.h").write_text("constexpr int kHelper = 1;\n")
    other = tmp_path / "other.h"
    other.write_text("constexpr int kOther = 1;\n")
    monkeypatch.setattr(_csrc, "_CSRC", tmp_path)
    before = _csrc.csrc_include("helper.h")

    other.write_text("constexpr int kOther = 2;\n")

    assert _csrc.csrc_include("helper.h") == before


@pytest.mark.smoke
def test_transitively_included_header_edit_changes_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header reached through a quoted include is part of the kernel's source."""
    (tmp_path / "helper.h").write_text('#pragma once\n#include "base.h"\n')
    base = tmp_path / "base.h"
    base.write_text("constexpr int kBase = 1;\n")
    monkeypatch.setattr(_csrc, "_CSRC", tmp_path)
    before = _csrc.csrc_include("helper.h")

    base.write_text("constexpr int kBase = 2;\n")
    after = _csrc.csrc_include("helper.h")

    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]


@pytest.mark.smoke
def test_include_cycle_terminates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two headers that include each other are each hashed once."""
    (tmp_path / "a.h").write_text('#pragma once\n#include "b.h"\n')
    (tmp_path / "b.h").write_text('#pragma once\n#include "a.h"\n')
    monkeypatch.setattr(_csrc, "_CSRC", tmp_path)

    assert _csrc.csrc_include("a.h")[-1] == _csrc.csrc_include("a.h")[-1]
