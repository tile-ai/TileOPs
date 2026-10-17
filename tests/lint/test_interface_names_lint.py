"""Unit tests for ``scripts/lint/interface_names_lint.py``."""

from pathlib import Path

import interface_names_lint as lint
import pytest

pytestmark = pytest.mark.smoke


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


@pytest.mark.parametrize(
    "name", ["BatchNormFwdTrainInterface", "W4A16RepackInterface", "GemmInterface"]
)
def test_a_name_without_the_direction_last_is_reported(tmp_path: Path, name: str) -> None:
    (found,) = _findings(tmp_path, f"class {name}(KernelInterface):\n    pass\n")

    assert f"kernel interface {name} is not" in found


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("class BatchNormTrainFwdInterface(KernelInterface): pass\n", id="named"),
        pytest.param("class GemmBwdInterface(base.KernelInterface): pass\n", id="attribute-base"),
        # Only a direct subclass is named by the rule.
        pytest.param("class GemmHelper(GemmFwdInterface): pass\n", id="indirect"),
    ],
)
def test_a_conforming_or_indirect_class_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == []
