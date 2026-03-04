import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def _is_unit_profile() -> bool:
    return os.getenv("CI_TEST_PROFILE", "").strip().lower() == "unit"


def _unit_manifest_path() -> Path:
    return Path(__file__).with_name("unit_nodeids.txt")


def _load_unit_nodeids() -> set[str]:
    path = _unit_manifest_path()
    if not path.exists():
        raise pytest.UsageError(f"CI_TEST_PROFILE=unit requires manifest file: {path}")

    nodeids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        nodeids.add(line)
    return nodeids


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _is_unit_profile():
        return

    keep_ids = _load_unit_nodeids()

    selected_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    for item in items:
        if item.nodeid in keep_ids:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items

    config._unit_selection_stats = {
        "kept": len(selected_items),
        "total": len(selected_items) + len(deselected_items),
        "manifest_size": len(keep_ids),
    }


def pytest_report_header(config: pytest.Config) -> str | None:
    if not _is_unit_profile():
        return None

    stats = getattr(config, "_unit_selection_stats", None)
    if not stats:
        return "CI_TEST_PROFILE=unit active: selecting tests from static unit manifest."

    return (
        "CI_TEST_PROFILE=unit active: "
        f"selected {stats['kept']}/{stats['total']} tests "
        f"(manifest entries: {stats['manifest_size']})."
    )
