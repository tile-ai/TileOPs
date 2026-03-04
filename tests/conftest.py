import os
from collections import defaultdict

import pytest
import torch


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def _is_unit_profile() -> bool:
    return os.getenv("CI_TEST_PROFILE", "").strip().lower() == "unit"


def _param_key(name, value) -> tuple[str, str]:
    return (name, repr(value))


def _group_id(item: pytest.Item) -> str:
    original_name = getattr(item, "originalname", item.name.split("[", 1)[0])
    return f"{item.path}::{original_name}"


def _build_item_coverage(item: pytest.Item) -> set[tuple[str, str]]:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return set()
    return {_param_key(name, value) for name, value in callspec.params.items()}


def _select_min_cover(items: list[pytest.Item]) -> set[int]:
    if not items:
        return set()

    coverages = [_build_item_coverage(item) for item in items]
    universe = set().union(*coverages) if coverages else set()
    if not universe:
        return set(range(len(items)))

    selected: list[int] = []
    uncovered = set(universe)
    remaining = set(range(len(items)))

    # Greedy stage: pick the case that covers the most uncovered parameter values.
    while uncovered and remaining:
        best_idx = min(
            remaining,
            key=lambda i: (
                -len(coverages[i] & uncovered),
                len(coverages[i]),
                items[i].nodeid,
            ),
        )
        gain = coverages[best_idx] & uncovered
        if not gain:
            break
        selected.append(best_idx)
        uncovered -= gain
        remaining.remove(best_idx)

    # Fallback for any uncovered residue (should be rare): ensure full coverage.
    if uncovered:
        for idx in sorted(remaining, key=lambda i: items[i].nodeid):
            if not uncovered:
                break
            gain = coverages[idx] & uncovered
            if gain:
                selected.append(idx)
                uncovered -= gain

    selected_set = set(selected)

    # Prune stage: drop redundant cases while preserving coverage.
    for idx in list(selected):
        trial = selected_set - {idx}
        trial_cover = set()
        for picked in trial:
            trial_cover.update(coverages[picked])
        if trial_cover >= universe:
            selected_set.remove(idx)

    return selected_set


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _is_unit_profile():
        return

    grouped_param_items: dict[str, list[pytest.Item]] = defaultdict(list)
    non_param_items: set[str] = set()

    for item in items:
        if getattr(item, "callspec", None) is None:
            non_param_items.add(item.nodeid)
            continue
        grouped_param_items[_group_id(item)].append(item)

    keep_ids = set(non_param_items)
    total_param = 0
    kept_param = 0

    for group_items in grouped_param_items.values():
        total_param += len(group_items)
        keep_indices = _select_min_cover(group_items)
        kept_param += len(keep_indices)
        for idx in keep_indices:
            keep_ids.add(group_items[idx].nodeid)

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
        "kept_param": kept_param,
        "total_param": total_param,
    }


def pytest_report_header(config: pytest.Config) -> str | None:
    if not _is_unit_profile():
        return None

    stats = getattr(config, "_unit_selection_stats", None)
    if not stats:
        return "CI_TEST_PROFILE=unit active: selecting minimal parameterized subset with full value coverage."

    return (
        "CI_TEST_PROFILE=unit active: "
        f"selected {stats['kept']}/{stats['total']} tests "
        f"(parameterized: {stats['kept_param']}/{stats['total_param']})."
    )
