from collections import defaultdict

import pytest
import torch


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate explicit test tier assignments."""
    tier_errors: list[str] = []
    tier_names = ("smoke", "full", "nightly")

    for item in items:
        path = str(item.path)
        if "tests/" not in path:
            continue
        tiers = [name for name in tier_names if item.get_closest_marker(name) is not None]
        if len(tiers) != 1:
            tier_errors.append(
                f"{item.nodeid}: expected exactly one tier marker, found {tiers or 'none'}"
            )

    ops_groups: dict[tuple[str, str], list[pytest.Item]] = defaultdict(list)
    for item in items:
        path = str(item.path)
        if "tests/ops/" not in path:
            continue
        test_name = getattr(item, "originalname", item.name)
        ops_groups[(path, test_name)].append(item)

    for (_path, _test_name), group in ops_groups.items():
        non_xfail_items = [
            item for item in group if item.get_closest_marker("xfail") is None
        ]
        if non_xfail_items:
            smoke_items = [
                item for item in group if item.get_closest_marker("smoke") is not None
            ]
            expected_smoke = non_xfail_items[0]
            if len(smoke_items) != 1 or smoke_items[0] is not expected_smoke:
                tier_errors.append(
                    f"{expected_smoke.nodeid}: smoke must be the first non-xfail case of each test"
                )

        first_tuned_item: pytest.Item | None = None
        full_tuned_items: list[pytest.Item] = []
        for item in group:
            callspec = getattr(item, "callspec", None)
            if callspec is None or "tune" not in callspec.params:
                continue

            tune = callspec.params["tune"]
            is_smoke = item.get_closest_marker("smoke") is not None
            if is_smoke and tune is True:
                tier_errors.append(f"{item.nodeid}: smoke cases must use tune=False")
            if is_smoke and item.get_closest_marker("xfail") is not None:
                tier_errors.append(f"{item.nodeid}: smoke cases must not be xfail")
            if tune is True:
                if first_tuned_item is None:
                    first_tuned_item = item
                if item.get_closest_marker("full") is not None:
                    full_tuned_items.append(item)
        if first_tuned_item is not None:
            if not full_tuned_items:
                tier_errors.append(
                    f"{first_tuned_item.nodeid}: the first tune=True case must be marked full"
                )
            elif len(full_tuned_items) > 1:
                tier_errors.append(
                    f"{group[0].path}::{group[0].originalname}: at most one tune=True case may be full"
                )
            elif full_tuned_items[0] is not first_tuned_item:
                tier_errors.append(
                    f"{first_tuned_item.nodeid}: the first tune=True case must be the only full tuned case"
                )

    if tier_errors:
        raise pytest.UsageError(
            "Invalid explicit test tier assignments detected:\n" + "\n".join(tier_errors)
        )
