"""The transform family's packaging contract.

The family carries `spec-only` entries only, so what an installed wheel has to prove is
that its module ships and that its entries name it. `scripts/ci/check_packaging_coverage.py`
requires one `packaging` case per family in `tileops._FAMILIES`.
"""

import pytest

from tileops.manifest import load_manifest


@pytest.mark.smoke
@pytest.mark.packaging(family="transform")
def test_the_transform_module_ships_and_exports_what_the_manifest_implements() -> None:
    import tileops.transform

    exported = set(tileops.transform.__all__)
    entries = {
        name: entry["status"]
        for name, entry in load_manifest().items()
        if entry["family"] == "transform"
    }
    assert entries, "the transform family has no manifest entry"
    for name, status in entries.items():
        assert (name in exported) == (status == "implemented"), name
