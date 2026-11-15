"""The indexing family's packaging contract: the module ships and exports exactly its
implemented manifest entries."""

import pytest

from tileops.manifest import load_manifest


@pytest.mark.smoke
@pytest.mark.packaging(family="indexing")
def test_the_indexing_module_ships_and_exports_what_the_manifest_implements() -> None:
    import tileops.indexing

    entries = {
        name: entry["status"]
        for name, entry in load_manifest().items()
        if entry["family"] == "indexing"
    }
    assert entries, "the indexing family has no manifest entry"
    implemented = {name for name, status in entries.items() if status == "implemented"}
    assert set(tileops.indexing.__all__) == implemented
