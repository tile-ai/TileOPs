"""The moe generators and their `requires` predicate (docs/design/manifest.md § Generators)."""

import random

import pytest

from tileops.manifest.primitives import GENERATOR_SHAPES, GENERATORS, PREDICATES
from tileops.manifest.values import ADTValue

pytestmark = pytest.mark.smoke

_layout_metadata = GENERATORS["moe.layout_metadata"]
_layout_valid = PREDICATES["moe.layout_valid"]


def _contiguous(packing: str, metadata_kind: str, alignment: int = 1) -> ADTValue:
    fields = {"packing": packing, "metadata_kind": metadata_kind, "alignment": alignment}
    return ADTValue("MGroupedLayout", {"contiguous": fields})


def _masked(max_m: int) -> ADTValue:
    return ADTValue("MGroupedLayout", {"masked": {"max_m": max_m}})


_LAYOUTS = {
    "tight-psum": _contiguous("tight", "physical_psum"),
    "tight-per-row": _contiguous("tight", "per_row"),
    "aligned-psum": _contiguous("aligned", "physical_psum", 4),
    "aligned-per-row": _contiguous("aligned", "per_row", 4),
}


@pytest.mark.parametrize("name", _LAYOUTS)
def test_layout_metadata_satisfies_its_predicate(name):
    """No rows, fewer tiles than experts, empty tail experts, several tiles per expert."""
    layout = _LAYOUTS[name]
    for rows, experts in [(0, 3), (8, 3), (12, 5), (40, 3)]:
        values = _layout_metadata(layout, rows, experts)
        assert (len(values),) == GENERATOR_SHAPES["moe.layout_metadata"](layout, rows, experts)
        assert _layout_valid(values, layout, rows, experts), (rows, experts)


def test_layout_metadata_psum_ends_are_exclusive_and_masked_counts_vary():
    assert _layout_metadata(_LAYOUTS["tight-psum"], 8, 2) == [4, 8]
    assert _layout_metadata(_LAYOUTS["aligned-psum"], 8, 3) == [2, 6, 8]
    masked = _masked(4)
    counts = _layout_metadata(masked, 12, 3)
    assert counts == [4, 2, 4] and _layout_valid(counts, masked, 12, 3)


@pytest.mark.parametrize(
    "layout,rows,experts",
    [
        (_LAYOUTS["aligned-psum"], 6, 2),  # not a whole number of tiles
        (_masked(4), 8, 3),  # not E * max_m
        (_LAYOUTS["tight-psum"], 4, 0),  # no experts
    ],
)
def test_layout_metadata_raises_outside_its_domain(layout, rows, experts):
    with pytest.raises(ValueError):
        _layout_metadata(layout, rows, experts)


@pytest.mark.parametrize(
    "layout,metadata,rows,expected",
    [
        pytest.param(_LAYOUTS["tight-psum"], [2, 2, 5], 5, True, id="tight-psum-ok"),
        pytest.param(_LAYOUTS["tight-psum"], [2, 2, 4], 5, False, id="tight-psum-short"),
        pytest.param(_LAYOUTS["tight-psum"], [2, 1, 5], 5, False, id="tight-psum-decreasing"),
        pytest.param(_LAYOUTS["aligned-psum"], [2, 6, 12], 12, True, id="aligned-psum-ok"),
        pytest.param(_LAYOUTS["aligned-psum"], [2, 3, 12], 12, False, id="aligned-psum-mid-tile"),
        pytest.param(_LAYOUTS["aligned-psum"], [2, 6, 13], 12, False, id="aligned-psum-overrun"),
        pytest.param(
            _contiguous("aligned", "per_row", 2),
            [0, 0, 1, 1, 3, 3],
            6,
            True,
            id="aligned-per-row-padded-tail",
        ),
        pytest.param(
            _contiguous("aligned", "per_row", 2),
            [0, 1, 1, 1, 3, 3],
            6,
            False,
            id="aligned-per-row-change-mid-tile",
        ),
        pytest.param(_LAYOUTS["tight-per-row"], [0, 0, 1, 2, 2], 5, True, id="tight-per-row-ok"),
        pytest.param(
            _LAYOUTS["tight-per-row"], [0, 0, 1, 3, 3], 5, False, id="tight-per-row-no-padding"
        ),
        pytest.param(_masked(4), [4, 0, 2], 12, True, id="masked-ok"),
        pytest.param(_masked(4), [5, 0, 2], 12, False, id="masked-over-capacity"),
    ],
)
def test_layout_valid_table(layout, metadata, rows, expected):
    assert _layout_valid(metadata, layout, rows, 3) is expected


def test_sample_indices_draws_distinct_values_in_range():
    values = GENERATORS["sample_indices"](random.Random(0), 6, 8)
    assert len(set(values)) == 6 and all(0 <= v < 8 for v in values)
    assert sorted(GENERATORS["sample_indices"](random.Random(0), 5, 5)) == list(range(5))
    with pytest.raises(ValueError):
        GENERATORS["sample_indices"](random.Random(0), 3, 2)
