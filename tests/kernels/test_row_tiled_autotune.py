"""The tile_n search space of the row-tiled reduction kernels."""

import pytest

from tileops._kernels.reduction._primitives import BlockConfigPlanner, RowTiledAutotuneMixin

pytestmark = pytest.mark.smoke


class _RowTiled(RowTiledAutotuneMixin):
    """The mixin's declared inputs, without building a kernel to get them."""

    _MAX_TILE_N_CANDIDATES = 3

    def __init__(self, n_padded: int, elem_bytes: int, smem_budget: int) -> None:
        self.N_padded = n_padded
        self._elem_bytes = elem_bytes
        self._smem_budget = smem_budget
        self._planner = BlockConfigPlanner(n_padded, elem_bytes, smem_budget)


def test_row_that_fits_untiled_at_every_thread_count_offers_no_tile():
    """A row one tile holds tunes block_m and threads only: a tiled width is another kernel."""
    assert _RowTiled(4096, 2, 227 * 1024)._tile_n_candidates() == [0]


def test_tiled_sweep_offers_only_buildable_tiles():
    """Every config a tiled sweep times carries a tile the tiled kernel can build.

    fp32 Welford rows of 11008 need tiling at 256 threads yet fit untiled at 512, and
    the full-row tile does not divide across 512 threads.
    """
    planner = BlockConfigPlanner(11008, 4, 227 * 1024, num_buffers=2, frag_slots=2)
    assert planner.needs_tiling
    for c in planner.autotune_configs():
        assert not planner.reject_tile_n(c["block_m"], c["tile_n"], c["threads"]), c
