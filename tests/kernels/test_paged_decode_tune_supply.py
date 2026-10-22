"""The paged decode kernels tune on requests that read distinct cache pages.

A candidate is ranked by the time it takes on the inputs the autotune supply builds. When
every request's block table names the same pages, the whole cache the candidates read is a
few pages that stay in L2, and the ranking stops following the cache traffic a served batch
pays. Each case builds a kernel, asks its supply for one set of inputs, and checks that the
requests' page lists are disjoint and inside the pool.
"""

import pytest
import torch

from tileops.kernels.attention import GQADecodePagedKernel, MHADecodePagedWSKernel

pytestmark = [pytest.mark.cuda_only, pytest.mark.smoke]

# Four requests of two pages each over a pool of eight pages: room for disjoint tables.
_BATCH, _PAGES, _PAGE_SIZE = 4, 2, 128
_POOL_ROWS = _BATCH * _PAGES * _PAGE_SIZE


def _assert_disjoint_pages(table: torch.Tensor) -> None:
    assert table.shape == (_BATCH, _PAGES)
    pages = table.flatten().tolist()
    assert len(set(pages)) == len(pages), f"requests share cache pages: {table.tolist()}"
    assert all(0 <= p < _POOL_ROWS // _PAGE_SIZE for p in pages)


def test_mha_paged_decode_tunes_on_distinct_pages() -> None:
    kernel = MHADecodePagedWSKernel(
        _BATCH, 8, 1, _POOL_ROWS, 64, _PAGE_SIZE, False, torch.float16, max_pages_per_req=_PAGES
    )
    config = kernel.config
    program = kernel.kernel(
        config["block_N"], config["num_split"], config["stages"], config["group"]
    )
    table_arg = 4  # block_table in the kernel signature
    _assert_disjoint_pages(kernel.autotune_supply_prog(program.params)[table_arg])


def test_gqa_paged_decode_tunes_on_distinct_pages() -> None:
    kernel = GQADecodePagedKernel(
        _BATCH, 8, 2, 1, _POOL_ROWS, 64, _PAGE_SIZE, _PAGES, False, torch.float16
    )
    program = kernel.kernel(**kernel.config)
    table_arg = 4  # block_table in the kernel signature
    _assert_disjoint_pages(kernel.autotune_supply_prog(program.params)[table_arg])
