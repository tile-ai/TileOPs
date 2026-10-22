"""The paged attention kernels tune on requests that read distinct cache pages.

A candidate is ranked by the time it takes on the inputs the autotune supply builds. When
every request's block table names the same pages, the whole cache the candidates read is a
few pages that stay in L2, and the ranking stops following the cache traffic a served batch
pays. Each case builds a kernel, asks its supply for one set of inputs, and checks the page
table: every page inside the pool, each request's pages distinct, and the requests' pages
disjoint wherever the pool holds them all.
"""

import pytest
import torch

from tileops.kernels.attention import (
    GQADecodePagedKernel,
    GQAPagedFwdKernel,
    MHADecodePagedWSKernel,
)

pytestmark = [pytest.mark.cuda_only, pytest.mark.smoke]

# Four requests of two pages each; the supplies read the page table at argument 4.
_BATCH, _WIDTH, _PAGE_SIZE = 4, 2, 128
_TABLE_ARG = 4


def _check_table(table: torch.Tensor, pool_pages: int) -> None:
    assert table.shape == (_BATCH, _WIDTH)
    assert int(table.min()) >= 0 and int(table.max()) < pool_pages
    for row in table.tolist():
        assert len(set(row)) == _WIDTH, f"a request repeats a page: {table.tolist()}"
    if pool_pages >= _BATCH * _WIDTH:
        pages = table.flatten().tolist()
        assert len(set(pages)) == len(pages), f"requests share cache pages: {table.tolist()}"


def test_mha_paged_decode_tunes_on_distinct_pages() -> None:
    # A pool of one request's pages, and one of every request's.
    for pool_pages in (_WIDTH, _BATCH * _WIDTH):
        kernel = MHADecodePagedWSKernel(
            _BATCH,
            8,
            1,
            pool_pages * _PAGE_SIZE,
            64,
            _PAGE_SIZE,
            False,
            torch.float16,
            max_pages_per_req=_WIDTH,
        )
        config = kernel.config
        program = kernel.kernel(
            config["block_N"], config["num_split"], config["stages"], config["group"]
        )
        inputs = kernel.autotune_supply_prog(program.params)
        _check_table(inputs[_TABLE_ARG], pool_pages)
        assert int(inputs[3].max()) <= _WIDTH * _PAGE_SIZE  # lengths within the table


def test_gqa_paged_decode_tunes_on_distinct_pages() -> None:
    for pool_pages in (_WIDTH, _BATCH * _WIDTH):
        kernel = GQADecodePagedKernel(
            _BATCH, 8, 2, 1, pool_pages * _PAGE_SIZE, 64, _PAGE_SIZE, _WIDTH, False, torch.float16
        )
        program = kernel.kernel(**kernel.config)
        inputs = kernel.autotune_supply_prog(program.params)
        _check_table(inputs[_TABLE_ARG], pool_pages)
        assert int(inputs[3].max()) <= _WIDTH * _PAGE_SIZE  # lengths within the table


def test_gqa_paged_tunes_on_distinct_pages() -> None:
    kernel = GQAPagedFwdKernel(_BATCH, 8, 2, 64, _PAGE_SIZE, _WIDTH, True)
    # The supply builds every argument itself, the pools among them.
    inputs = kernel.autotune_supply_prog([None] * 12)
    pool_pages = inputs[1].shape[0] // _PAGE_SIZE
    _check_table(inputs[_TABLE_ARG], pool_pages)
    assert pool_pages >= _BATCH * _WIDTH
