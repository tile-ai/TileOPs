"""Helpers for building paged KV cache block tables and page pools."""

import torch

from workloads.device import run_device
from workloads.workload_base import WORKLOAD_SEED


def make_fragmented_block_table(
    batch: int, pages_per_req: int, pool_pages: int, seed: int = WORKLOAD_SEED
) -> torch.Tensor:
    """Block table over a fragmented page pool, the layout a serving cache has.

    A pool holding ``batch * pages_per_req`` pages gives every request a
    disjoint set; a smaller pool gives every request its own permutation of the
    same pages. The fixed seed keeps two runs on the same layout.
    """
    if pages_per_req > pool_pages:
        raise ValueError(f"a request needs {pages_per_req} pages but the pool holds {pool_pages}")
    generator = torch.Generator().manual_seed(seed)
    if pool_pages >= batch * pages_per_req:
        pages = torch.randperm(pool_pages, generator=generator)[: batch * pages_per_req]
        table = pages.reshape(batch, pages_per_req)
    else:
        table = torch.stack(
            [torch.randperm(pool_pages, generator=generator)[:pages_per_req] for _ in range(batch)]
        )
    return table.to(device=run_device(), dtype=torch.int32).contiguous()


def paged_cache_row(
    block_table: torch.Tensor, batch_idx: int, logical_pos: int, page_size: int
) -> int:
    """Row of the page pool holding logical position *logical_pos* of *batch_idx*."""
    logical_page = logical_pos // page_size
    page_offset = logical_pos % page_size
    physical_page = int(block_table[batch_idx, logical_page].item())
    return physical_page * page_size + page_offset
