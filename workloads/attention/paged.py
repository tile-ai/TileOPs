"""Helpers for building paged KV cache block tables and page pools."""

import torch

from workloads.workload_base import WORKLOAD_SEED


def make_interleaved_block_table(
    batch: int, max_pages_per_req: int, *, device: torch.device | str = "cuda"
) -> torch.Tensor:
    """Block table whose logical pages sit out of order in physical memory.

    Each request owns a contiguous run of physical pages and reads them
    even-indices-first, so a kernel that ignores the table and walks physical
    pages in order produces a different answer than one that honours it.
    """
    rows = []
    for b in range(batch):
        start = b * max_pages_per_req
        pages = list(range(start, start + max_pages_per_req))
        rows.append(pages[::2] + pages[1::2])
    return torch.tensor(rows, device=device, dtype=torch.int32).contiguous()


def make_fragmented_block_table(
    batch: int,
    pages_per_req: int,
    pool_pages: int,
    seed: int = WORKLOAD_SEED,
    *,
    device: torch.device | str = "cuda",
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
    return table.to(device=device, dtype=torch.int32).contiguous()


def paged_cache_row(
    block_table: torch.Tensor, batch_idx: int, logical_pos: int, page_size: int
) -> int:
    """Row of the page pool holding logical position *logical_pos* of *batch_idx*."""
    logical_page = logical_pos // page_size
    page_offset = logical_pos % page_size
    physical_page = int(block_table[batch_idx, logical_page].item())
    return physical_page * page_size + page_offset


def make_unit_cache_scales(
    *, device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """The K and V dequantisation scales of an unquantised cache."""
    scale = torch.ones((1,), device=device, dtype=torch.float32)
    return scale, scale.clone()


def fill_paged_cache_from_logical(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    k_old: list[torch.Tensor],
    v_old: list[torch.Tensor],
    block_table: torch.Tensor,
    page_size: int,
) -> None:
    """Scatter each request's logical cache rows into the pages *block_table* names.

    ``k_old`` and ``v_old`` carry one ``[cache_len, heads_kv, dim]`` tensor per
    request, in batch order, holding that request's cache in logical position
    order. ``k_pages`` and ``v_pages`` are written in place.
    """
    for b, (k_b, v_b) in enumerate(zip(k_old, v_old, strict=True)):
        for pos in range(k_b.shape[0]):
            row = paged_cache_row(block_table, b, pos, page_size)
            k_pages[row].copy_(k_b[pos])
            v_pages[row].copy_(v_b[pos])
