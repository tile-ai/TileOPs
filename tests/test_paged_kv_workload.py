import pytest
import torch

from workloads.attention.gqa.paged import GQAPagedFwdWorkload
from workloads.attention.paged_kv_cache import make_fragmented_block_table
from workloads.device import run_device


@pytest.mark.smoke
def test_fragmented_block_table() -> None:
    batch, pages_per_req = 4, 8
    pool_pages = batch * pages_per_req

    disjoint = make_fragmented_block_table(batch, pages_per_req, pool_pages)
    assert disjoint.shape == (batch, pages_per_req)
    assert sorted(disjoint.flatten().tolist()) == list(range(pool_pages))
    assert not torch.equal(
        disjoint,
        torch.arange(pool_pages, dtype=torch.int32, device=run_device()).reshape(batch, -1),
    )

    shared = make_fragmented_block_table(batch, pages_per_req, pages_per_req).tolist()
    for row in shared:
        assert sorted(row) == list(range(pages_per_req))
    assert not all(row == sorted(row) for row in shared)

    assert torch.equal(disjoint, make_fragmented_block_table(batch, pages_per_req, pool_pages))

    case = GQAPagedFwdWorkload(16, 4, 128, [64, 64], [192, 192], 64, 3, 6, torch.float16)
    table = case.gen_inputs()[3]
    assert sorted(table.flatten().tolist()) == list(range(6))
