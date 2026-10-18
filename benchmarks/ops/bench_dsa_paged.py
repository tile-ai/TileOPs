"""Benchmark TileOPs DeepSeek sparse attention (DSA) over a paged FP8 cache against FlashMLA."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import vllm_op
from tileops.ops import DSAPagedFwdOp


def _flashmla(case: bench.Case) -> bench.Implementation:
    """FlashMLA's sparse FP8 decode on the case's cache.

    Positions resolve to physical rows outside the timed call. The SM90 kernel takes no
    ``topk_length``, so it reads every slot.
    """
    run = vllm_op("flash_mla_with_kvcache", "v1.attention.ops.flashmla")
    metadata = vllm_op("get_mla_metadata", "v1.attention.ops.flashmla")
    q, kv_cache, block_table, cache_seqlens, indices = case.inputs
    page_size = kv_cache.shape[1]
    pos = indices.long()
    live = (pos >= 0) & (pos < cache_seqlens.long()[:, None, None])
    slot = pos.masked_fill(~live, 0)
    page = torch.gather(block_table.long(), 1, (slot // page_size).flatten(1)).view_as(slot)
    rows = torch.where(live, page * page_size + slot % page_size, -1).to(torch.int32)
    schedule, _ = metadata()
    cache = kv_cache.unsqueeze(2)
    sm_scale = case.arguments["sm_scale"]
    head_dim_v = 512  # the latent width, which is also the value width

    def fn():
        o, lse = run(
            q, cache, None, None, head_dim_v, schedule, softmax_scale=sm_scale, is_fp8_kvcache=True,
            indices=rows,
        )  # fmt: skip
        # FlashMLA returns the log-sum-exp as (batch, heads, seq_len).
        return o, lse.transpose(1, 2)

    return bench.Implementation(run=fn, args=())


@pytest.mark.parametrize("case", bench.cases(DSAPagedFwdOp), ids=lambda case: case.id)
def test_dsa_paged_bench(case) -> None:
    op = DSAPagedFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"flashmla": _flashmla(case)})
