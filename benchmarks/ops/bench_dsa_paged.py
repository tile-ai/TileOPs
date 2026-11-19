"""Benchmark TileOPs DeepSeek sparse attention (DSA) over a paged cache against FlashMLA."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import vllm_op
from tileops.attention import DSAPagedFwdOp


def _physical_rows(case: bench.Case) -> torch.Tensor:
    """Each slot's physical cache row, or -1 where it is dead."""
    _, kv_cache, block_table, cache_seqlens, indices, _ = case.inputs
    page_size = kv_cache.shape[1]
    pos = indices.long()
    live = (pos >= 0) & (pos < cache_seqlens.long()[:, None, None])
    slot = pos.masked_fill(~live, 0)
    page = torch.gather(block_table.long(), 1, (slot // page_size).flatten(1)).view_as(slot)
    return torch.where(live, page * page_size + slot % page_size, -1).to(torch.int32)


def _flashmla(case: bench.Case) -> "bench.Implementation":
    """FlashMLA on the case's cache: its sparse FP8 decode, or for bfloat16 rows, which that
    decode does not take, its sparse attention over bfloat16 rows.

    Positions resolve to physical rows outside the timed call. The SM90 FP8 decode takes no
    ``topk_length``, so it reads every slot.
    """
    q, kv_cache, _, _, _, attn_sink = case.inputs
    rows = _physical_rows(case)
    if kv_cache.dtype != torch.uint8:
        return _flashmla_bf16(case, rows)
    run = vllm_op("flash_mla_with_kvcache", "v1.attention.ops.flashmla")
    metadata = vllm_op("get_mla_metadata", "v1.attention.ops.flashmla")
    schedule, _ = metadata()
    cache = kv_cache.unsqueeze(2)
    sm_scale, head_dim_v = case.arguments["sm_scale"], case.arguments["head_dim_v"]

    def fn():
        o, lse = run(
            q, cache, None, None, head_dim_v, schedule, softmax_scale=sm_scale, is_fp8_kvcache=True,
            indices=rows, attn_sink=attn_sink,
        )  # fmt: skip
        # FlashMLA returns the log-sum-exp as (batch, heads, seq_len), the op's layout.
        return o, lse

    return bench.Implementation(run=fn, args=())


def _flashmla_bf16(case: bench.Case, rows: torch.Tensor) -> "bench.Implementation":
    """FlashMLA's ``flash_mla_sparse_fwd`` over the cache as one flat row list, each query
    row a query of its own.

    The selection pads to a multiple of 128 slots, which the SM90 kernel requires, with dead
    slots that ``topk_length`` leaves unread; both are built outside the timed call.
    """
    q, kv_cache, _, _, indices, attn_sink = case.inputs
    run = vllm_op("flash_mla_sparse_fwd", "v1.attention.ops.flashmla")
    batch, seq_len, heads, dk = q.shape
    topk = indices.shape[-1]
    padded = torch.full(
        (batch * seq_len, 1, -(-topk // 128) * 128), -1, dtype=torch.int32, device=q.device
    )
    padded[:, 0, :topk] = rows.view(batch * seq_len, topk)
    lengths = torch.full((batch * seq_len,), topk, dtype=torch.int32, device=q.device)
    flat_q, flat_kv = q.view(batch * seq_len, heads, dk), kv_cache.view(-1, 1, dk)
    sm_scale = case.arguments["sm_scale"]
    sm_scale = dk**-0.5 if sm_scale is None else sm_scale
    head_dim_v = case.arguments["head_dim_v"]

    def fn():
        o, _, lse = run(flat_q, flat_kv, padded, sm_scale, head_dim_v, attn_sink, lengths)
        # Its lse is (query, heads); the op's is (batch, heads, seq_len).
        return o.view(batch, seq_len, heads, head_dim_v), lse.view(batch, seq_len, heads).transpose(
            1, 2
        )

    return bench.Implementation(run=fn, args=())


@pytest.mark.parametrize("case", bench.cases(DSAPagedFwdOp), ids=lambda case: case.id)
def test_dsa_paged_bench(case) -> None:
    op = DSAPagedFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"flashmla": _flashmla(case)})
