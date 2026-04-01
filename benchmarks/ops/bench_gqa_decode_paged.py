from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_decode_paged import GqaDecodePagedTest
from tileops.ops import GroupQueryAttentionDecodePagedWithKVCacheOp


class GqaDecodePagedBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seqlen_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        num_pages = t.seqlen_kv // t.page_size
        # Q, output: batch * heads * dim; K,V: seqlen_kv * heads_kv * dim; block_table, real_seqlen_kv: int32
        return (t.batch * t.heads * t.dim * 2 +
                2 * t.seqlen_kv * t.heads_kv * t.dim) * t.dtype.itemsize + \
            t.batch * num_pages * 4 + t.batch * 4


def _fa3_gqa_decode_paged(test, k, v):
    """Set up FA3 paged decode. Returns callable or None.

    FA3 requires page_block_size to be a multiple of 256.
    """
    if test.page_size % 256 != 0:
        return None
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: PLC0415
    except ImportError:
        return None

    num_pages = k.shape[0] // test.page_size
    k_paged = k.view(num_pages, test.page_size, test.heads_kv, test.dim)
    v_paged = v.view(num_pages, test.page_size, test.heads_kv, test.dim)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim) — add seq dim for flash_attn
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int()).squeeze(1)

    return baseline_fn


def _flashinfer_gqa_decode_paged(test, q, k, v, real_seqlen_kv, block_table):
    """Set up FlashInfer paged decode wrapper. Returns callable or None."""
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    batch = q.shape[0]
    num_pages = k.shape[0] // test.page_size
    k_paged = k.view(num_pages, test.page_size, test.heads_kv, test.dim)
    v_paged = v.view(num_pages, test.page_size, test.heads_kv, test.dim)
    kv_data = (k_paged, v_paged)

    pages_per_batch = (real_seqlen_kv.int() + test.page_size - 1) // test.page_size
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    indptr[1:] = torch.cumsum(pages_per_batch, dim=0)

    indices_list = []
    for b in range(batch):
        n = pages_per_batch[b].item()
        indices_list.append(block_table[b, :n])
    indices = torch.cat(indices_list)

    last_page_len = (real_seqlen_kv.int() - 1) % test.page_size + 1

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=test.heads,
        num_kv_heads=test.heads_kv,
        head_dim=test.dim,
        page_size=test.page_size,
        q_data_type=test.dtype,
    )

    def run_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim)
        return wrapper.run(q, kv_data)

    return run_fn


# GQA paged decode benchmark parameters.
#
# Paged KV cache is the standard in production serving (vLLM, SGLang).
# Batch sizes reflect real multi-request serving: B=4-64.
#
# Three page_size tiers:
#   64  — TileOPs current minimum supported page_size (block_N constraint)
#   256 — FlashInfer's optimized page_size, also FA3-compatible (multiple of 256)
#   16  — vLLM/SGLang default; skip-marked until kernel supports page_size<64
#
# Head profiles and KV cache lengths match non-paged decode configs.
_GQA_DECODE_PAGED_BENCH_PARAMS = [
    # ── page_size=64 (TileOPs minimum) ──
    # 8B-class online serving (B=32, 4K context)
    pytest.param(32, 32, 8, 4096, 128, 64, torch.float16, True, id="serving-8b-p64"),
    # 8B-class long-context serving (B=8, 32K context)
    pytest.param(8, 32, 8, 32768, 128, 64, torch.float16, True, id="serving-8b-long-p64"),
    # 8B-class high-throughput batch (B=64, short output)
    pytest.param(64, 32, 8, 2048, 128, 64, torch.float16, True, id="throughput-8b-p64"),
    # 70B-class online serving
    pytest.param(8, 64, 8, 4096, 128, 64, torch.float16, True, id="serving-70b-p64"),
    # ── page_size=256 (FlashInfer optimized, FA3 compatible) ──
    # 8B-class online serving
    pytest.param(32, 32, 8, 4096, 128, 256, torch.float16, True, id="serving-8b-p256"),
    # 70B-class online serving
    pytest.param(8, 64, 8, 4096, 128, 256, torch.float16, True, id="serving-70b-p256"),
    # 405B-class online serving (B=4, limited by model size)
    pytest.param(4, 128, 8, 4096, 128, 256, torch.float16, True, id="serving-405b-p256"),
    # ── page_size=16 (vLLM/SGLang default) — skip until kernel support ──
    pytest.param(32, 32, 8, 4096, 128, 16, torch.float16, True, id="serving-8b-p16",
                 marks=pytest.mark.skip(reason="page_size=16 not yet supported by kernel")),
    pytest.param(64, 32, 8, 2048, 128, 16, torch.float16, True, id="throughput-8b-p16",
                 marks=pytest.mark.skip(reason="page_size=16 not yet supported by kernel")),
    pytest.param(8, 64, 8, 4096, 128, 16, torch.float16, True, id="serving-70b-p16",
                 marks=pytest.mark.skip(reason="page_size=16 not yet supported by kernel")),
]


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune",
    _GQA_DECODE_PAGED_BENCH_PARAMS,
)
def test_gqa_decode_paged_bench(batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                                page_size: int, dtype: torch.dtype, tune: bool) -> None:
    test = GqaDecodePagedTest(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
    bm = GqaDecodePagedBenchmark(test)
    inputs = test.gen_inputs()
    q, k, v, real_seqlen_kv, block_table = inputs

    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_gqa_decode_paged(test, k, v)
    if fa3_fn is not None:
        result_fa3 = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fa3, tag="fa3")

    fi_fn = _flashinfer_gqa_decode_paged(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
