import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MhaDecodePagedFwdOp
from workloads.ops.mha_decode_paged import MhaDecodePagedTest


class _MhaDecodePagedTestBaseline(MhaDecodePagedTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then SDPA."""
        batch, seqlen_q, heads, dim = q.shape
        seqlen_kv = k.shape[0]
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b:i_b + 1, :, :, :]
            k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(real_seqlen_kv[i_b].item() / self.page_size)
            for i_paged in range(num_pages):
                start_pos = block_table[i_b, i_paged].item() * self.page_size
                end_pos = min(start_pos + self.page_size, seqlen_kv)
                page_len = end_pos - start_pos
                k_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = k[start_pos:end_pos, :, :]
                v_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = v[start_pos:end_pos, :, :]
            k_logical = k_logical[:real_seqlen_kv[i_b].item(), :, :]
            v_logical = v_logical[:real_seqlen_kv[i_b].item(), :, :]
            k_b = k_logical.unsqueeze(0)
            v_b = v_logical.unsqueeze(0)
            q_bhsd = q_b.transpose(1, 2)
            k_bhsd = k_b.transpose(1, 2)
            v_bhsd = v_b.transpose(1, 2)
            out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.transpose(1, 2).contiguous()
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)


class MhaDecodePagedBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seqlen_q * t.seqlen_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        num_pages = t.seqlen_kv // t.page_size
        # Q, output: batch * seqlen_q * heads * dim; K,V: seqlen_kv * heads * dim; block_table, real_seqlen_kv: int32
        return (t.batch * t.seqlen_q * t.heads * t.dim * 2 +
                2 * t.seqlen_kv * t.heads * t.dim) * t.dtype.itemsize + \
            t.batch * num_pages * 4 + t.batch * 4


def _fa3_mha_decode_paged(test, k, v):
    """Set up FA3 paged decode. Returns callable or None.

    FA3 requires page_block_size to be a multiple of 256.
    """
    if test.page_size % 256 != 0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache  # noqa: PLC0415
    except ImportError:
        return None

    num_pages = k.shape[0] // test.page_size
    k_paged = k.view(num_pages, test.page_size, test.heads, test.dim)
    v_paged = v.view(num_pages, test.page_size, test.heads, test.dim)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        out = flash_attn_with_kvcache(
            q, k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            page_table=block_table.int())
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_mha_decode_paged(test, q, k, v, real_seqlen_kv, block_table):
    """Set up FlashInfer paged decode wrapper. Returns callable or None."""
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    batch = q.shape[0]
    num_pages = k.shape[0] // test.page_size
    k_paged = k.view(num_pages, test.page_size, test.heads, test.dim)
    v_paged = v.view(num_pages, test.page_size, test.heads, test.dim)
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
        num_kv_heads=test.heads,
        head_dim=test.dim,
        page_size=test.page_size,
        q_data_type=test.dtype,
    )

    def run_fn(q, k, v, real_seqlen_kv, block_table):
        return wrapper.run(q.squeeze(1), kv_data).unsqueeze(1)

    return run_fn


_MHA_DECODE_PAGED_BENCH_PARAMS = [
    pytest.param(1, 16, 1, 512, 128, 128, False, torch.float16, True, id="single-token-page128"),
    pytest.param(2, 8, 1, 1024, 64, 256, False, torch.float16, True, id="batch2-page256"),
    pytest.param(1, 8, 1, 1024, 64, 256, False, torch.float16, True, id="longer-cache"),
    pytest.param(1, 8, 1, 512, 64, 256, False, torch.float16, True, id="shorter-cache"),
]


@pytest.mark.parametrize(
    "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
    _MHA_DECODE_PAGED_BENCH_PARAMS,
)
def test_mha_decode_paged_bench(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                                page_size: int, is_causal: bool, dtype: torch.dtype,
                                tune: bool) -> None:
    test = _MhaDecodePagedTestBaseline(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    bm = MhaDecodePagedBenchmark(test)
    inputs = test.gen_inputs()
    q, k, v, real_seqlen_kv, block_table = inputs

    op = MhaDecodePagedFwdOp(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_mha_decode_paged(test, k, v)
    if fa3_fn is not None:
        result_fa3 = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fa3, tag="fa3")

    fi_fn = _flashinfer_mha_decode_paged(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
