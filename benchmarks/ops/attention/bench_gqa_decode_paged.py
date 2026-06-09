import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import gqa_decode_paged_args, manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionDecodePagedWithKVCacheFwdOp
from workloads.attention.gqa_decode_paged import GroupedQueryAttentionDecodePagedTest

_OP_NAME = "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp"


class _GroupedQueryAttentionDecodePagedTestBaseline(GroupedQueryAttentionDecodePagedTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then GQA (expand to heads) + SDPA."""
        batch, _, dim = q.shape
        seqlen_kv, _, _ = k.shape
        kv_group_num = self.heads // self.heads_kv
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b:i_b + 1, :, :]
            k_logical = torch.zeros(seqlen_kv, self.heads_kv, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, self.heads_kv, dim, dtype=q.dtype, device=q.device)
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
            group_id = torch.arange(self.heads, dtype=torch.long, device=q.device) // kv_group_num
            k_bhsd = k_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
            v_bhsd = v_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
            q_bhsd = q_b.unsqueeze(2)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.squeeze(2)
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)


def _fa3_gqa_decode_paged(test, k, v):
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
    k_paged = k.view(num_pages, test.page_size, test.heads_kv, test.dim)
    v_paged = v.view(num_pages, test.page_size, test.heads_kv, test.dim)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim) — add seq dim for flash_attn
        out = flash_attn_with_kvcache(
            q.unsqueeze(1), k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            page_table=block_table.int())
        out = out[0] if isinstance(out, tuple) else out
        return out.squeeze(1)

    return baseline_fn


def _flashinfer_gqa_decode_paged(test, q, k, v, real_seqlen_kv, block_table):
    """Set up FlashInfer paged decode wrapper. Returns callable or None.

    FlashInfer decode kernel supports group_size (Q/KV head ratio) up to 8.
    """
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    if test.heads // test.heads_kv > 8:
        return None  # FlashInfer decode kernel does not support group_size > 8

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


_GQA_DECODE_PAGED_BENCH_PARAMS = manifest_params(
    load_workloads(_OP_NAME),
    gqa_decode_paged_args,
)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune",
    _GQA_DECODE_PAGED_BENCH_PARAMS,
)
def test_gqa_decode_paged_bench(batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                                page_size: int, dtype: torch.dtype, tune: bool) -> None:
    test = _GroupedQueryAttentionDecodePagedTestBaseline(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
    inputs = test.gen_inputs()
    q, k, v, real_seqlen_kv, block_table = inputs

    op = GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(
        batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
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
