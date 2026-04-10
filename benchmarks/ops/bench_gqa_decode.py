from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp
from workloads.ops.gqa_decode import GqaDecodeTest


class _GqaDecodeTestBaseline(GqaDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).squeeze(1).contiguous()
        return output


class GqaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # Q: batch * 1 * heads * dim
        # K, V: batch * seq_len_kv * heads_kv * dim
        # Output: batch * 1 * heads * dim
        return 2 * t.batch * t.dim * t.dtype.itemsize * (
            t.heads + t.heads_kv * t.seq_len_kv)


def _fa3_gqa_decode_fwd(test):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        # Q is (B, H, D) — add seq dim for flash_attn
        out = flash_attn_func(q.unsqueeze(1), k, v)
        out = out[0] if isinstance(out, tuple) else out
        return out.squeeze(1)

    return baseline_fn


def _flashinfer_gqa_decode_fwd(test, q, k, v):
    """Set up FlashInfer batched paged decode for non-paged KV cache.

    Reshapes the contiguous (B, S_kv, H_kv, D) KV into paged format with
    page_size=256, then uses the specialized decode kernel.
    FlashInfer decode kernel supports group_size (Q/KV head ratio) up to 8.
    """
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    # Q is (B, H, D) — single token per request
    # K/V is (B, S_kv, H_kv, D)
    B, H, D = q.shape
    Hkv = k.shape[2]
    if H // Hkv > 8:
        return None  # FlashInfer decode kernel does not support group_size > 8
    Skv = k.shape[1]
    page_size = 256
    pages_per_seq = (Skv + page_size - 1) // page_size

    # Reshape (B, S_kv, H_kv, D) → (B * pages_per_seq, page_size, H_kv, D)
    # Pad S_kv to a multiple of page_size if needed
    if Skv % page_size != 0:
        pad = page_size * pages_per_seq - Skv
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    k_paged = k.reshape(B * pages_per_seq, page_size, Hkv, D)
    v_paged = v.reshape(B * pages_per_seq, page_size, Hkv, D)
    kv_data = (k_paged, v_paged)

    total_pages = B * pages_per_seq
    indptr = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * pages_per_seq
    indices = torch.arange(0, total_pages, dtype=torch.int32, device=q.device)
    last_page_len_val = Skv - (pages_per_seq - 1) * page_size
    last_page_len = torch.full((B,), last_page_len_val, dtype=torch.int32, device=q.device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=indptr, indices=indices, last_page_len=last_page_len,
        num_qo_heads=H, num_kv_heads=Hkv, head_dim=D,
        page_size=page_size,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(q, kv_data)

    return run_fn


# GQA decode (non-paged) benchmark parameters.
#
# Non-paged KV cache is used for single-request inference (no serving framework).
# B=1 exclusively — multi-request scenarios use paged KV cache instead.
# Configs target the three standard head profiles (see bench_gqa.py) with
# KV cache lengths representing typical chat (4K) and long-context (32K) use.
_GQA_DECODE_BENCH_PARAMS = [
    # Single-user chat (8B-class, 4K context)
    pytest.param(1, 32, 8, 4096, 128, torch.float16, True, id="llama8b-4k"),
    # Long-context generation (8B-class, 32K context)
    pytest.param(1, 32, 8, 32768, 128, torch.float16, True, id="llama8b-32k"),
    # 70B-class single-request decode
    pytest.param(1, 64, 8, 4096, 128, torch.float16, True, id="llama70b-4k"),
    # 405B-class single-request decode
    pytest.param(1, 128, 8, 8192, 128, torch.float16, True, id="llama405b-8k"),
]


@pytest.mark.parametrize("batch, heads, heads_kv, seq_len_kv, dim, dtype, tune", _GQA_DECODE_BENCH_PARAMS)
def test_gqa_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = _GqaDecodeTestBaseline(batch, heads, heads_kv, seq_len_kv, dim, dtype)
    bm = GqaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(batch, heads, heads_kv, seq_len_kv, dim, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_gqa_decode_fwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    fi_fn = _flashinfer_gqa_decode_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
