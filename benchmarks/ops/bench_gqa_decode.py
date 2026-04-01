from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_decode import GqaDecodeTest
from tileops.ops import GroupQueryAttentionDecodeWithKVCacheOp


class GqaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Q: batch * 1 * heads * dim
        # K, V: batch * seq_len_kv * heads_kv * dim
        # Output: batch * 1 * heads * dim
        return 2 * t.batch * t.dim * t.dtype.itemsize * (
            t.heads + t.heads_kv * t.seq_len_kv)


def _fa3_gqa_decode_fwd(test):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        # Q is (B, H, D) — add seq dim for flash_attn
        return flash_attn_func(q.unsqueeze(1), k, v).squeeze(1)

    return baseline_fn


def _flashinfer_gqa_decode_fwd(test, q, k, v):
    """Set up FlashInfer batched ragged prefill (seqlen_q=1). Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    # Q is (B, H, D) — single token per request
    B, H, D = q.shape
    Hkv = k.shape[2]
    Skv = k.shape[1]
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * Skv

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens_q, kv_indptr=cu_seqlens_k,
        num_qo_heads=H, num_kv_heads=Hkv, head_dim_qk=D,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        # Q: (B, H, D) → (B, H, D) — already packed (1 token per request)
        return wrapper.run(q, k.reshape(-1, Hkv, D), v.reshape(-1, Hkv, D))

    return run_fn


_GQA_DECODE_BENCH_PARAMS = [
    pytest.param(1, 32, 8, 8192, 128, torch.float16, False, id="single-batch-fp16"),
    pytest.param(4, 32, 4, 4096, 128, torch.bfloat16, False, id="bf16-mid-cache"),
    pytest.param(8, 64, 16, 8192, 128, torch.float16, False, id="multi-batch-fp16"),
]


@pytest.mark.parametrize("batch, heads, heads_kv, seq_len_kv, dim, dtype, tune", _GQA_DECODE_BENCH_PARAMS)
def test_gqa_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GqaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dtype)
    bm = GqaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupQueryAttentionDecodeWithKVCacheOp(batch, heads, heads_kv, seq_len_kv, dim, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_gqa_decode_fwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")
    else:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")

    fi_fn = _flashinfer_gqa_decode_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
