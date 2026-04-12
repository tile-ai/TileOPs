from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MultiHeadAttentionDecodeWithKVCacheFwdOp
from workloads.ops.attention.mha_decode import MhaDecodeTest


class _MhaDecodeTestBaseline(MhaDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.transpose(1, 2)  # [B, H, S_q, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output


class MhaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len_q * t.seq_len_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # Q: batch * seq_len_q * heads * dim
        # K, V: batch * seq_len_kv * heads * dim
        # Output: batch * seq_len_q * heads * dim
        return (t.batch * t.heads * (2 * t.seq_len_q + 2 * t.seq_len_kv) * t.dim *
                t.dtype.itemsize)


def _fa3_mha_decode_fwd(test):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(q, k, v)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_mha_decode_fwd(test, q, k, v):
    """Set up FlashInfer batched prefill wrapper. Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * Sq
    cu_seqlens_k = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * Skv

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens_q, kv_indptr=cu_seqlens_k,
        num_qo_heads=H, num_kv_heads=H, head_dim_qk=D,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D), k.reshape(-1, H, D), v.reshape(-1, H, D),
        ).reshape(B, Sq, H, D)

    return run_fn


_MHA_DECODE_BENCH_PARAMS = [
    pytest.param(1, 32, 128, 8192, 128, torch.float16, True, id="fp16-long-cache"),
    pytest.param(1, 32, 128, 8192, 128, torch.bfloat16, True, id="bf16-long-cache"),
    pytest.param(1, 32, 128, 5, 128, torch.float16, True, id="short-kv-tail"),
]


@pytest.mark.parametrize("b, h, s_q, s_kv, d, dtype, tune", _MHA_DECODE_BENCH_PARAMS)
def test_mha_decode_bench(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                          tune: bool) -> None:
    test = _MhaDecodeTestBaseline(b, h, s_q, s_kv, d, dtype)
    bm = MhaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionDecodeWithKVCacheFwdOp(b, h, s_q, s_kv, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_mha_decode_fwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    fi_fn = _flashinfer_mha_decode_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
