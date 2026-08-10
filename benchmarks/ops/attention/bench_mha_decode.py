import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params, mha_decode_args
from tileops.manifest import load_workloads
from tileops.ops import MultiHeadAttentionDecodeWithKVCacheFwdOp
from workloads.attention.mha import MhaDecodeWorkload

_OP_NAME = "MultiHeadAttentionDecodeWithKVCacheFwdOp"


def _fa3_mha_decode_fwd(test):
    """Return FA3 forward baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(q, k, v)
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_mha_decode_fwd(test, q, k, v):
    """Set up FlashInfer batched prefill wrapper. Returns callable or None."""
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
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


_MHA_DECODE_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), mha_decode_args)


@pytest.mark.parametrize("b, h, s_q, s_kv, d, dtype, tune", _MHA_DECODE_BENCH_PARAMS)
def test_mha_decode_bench(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                          tune: bool) -> None:
    test = MhaDecodeWorkload(b, h, s_q, s_kv, d, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionDecodeWithKVCacheFwdOp(b, h, s_q, s_kv, d, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
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
