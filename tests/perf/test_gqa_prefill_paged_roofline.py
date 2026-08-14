import pytest

from tileops.manifest import load_workloads
from tileops.perf.formulas import (
    gqa_fwd_roofline,
    gqa_prefill_paged_with_kv_cache_fwd_roofline,
    gqa_prefill_varlen_fwd_roofline,
)

pytestmark = pytest.mark.smoke

_PAGED_PREFILL_OP = "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp"
_MIXED_QWEN_LABEL = "qwen35-9b-prefill-paged-fullattn-mixed-b8-p64-partial-rope64"
_BENCH_Q_LENS = [256, 512, 768, 1024, 384, 640, 896, 128]
_BENCH_CACHE_LENS = [4096, 8192, 16384, 32768, 12288, 24576, 30720, 2048]


def _workload_by_label(label: str) -> dict:
    for workload in load_workloads(_PAGED_PREFILL_OP):
        if workload.get("label") == label:
            return workload
    raise AssertionError(f"workload {label!r} not found")


def test_gqa_prefill_paged_mixed_manifest_matches_benchmark_lengths() -> None:
    workload = _workload_by_label(_MIXED_QWEN_LABEL)

    assert workload["q_lens"] == _BENCH_Q_LENS
    assert workload["cache_lens"] == _BENCH_CACHE_LENS
    assert sum(workload["q_lens"]) == workload["total_q"]


def test_gqa_prefill_paged_roofline_accepts_mixed_manifest_workload() -> None:
    workload = _workload_by_label(_MIXED_QWEN_LABEL)

    flops, nbytes = gqa_prefill_paged_with_kv_cache_fwd_roofline(**workload)

    assert flops > 0
    assert nbytes > 0


def test_gqa_prefill_varlen_native_fp8_roofline_counts_window_and_output_dtype() -> None:
    flops, nbytes = gqa_prefill_varlen_fwd_roofline(
        q_shape=(2, 4, 8),
        k_shape=(4, 2, 8),
        batch=1,
        max_seqlen_q=2,
        max_seqlen_kv=4,
        q_lens=[2],
        kv_lens=[4],
        is_causal=True,
        window_size_left=1,
        window_size_right=-1,
        dtype="float8_e4m3fn",
        output_dtype="float16",
    )

    # Bottom-right rows see {1,2} and {2,3}: four QK/PV pairs.
    assert flops == 4 * 4 * 4 * 8
    # FP8 Q/K/V + FP16 output + two cu-seqlens + three group-scale tensors.
    assert nbytes == 360


def test_gqa_prefill_dense_native_fp8_roofline_counts_rectangular_window() -> None:
    flops, nbytes = gqa_fwd_roofline(
        q_shape=(1, 2, 4, 8),
        kv_shape=(1, 4, 2, 8),
        is_causal=True,
        window_size_left=1,
        window_size_right=-1,
        input_dtype="float8_e4m3fn",
        dtypes=["float16"],
    )

    # Bottom-right rows see {1,2} and {2,3}: four QK/PV pairs.
    assert flops == 4 * 4 * 4 * 8
    # FP8 Q/K/V + FP16 output + three [B, H_kv] float32 scales.
    assert nbytes == 344


def test_gqa_prefill_paged_native_fp8_roofline_uses_cache_dtype_and_append_policy() -> None:
    flops, nbytes = gqa_prefill_paged_with_kv_cache_fwd_roofline(
        total_q=2,
        batch=1,
        heads=4,
        heads_kv=2,
        dim=8,
        max_pages_per_req=1,
        page_size=64,
        max_seqlen_q=2,
        q_lens=[2],
        cache_lens=[4],
        is_causal=True,
        window_size_left=1,
        window_size_right=-1,
        append_kv=False,
        dtype="float8_e4m3fn",
        cache_dtype="float8_e4m3fn",
        output_dtype="float16",
    )

    assert flops == 4 * 4 * 4 * 8
    assert nbytes == 424
