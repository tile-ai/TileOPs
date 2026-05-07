import pytest

from tileops.manifest import load_workloads
from tileops.perf.formulas import gqa_prefill_paged_with_kv_cache_fwd_roofline

pytestmark = pytest.mark.smoke

_PAGED_PREFILL_OP = "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp"
_MIXED_QWEN_LABEL = "qwen35-9b-prefill-paged-fullattn-mixed-b8-p64-partial-rope64-fp16"
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

    roofline = gqa_prefill_paged_with_kv_cache_fwd_roofline(**workload)

    assert roofline["flops"] > 0
    assert roofline["bytes"] > 0
