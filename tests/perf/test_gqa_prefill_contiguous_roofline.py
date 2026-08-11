import pytest

from tileops.manifest import load_workloads
from tileops.perf.formulas import gqa_prefill_with_kv_cache_fwd_roofline

pytestmark = pytest.mark.smoke

_OP = "GroupedQueryAttentionPrefillWithKVCacheFwdOp"


def test_gqa_prefill_contiguous_manifest_roofline() -> None:
    for workload in load_workloads(_OP):
        flops, nbytes = gqa_prefill_with_kv_cache_fwd_roofline(**workload)
        assert flops > 0
        assert nbytes > 0


def test_gqa_prefill_contiguous_roofline_counts_visible_scores() -> None:
    flops, nbytes = gqa_prefill_with_kv_cache_fwd_roofline(
        batch=2,
        seq_len_new=3,
        seqlen_kv=16,
        cache_lens=[4, 7],
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=True,
        dtype="float16",
    )
    visible_scores = (3 * 4 + 3 * 4 // 2) + (3 * 7 + 3 * 4 // 2)
    assert flops == 4 * 8 * visible_scores * 64
    assert nbytes > 0
