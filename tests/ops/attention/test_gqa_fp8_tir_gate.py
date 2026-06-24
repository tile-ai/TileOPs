"""Focused test for the tvm.tir availability gate that skips the fp8 GQA / topk smoke cases.

Unlike test_gqa_fp8.py, this test is NOT skipped when tvm.tir is unavailable — it asserts the
gate contract itself (probe agrees with the kernel-module sentinel, and the sentinel raises a
targeted error), so the skip wiring is covered on CPU runners as well as the GPU fleet.
"""

import pytest

from tileops.kernels.attention import gqa_fwd_fp8
from tileops.testing.tir_compat import tir_available


@pytest.mark.smoke
def test_tir_gate_matches_gqa_sentinel() -> None:
    available = tir_available()
    is_sentinel = type(gqa_fwd_fp8.tir).__name__ == "_TirUnavailable"
    # The probe and the kernel module must agree: the sentinel is installed iff tvm.tir is missing.
    assert available != is_sentinel
    if is_sentinel:
        # Touching tir.* must raise a targeted, actionable ImportError (not a bare AttributeError).
        with pytest.raises(ImportError, match="tvm.tir is unavailable"):
            _ = gqa_fwd_fp8.tir.call_extern
    else:
        assert hasattr(gqa_fwd_fp8.tir, "call_extern")
