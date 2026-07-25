"""Contract tests for signature-derived workload binding, on the real manifest."""

from __future__ import annotations

import pytest

from benchmarks.benchmark_base import workloads_to_params

pytestmark = pytest.mark.smoke


def test_single_input_ops_are_supported():
    params = workloads_to_params("SumFwdOp")
    assert params, "SumFwdOp must yield at least one workload"


def test_single_input_with_extra_params():
    params = workloads_to_params("SumFwdOp", include_extra=True)
    # Each pytest.param carries (shape, dtype, extra) with extra a dict.
    for p in params:
        assert len(p.values) == 3
        shape, _, extra = p.values
        assert isinstance(shape, tuple)
        assert isinstance(extra, dict)


def test_multi_input_op_raises_keyerror():
    """Multi-input ops (q/k/v) raise instead of binding a wrong tensor."""
    with pytest.raises(KeyError, match="exactly one manifest tensor input"):
        workloads_to_params("GroupedQueryAttentionFwdOp")
