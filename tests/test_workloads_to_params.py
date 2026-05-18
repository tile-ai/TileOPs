"""Contract tests for :func:`benchmarks.benchmark_base.workloads_to_params`.

The helper is scoped to single-input ops whose tensor input is named
``x``. Multi-input ops (e.g. attention families declaring ``q_shape`` /
``kv_shape``) are out of scope and must surface a clear ``KeyError``.
"""

from __future__ import annotations

import pytest

from benchmarks.benchmark_base import _workload_extra_params, workloads_to_params

pytestmark = pytest.mark.smoke


def test_single_input_ops_are_supported():
    params = workloads_to_params("SumFwdOp")
    assert params, "SumFwdOp must yield at least one workload"


def test_single_input_with_extra_params():
    params = workloads_to_params("SumFwdOp", include_extra=True)
    # Confirm each pytest.param carries (shape, dtype, extra) where extra
    # is a dict (possibly empty) of op params.
    for p in params:
        assert len(p.values) == 3
        _, _, extra = p.values
        assert isinstance(extra, dict)


def test_multi_input_op_raises_keyerror():
    """GroupedQueryAttentionFwdOp declares q_shape/kv_shape, not x_shape.
    The harness must surface a clear KeyError instead of silently binding
    the wrong tensor name."""
    with pytest.raises(KeyError, match="x_shape"):
        workloads_to_params("GroupedQueryAttentionFwdOp")


def test_extra_params_strips_reserved_keys_only():
    w = {
        "x_shape": [2048, 4096],
        "dtypes": ["bfloat16"],
        "label": "demo",
        "dim": 0,
        "keepdim": True,
    }
    extra = _workload_extra_params(w)
    assert extra == {"dim": 0, "keepdim": True}


def test_extra_params_preserves_non_x_shape_keys():
    """Any non-reserved key (even ``q_shape``) is surfaced as an op param;
    the harness does not special-case other ``*_shape`` keys."""
    w = {
        "x_shape": [2, 4],
        "dtypes": ["bfloat16"],
        "q_shape": [1, 2],
    }
    extra = _workload_extra_params(w)
    assert extra == {"q_shape": [1, 2]}


def test_keepdim_workload_is_surfaced_as_op_param(monkeypatch):
    """``keepdim`` on a workload entry must flow through as an op param so
    the benchmark baseline can see it.

    Uses a synthetic workload list (patched in place of ``load_workloads``)
    so the assertion describes the helper's contract, not the contents or
    ordering of the ops manifest (`tileops/manifest/`).
    """
    synthetic = [
        {"x_shape": [8, 16], "dtypes": ["bfloat16"], "label": "no-extras"},
        {
            "x_shape": [8, 16],
            "dtypes": ["bfloat16"],
            "label": "with-keepdim",
            "dim": 0,
            "keepdim": True,
        },
    ]
    import benchmarks.benchmark_base as bb

    monkeypatch.setattr(bb, "load_workloads", lambda op: synthetic)

    params = workloads_to_params("FakeOp", include_extra=True)
    extras_by_label = {p.id: p.values[2] for p in params}
    assert extras_by_label == {
        "no-extras-bfloat16": {},
        "with-keepdim-bfloat16": {"dim": 0, "keepdim": True},
    }
