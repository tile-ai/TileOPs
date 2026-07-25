"""Contract tests for :func:`benchmarks.benchmark_base.workloads_to_params`.

The helper serves single-tensor-input ops. The tensor input's name comes
from the manifest ``signature.inputs`` (never a hardcoded list); the
workload entry must declare ``{input}_shape`` and every other key must be a
declared signature param. Multi-input ops (e.g. attention families
declaring ``q_shape`` / ``kv_shape``) are out of scope and must surface a
clear ``KeyError``.
"""

from __future__ import annotations

import pytest

from benchmarks.benchmark_base import _workload_extra_params, workloads_to_params

pytestmark = pytest.mark.smoke


def _patch_manifest(monkeypatch, op_name, input_name, params=(), workloads=()):
    """Stub the manifest lookup and workload list for a synthetic op."""
    import benchmarks.benchmark_base as bb

    entry = {
        "signature": {
            "inputs": {input_name: {"dtype": "float16"}},
            "params": {p: {"type": "int"} for p in params},
        }
    }
    monkeypatch.setattr(bb, "load_manifest", lambda: {op_name: entry})
    monkeypatch.setattr(bb, "load_workloads", lambda op: list(workloads))


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


def test_shape_key_is_derived_from_signature_input_name(monkeypatch):
    """Ops whose PyTorch signature names the tensor ``input`` (e.g.
    ``F.dropout(input, ...)``) declare ``input_shape`` in the manifest; the
    key is derived from ``signature.inputs``, not from an allowlist."""
    _patch_manifest(
        monkeypatch, "FakeInputOp", "input", params=("p",),
        workloads=[{"input_shape": [1024, 4096], "p": 0.5,
                    "dtypes": ["float16"], "label": "drp"}],
    )

    params = workloads_to_params("FakeInputOp", include_extra=True)
    assert len(params) == 1
    shape, dtype, extra = params[0].values
    assert shape == (1024, 4096)
    assert extra == {"p": 0.5}, "input_shape must be stripped from extras"


def test_wrong_shape_key_raises_keyerror(monkeypatch):
    """A workload keyed ``x_shape`` while the signature input is ``input``
    is a manifest bug and must be rejected, not silently accepted."""
    _patch_manifest(
        monkeypatch, "FakeInputOp", "input",
        workloads=[{"x_shape": [8], "dtypes": ["float16"], "label": "bad"}],
    )

    with pytest.raises(KeyError, match="input_shape"):
        workloads_to_params("FakeInputOp")


def test_unknown_workload_key_raises_keyerror(monkeypatch):
    """A workload key that is neither the shape key, a harness key, nor a
    declared signature param (e.g. a typo) must fail at collection."""
    _patch_manifest(
        monkeypatch, "FakeOp", "x", params=("dim",),
        workloads=[{"x_shape": [8], "dtypes": ["float16"],
                    "label": "typo", "dmi": 0}],
    )

    with pytest.raises(KeyError, match="dmi"):
        workloads_to_params("FakeOp")


def test_multi_input_op_raises_keyerror():
    """GroupedQueryAttentionFwdOp declares three tensor inputs (q, k, v).
    The harness must surface a clear KeyError instead of silently binding
    the wrong tensor name."""
    with pytest.raises(KeyError, match="exactly one tensor input"):
        workloads_to_params("GroupedQueryAttentionFwdOp")


def test_extra_params_strips_reserved_keys_only():
    w = {
        "x_shape": [2048, 4096],
        "dtypes": ["bfloat16"],
        "label": "demo",
        "dim": 0,
        "keepdim": True,
    }
    extra = _workload_extra_params(w, "x_shape")
    assert extra == {"dim": 0, "keepdim": True}


def test_extra_params_strips_only_the_signature_shape_key():
    """Only the shape key derived from the signature is reserved; another
    ``*_shape`` key is surfaced as an op param (and would be rejected by
    ``workloads_to_params`` unless it is a declared signature param)."""
    w = {
        "x_shape": [2, 4],
        "dtypes": ["bfloat16"],
        "q_shape": [1, 2],
    }
    extra = _workload_extra_params(w, "x_shape")
    assert extra == {"q_shape": [1, 2]}


def test_keepdim_workload_is_surfaced_as_op_param(monkeypatch):
    """``keepdim`` on a workload entry must flow through as an op param so
    the benchmark baseline can see it.

    Uses a synthetic manifest + workload list so the assertion describes
    the helper's contract, not the contents or ordering of the ops
    manifest (`tileops/manifest/`).
    """
    _patch_manifest(
        monkeypatch, "FakeOp", "x", params=("dim", "keepdim"),
        workloads=[
            {"x_shape": [8, 16], "dtypes": ["bfloat16"], "label": "no-extras"},
            {
                "x_shape": [8, 16],
                "dtypes": ["bfloat16"],
                "label": "with-keepdim",
                "dim": 0,
                "keepdim": True,
            },
        ],
    )

    params = workloads_to_params("FakeOp", include_extra=True)
    extras_by_label = {p.id: p.values[2] for p in params}
    assert extras_by_label == {
        "no-extras-bfloat16": {},
        "with-keepdim-bfloat16": {"dim": 0, "keepdim": True},
    }
