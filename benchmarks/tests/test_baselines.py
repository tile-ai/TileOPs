"""Tests for :mod:`benchmarks.baselines`.

The import-order tests run in subprocesses: the failure they pin down is a process
abort that no ``except`` here would survive.
"""

import subprocess
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

from benchmarks.baselines import (
    _FlagGemsImportOrder,
    assert_output_spec,
    compiled_reference,
    flaggems_op,
    vllm_op,
)
from workloads.numerics import Exact, reference_tolerance, verify

# flag_gems refuses to import without a device, so its tests need one.
_BOTH_LIBRARIES = (
    find_spec("flag_gems") is not None
    and find_spec("vllm") is not None
    and torch.cuda.is_available()
)
_needs_both = pytest.mark.skipif(
    not _BOTH_LIBRARIES, reason="needs both flag_gems and vllm installed, on a GPU"
)


def _run(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_guard_is_armed_on_import():
    """Importing the module installs the finder, and installing twice is a no-op."""
    finders = [f for f in sys.meta_path if isinstance(f, _FlagGemsImportOrder)]
    assert len(finders) == 1


@_needs_both
def test_flag_gems_before_vllm_aborts_without_the_guard():
    """The hazard the guard exists for. If this ever passes, drop the guard."""
    result = _run("import flag_gems.ops; import vllm._custom_ops; print('survived')")
    # A negative code is death by signal — the abort itself, not a Python error
    # that happens to leave a traceback.
    assert result.returncode < 0, (
        f"flag_gems before vllm exited {result.returncode} rather than aborting; if it "
        "no longer aborts, the import-order guard in benchmarks.baselines and the vllm "
        f"import it costs are no longer needed. stderr: {result.stderr[-500:]}"
    )
    assert "survived" not in result.stdout


@_needs_both
def test_the_guard_makes_either_import_order_safe():
    """With the guard armed, importing flag_gems first is no longer fatal."""
    result = _run(
        "import benchmarks.baselines; "
        "import flag_gems.ops; "
        "import vllm._custom_ops; "
        "print('survived')"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "survived" in result.stdout


@pytest.mark.skipif(find_spec("vllm") is None, reason="needs vllm installed")
def test_vllm_op_reports_the_order_instead_of_aborting(monkeypatch):
    """A process that lost the race gets a message, not a segfault."""
    monkeypatch.setitem(sys.modules, "flag_gems", object())
    monkeypatch.delitem(sys.modules, "vllm._custom_ops", raising=False)
    with pytest.raises(RuntimeError, match="imported before vllm"):
        vllm_op("rms_norm")


@pytest.mark.skipif(find_spec("flag_gems") is None, reason="needs flag_gems installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="flag_gems needs a device")
def test_flaggems_op_refuses_a_pointwise_entry_point():
    """A pointwise kernel would abort the process on the timing loop's second call."""
    with pytest.raises(RuntimeError, match="LibEntry"):
        flaggems_op("exp")
    # A reduction goes through a different launcher and resolves.
    assert flaggems_op("sum_dim") is not None


def test_reference_tolerance_follows_the_dtype():
    assert reference_tolerance(torch.float16) == {"rtol": 1e-3, "atol": 1e-3}
    # The other branch: an unlisted dtype leaves assert_close on its own defaults.
    assert reference_tolerance(torch.int32) == {"atol": 0.0, "rtol": 0.0}


def test_shared_verification_rejects_undeclared_extra_outputs():
    value = torch.ones(4)
    other = torch.zeros(4)

    def one_output(x):
        return x

    def two_outputs(x):
        return (x, other)

    with pytest.raises(ValueError, match="outputs"):
        verify(two_outputs, (value,), reference=one_output, evidence=Exact())
    with pytest.raises(AssertionError):
        verify(lambda x: x + 1, (value,), reference=one_output, evidence=Exact())
    verify(two_outputs, (value,), reference=two_outputs, evidence=Exact())
    with pytest.raises(AssertionError, match="Tensor-likes"):
        verify(lambda x: (x, other + 1), (value,), reference=two_outputs, evidence=Exact())
    with pytest.raises(ValueError, match="outputs"):
        verify(one_output, (value,), reference=two_outputs, evidence=Exact())


def test_compiled_reference_refuses_a_reference_dynamo_splits():
    """A row tagged torch-compile fails rather than time part of itself eager."""

    def one_graph(x):
        return x * 2 + 1

    def data_dependent(x):
        # A branch on a value dynamo cannot know at trace time splits the graph.
        if x.sum() > 0:
            return x * 2
        return x * 3

    value = torch.ones(4)
    assert torch.equal(compiled_reference(one_graph)(value), one_graph(value))
    with pytest.raises(AssertionError, match="graph"):
        compiled_reference(data_dependent)(value)


def test_assert_output_spec_rejects_another_dtype_or_shape():
    spec = SimpleNamespace(shape=(2, 3), dtype="float16")

    assert_output_spec(torch.zeros(2, 3, dtype=torch.float16), spec, "tag")
    with pytest.raises(AssertionError, match="float32"):
        assert_output_spec(torch.zeros(2, 3), spec, "tag")
    with pytest.raises(AssertionError, match=r"\(3, 3\)"):
        assert_output_spec(torch.zeros(3, 3, dtype=torch.float16), spec, "tag")
    with pytest.raises(AssertionError, match="not a tensor"):
        assert_output_spec((torch.zeros(2, 3, dtype=torch.float16),), spec, "tag")


def test_compiled_reference_warmup_updates_state_once():
    """Graph validation must not secretly execute the stateful reference first."""

    def advance(state):
        return state.add_(1)

    state = torch.ones(4)
    result = compiled_reference(advance)(state)
    torch.testing.assert_close(state, torch.full_like(state, 2))
    torch.testing.assert_close(result, state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA SDPA backward")
def test_gqa_backward_adapter_survives_input_restore_and_returns_bshd():
    from benchmarks.ops.bench_gqa import _torch_gqa_bwd
    from workloads.attention.gqa.bwd import GQABwdWorkload

    workload = GQABwdWorkload(1, 4, 2, 32, 64, True, torch.float16)
    inputs = workload.gen_inputs()
    backward = _torch_gqa_bwd(workload, *inputs[:3])
    verify(
        backward, inputs, reference=workload.ref_program, evidence=workload.verification(*inputs)
    )
