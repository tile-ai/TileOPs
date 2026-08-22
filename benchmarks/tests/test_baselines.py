"""Tests for :mod:`benchmarks.baselines`.

The import-order tests run in subprocesses: the failure they pin down is a process
abort that no ``except`` here would survive.
"""

import subprocess
import sys
from importlib.util import find_spec

import pytest
import torch

from benchmarks.baselines import (
    _FlagGemsImportOrder,
    assert_matches_reference,
    flaggems_op,
    reference_tolerance,
    vllm_op,
)

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
    assert reference_tolerance(torch.int32) == {}


def test_assert_matches_reference_compares_every_output_the_reference_returns():
    value = torch.ones(4)
    other = torch.zeros(4)

    def one_output(x):
        return x

    def two_outputs(x):
        return x, other

    # A baseline that returns more than the reference names is fine.
    assert_matches_reference(two_outputs, one_output, value)
    with pytest.raises(AssertionError):
        assert_matches_reference(lambda x: x + 1, one_output, value)

    # The output the reference names second is checked too: comparing only the
    # first would accept this baseline.
    assert_matches_reference(two_outputs, two_outputs, value)
    with pytest.raises(AssertionError, match="output 1"):
        assert_matches_reference(lambda x: (x, other + 1), two_outputs, value)

    # Returning fewer outputs than the reference is a mismatch, not a pass.
    with pytest.raises(AssertionError, match="output"):
        assert_matches_reference(one_output, two_outputs, value)
