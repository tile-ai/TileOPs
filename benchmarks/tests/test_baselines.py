"""Tests for :mod:`benchmarks.baselines` and the GEMM baseline domains in `bench_gemm`.

The import-order tests run in subprocesses: the failure they pin down is a process
abort that no ``except`` here would survive.
"""

import subprocess
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

from benchmarks import baselines
from benchmarks.baselines import (
    _FlagGemsImportOrder,
    compiled_reference,
    flaggems_op,
    vllm_op,
)
from workloads.numerics import Exact, Request, reference_tolerance, verify

# flag_gems refuses to import without a device, so its tests need one.
_BOTH_LIBRARIES = (
    find_spec("flag_gems") is not None
    and find_spec("vllm") is not None
    and torch.cuda.is_available()
)
_needs_both = pytest.mark.skipif(
    not _BOTH_LIBRARIES, reason="needs both flag_gems and vllm installed, on a GPU"
)


def _verify_one(subject, inputs, *, reference, evidence, subject_inputs=None):
    """Check one call the way ``verify`` checks each of its requests."""
    args = inputs if subject_inputs is None else subject_inputs
    return verify(
        reference, inputs, evidence=evidence, requests={"subject": Request(subject, args)}
    )["subject"]


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
        _verify_one(two_outputs, (value,), reference=one_output, evidence=Exact())
    with pytest.raises(AssertionError):
        _verify_one(lambda x: x + 1, (value,), reference=one_output, evidence=Exact())
    _verify_one(two_outputs, (value,), reference=two_outputs, evidence=Exact())
    with pytest.raises(AssertionError, match="Tensor-likes"):
        _verify_one(lambda x: (x, other + 1), (value,), reference=two_outputs, evidence=Exact())
    with pytest.raises(ValueError, match="outputs"):
        _verify_one(one_output, (value,), reference=two_outputs, evidence=Exact())


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
    backward = _torch_gqa_bwd(workload, *inputs[1:4])
    _verify_one(
        backward, inputs, reference=workload.ref_program, evidence=workload.verification(*inputs)
    )


@pytest.mark.parametrize(
    "device, raised",
    [("cuda", RuntimeError), ("privateuseone", pytest.skip.Exception)],
    ids=["cuda-fails", "elsewhere-skips"],
)
def test_an_absent_selected_library_fails_the_row_on_cuda_and_skips_it_elsewhere(
    monkeypatch, device, raised
):
    """The runner image installs every selected library for CUDA; elsewhere absent means
    not there, and the row says which library."""
    monkeypatch.setattr(baselines, "bench_device", lambda: torch.device(device, 0))
    with pytest.raises(raised, match="nolib is not installed"):
        baselines._resolve("tileops_no_such_library", "op", "nolib")


def test_an_installed_library_that_fails_to_import_is_broken_on_any_device(monkeypatch):
    """Only the library's own package being absent makes it unavailable."""
    monkeypatch.setattr(baselines, "bench_device", lambda: torch.device("privateuseone", 0))
    with pytest.raises(RuntimeError, match="fails to import"):
        baselines.baseline_module("torch.tileops_no_such_module", "torch")


def test_a_library_under_a_shared_namespace_is_absent_when_its_own_package_is(monkeypatch):
    """``nvidia`` may come from another NVIDIA package; cuDNN is ``nvidia.cudnn``."""
    monkeypatch.setattr(baselines, "bench_device", lambda: torch.device("privateuseone", 0))
    with pytest.raises(pytest.skip.Exception, match="nolib is not installed"):
        baselines.baseline_module(
            "torch.tileops_no_such_package", "nolib", package="torch.tileops_no_such_package"
        )


def test_a_cuda_only_library_cannot_run_off_cuda_even_where_it_is_installed(monkeypatch):
    """Such a library's kernels run on CUDA alone, so only the device decides."""
    monkeypatch.setattr(baselines, "bench_device", lambda: torch.device("privateuseone", 0))
    with pytest.raises(pytest.skip.Exception, match="flashinfer runs on cuda"):
        baselines.baseline_module("torch", "flashinfer")


@pytest.mark.parametrize(
    "device, module, library, name, found",
    [
        ("privateuseone", "torch", "flashinfer", "Tensor", None),
        ("cuda", "tileops_no_such_library", "nolib", "Tensor", None),
        ("cuda", "torch", "torch", "Tensor", torch.Tensor),
        ("cuda", "torch", "torch", "no_such_name", None),
    ],
    ids=["cuda-only-elsewhere", "absent", "present", "missing-name"],
)
def test_a_preferred_baseline_that_cannot_run_here_drops_its_tag(
    monkeypatch, device, module, library, name, found
):
    monkeypatch.setattr(baselines, "bench_device", lambda: torch.device(device, 0))
    assert baselines.optional_baseline(module, library, name) is found


@pytest.mark.parametrize(
    "n, k, group, expected",
    [
        pytest.param(128, 256, 128, None, id="smallest-complete-reorder"),
        pytest.param(7168, 3456, 128, "K divisible by 256", id="partial-reorder-row"),
        pytest.param(16, 256, 128, "N divisible by 128", id="partial-n-tile"),
        pytest.param(128, 256, 64, "group_size 128", id="unsupported-group"),
    ],
)
def test_w4a8_baseline_domain_rejects_partial_reorder_rows(monkeypatch, n, k, group, expected):
    """The benchmark refuses unsafe library preparation before launching a CUDA kernel."""
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(bench_gemm, "get_sm_version", lambda: 90)
    reason = bench_gemm._vllm_w4a8_unsupported_reason(n, k, group)
    assert reason is None if expected is None else expected in reason


@pytest.mark.parametrize(
    "n, k, group_size, expected",
    [
        pytest.param(64, 128, 32, None, id="group32"),
        pytest.param(64, 128, 128, None, id="group128"),
        pytest.param(64, 256, 256, "positive group sizes", id="unsupported-group"),
        pytest.param(16, 128, 128, "N % 64", id="partial-n"),
        pytest.param(64, 0, 128, "positive N", id="empty-k"),
        pytest.param(64, 129, 128, "K % 128", id="partial-k"),
    ],
)
def test_marlin_baseline_reads_the_callers_group_size(monkeypatch, n, k, group_size, expected):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(bench_gemm, "get_sm_version", lambda: 90)
    reason = bench_gemm._marlin_w4a16_unsupported_reason(n, k, group_size)
    assert reason is None if expected is None else expected in reason


def _deepgemm_fp8_on_sm90(monkeypatch, m, n, k, out_dtype, mode):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(bench_gemm, "get_sm_version", lambda: 90)
    monkeypatch.setattr(bench_gemm, "deepgemm_op", lambda name: None)
    workload = SimpleNamespace(m=m, n=n, k=k, out_dtype=out_dtype, scale_mode=mode, bias=False)
    return bench_gemm._deepgemm_fp8(workload)


@pytest.mark.parametrize(
    "m, n, k, out_dtype, mode",
    [
        pytest.param(64, 64, 144, torch.bfloat16, "per_tensor", id="partial-scale-group"),
        pytest.param(64, 68, 144, torch.float16, "per_tensor", id="fp32-intermediate"),
        pytest.param(64, 68, 144, torch.bfloat16, "block128", id="element-scales"),
        pytest.param(32, 64, 144, torch.bfloat16, "per_tensor", id="small-m-1d2d"),
    ],
)
def test_deepgemm_fp8_baseline_accepts_its_aligned_tail_domain(
    monkeypatch, m, n, k, out_dtype, mode
):
    assert callable(_deepgemm_fp8_on_sm90(monkeypatch, m, n, k, out_dtype, mode))


@pytest.mark.parametrize(
    "m, n, k, out_dtype, mode, error",
    [
        pytest.param(
            64, 68, 144, torch.bfloat16, "per_tensor", "divisible", id="bf16-output-stride"
        ),
        pytest.param(64, 64, 129, torch.bfloat16, "per_tensor", "divisible", id="operand-stride"),
        pytest.param(32, 64, 144, torch.bfloat16, "block128", "M <= 32", id="small-m-1d1d"),
    ],
)
def test_deepgemm_fp8_baseline_refuses_rows_it_cannot_run(
    monkeypatch, m, n, k, out_dtype, mode, error
):
    with pytest.raises(ValueError, match=error):
        _deepgemm_fp8_on_sm90(monkeypatch, m, n, k, out_dtype, mode)


@pytest.mark.parametrize(
    "m, n, k, trans_a, trans_b, supported",
    [
        pytest.param(4, 64, 136, False, True, True, id="nt"),
        pytest.param(4, 64, 129, False, True, False, id="nt-operand-stride"),
        pytest.param(8, 64, 129, True, False, True, id="tn-unaligned-logical-k"),
        pytest.param(7, 64, 128, True, False, False, id="tn-operand-stride"),
        pytest.param(8, 65, 128, False, False, False, id="output-stride"),
        pytest.param(0, 65, 129, True, False, True, id="empty-fast-path"),
    ],
)
def test_quack_baseline_checks_storage_strides(m, n, k, trans_a, trans_b, supported):
    from benchmarks.ops import bench_gemm

    workload = SimpleNamespace(
        m=m, n=n, k=k, trans_a=trans_a, trans_b=trans_b, dtype=torch.bfloat16
    )
    assert (bench_gemm._quack_gemm_unsupported_reason(workload) is None) == supported


@pytest.mark.parametrize(
    "n, k, supported",
    [(32, 128, True), (31, 128, False), (32, 129, False)],
)
def test_flashinfer_low_latency_preparation_domain(monkeypatch, n, k, supported):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(bench_gemm, "get_sm_version", lambda index: 100)
    sm100 = SimpleNamespace(type="cuda", index=0)  # lint: device-literal allow: no tensor
    assert (
        bench_gemm._flashinfer_fp8_per_tensor_unsupported_reason(sm100, n, k) is None
    ) == supported


def test_flashinfer_low_latency_refuses_a_non_cuda_device_before_reading_arch(monkeypatch):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(
        bench_gemm, "get_sm_version", lambda index: pytest.fail("read CUDA arch for a CPU call")
    )
    reason = bench_gemm._flashinfer_fp8_per_tensor_unsupported_reason(torch.device("cpu"), 32, 128)
    assert "runs on cuda" in reason
