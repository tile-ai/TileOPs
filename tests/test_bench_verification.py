"""The pytest and benchmark adapters must consume the same workload contract."""

from types import SimpleNamespace

import pytest
import torch

from workloads.numerics import Exact

pytestmark = pytest.mark.smoke


def _workload():
    return SimpleNamespace(
        gen_inputs=lambda: (torch.ones(1),),
        ref_program=lambda x: x,
        verification=lambda *inputs: Exact(atol=0, rtol=0),
        arguments=dict,
    )


@pytest.mark.parametrize("consumer", ["test", "benchmark"])
def test_both_consumers_reject_the_same_wrong_result(consumer, monkeypatch):
    from tileops.ops.elementwise import AbsFwdOp

    if consumer == "test":
        from tests.workload_test_base import TestBase

        with pytest.raises(AssertionError):
            TestBase.check(_workload(), AbsFwdOp(), torch.ones(1), runs=lambda x: x + 0.001)
    else:
        from benchmarks import api
        from benchmarks._cases import Entry

        monkeypatch.setattr(
            api, "bench_kernel", lambda *_a, **_k: pytest.fail("timed wrong output")
        )

        # The op's own row is right; the wrong implementation alone must be rejected.
        def right(op, case):
            return api.Implementation(run=lambda x: x + 0)

        call = SimpleNamespace(signature=SimpleNamespace(name="AbsFwdOp"))
        case = api.Case("probe", call, Entry(lambda _call: _workload(), binder=right))
        with pytest.raises(AssertionError):
            api.Runner(AbsFwdOp(), case).compare({"wrong": lambda x: x + 0.001})


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
        pytest.param(64, 128, 64, None, id="group64"),
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


@pytest.mark.parametrize(
    "n, k, out_dtype, mode, bias, supported",
    [
        pytest.param(64, 144, torch.bfloat16, "per_tensor", False, True, id="partial-scale-group"),
        pytest.param(68, 144, torch.float16, "per_tensor", False, True, id="fp32-intermediate"),
        pytest.param(68, 144, torch.bfloat16, "block128", False, True, id="element-scales"),
        pytest.param(68, 144, torch.bfloat16, "per_tensor", False, False, id="bf16-output-stride"),
        pytest.param(64, 129, torch.bfloat16, "per_tensor", False, False, id="operand-stride"),
    ],
)
def test_deepgemm_fp8_baseline_preserves_its_aligned_tail_domain(
    monkeypatch, n, k, out_dtype, mode, bias, supported
):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(bench_gemm, "get_sm_version", lambda: 90)
    monkeypatch.setattr(bench_gemm, "deepgemm_op", lambda name: None)
    workload = SimpleNamespace(n=n, k=k, out_dtype=out_dtype, scale_mode=mode, bias=bias)
    if supported:
        assert callable(bench_gemm._deepgemm_fp8(workload))
    else:
        with pytest.raises(ValueError, match="divisible"):
            bench_gemm._deepgemm_fp8(workload)


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
@pytest.mark.cuda_only
def test_flashinfer_low_latency_preparation_domain(monkeypatch, n, k, supported):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(bench_gemm, "get_sm_version", lambda index: 100)
    assert (
        bench_gemm._flashinfer_fp8_per_tensor_unsupported_reason(
            SimpleNamespace(type="cuda", index=0), n, k
        )
        is None
    ) == supported


def test_flashinfer_low_latency_refuses_a_non_cuda_device_before_reading_arch(monkeypatch):
    from benchmarks.ops import bench_gemm

    monkeypatch.setattr(
        bench_gemm, "get_sm_version", lambda index: pytest.fail("read CUDA arch for a CPU call")
    )
    reason = bench_gemm._flashinfer_fp8_per_tensor_unsupported_reason(torch.device("cpu"), 32, 128)
    assert "runs on cuda" in reason
