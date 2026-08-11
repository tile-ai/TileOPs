"""Unit tests for benchmarks.benchmark_base.

Verifies that the generic ``BenchmarkBase`` / ``ManifestBenchmark`` accept
any duck-typed workload rather than requiring ``WorkloadBase`` inheritance.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    _attributed_latency_samples_ms,
    _bench_meta,
    _NativeCUPTIAttributionError,
    _ShiftingTensorPool,
    bench_kernel,
    workloads_to_params,
)

# Duck-typed test workloads


class _DuckShapeDtype:
    """Object with shape and dtype but NOT a WorkloadBase subclass."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeRooflineOp:
    """Minimal op-like object for ManifestBenchmark unit tests."""

    def __init__(self, roofline: tuple[int, int] = (128, 256)):
        self.calls = 0
        self._roofline = roofline

    def eval_roofline(self) -> tuple[int, int]:
        self.calls += 1
        return self._roofline


# ManifestBenchmark contract tests


@pytest.mark.smoke
def test_manifest_benchmark_accepts_duck_typed_workload():
    """ManifestBenchmark reads roofline off the op, never off the workload."""
    w = _DuckShapeDtype((4, 8, 1024), torch.float16)
    op = _FakeRooflineOp((123, 456))
    bm = ManifestBenchmark("TestOp", op, w)
    assert bm.workload is w
    assert bm.calculate_flops() == 123.0
    assert bm.calculate_memory() == 456.0
    assert op.calls == 1


@pytest.mark.smoke
def test_workloads_to_params_include_extra_propagates_dim():
    """When a workload entry carries ``dim``, ``include_extra=True`` should
    surface it in the pytest param triple.
    """
    # End-to-end with the manifest: include_extra=True must still yield
    # well-formed triples with the (shape, dtype, extra) mapping. The
    # contract being asserted is per-triple shape/dtype/extra typing; it
    # must not depend on the ordering of SumFwdOp.workloads (which is QA
    # curated and may be reordered without regressing the helper).
    triples = workloads_to_params("SumFwdOp", include_extra=True)
    assert len(triples) > 0
    assert any("dim" in p.values[2] for p in triples), (
        "at least one SumFwdOp workload must propagate a dim param"
    )
    for p in triples:
        shape, dtype, extra = p.values
        assert isinstance(shape, tuple)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(extra, dict)
    # A workload with no extras must yield an empty dict, not a missing slot.
    assert any(p.values[2] == {} for p in triples)


@pytest.mark.smoke
def test_manifest_benchmark_propagates_op_eval_error():
    w = _DuckShapeDtype((4, 8), torch.float16)

    class _BrokenOp:
        def eval_roofline(self):
            raise RuntimeError("shape not bound")

    bm = ManifestBenchmark("SumFwdOp", _BrokenOp(), w)
    with pytest.raises(RuntimeError, match="shape not bound"):
        bm.calculate_flops()


def test_multi_input_op_raises_keyerror():
    """Multi-input ops (q/k/v) raise instead of binding a wrong tensor."""
    with pytest.raises(KeyError, match="exactly one manifest tensor input"):
        workloads_to_params("GroupedQueryAttentionFwdOp")


def _kernel(name: str, start_ns: int, end_ns: int) -> dict:
    return {"kind": "kernel", "name": name, "start_ns": start_ns, "end_ns": end_ns}


def _seq(*names: str) -> tuple[str, ...]:
    """Discovered sequences hold activity identities, not bare kernel names."""
    return tuple(f"kernel:{name}" for name in names)


def test_attribution_excludes_prepare_and_keeps_the_operator_gap():
    trace = {
        "dropped": 0,
        "kernels": [
            _kernel("copy", 1_000, 2_000),
            _kernel("fill", 2_100, 3_000),
            _kernel("op-a", 4_000, 6_000),
            _kernel("op-b", 9_000, 10_000),
            _kernel("copy", 20_000, 21_000),
            _kernel("fill", 21_100, 22_000),
            _kernel("op-a", 23_000, 24_000),
            _kernel("op-b", 29_000, 31_000),
        ],
    }

    samples_ms = _attributed_latency_samples_ms(
        trace,
        _seq("op-a", "op-b"),
        n_repeat=2,
        expected_prepare_sequence=_seq("copy", "fill"),
    )

    # Operator envelopes are 6 us and 8 us. Prepare kernels are validated but
    # excluded, while the 3/5 us inter-kernel gaps remain part of op latency.
    assert samples_ms == pytest.approx([0.006, 0.008])


@pytest.mark.parametrize(
    "kernels, expected_sequence",
    [
        # CUPTI may publish two concurrent kernels in either start-time order.
        ([_kernel("b", 1_000, 3_000), _kernel("a", 1_500, 2_500)], _seq("a", "b")),
        # A repeated kernel name is matched by occurrence, not by identity.
        (
            [
                _kernel("a", 1_000, 2_000),
                _kernel("b", 1_500, 3_000),
                _kernel("a", 2_000, 2_500),
            ],
            _seq("a", "a", "b"),
        ),
    ],
)
def test_attribution_accepts_overlapping_activities_in_either_order(
    kernels, expected_sequence,
):
    samples_ms = _attributed_latency_samples_ms(
        {"dropped": 0, "kernels": kernels}, expected_sequence, n_repeat=1,
    )
    assert samples_ms == pytest.approx([0.002])


@pytest.mark.parametrize(
    "trace, expected_sequence, message",
    [
        # A dropped record makes the whole trial unattributable.
        ({"dropped": 1, "kernels": []}, _seq("a"), "dropped 1 records"),
        # One activity short of the discovered sequence.
        (
            {"dropped": 0, "kernels": [_kernel("a", 1_000, 2_000)]},
            _seq("a", "b"),
            "activity count does not match",
        ),
        # A dynamic path launched an extra kernel.
        (
            {
                "dropped": 0,
                "kernels": [
                    _kernel("a", 1_000, 2_000),
                    _kernel("b", 2_000, 3_000),
                    _kernel("extra", 3_000, 4_000),
                ],
            },
            _seq("a", "b"),
            # The observed sequence names the unexpected activity, so a CI
            # abort is diagnosable from the log alone.
            r"activity count does not match.*observed=.*kernel:extra",
        ),
        # Right count, different kernels.
        (
            {
                "dropped": 0,
                "kernels": [_kernel("a", 1_000, 2_000), _kernel("a", 2_000, 3_000)],
            },
            _seq("a", "b"),
            "attributed 0/1",
        ),
        # Serially reordered kernels are a real sequence change, not a
        # publication-order artifact.
        (
            {
                "dropped": 0,
                "kernels": [_kernel("b", 1_000, 2_000), _kernel("a", 2_000, 3_000)],
            },
            _seq("a", "b"),
            "attributed 0/1",
        ),
    ],
)
def test_attribution_fails_closed(trace, expected_sequence, message):
    with pytest.raises(_NativeCUPTIAttributionError, match=message):
        _attributed_latency_samples_ms(trace, expected_sequence, n_repeat=1)


def test_shifting_tensor_pool_preserves_layout_values_and_alignment():
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    pool = _ShiftingTensorPool((source, 7), total_iterations=3, seed=123)
    pointers = []

    for _ in range(3):
        shifted, scalar = pool.next_args()
        assert scalar == 7
        assert shifted.stride() == source.stride()
        torch.testing.assert_close(shifted, source)
        pointers.append(shifted.data_ptr())
        shifted.zero_()

    assert len(set(pointers)) == 3
    assert all(
        (pointer - pointers[0]) % _ShiftingTensorPool._POOL_ALIGNMENT == 0
        for pointer in pointers[1:]
    )
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    torch.testing.assert_close(source, expected)
    with pytest.raises(RuntimeError, match="ShiftingTensorPool exhausted"):
        pool.next_args()


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_fails_closed_by_default(monkeypatch):
    """A callable launching no CUDA kernel cannot be attributed by CUPTI."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0")
    with pytest.raises(RuntimeError, match="CUDA-events fallback is disabled"):
        bench_kernel(lambda: sum(range(64)))


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_falls_back_when_enabled(monkeypatch):
    """CUDA-event fallback remains available for local diagnosis."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "1")
    samples = bench_kernel(lambda: sum(range(64)))
    assert len(samples) >= 10 and all(s >= 0.0 for s in samples)
    assert _bench_meta.timing == "cuda-events"


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_runtime_error_propagates():
    """Genuine RuntimeErrors must reach the caller, not the fallback path."""
    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom)
