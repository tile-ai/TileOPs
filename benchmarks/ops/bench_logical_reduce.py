"""Benchmarks for logical reduce ops (any, all, count_nonzero).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.reduction.all_op import AllFwdOp
from tileops.ops.reduction.any_op import AnyFwdOp
from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp
from workloads.ops.logical_reduce import AllWorkload, AnyWorkload, CountNonzeroWorkload

# ===================================================================
# Op name constants
# ===================================================================

_ANY_OP = "AnyFwdOp"
_ALL_OP = "AllFwdOp"
_COUNT_NONZERO_OP = "CountNonzeroFwdOp"


# ===================================================================
# Roofline helper
# ===================================================================


def _roofline_vars(workload) -> dict:
    """Extract roofline variables from a workload (shape + dtype -> M, N, elem_bytes)."""
    elem_bytes = torch.tensor([], dtype=workload.dtype).element_size()
    N = workload.shape[-1]
    M = 1
    for s in workload.shape[:-1]:
        M *= s
    return dict(M=M, N=N, elem_bytes=elem_bytes)


# ===================================================================
# Benchmark classes -- use manifest roofline for FLOP/memory counts
# ===================================================================


class AnyBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _ANY_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class AllBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _ALL_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class CountNonzeroBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _COUNT_NONZERO_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


# ===================================================================
# Manifest-driven parametrize helper
# ===================================================================


def _workloads_to_params(workloads):
    """Convert manifest workload dicts to pytest params: (shape, dtype)."""
    params = []
    for w in workloads:
        shape = tuple(w["x_shape"])
        label = w.get("label", "x".join(str(s) for s in shape))
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(
                shape, dtype,
                id=f"{label}-{dtype_str}",
            ))
    return params


# ===================================================================
# Any benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_ANY_OP)))
def test_any_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AnyWorkload(shape, dtype)
    bm = AnyBenchmark(test)
    inputs = test.gen_inputs()

    op = AnyFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.bool().any(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# All benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_ALL_OP)))
def test_all_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AllWorkload(shape, dtype)
    bm = AllBenchmark(test)
    inputs = test.gen_inputs()

    op = AllFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.bool().all(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# CountNonzero benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_COUNT_NONZERO_OP)))
def test_count_nonzero_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = CountNonzeroWorkload(shape, dtype)
    bm = CountNonzeroBenchmark(test)
    inputs = test.gen_inputs()

    op = CountNonzeroFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.count_nonzero(x, dim=-1).to(torch.int64)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
