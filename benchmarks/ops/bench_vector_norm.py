"""Benchmarks for vector norm ops (l1_norm, l2_norm, inf_norm).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.reduction.inf_norm import InfNormFwdOp
from tileops.ops.reduction.l1_norm import L1NormFwdOp
from tileops.ops.reduction.l2_norm import L2NormFwdOp
from workloads.ops.vector_norm import InfNormTest, L1NormTest, L2NormTest

# ===================================================================
# Op name constants
# ===================================================================

_L1_NORM_OP = "L1NormFwdOp"
_L2_NORM_OP = "L2NormFwdOp"
_INF_NORM_OP = "InfNormFwdOp"


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
# Benchmark classes — use manifest roofline for FLOP/memory counts
# ===================================================================


class L1NormBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _L1_NORM_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class L2NormBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _L2_NORM_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class InfNormBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _INF_NORM_OP, **_roofline_vars(self.workload))
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
# L1 Norm benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_L1_NORM_OP)))
def test_l1_norm_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = L1NormTest(shape, dtype)
    bm = L1NormBenchmark(test)
    inputs = test.gen_inputs()

    op = L1NormFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# L2 Norm benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_L2_NORM_OP)))
def test_l2_norm_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = L2NormTest(shape, dtype)
    bm = L2NormBenchmark(test)
    inputs = test.gen_inputs()

    op = L2NormFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Inf Norm benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_INF_NORM_OP)))
def test_inf_norm_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = InfNormTest(shape, dtype)
    bm = InfNormBenchmark(test)
    inputs = test.gen_inputs()

    op = InfNormFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
