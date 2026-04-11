"""Benchmarks for the 8 basic reduce ops.

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.reduction.reduce import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)
from workloads.ops.reduce import (
    AmaxTest,
    AminTest,
    MeanTest,
    ProdTest,
    StdTest,
    SumTest,
    VarMeanTest,
    VarTest,
)

# ===================================================================
# Op name constants
# ===================================================================

_SUM_OP = "SumFwdOp"
_MEAN_OP = "MeanFwdOp"
_AMAX_OP = "AmaxFwdOp"
_AMIN_OP = "AminFwdOp"
_PROD_OP = "ProdFwdOp"
_STD_OP = "StdFwdOp"
_VAR_OP = "VarFwdOp"
_VAR_MEAN_OP = "VarMeanFwdOp"


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


class SumBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _SUM_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class MeanBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _MEAN_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class AmaxBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _AMAX_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class AminBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _AMIN_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class ProdBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _PROD_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class StdBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _STD_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class VarBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _VAR_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class VarMeanBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _VAR_MEAN_OP, **_roofline_vars(self.workload))
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
# Sum benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_SUM_OP)))
def test_sum_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = SumTest(shape, dtype)
    bm = SumBenchmark(test)
    inputs = test.gen_inputs()

    op = SumFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().sum(dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Mean benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_MEAN_OP)))
def test_mean_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = MeanTest(shape, dtype)
    bm = MeanBenchmark(test)
    inputs = test.gen_inputs()

    op = MeanFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().mean(dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Amax benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_AMAX_OP)))
def test_amax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AmaxTest(shape, dtype)
    bm = AmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = AmaxFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.amax(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Amin benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_AMIN_OP)))
def test_amin_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AminTest(shape, dtype)
    bm = AminBenchmark(test)
    inputs = test.gen_inputs()

    op = AminFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.amin(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Prod benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_PROD_OP)))
def test_prod_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = ProdTest(shape, dtype)
    bm = ProdBenchmark(test)
    inputs = test.gen_inputs()

    op = ProdFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().prod(dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Std benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_STD_OP)))
def test_std_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = StdTest(shape, dtype)
    bm = StdBenchmark(test)
    inputs = test.gen_inputs()

    op = StdFwdOp(dtype=dtype, correction=1)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().std(dim=-1, correction=1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Var benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_VAR_OP)))
def test_var_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = VarTest(shape, dtype)
    bm = VarBenchmark(test)
    inputs = test.gen_inputs()

    op = VarFwdOp(dtype=dtype, correction=1)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().var(dim=-1, correction=1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# VarMean benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_VAR_MEAN_OP)))
def test_var_mean_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = VarMeanTest(shape, dtype)
    bm = VarMeanBenchmark(test)
    inputs = test.gen_inputs()

    op = VarMeanFwdOp(dtype=dtype, correction=1)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        v = x.float().var(dim=-1, correction=1).to(x.dtype)
        m = x.float().mean(dim=-1).to(x.dtype)
        return (v, m)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
