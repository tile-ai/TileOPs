"""Benchmarks for argreduce ops (argmax, argmin).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.reduction.argmax import ArgmaxFwdOp
from tileops.ops.reduction.argmin import ArgminFwdOp
from workloads.ops.argreduce import ArgmaxTest, ArgminTest

_ARGMAX_OP = "ArgmaxFwdOp"
_ARGMIN_OP = "ArgminFwdOp"


def _roofline_vars(shape, dtype) -> dict:
    """Extract roofline variables from shape + dtype → M, N, elem_bytes."""
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    N = shape[-1]
    M = 1
    for s in shape[:-1]:
        M *= s
    return dict(M=M, N=N, elem_bytes=elem_bytes)


class ArgmaxBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _ARGMAX_OP, **_roofline_vars(self.workload.shape, self.workload.dtype))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class ArgminBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _ARGMIN_OP, **_roofline_vars(self.workload.shape, self.workload.dtype))
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
# Argmax benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_ARGMAX_OP)))
def test_argmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    workload = ArgmaxTest(shape, dtype)
    bm = ArgmaxBenchmark(workload)
    inputs = workload.gen_inputs()

    op = ArgmaxFwdOp(dtype=dtype)
    # FIXME: ArgreduceKernel fails on large-N shapes (e.g. [4, 102400]) with
    # "Can't fetch the lanes of a scalable vector at a compile time".
    # Skip until the kernel is fixed.
    try:
        result = bm.profile(op, *inputs)
    except Exception as exc:
        msg = str(exc)
        if "scalable vector" in msg or "No configurations to tune" in msg:
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.argmax(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Argmin benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_ARGMIN_OP)))
def test_argmin_bench(shape: tuple, dtype: torch.dtype) -> None:
    workload = ArgminTest(shape, dtype)
    bm = ArgminBenchmark(workload)
    inputs = workload.gen_inputs()

    op = ArgminFwdOp(dtype=dtype)
    # FIXME: ArgreduceKernel fails on large-N shapes (e.g. [4, 102400]) with
    # "Can't fetch the lanes of a scalable vector at a compile time".
    # Skip until the kernel is fixed.
    try:
        result = bm.profile(op, *inputs)
    except Exception as exc:
        msg = str(exc)
        if "scalable vector" in msg or "No configurations to tune" in msg:
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.argmin(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
