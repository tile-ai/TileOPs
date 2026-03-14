"""Benchmarks for 11 independent elementwise ops.

Profiles TileOPs vs PyTorch baselines across 3 shape tiers and all supported dtypes.
"""

from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase
from tileops.ops.elementwise import (
    AlibiOp,
    ClampOp,
    EluOp,
    HardtanhOp,
    LeakyReluOp,
    MaskedFillOp,
    NanToNumOp,
    PreluOp,
    SinusoidalOp,
    SoftplusOp,
    WhereOp,
)

_SHAPES = (262_144, 1_048_576, 4_000_000)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class UnaryBenchCase:
    """Test harness for unary-like ops."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)


class UnaryBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.test.n_total * self.test.dtype.itemsize * 2


# ---------------------------------------------------------------------------
# Unary-like ops: leaky_relu, elu, hardtanh, softplus, clamp, nan_to_num
# ---------------------------------------------------------------------------

def _unary_params():
    """Generate params: 6 ops × 3 shapes × 3 dtypes. Smoke = smallest shape + fp16."""
    params = []
    for op_name in ("leaky_relu", "elu", "hardtanh", "softplus", "clamp", "nan_to_num"):
        for shape in _SHAPES:
            for dtype in _DTYPES:
                mark = pytest.mark.smoke if (shape == _SHAPES[0] and dtype == torch.float16) else pytest.mark.full
                params.append(pytest.param(op_name, shape, dtype, marks=mark))
    return params


class UnaryIndependentBenchFixture(FixtureBase):
    PARAMS = [("op_name, n_total, dtype", _unary_params())]


_UNARY_OPS = {
    "leaky_relu": (LeakyReluOp, lambda x: F.leaky_relu(x, 0.01), {}),
    "elu": (EluOp, lambda x: F.elu(x, 1.0), {}),
    "hardtanh": (HardtanhOp, lambda x: F.hardtanh(x, -1.0, 1.0), {"min_val": -1.0, "max_val": 1.0}),
    "softplus": (SoftplusOp, lambda x: F.softplus(x, 1.0, 20.0), {}),
    "clamp": (ClampOp, lambda x: torch.clamp(x, -0.5, 0.5), {"min_val": -0.5, "max_val": 0.5}),
    "nan_to_num": (NanToNumOp, lambda x: torch.nan_to_num(x, 0.0, 1e4, -1e4), {}),
}


@UnaryIndependentBenchFixture
def test_unary_independent_bench(op_name: str, n_total: int, dtype: torch.dtype) -> None:
    op_cls, baseline_fn, extra_kwargs = _UNARY_OPS[op_name]
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = op_cls(N_total=n_total, dtype=dtype, **extra_kwargs)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# prelu (2 inputs: x + weight)
# ---------------------------------------------------------------------------

class PreluBenchCase:

    def __init__(self, n_total: int, num_channels: int, dtype: torch.dtype):
        self.n_total = n_total
        self.num_channels = num_channels
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        h = self.n_total // self.num_channels
        x = torch.randn(1, self.num_channels, h, device="cuda", dtype=self.dtype)
        weight = torch.randn(self.num_channels, device="cuda", dtype=self.dtype).abs() * 0.25
        return x, weight


class PreluBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.n_total * t.dtype.itemsize * 2 + t.num_channels * t.dtype.itemsize


def _prelu_params():
    params = []
    for shape in _SHAPES:
        for dtype in _DTYPES:
            mark = pytest.mark.smoke if (shape == _SHAPES[0] and dtype == torch.float16) else pytest.mark.full
            params.append(pytest.param(shape, 64, dtype, marks=mark))
    return params


class PreluBenchFixture(FixtureBase):
    PARAMS = [("n_total, num_channels, dtype", _prelu_params())]


@PreluBenchFixture
def test_prelu_bench(n_total: int, num_channels: int, dtype: torch.dtype) -> None:
    test = PreluBenchCase(n_total, num_channels, dtype)
    bm = PreluBenchmark(test)
    x, weight = test.gen_inputs()

    h = n_total // num_channels
    op = PreluOp(shape=(1, num_channels, h), dtype=dtype, num_channels=num_channels)
    result = bm.profile(op, x, weight)
    BenchmarkReport.record("prelu", locals(), result, tag="tileops")

    result_bl = bm.profile(F.prelu, x, weight)
    BenchmarkReport.record("prelu", locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# where (3 inputs: cond, x, y)
# ---------------------------------------------------------------------------

class WhereBenchCase:

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        cond = torch.rand(self.n_total, device="cuda") > 0.5
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        y = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        return cond, x, y


class WhereBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.n_total * (t.dtype.itemsize * 2 + 1) + t.n_total * t.dtype.itemsize


def _shape_dtype_params():
    params = []
    for shape in _SHAPES:
        for dtype in _DTYPES:
            mark = pytest.mark.smoke if (shape == _SHAPES[0] and dtype == torch.float16) else pytest.mark.full
            params.append(pytest.param(shape, dtype, marks=mark))
    return params


class WhereBenchFixture(FixtureBase):
    PARAMS = [("n_total, dtype", _shape_dtype_params())]


@WhereBenchFixture
def test_where_bench(n_total: int, dtype: torch.dtype) -> None:
    test = WhereBenchCase(n_total, dtype)
    bm = WhereBenchmark(test)
    cond, x, y = test.gen_inputs()

    op = WhereOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, cond, x, y)
    BenchmarkReport.record("where", locals(), result, tag="tileops")

    result_bl = bm.profile(torch.where, cond, x, y)
    BenchmarkReport.record("where", locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# masked_fill (2 inputs: x + mask)
# ---------------------------------------------------------------------------

class MaskedFillBenchCase:

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        mask = torch.rand(self.n_total, device="cuda") > 0.5
        return x, mask


class MaskedFillBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.n_total * (t.dtype.itemsize + 1) + t.n_total * t.dtype.itemsize


class MaskedFillBenchFixture(FixtureBase):
    PARAMS = [("n_total, dtype", _shape_dtype_params())]


@MaskedFillBenchFixture
def test_masked_fill_bench(n_total: int, dtype: torch.dtype) -> None:
    test = MaskedFillBenchCase(n_total, dtype)
    bm = MaskedFillBenchmark(test)
    x, mask = test.gen_inputs()

    op = MaskedFillOp(N_total=n_total, dtype=dtype, fill_value=-65000.0)
    result = bm.profile(op, x, mask)
    BenchmarkReport.record("masked_fill", locals(), result, tag="tileops")

    def baseline_fn(x, mask):
        return x.masked_fill(mask, -65000.0)

    result_bl = bm.profile(baseline_fn, x, mask)
    BenchmarkReport.record("masked_fill", locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# alibi & sinusoidal (generative: no input tensors)
# ---------------------------------------------------------------------------

class GenerativeBenchCase:

    def __init__(self, seq_len: int, dim: int, dtype: torch.dtype):
        self.n_total = seq_len * dim
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple:
        return ()


class GenerativeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.test.n_total * self.test.dtype.itemsize


def _generative_params():
    """alibi: 3 shapes × 3 dtypes, sinusoidal: 3 shapes × 3 dtypes."""
    alibi_shapes = [(512, 64), (2048, 64), (4096, 128)]
    sinusoidal_shapes = [(512, 256), (2048, 256), (4096, 512)]
    params = []
    for op_name, shapes in [("alibi", alibi_shapes), ("sinusoidal", sinusoidal_shapes)]:
        for seq_len, dim in shapes:
            for dtype in _DTYPES:
                mark = pytest.mark.smoke if (seq_len == shapes[0][0] and dtype == torch.float16) else pytest.mark.full
                params.append(pytest.param(op_name, seq_len, dim, dtype, marks=mark))
    return params


class GenerativeBenchFixture(FixtureBase):
    PARAMS = [("op_name, seq_len, dim, dtype", _generative_params())]


def _alibi_reference(seq_len: int, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    slopes = 2.0 ** (-(8.0 * torch.arange(1, num_heads + 1, device="cuda", dtype=torch.float32) / num_heads))
    bias = positions.unsqueeze(0) * slopes.unsqueeze(1)
    return bias.to(dtype)


def _sinusoidal_reference(seq_len: int, d_model: int, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(seq_len, device="cuda", dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, d_model, 2, device="cuda", dtype=torch.float32)
    angles = pos / torch.pow(10000.0, dim / d_model)
    pe = torch.zeros(seq_len, d_model, device="cuda", dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, :d_model // 2])
    return pe.to(dtype)


@GenerativeBenchFixture
def test_generative_bench(op_name: str, seq_len: int, dim: int, dtype: torch.dtype) -> None:
    test = GenerativeBenchCase(seq_len, dim, dtype)
    bm = GenerativeBenchmark(test)

    if op_name == "alibi":
        op = AlibiOp(seq_len=seq_len, num_heads=dim, dtype=dtype)

        def baseline_fn():
            return _alibi_reference(seq_len, dim, dtype)
    else:
        op = SinusoidalOp(seq_len=seq_len, d_model=dim, dtype=dtype)

        def baseline_fn():
            return _sinusoidal_reference(seq_len, dim, dtype)

    result = bm.profile(op)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
