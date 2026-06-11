"""Benchmarks for 11 independent elementwise ops.

Profiles TileOPs vs PyTorch baselines using DNN-realistic 2-D shapes
(tokens x hidden_dim) across all supported dtypes.
"""

from math import prod
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.elementwise import (
    AlibiFwdOp,
    ClampFwdOp,
    ClampMaxFwdOp,
    ClampMinFwdOp,
    ClampScalarFwdOp,
    EluFwdOp,
    HardtanhFwdOp,
    LeakyReluFwdOp,
    MaskedFillScalarFwdOp,
    NanToNumFwdOp,
    PreluFwdOp,
    SinusoidalFwdOp,
    SoftplusFwdOp,
    WhereFwdOp,
)
from workloads.workload_base import FixtureBase

# DNN-realistic shapes: (tokens, hidden_dim).
# small=4096 (pow2), medium=10240 (pow2), large=11008 (non-pow2,
# LLaMA-7B intermediate) so each op exercises a non-pow2 shape.
_UNARY_SHAPES = [(1024, 4096), (1024, 10240), (1024, 11008)]
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


# ---------------------------------------------------------------------------
# Benchmark base classes
# ---------------------------------------------------------------------------

class UnaryBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


class UnaryBenchmark(BenchmarkBase[UnaryBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.workload.n_total * self.workload.dtype.itemsize * 2


# ---------------------------------------------------------------------------
# Unary-like ops: leaky_relu, elu, hardtanh, softplus, clamp, nan_to_num
# ---------------------------------------------------------------------------

def _unary_params():
    params = []
    for op_name in ("leaky_relu", "elu", "hardtanh", "softplus", "clamp", "nan_to_num"):
        for shape in _UNARY_SHAPES:
            for dtype in _DTYPES:
                mark = pytest.mark.smoke if (shape == _UNARY_SHAPES[0] and dtype == torch.float16) else pytest.mark.full
                params.append(pytest.param(op_name, shape, dtype, marks=mark))
    return params


class UnaryIndependentBenchFixture(FixtureBase):
    PARAMS = [("op_name, shape, dtype", _unary_params())]


_UNARY_OPS = {
    "leaky_relu": (LeakyReluFwdOp, lambda x: F.leaky_relu(x, 0.01), {}),
    "elu": (EluFwdOp, lambda x: F.elu(x, 1.0), {}),
    "hardtanh": (HardtanhFwdOp, lambda x: F.hardtanh(x, -1.0, 1.0), {"min_val": -1.0, "max_val": 1.0}),
    "softplus": (SoftplusFwdOp, lambda x: F.softplus(x, 1.0, 20.0), {}),
    "clamp": (ClampScalarFwdOp, lambda x: torch.clamp(x, -0.5, 0.5), {"min": -0.5, "max": 0.5}),
    "nan_to_num": (NanToNumFwdOp, lambda x: torch.nan_to_num(x, 0.0, 1e4, -1e4), {}),
}


@UnaryIndependentBenchFixture
def test_unary_independent_bench(op_name: str, shape: tuple, dtype: torch.dtype) -> None:
    n_total = prod(shape)
    op_cls, baseline_fn, extra_kwargs = _UNARY_OPS[op_name]
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    if op_cls.__name__ == "ClampScalarFwdOp":
        op = op_cls(input=shape, dtype=dtype, **extra_kwargs)
    else:
        op = op_cls(N_total=n_total, dtype=dtype, **extra_kwargs)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# Tensor-bound clamp ops (manifest-driven): ClampFwdOp, ClampMinFwdOp,
# ClampMaxFwdOp. Workload shapes are loaded from tileops/manifest/ so the
# bench coverage stays aligned with the spec (post-broadcast N_total ==
# product(out_shape)). FLOP/byte counts come from each op's eval_roofline().
# ---------------------------------------------------------------------------

_CLAMP_FWD_OP = "ClampFwdOp"
_CLAMP_MIN_OP = "ClampMinFwdOp"
_CLAMP_MAX_OP = "ClampMaxFwdOp"


class TensorClampBenchCase:
    """Workload adapter for Tensor-bound clamp ops.

    Holds the post-broadcast output shape so :class:`ManifestBenchmark`
    can read a single ``n_total`` while the bench builds per-operand
    tensors from the manifest-declared ``input_shape`` / ``min_shape`` /
    ``max_shape`` keys.
    """

    def __init__(
        self,
        input_shape: tuple,
        dtype: torch.dtype,
        min_shape: Optional[tuple] = None,
        max_shape: Optional[tuple] = None,
    ):
        self.input_shape = input_shape
        self.min_shape = min_shape
        self.max_shape = max_shape
        broadcast_args = [input_shape]
        if min_shape is not None:
            broadcast_args.append(min_shape)
        if max_shape is not None:
            broadcast_args.append(max_shape)
        self.shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        tensors: list[torch.Tensor] = [x]
        if self.min_shape is not None:
            tensors.append(
                torch.randn(self.min_shape, device="cuda", dtype=self.dtype) - 0.5
            )
        if self.max_shape is not None:
            tensors.append(
                torch.randn(self.max_shape, device="cuda", dtype=self.dtype) + 0.5
            )
        return tuple(tensors)


def _workloads_to_clamp_params(
    workloads: list, *, needs_min: bool, needs_max: bool,
) -> list:
    """Convert manifest workload dicts to clamp-bench pytest params."""
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        min_shape = tuple(w["min_shape"]) if needs_min else None
        max_shape = tuple(w["max_shape"]) if needs_max else None
        label = w.get("label", "x".join(str(s) for s in input_shape))
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            # Smoke = first workload + fp16; everything else is full-mode.
            mark = (
                pytest.mark.smoke
                if (idx == 0 and dtype is torch.float16)
                else pytest.mark.full
            )
            params.append(
                pytest.param(
                    input_shape, min_shape, max_shape, dtype,
                    marks=mark, id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "input_shape, min_shape, max_shape, dtype",
    _workloads_to_clamp_params(
        load_workloads(_CLAMP_FWD_OP), needs_min=True, needs_max=True,
    ),
)
def test_clamp_tensor_bench(
    input_shape: tuple,
    min_shape: tuple,
    max_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = TensorClampBenchCase(input_shape, dtype, min_shape=min_shape, max_shape=max_shape)
    x, t_min, t_max = test.gen_inputs()

    op = ClampFwdOp(input=input_shape, min=min_shape, max=max_shape, dtype=dtype)
    bm = ManifestBenchmark(_CLAMP_FWD_OP, op, test)
    result = bm.profile(op, x, t_min, t_max)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x, t_min, t_max):
        return torch.clamp(x, t_min, t_max)

    result_bl = bm.profile(baseline_fn, x, t_min, t_max)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize(
    "input_shape, min_shape, _max_shape, dtype",
    _workloads_to_clamp_params(
        load_workloads(_CLAMP_MIN_OP), needs_min=True, needs_max=False,
    ),
)
def test_clamp_min_bench(
    input_shape: tuple,
    min_shape: tuple,
    _max_shape: Optional[tuple],
    dtype: torch.dtype,
) -> None:
    test = TensorClampBenchCase(input_shape, dtype, min_shape=min_shape)
    x, t_min = test.gen_inputs()

    op = ClampMinFwdOp(input=input_shape, min=min_shape, dtype=dtype)
    bm = ManifestBenchmark(_CLAMP_MIN_OP, op, test)
    result = bm.profile(op, x, t_min)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x, t_min):
        return torch.maximum(x, t_min)

    result_bl = bm.profile(baseline_fn, x, t_min)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize(
    "input_shape, _min_shape, max_shape, dtype",
    _workloads_to_clamp_params(
        load_workloads(_CLAMP_MAX_OP), needs_min=False, needs_max=True,
    ),
)
def test_clamp_max_bench(
    input_shape: tuple,
    _min_shape: Optional[tuple],
    max_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = TensorClampBenchCase(input_shape, dtype, max_shape=max_shape)
    x, t_max = test.gen_inputs()

    op = ClampMaxFwdOp(input=input_shape, max=max_shape, dtype=dtype)
    bm = ManifestBenchmark(_CLAMP_MAX_OP, op, test)
    result = bm.profile(op, x, t_max)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x, t_max):
        return torch.minimum(x, t_max)

    result_bl = bm.profile(baseline_fn, x, t_max)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# prelu (2 inputs: x + weight)
# ---------------------------------------------------------------------------

_PRELU_SHAPES = [(1024, 128), (1024, 4096), (1024, 10240), (1024, 11008)]


class PreluBenchCase:
    def __init__(self, shape: tuple, num_channels: int, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.num_channels = num_channels
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape, device="cuda", dtype=self.dtype)
        weight = torch.randn(self.num_channels, device="cuda", dtype=self.dtype).abs() * 0.25
        return x, weight


class PreluBenchmark(BenchmarkBase[PreluBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return t.n_total * t.dtype.itemsize * 2 + t.num_channels * t.dtype.itemsize


def _prelu_params():
    params = []
    for tokens, hidden in _PRELU_SHAPES:
        for dtype in _DTYPES:
            mark = pytest.mark.smoke if (hidden == _PRELU_SHAPES[0][1] and dtype == torch.float16) else pytest.mark.full
            params.append(pytest.param((tokens, hidden), hidden, dtype, marks=mark))
    return params


class PreluBenchFixture(FixtureBase):
    PARAMS = [("shape, num_channels, dtype", _prelu_params())]


@PreluBenchFixture
def test_prelu_bench(shape: tuple, num_channels: int, dtype: torch.dtype) -> None:
    test = PreluBenchCase(shape, num_channels, dtype)
    bm = PreluBenchmark(test)
    x, weight = test.gen_inputs()

    # PReLU shape convention: (batch, channels, spatial)
    prelu_shape = (1, num_channels, shape[0])
    n_total = prod(shape)
    op = PreluFwdOp(shape=prelu_shape, dtype=dtype, num_channels=num_channels)
    result = bm.profile(op, x.reshape(prelu_shape), weight)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(F.prelu, x.reshape(prelu_shape), weight)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# where (3 inputs: cond, x, y)
# ---------------------------------------------------------------------------

class WhereBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        cond = torch.rand(self.shape, device="cuda") > 0.5
        x = torch.randn(self.shape, device="cuda", dtype=self.dtype)
        y = torch.randn(self.shape, device="cuda", dtype=self.dtype)
        return cond, x, y


class WhereBenchmark(BenchmarkBase[WhereBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return t.n_total * (t.dtype.itemsize * 2 + 1) + t.n_total * t.dtype.itemsize


def _shape_dtype_params(shapes):
    params = []
    for shape in shapes:
        for dtype in _DTYPES:
            mark = pytest.mark.smoke if (shape == shapes[0] and dtype == torch.float16) else pytest.mark.full
            params.append(pytest.param(shape, dtype, marks=mark))
    return params


class WhereBenchFixture(FixtureBase):
    PARAMS = [("shape, dtype", _shape_dtype_params(_UNARY_SHAPES))]


@WhereBenchFixture
def test_where_bench(shape: tuple, dtype: torch.dtype) -> None:
    n_total = prod(shape)
    test = WhereBenchCase(shape, dtype)
    bm = WhereBenchmark(test)
    cond, x, y = test.gen_inputs()

    op = WhereFwdOp(condition=tuple(shape), input=tuple(shape), other=tuple(shape), dtype=dtype)
    result = bm.profile(op, cond, x, y)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(torch.where, cond, x, y)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# masked_fill (2 inputs: x + mask)
# ---------------------------------------------------------------------------

class MaskedFillBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape, device="cuda", dtype=self.dtype)
        mask = torch.rand(self.shape, device="cuda") > 0.5
        return x, mask


class MaskedFillBenchmark(BenchmarkBase[MaskedFillBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return t.n_total * (t.dtype.itemsize + 1) + t.n_total * t.dtype.itemsize


class MaskedFillBenchFixture(FixtureBase):
    PARAMS = [("shape, dtype", _shape_dtype_params(_UNARY_SHAPES))]


@MaskedFillBenchFixture
def test_masked_fill_bench(shape: tuple, dtype: torch.dtype) -> None:
    n_total = prod(shape)
    test = MaskedFillBenchCase(shape, dtype)
    bm = MaskedFillBenchmark(test)
    x, mask = test.gen_inputs()

    op = MaskedFillScalarFwdOp(input=tuple(shape), mask=tuple(shape), value=-65000.0, dtype=dtype)
    result = bm.profile(op, x, mask)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x, mask):
        return x.masked_fill(mask, -65000.0)

    result_bl = bm.profile(baseline_fn, x, mask)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


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


class GenerativeBenchmark(BenchmarkBase[GenerativeBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.workload.n_total * self.workload.dtype.itemsize


def _generative_params():
    alibi_shapes = [(512, 64), (2048, 64), (4096, 128)]
    sinusoidal_shapes = [(512, 256), (2048, 300), (4096, 512)]
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
    """Full ALiBi bias: (num_heads, seq_len, seq_len), bias[h,i,j] = -slope_h * |i-j|."""
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (S, S)
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device="cuda", dtype=torch.float32) / num_heads,
    )
    bias = (-slopes[:, None, None] * dist[None, :, :])  # (H, S, S)
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

    if op_name == "alibi":
        # ALiBi outputs (num_heads, seq_len, seq_len); override n_total.
        test.n_total = dim * seq_len * seq_len
        shape = (dim, seq_len, seq_len)
        op = AlibiFwdOp(seq_len=seq_len, num_heads=dim, dtype=dtype)

        def baseline_fn():
            return _alibi_reference(seq_len, dim, dtype)
    else:
        # Sinusoidal positional embedding: (seq_len, d_model).
        shape = (seq_len, dim)
        op = SinusoidalFwdOp(seq_len=seq_len, d_model=dim, dtype=dtype)

        def baseline_fn():
            return _sinusoidal_reference(seq_len, dim, dtype)

    bm = GenerativeBenchmark(test)
    result = bm.profile(op)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch-ref")


# ---------------------------------------------------------------------------
# fp8 benchmarks: representative independent ops with e4m3fn / e5m2
# Baseline: PyTorch fp16-compute-then-cast (no native fp8 elementwise in PyTorch)
# ---------------------------------------------------------------------------

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_UNSUPPORTED_FP8_SKIP = pytest.mark.skip(
    reason=(
        "TileOPs elementwise ops currently reject fp8 dtypes; "
        "benchmark is kept as an explicit unsupported case"
    )
)


class Fp8UnaryBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0
        return (x.to(self.dtype),)


class Fp8UnaryBenchmark(BenchmarkBase[Fp8UnaryBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # fp8 in (1B) + fp8 out (1B) per element
        return self.workload.n_total * 2


_FP8_UNARY_OPS = {
    "leaky_relu": (LeakyReluFwdOp, lambda x: F.leaky_relu(x, 0.01), {}),
    "elu": (EluFwdOp, lambda x: F.elu(x, 1.0), {}),
    "clamp": (ClampScalarFwdOp, lambda x: torch.clamp(x, -0.5, 0.5), {"min": -0.5, "max": 0.5}),
}


def _fp8_unary_params():
    params = []
    for op_name in ("leaky_relu", "elu", "clamp"):
        for shape in _UNARY_SHAPES:
            for dtype in _FP8_DTYPES:
                mark = (
                    pytest.mark.smoke
                    if (shape == _UNARY_SHAPES[0] and dtype == torch.float8_e4m3fn)
                    else pytest.mark.full
                )
                params.append(
                    pytest.param(
                        op_name, shape, dtype,
                        marks=[mark, _UNSUPPORTED_FP8_SKIP],
                    )
                )
    return params


class Fp8UnaryIndependentBenchFixture(FixtureBase):
    PARAMS = [("op_name, shape, dtype", _fp8_unary_params())]


@Fp8UnaryIndependentBenchFixture
def test_fp8_unary_independent_bench(
    op_name: str, shape: tuple, dtype: torch.dtype
) -> None:
    n_total = prod(shape)
    op_cls, baseline_fn, extra_kwargs = _FP8_UNARY_OPS[op_name]
    test = Fp8UnaryBenchCase(shape, dtype)
    bm = Fp8UnaryBenchmark(test)
    inputs = test.gen_inputs()

    if op_cls.__name__ == "ClampScalarFwdOp":
        op = op_cls(input=shape, dtype=dtype, **extra_kwargs)
    else:
        op = op_cls(N_total=n_total, dtype=dtype, **extra_kwargs)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(f"{op_name}_fp8", locals(), result, tag="tileops")

    # Baseline: PyTorch fp16 compute then cast back to fp8
    def baseline(x):
        return baseline_fn(x.to(torch.float16)).to(dtype)

    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(f"{op_name}_fp8", locals(), result_bl, tag="torch-ref")


# ---------------------------------------------------------------------------
# fp8 where / masked_fill (selection ops - pass fp8 through directly)
# ---------------------------------------------------------------------------


class Fp8WhereBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        cond = torch.rand(self.shape, device="cuda") > 0.5
        x = (torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0).to(
            self.dtype
        )
        y = (torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0).to(
            self.dtype
        )
        return cond, x, y


class Fp8WhereBenchmark(BenchmarkBase[Fp8WhereBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # cond (1B) + fp8 x (1B) + fp8 y (1B) + fp8 out (1B)
        return self.workload.n_total * 4


class Fp8MaskedFillBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = (torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0).to(
            self.dtype
        )
        mask = torch.rand(self.shape, device="cuda") > 0.5
        return x, mask


class Fp8MaskedFillBenchmark(BenchmarkBase[Fp8MaskedFillBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # fp8 x (1B) + mask (1B) + fp8 out (1B)
        return self.workload.n_total * 3


def _fp8_selection_params():
    params = []
    for op_name in ("where", "masked_fill"):
        for shape in _UNARY_SHAPES:
            for dtype in _FP8_DTYPES:
                mark = (
                    pytest.mark.smoke
                    if (shape == _UNARY_SHAPES[0] and dtype == torch.float8_e4m3fn)
                    else pytest.mark.full
                )
                marks = [mark]
                if op_name == "masked_fill":
                    marks.append(_UNSUPPORTED_FP8_SKIP)
                params.append(pytest.param(op_name, shape, dtype, marks=marks))
    return params


class Fp8SelectionBenchFixture(FixtureBase):
    PARAMS = [("op_name, shape, dtype", _fp8_selection_params())]


@Fp8SelectionBenchFixture
def test_fp8_selection_bench(
    op_name: str, shape: tuple, dtype: torch.dtype
) -> None:
    n_total = prod(shape)

    if op_name == "where":
        # WhereFwdOp manifest does not declare fp8 dtype support, so this
        # branch records only the torch.where baseline. Instantiating
        # WhereFwdOp with an fp8 dtype here would violate the manifest
        # contract; the manifest is the spec, not the code.
        test = Fp8WhereBenchCase(shape, dtype)
        bm = Fp8WhereBenchmark(test)
        cond, x, y = test.gen_inputs()

        # torch.where supports fp8 natively (pure selection, no arithmetic)
        def baseline(cond, x, y):
            return torch.where(cond, x, y)

        result_bl = bm.profile(baseline, cond, x, y)
        BenchmarkReport.record("where_fp8", locals(), result_bl, tag="torch-ref")
    else:
        test = Fp8MaskedFillBenchCase(shape, dtype)
        bm = Fp8MaskedFillBenchmark(test)
        x, mask = test.gen_inputs()

        op = MaskedFillScalarFwdOp(input=tuple(shape), mask=tuple(shape), value=-100.0, dtype=dtype)
        result = bm.profile(op, x, mask)
        BenchmarkReport.record(op, locals(), result, tag="tileops")

        def baseline(x, mask):
            return x.to(torch.float16).masked_fill(mask, -100.0).to(dtype)

        result_bl = bm.profile(baseline, x, mask)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
