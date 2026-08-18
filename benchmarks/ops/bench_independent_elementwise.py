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
from workloads.elementwise import (
    Fp8MaskedFillBenchCase,
    Fp8UnaryBenchCase,
    Fp8WhereBenchCase,
    ShapedRandnWorkload,
    TensorClampBenchCase,
    _GenerativeWorkload,
)
from workloads.workload_base import FixtureBase

# DNN-realistic shapes: (tokens, hidden_dim).
# small=4096 (pow2), medium=10240 (pow2), large=11008 (non-pow2,
# LLaMA-7B intermediate) so each op exercises a non-pow2 shape.
_UNARY_SHAPES = [(1024, 4096), (1024, 10240), (1024, 11008)]
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


# Benchmark base classes


class UnaryBenchmark(BenchmarkBase[ShapedRandnWorkload]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.workload.n_total * self.workload.dtype.itemsize * 2


# Tensor-bound clamp. N_total is post-broadcast, i.e. product(out_shape).

_CLAMP_FWD_OP = "ClampFwdOp"


def _workloads_to_clamp_params(workloads: list) -> list:
    """Convert manifest workload dicts to clamp-bench pytest params.

    A row passes a bound exactly when it declares that bound's shape (R18.1).
    """
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        min_shape = tuple(w["min_shape"]) if "min_shape" in w else None
        max_shape = tuple(w["max_shape"]) if "max_shape" in w else None
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
    _workloads_to_clamp_params(load_workloads(_CLAMP_FWD_OP)),
)
def test_clamp_tensor_bench(
    input_shape: tuple,
    min_shape: Optional[tuple],
    max_shape: Optional[tuple],
    dtype: torch.dtype,
) -> None:
    test = TensorClampBenchCase(
        input_shape, dtype, min_shape=min_shape, max_shape=max_shape,
    )
    # gen_inputs yields only the bounds this row passes; widen to (x, min, max).
    x, *bounds = test.gen_inputs()
    t_min = bounds.pop(0) if min_shape is not None else None
    t_max = bounds.pop(0) if max_shape is not None else None

    op = ClampFwdOp(input=input_shape, min=min_shape, max=max_shape)
    bm = ManifestBenchmark(_CLAMP_FWD_OP, op, test)

    def baseline_fn(x, t_min, t_max):
        return torch.clamp(x, t_min, t_max)

    bm.compare(
        {"tileops": op, "torch": baseline_fn}, x, t_min, t_max,
        record_as=op, params=locals(),
    )


# alibi & sinusoidal (generative: no input tensors)

_ALIBI_OP = "AlibiFwdOp"
_SINUSOIDAL_OP = "SinusoidalFwdOp"


def _generative_params(workloads: list, keys: tuple) -> list:
    """Manifest workloads -> params; first workload smoke, rest full."""
    params = []
    for i, w in enumerate(workloads):
        values = [w[k] for k in keys]
        dtype = getattr(torch, w["dtypes"][0])
        mark = pytest.mark.smoke if i == 0 else pytest.mark.full
        params.append(pytest.param(*values, dtype, marks=mark,
                                   id=w.get("label", f"w{i}")))
    return params


class AlibiBenchFixture(FixtureBase):
    PARAMS = [("seq_len, num_heads, dtype",
               _generative_params(load_workloads(_ALIBI_OP),
                                  ("seq_len", "num_heads")))]


class SinusoidalBenchFixture(FixtureBase):
    PARAMS = [("seq_len, d_model, dtype",
               _generative_params(load_workloads(_SINUSOIDAL_OP),
                                  ("seq_len", "d_model")))]


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


@AlibiBenchFixture
def test_alibi_bench(seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    op = AlibiFwdOp(seq_len=seq_len, num_heads=num_heads, dtype=dtype)
    workload = _GenerativeWorkload((num_heads, seq_len, seq_len), dtype)
    bm = ManifestBenchmark(_ALIBI_OP, op, workload)

    bm.compare({"tileops": op, "torch-ref": lambda: _alibi_reference(seq_len, num_heads, dtype)}, record_as=op, params=locals())


@SinusoidalBenchFixture
def test_sinusoidal_bench(seq_len: int, d_model: int, dtype: torch.dtype) -> None:
    op = SinusoidalFwdOp(seq_len=seq_len, d_model=d_model, dtype=dtype)
    workload = _GenerativeWorkload((seq_len, d_model), dtype)
    bm = ManifestBenchmark(_SINUSOIDAL_OP, op, workload)

    bm.compare({"tileops": op, "torch-ref": lambda: _sinusoidal_reference(seq_len, d_model, dtype)}, record_as=op, params=locals())


# fp8 benchmarks: representative independent ops with e4m3fn / e5m2
# Baseline: PyTorch fp16-compute-then-cast (no native fp8 elementwise in PyTorch)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_UNSUPPORTED_FP8_SKIP = pytest.mark.skip(
    reason=(
        "TileOPs elementwise ops currently reject fp8 dtypes; "
        "benchmark is kept as an explicit unsupported case"
    )
)
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
    """Both fp8 dtypes per op (e5m2 takes the non-saturating cast path);
    shape swept on one op, since all three share one kernel."""
    ref_shape = _UNARY_SHAPES[0]
    params = []
    for op_name in ("leaky_relu", "elu", "clamp"):
        for dtype in _FP8_DTYPES:
            mark = (pytest.mark.smoke if dtype == torch.float8_e4m3fn
                    else pytest.mark.full)
            params.append(pytest.param(
                op_name, ref_shape, dtype, marks=[mark, _UNSUPPORTED_FP8_SKIP]))
    for shape in _UNARY_SHAPES[1:]:
        params.append(pytest.param(
            "leaky_relu", shape, torch.float8_e4m3fn,
            marks=[pytest.mark.full, _UNSUPPORTED_FP8_SKIP]))
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
        op = op_cls(input=shape, **extra_kwargs)
    else:
        op = op_cls(N_total=n_total, **extra_kwargs)

    # Baseline: PyTorch fp16 compute then cast back to fp8
    def baseline(x):
        return baseline_fn(x.to(torch.float16)).to(dtype)

    bm.compare({"tileops": op, "torch-ref": baseline}, *inputs, record_as=f'{op_name}_fp8', params=locals())


# fp8 where / masked_fill (selection ops - pass fp8 through directly)


class Fp8WhereBenchmark(BenchmarkBase[Fp8WhereBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # cond (1B) + fp8 x (1B) + fp8 y (1B) + fp8 out (1B)
        return self.workload.n_total * 4


class Fp8MaskedFillBenchmark(BenchmarkBase[Fp8MaskedFillBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # fp8 x (1B) + mask (1B) + fp8 out (1B)
        return self.workload.n_total * 3


def _fp8_selection_params():
    """Both fp8 dtypes per op at the reference shape; the selection kernels are
    shape-agnostic beyond total element count."""
    ref_shape = _UNARY_SHAPES[0]
    params = []
    for op_name in ("where", "masked_fill"):
        for dtype in _FP8_DTYPES:
            marks = [pytest.mark.smoke if dtype == torch.float8_e4m3fn
                     else pytest.mark.full]
            if op_name == "masked_fill":
                marks.append(_UNSUPPORTED_FP8_SKIP)
            params.append(pytest.param(op_name, ref_shape, dtype, marks=marks))
    return params


class Fp8SelectionBenchFixture(FixtureBase):
    PARAMS = [("op_name, shape, dtype", _fp8_selection_params())]


@Fp8SelectionBenchFixture
def test_fp8_selection_bench(
    op_name: str, shape: tuple, dtype: torch.dtype
) -> None:
    n_total = prod(shape)

    if op_name == "where":
        # WhereFwdOp declares no fp8 support, so record only the torch
        # baseline — instantiating it with an fp8 dtype would break the spec.
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

        op = MaskedFillScalarFwdOp(input=tuple(shape), mask=tuple(shape), value=-100.0)

        def baseline(x, mask):
            return x.to(torch.float16).masked_fill(mask, -100.0).to(dtype)

        bm.compare({"tileops": op, "torch-ref": baseline}, x, mask,
                   record_as=op, params=locals())
