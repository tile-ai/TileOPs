"""Benchmarks for 11 independent elementwise ops.

Profiles TileOPs vs PyTorch baselines using DNN-realistic 2-D shapes
(tokens x hidden_dim) across all supported dtypes.

Each row is timed against torch eager and the same reference through inductor,
including the two generative ops (alibi, sinusoidal), which take no inputs.
"""

from typing import Optional

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    fields,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise import (
    AlibiFwdOp,
    ClampFwdOp,
    SinusoidalFwdOp,
)
from workloads.elementwise import (
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


# Tensor-bound clamp. N_total is post-broadcast, i.e. product(out_shape).


def _clamp_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(input_shape, min_shape, max_shape, dtype)``; a row passes a bound
    exactly when it declares that bound's shape."""
    return (
        tuple(w["input_shape"]),
        tuple(w["min_shape"]) if "min_shape" in w else None,
        tuple(w["max_shape"]) if "max_shape" in w else None,
        dtype,
    )


def _clamp_marks(w: dict, dtype: torch.dtype, index: int) -> tuple:
    """The first row's fp16 case is the smoke case; every other case is full."""
    return (pytest.mark.smoke if index == 0 and dtype is torch.float16 else pytest.mark.full,)


@pytest.mark.parametrize(
    "input_shape, min_shape, max_shape, dtype",
    workload_params(load_workloads(ClampFwdOp), _clamp_args, marks=_clamp_marks),
)
def test_clamp_tensor_bench(
    input_shape: tuple,
    min_shape: Optional[tuple],
    max_shape: Optional[tuple],
    dtype: torch.dtype,
) -> None:
    test = TensorClampBenchCase(
        input_shape,
        dtype,
        min_shape=min_shape,
        max_shape=max_shape,
    )
    # gen_inputs yields only the bounds this row passes; widen to (x, min, max).
    x, *bounds = test.gen_inputs()
    t_min = bounds.pop(0) if min_shape is not None else None
    t_max = bounds.pop(0) if max_shape is not None else None

    op = ClampFwdOp()
    bm = ManifestBenchmark(op, test)

    def baseline_fn(x, t_min, t_max):
        return torch.clamp(x, t_min, t_max)

    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        t_min,
        t_max,
    )


# alibi & sinusoidal (generative: no input tensors)


class AlibiBenchFixture(FixtureBase):
    PARAMS = [
        (
            "seq_len, num_heads, dtype",
            workload_params(
                load_workloads(AlibiFwdOp),
                fields("seq_len", "num_heads", dtype_last=True),
                smoke_first=True,
            ),
        )
    ]


class SinusoidalBenchFixture(FixtureBase):
    PARAMS = [
        (
            "seq_len, d_model, dtype",
            workload_params(
                load_workloads(SinusoidalFwdOp),
                fields("seq_len", "d_model", dtype_last=True),
                smoke_first=True,
            ),
        )
    ]


def _alibi_reference(seq_len: int, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """Full ALiBi bias: (num_heads, seq_len, seq_len), bias[h,i,j] = -slope_h * |i-j|."""
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (S, S)
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device="cuda", dtype=torch.float32) / num_heads,
    )
    bias = -slopes[:, None, None] * dist[None, :, :]  # (H, S, S)
    return bias.to(dtype)


def _sinusoidal_reference(seq_len: int, d_model: int, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(seq_len, device="cuda", dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, d_model, 2, device="cuda", dtype=torch.float32)
    angles = pos / torch.pow(10000.0, dim / d_model)
    pe = torch.zeros(seq_len, d_model, device="cuda", dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return pe.to(dtype)


@AlibiBenchFixture
def test_alibi_bench(seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    op = AlibiFwdOp(seq_len=seq_len, num_heads=num_heads, dtype=dtype)
    workload = _GenerativeWorkload((num_heads, seq_len, seq_len), dtype)
    bm = ManifestBenchmark(op, workload)

    def baseline_fn():
        return _alibi_reference(seq_len, num_heads, dtype)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        }
    )


@SinusoidalBenchFixture
def test_sinusoidal_bench(seq_len: int, d_model: int, dtype: torch.dtype) -> None:
    op = SinusoidalFwdOp(seq_len=seq_len, d_model=d_model, dtype=dtype)
    workload = _GenerativeWorkload((seq_len, d_model), dtype)
    bm = ManifestBenchmark(op, workload)

    def baseline_fn():
        return _sinusoidal_reference(seq_len, d_model, dtype)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        }
    )
