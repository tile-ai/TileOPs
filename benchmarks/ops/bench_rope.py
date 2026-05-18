"""Benchmarks for 5 RoPE variants (1D layout).

Profiles TileOPs RoPE vs manual PyTorch reference on DNN-realistic shapes.
Tests neox variant as representative; all variants share the same kernel.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.rope import RopeNeoxOp
from workloads.workload_base import FixtureBase

# DNN-realistic: (seq_len, head_dim) — typical attention head sizes.
# Includes a non-pow2 seq_len (3000) to exercise tail handling.
_SHAPES = [(2048, 64), (2048, 128), (4096, 128), (3000, 128)]
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class RopeBenchCase:
    def __init__(self, shape: tuple[int, int], dtype: torch.dtype):
        self.shape = shape
        self.seq_len, self.head_dim = shape
        self.n_total = self.seq_len * self.head_dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(*self.shape, device="cuda", dtype=self.dtype),)


class RopeBenchmark(BenchmarkBase[RopeBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        # 4 ops per element: 2 muls + 1 add + 1 negate/select
        return self.workload.n_total * 4

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = t.dtype.itemsize
        # Read x + cos + sin + write y
        cos_sin_elems = t.seq_len * (t.head_dim // 2) * 2
        return (2 * t.n_total + cos_sin_elems) * elem


def _precompute_rope_neox_cos_sin(
    seq_len: int, head_dim: int, dtype: torch.dtype, base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables (matches RopeNeoxOp caching behavior)."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half))
    t = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_full = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(dtype)
    sin_full = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(dtype)
    return cos_full, sin_full


def _rope_neox_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply neox RoPE rotation with pre-computed cos/sin."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def _rope_params():
    params = []
    smoke_shape = _SHAPES[0]
    for shape in _SHAPES:
        for dtype in _DTYPES:
            mark = (
                pytest.mark.smoke
                if (shape == smoke_shape and dtype == torch.float16)
                else pytest.mark.full
            )
            params.append(pytest.param(
                shape, dtype,
                id=f"{shape[0]}x{shape[1]}-{dtype}",
                marks=mark,
            ))
    return params


class RopeBenchFixture(FixtureBase):
    PARAMS = [("shape, dtype", _rope_params())]


@RopeBenchFixture
def test_rope_bench(shape: tuple[int, int], dtype: torch.dtype) -> None:
    test = RopeBenchCase(shape, dtype)
    bm = RopeBenchmark(test)
    (x,) = test.gen_inputs()

    seq_len, head_dim = shape
    op = RopeNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    cos, sin = _precompute_rope_neox_cos_sin(seq_len, head_dim, dtype)

    def baseline_fn(x):
        return _rope_neox_apply(x, cos, sin)

    result_bl = bm.profile(baseline_fn, x)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
