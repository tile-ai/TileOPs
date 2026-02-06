"""Test DeepSeek NSA GQA Window Sliding operation."""

import sys

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import GQAWindowSlidingBenchmark
from top.ops import GQAWindowSlidingOp


@pytest.mark.parametrize(
    ("batch_size", "groups", "uq", "ukv", "heads", "dim", "is_causal", "window_size_left",
     "window_size_right", "dtype", "accum_dtype", "tune"),
    [
        (1, 16, 1024, 1024, 64, 128, True, 32, -1, torch.float16, torch.float32, False),
        (3, 16, 8192, 8192, 64, 128, True, 2048, 0, torch.float16, torch.float32, False),
        (3, 16, 8192, 8192, 64, 128, False, -1, -1, torch.float16, torch.float32, False),
    ],
)
def test_nsa_gqa_window_sliding_op(
    batch_size: int,
    groups: int,
    uq: int,
    ukv: int,
    heads: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:

    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    params = {
        "batch_size": batch_size,
        "groups": groups,
        "uq": uq,
        "ukv": ukv,
        "heads": heads,
        "dim": dim,
        "is_causal": is_causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }
    benchmark = GQAWindowSlidingBenchmark(**params)
    op = GQAWindowSlidingOp(**params)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=5e-4, rtol=1e-5)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
