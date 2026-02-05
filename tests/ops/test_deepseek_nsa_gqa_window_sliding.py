"""Test DeepSeek NSA GQA Window Sliding operation."""

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import GQAWindowSlidingBenchmark
from top.ops import GQAWindowSlidingOp


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


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
    benchmark.check(op, *inputs)
    benchmark.baseline_profile(*inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":

    test_nsa_gqa_window_sliding_op(
        batch_size=1,
        groups=16,
        uq=1024,
        ukv=1024,
        heads=64,
        dim=128,
        is_causal=True,
        window_size_left=32,
        window_size_right=-1,
        dtype=torch.float16,
        accum_dtype=torch.float32,
        tune=False)
    test_nsa_gqa_window_sliding_op(
        batch_size=3,
        groups=16,
        uq=8192,
        ukv=8192,
        heads=64,
        dim=128,
        is_causal=True,
        window_size_left=2048,
        window_size_right=0,
        dtype=torch.float16,
        accum_dtype=torch.float32,
        tune=False)
    test_nsa_gqa_window_sliding_op(
        batch_size=3,
        groups=16,
        uq=8192,
        ukv=8192,
        heads=64,
        dim=128,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        dtype=torch.float16,
        accum_dtype=torch.float32,
        tune=False)
