"""Test NativeSparseAttention operation."""
import sys

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import NSAFwdVarlenBenchmark
from top.ops import NSAFwdVarlenOp


@pytest.mark.parametrize(
    ("batch, heads, c_seq_len, dim, is_causal, scale, block_size, "
     "groups, selected_blocks, dtype, accum_dtype, tune"),
    [
        (1, 16, 1024, 64, True, 0.1, 32, 16, 1, torch.float16, torch.float32, False),
        (4, 16, 8192, 64, True, 0.1, 32, 16, 1, torch.float16, torch.float32, False),
        (2, 16, 8192, 64, True, 0.1, 32, 16, 4, torch.float16, torch.float32, False),
    ],
)
def test_nsa_varlen_op(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:

    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    params = {
        "batch": batch,
        "heads": heads,
        "c_seq_len": c_seq_len,
        "dim": dim,
        "is_causal": is_causal,
        "scale": scale,
        "block_size": block_size,
        "groups": groups,
        "selected_blocks": selected_blocks,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }
    benchmark = NSAFwdVarlenBenchmark(**params)
    op = NSAFwdVarlenOp(**params)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=5e-4, rtol=1e-5)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
