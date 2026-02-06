import sys

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import NSATopkVarlenBenchmark
from top.ops import NSATopkVarlenOp


@pytest.mark.parametrize(
    ("seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bc, bs, bk, "
     "dtype, accum_dtype, tune"),
    [
        (5, 1024, 32, 128, 16, 1, 16, 32, 32, 128, torch.float16, torch.float32, False),
        (3, 512, 32, 128, 16, 1, 16, 32, 32, 128, torch.float16, torch.float32, False),
    ],
)
def test_nsa_topk_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bc: int,
    bs: int,
    bk: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:

    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    params = {
        "seq_num": seq_num,
        "c_seq_len": c_seq_len,
        "heads": heads,
        "dim": dim,
        "group": group,
        "scale": scale,
        "selected_block_num": selected_block_num,
        "bc": bc,
        "bs": bs,
        "bk": bk,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }
    benchmark = NSATopkVarlenBenchmark(**params)
    inputs = benchmark.gen_inputs()
    op = NSATopkVarlenOp(**params, chunk_num=benchmark.chunk_num)
    benchmark.check(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
