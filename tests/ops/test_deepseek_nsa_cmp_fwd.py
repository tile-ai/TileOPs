import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import NSACmpFwdVarlenBenchmark
from top.ops import NSACmpFwdVarlenOp


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    ("seq_num, c_seq_len, heads, dim_k, dim_v, chunk_num, group, scale, bc, bs, bk, bv, "
     "dtype, accum_dtype, tune"),
    [
        (5, 1024, 32, 128, 128, 1024 // 32, 16, 128**-0.5, 32, 32, 128, 128, torch.float16, torch.float32, False),
        (3, 512, 32, 128, 128, 512 // 32, 16, 128**-0.5, 32, 32, 128, 128, torch.float16, torch.float32, False),
    ],
)
def test_nsa_cmp_fwd_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    bk: int,
    bv: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:

    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    params = {
        "seq_num": seq_num,
        "c_seq_len": c_seq_len,
        "heads": heads,
        "dim_k": dim_k,
        "dim_v": dim_v,
        "group": group,
        "scale": scale,
        "bc": bc,
        "bs": bs,
        "bk": bk,
        "bv": bv,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }
    benchmark = NSACmpFwdVarlenBenchmark(**params)
    inputs = benchmark.gen_inputs()
    # Update chunk_num based on generated inputs
    params["chunk_num"] = benchmark.chunk_num
    op = NSACmpFwdVarlenOp(**params)
    benchmark.check(op, *inputs)


if __name__ == "__main__":
    test_nsa_cmp_fwd_varlen_op(
        seq_num=9,
        c_seq_len=8192,
        heads=32,
        dim_k=128,
        dim_v=128,
        group=16,
        scale=128**-0.5,
        bc=32,
        bs=32,
        bk=128,
        bv=128,
        dtype=torch.float16,
        accum_dtype=torch.float32,
        tune=False
    )
