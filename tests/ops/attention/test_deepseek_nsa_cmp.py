import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSACmpFwdVarlenOp
from workloads.attention.deepseek import NsaCmpFwdWorkload


class NsaCmpFwdTest(NsaCmpFwdWorkload, TestBase):
    pass


class NsaCmpFwdFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, "
            "dtype, accum_dtype, tune",
            [
                pytest.param(
                    9,
                    8192,
                    32,
                    128,
                    128,
                    16,
                    128**-0.5,
                    32,
                    32,
                    torch.float16,
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                ),
            ],
        ),
    ]


@NsaCmpFwdFixture
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
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaCmpFwdTest(
        seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, dtype, accum_dtype
    )
    inputs = test.gen_inputs()

    op = NSACmpFwdVarlenOp(
        seq_num=seq_num,
        c_seq_len=c_seq_len,
        heads=heads,
        dim_k=dim_k,
        dim_v=dim_v,
        group=group,
        scale=scale,
        bc=bc,
        bs=bs,
        accum_dtype=accum_dtype,
        tune=tune,
        chunk_num=test.chunk_num,
    )
    test.check(op, *inputs, atol=4e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
