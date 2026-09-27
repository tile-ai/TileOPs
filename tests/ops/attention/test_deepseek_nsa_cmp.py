import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSACmpVarlenFwdOp
from workloads.attention.deepseek import NsaCmpFwdWorkload


class NsaCmpFwdTest(NsaCmpFwdWorkload, TestBase):
    pass


class NsaCmpFwdFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype, tune",
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
                    torch.float16,
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
    bs: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaCmpFwdTest(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype)
    inputs = test.gen_inputs()

    op = NSACmpVarlenFwdOp(scale=scale, bs=bs, tune=tune)
    test.check(op, *inputs, atol=4e-3, rtol=1e-5)
