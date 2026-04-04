
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp
from workloads.ops.deepseek_mla_decode import MlaDecodeTest as _MlaDecodeTestWorkload


class MlaDecodeTest(_MlaDecodeTestWorkload, TestBase):
    pass


class MlaDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune", [
            pytest.param(32, 128, 1, 8192, 512, 64, torch.float16, False, marks=pytest.mark.smoke),
        ]),
    ]


@MlaDecodeFixture
def test_mla_decode(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                    dim_pe: int, dtype: torch.dtype, tune: bool):
    test = MlaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
