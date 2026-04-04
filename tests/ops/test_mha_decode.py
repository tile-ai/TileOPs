
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadAttentionDecodeWithKVCacheOp
from workloads.ops.mha_decode import MhaDecodeTest as _MhaDecodeTestWorkload


class MhaDecodeTest(_MhaDecodeTestWorkload, TestBase):
    pass


class MhaDecodeFixture(FixtureBase):
    PARAMS = [
        ("b, h, s_q, s_kv, d, dtype, tune", [
            pytest.param(1, 32, 128, 8192, 128, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 32, 128, 8192, 128, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 32, 128, 5, 128, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@MhaDecodeFixture
def test_mha_decode(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                    tune: bool) -> None:
    test = MhaDecodeTest(b, h, s_q, s_kv, d, dtype)
    op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=2e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
