
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp
from workloads.ops.gqa import GqaBwdTest as _GqaBwdTestWorkload
from workloads.ops.gqa import GqaFwdTest as _GqaFwdTestWorkload


class GqaBwdTest(_GqaBwdTestWorkload, TestBase):
    pass
class GqaFwdTest(_GqaFwdTestWorkload, TestBase):
    pass


class GqaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class GqaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GqaFwdFixture
def test_gqa_fwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    test = GqaFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@GqaBwdFixture
def test_gqa_bwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    test = GqaBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
