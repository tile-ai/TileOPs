"""Test ManifoldConstrainedHyperConnection Post operation."""


import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import ManifoldConstrainedHyperConnectionPostOp
from workloads.ops.mhc_post import MhcPostTest as _MhcPostTestWorkload


class MhcPostTest(_MhcPostTestWorkload, TestBase):
    pass


class MhcPostFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (mhc post uses bf16 and needs looser checks)."""
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, \
        f"cosine similarity too low: {cos_sim.min().item()}"


@MhcPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                     tune: bool) -> None:
    test = MhcPostTest(batch, n_expand, c_x, dtype)
    op = ManifoldConstrainedHyperConnectionPostOp(
        batch, n_expand, c_x, dtype=str(dtype).split('.')[-1])
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
