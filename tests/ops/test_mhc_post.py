"""Test ManifoldConstrainedHyperConnection Post operation."""


import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MHCPostOp
from workloads.mhc import MHCPostTest as _MHCPostTestWorkload


class MHCPostTest(_MHCPostTestWorkload, TestBase):
    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref


class MHCPostFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
        ]),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (mhc post uses bf16 and needs looser checks)."""
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, \
        f"cosine similarity too low: {cos_sim.min().item()}"


@MHCPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                     tune: bool) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostOp(batch, n_expand, c_x, dtype=dtype, tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
