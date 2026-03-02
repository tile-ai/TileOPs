"""Test ManifoldConstrainedHyperConnection Post operation."""

from typing import Tuple

import torch
import torch.nn.functional as F
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import ManifoldConstrainedHyperConnectionPostOp


class MhcPostFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            (1, 4, 1280, torch.bfloat16, False),
            (2, 4, 1920, torch.bfloat16, False),
            (4, 4, 2560, torch.bfloat16, False),
        ]),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (mhc post uses bf16 and needs looser checks)."""
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, \
        f"cosine similarity too low: {cos_sim.min().item()}"


class MhcPostTest(TestBase):

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=self.dtype)
        h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
        x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=self.dtype)
        return x_layer_out, h_post, x_res

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref


@MhcPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                     tune: bool) -> None:
    test = MhcPostTest(batch, n_expand, c_x, dtype)
    op = ManifoldConstrainedHyperConnectionPostOp(
        batch, n_expand, c_x, dtype=str(dtype).split('.')[-1])
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
