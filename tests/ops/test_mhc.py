"""Tests for the MHC pre/post ops."""


import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MHCPostOp, MHCPreOp
from tileops.testing.mhc_reference import mhc_post_ref, mhc_pre_ref
from workloads.mhc import MHCPostWorkload, MHCPreWorkload


class MHCPreTest(MHCPreWorkload, TestBase):
    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return mhc_pre_ref(
            self.batch, self.n_expand, self.c_x,
            phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, eps,
        )


class MHCPreFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (MHC ops use bf16 and need looser checks)."""
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, \
        f"cosine similarity too low: {cos_sim.min().item()}"


@MHCPreFixture
def test_mhc_pre_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                    tune: bool) -> None:
    test = MHCPreTest(batch, n_expand, c_x, dtype)
    op = MHCPreOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


class MHCPostTest(MHCPostWorkload, TestBase):
    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        return mhc_post_ref(
            self.batch, self.n_expand, self.c_x, x_layer_out, h_post, x_res,
        )


class MHCPostFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]





@MHCPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                     tune: bool) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
