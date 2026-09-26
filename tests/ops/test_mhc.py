"""Tests for the MHC pre/post ops."""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MHCPostFwdOp, MHCPreFwdOp
from workloads.mhc import MHCPostWorkload, MHCPreWorkload


class MHCPreTest(MHCPreWorkload, TestBase):
    pass


class MHCPreFixture(FixtureBase):
    PARAMS = [
        (
            "batch, n_expand, c_x, dtype, tune",
            [
                pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(6, 4, 1000, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


def _compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare bf16 outputs by cosine similarity and float32 ones elementwise.

    The float32 ``h_post`` rows can saturate to near-zero vectors, whose cosine is
    undefined, so they are compared by value instead.
    """
    if output.dtype == torch.float32:
        torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
        return
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, f"cosine similarity too low: {cos_sim.min().item()}"


@MHCPreFixture
def test_mhc_pre_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype, tune: bool) -> None:
    test = MHCPreTest(batch, n_expand, c_x, dtype)
    op = MHCPreFwdOp(
        test.alpha_pre,
        test.alpha_post,
        test.alpha_res,
        test.sinkhorn_repeat,
        test.sinkhorn_eps,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), compare=_compare)


class MHCPostTest(MHCPostWorkload, TestBase):
    pass


class MHCPostFixture(FixtureBase):
    PARAMS = [
        (
            "batch, n_expand, c_x, dtype, tune",
            [
                pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
                # A c_x that no column tile divides: the last tile runs past it.
                pytest.param(6, 4, 1000, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@MHCPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype, tune: bool) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostFwdOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_compare)
