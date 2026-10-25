"""Tests for the MHC pre/post ops."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.sequence_modeling import MHCPostFwdOp, MHCPreFwdOp
from workloads.sequence_modeling.mhc import MHCPostWorkload, MHCPreWorkload


class MHCPreTest(MHCPreWorkload, TestBase):
    pass


class MHCPreFixture(FixtureBase):
    PARAMS = [
        (
            "batch, n_expand, c_x, dtype, tune",
            [
                pytest.param(
                    1,
                    4,
                    1280,
                    torch.bfloat16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="sequence_modeling")],
                ),
                pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(6, 4, 1000, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@MHCPreFixture
def test_mhc_pre_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype, tune: bool) -> None:
    test = MHCPreTest(batch, n_expand, c_x, dtype)
    op = MHCPreFwdOp(
        test.alpha_pre,
        test.alpha_post,
        test.alpha_res,
        test.sinkhorn_repeat,
        test.sinkhorn_eps,
    )
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


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
                pytest.param(1, 40, 128, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1, 4, 999, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@MHCPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype, tune: bool) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_mhc_post_op_misaligned_storage() -> None:
    """Contiguous views with an unaligned storage offset, at a c_x read in pairs."""
    test = MHCPostTest(2, 4, 998, torch.bfloat16)
    x_layer_out, h_post, x_res = test.gen_inputs()
    x_layer_out, x_res = (
        torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view_as(t).copy_(t)
        for t in (x_layer_out, x_res)
    )
    test.check(MHCPostFwdOp(), x_layer_out, h_post, x_res)
