"""Tests for the fused residual add, RMSNorm and dynamic per-token quantization."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_rms_norm_per_token_quant import FusedAddRMSNormPerTokenQuantFwdOp
from workloads.norm import FusedAddRMSNormPerTokenQuantWorkload
from workloads.numerics import compare_outputs

FP8, INT8 = torch.float8_e4m3fn, torch.int8


class FusedAddRMSNormPerTokenQuantTest(FusedAddRMSNormPerTokenQuantWorkload, TestBase):
    pass


class FusedAddRMSNormPerTokenQuantFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype, quant_dtype, scale_ub",
            [
                # A few rows (at most one an SM on current GPUs): each row spreads over 512
                # threads.
                pytest.param((64, 4096), torch.bfloat16, FP8, None, marks=pytest.mark.smoke),
                pytest.param((64, 8192), torch.float16, INT8, None, marks=pytest.mark.smoke),
                # A row the threads' accesses do not cover evenly skips the accesses past
                # its end: single elements at 257, 16-byte accesses at 3000, 8-byte at 4100,
                # 4-byte at 4098.
                pytest.param((1, 257), torch.bfloat16, INT8, None, marks=pytest.mark.smoke),
                pytest.param((33, 3000), torch.bfloat16, INT8, None, marks=pytest.mark.full),
                pytest.param((3, 4100), torch.float16, FP8, None, marks=pytest.mark.full),
                pytest.param((2, 4098), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # Several rows an SM take the configured 128 threads a row, here with a
                # partial last access step.
                pytest.param((512, 3000), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # A row of one warp folds without shared memory; odd single-element accesses
                # take the scalar FP8 path.
                pytest.param((4, 31), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # Leading axes flatten into rows and come back on every output.
                pytest.param((2, 7, 4096), torch.float16, FP8, None, marks=pytest.mark.full),
                # A bound below every row's amax: each scale is the bound, and the codes
                # past it saturate.
                pytest.param((16, 4096), torch.bfloat16, FP8, 0.5, marks=pytest.mark.full),
                # On SM90 a row wider than 8192 elements, with SMs to spare, splits across
                # a cluster whose CTAs exchange both partials.
                pytest.param((2, 16384), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # A row wider than 8192 elements stays whole when a split would leave each
                # CTA more than 4096 of them.
                pytest.param((64, 16384), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # On SM90 a row wider than one CTA holds takes the narrowest cluster that
                # holds it, however many rows; elsewhere it is read in tiles.
                pytest.param((40, 40000), torch.float16, INT8, None, marks=pytest.mark.full),
                # On SM90 a row that needs a cluster widens it further while SMs are spare.
                pytest.param((1, 65536), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # A row no CTA or cluster holds in whole 4-byte accesses is read in tiles,
                # three times.
                pytest.param((3, 32770), torch.bfloat16, FP8, None, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddRMSNormPerTokenQuantFixture
def test_fused_add_rms_norm_per_token_quant(shape, dtype, quant_dtype, scale_ub) -> None:
    test = FusedAddRMSNormPerTokenQuantTest(shape, dtype, quant_dtype, scale_ub=scale_ub)
    op = FusedAddRMSNormPerTokenQuantFwdOp(quant_dtype=quant_dtype, scale_ub=scale_ub)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("quant_dtype", [FP8, INT8])
def test_a_row_summing_to_zero_takes_the_floor_scale(quant_dtype) -> None:
    """``residual = -x`` zeroes a row: its scale is the format's floor and its codes zero."""
    workload = FusedAddRMSNormPerTokenQuantWorkload((4, 4096), torch.bfloat16, quant_dtype)
    x, residual, weight = workload.gen_inputs()
    residual[0] = -x[0]
    got = FusedAddRMSNormPerTokenQuantFwdOp(quant_dtype=quant_dtype)(x, residual, weight)
    compare_outputs(got, workload.ref_program(x, residual, weight), workload.verification(x))
    floor = 1.0 / (448.0 * 512.0) if quant_dtype == FP8 else torch.finfo(torch.float32).eps
    assert got[1][0].item() == pytest.approx(floor, rel=0)
    assert not got[0][0].float().any()


@pytest.mark.parametrize("quant_dtype", [FP8, INT8])
@pytest.mark.parametrize(
    "n",
    [
        pytest.param(1024, marks=pytest.mark.smoke),
        # A row read in tiles takes its codes in a separate pass from its amax.
        pytest.param(32770, marks=pytest.mark.full),
    ],
)
def test_nonfinite_values_take_vllms_codes(n, quant_dtype) -> None:
    """A NaN is skipped by the amax and saturates as vLLM's conversion does; an infinite sum
    turns its row's normalized values into NaN and zeros; an infinite weight makes a row's
    amax and scale infinite, and its codes those of ``y * 0``."""
    workload = FusedAddRMSNormPerTokenQuantWorkload((4, n), torch.float16, quant_dtype)
    x, residual, weight = workload.gen_inputs()
    x[0, 5] = float("nan")
    x[1, 9] = float("inf")
    weight[7] = float("inf")
    got = FusedAddRMSNormPerTokenQuantFwdOp(quant_dtype=quant_dtype)(x, residual, weight)
    compare_outputs(got, workload.ref_program(x, residual, weight), workload.verification(x))
