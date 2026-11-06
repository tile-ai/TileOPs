"""SM89 dense decode sinks use the existing general decode programs."""

import math

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.attention import GQADenseFwdOp
from tileops.backend import BUILTIN
from workloads.attention.gqa.dense import GQADensePrefillWorkload


@pytest.mark.in_tree_kernels
@pytest.mark.sm89
@pytest.mark.parametrize(
    "shape,dtype",
    [
        pytest.param((2, 1, 17, 8, 2, 64), torch.float16, id="unsplit", marks=pytest.mark.smoke),
        pytest.param(
            (2, 1, 257, 6, 2, 64), torch.bfloat16, id="split-tail", marks=pytest.mark.smoke
        ),
        pytest.param(
            (1, 1, 17, 64, 1, 256), torch.float16, id="empty-split", marks=pytest.mark.full
        ),
    ],
)
def test_gqa_dense_sinks_on_sm89(shape, dtype) -> None:
    """Cover a partial head group and the D256 split with no assigned keys."""
    workload = GQADensePrefillWorkload(*shape, dtype, has_sinks=True)
    op = GQADenseFwdOp(is_causal=True, target=BUILTIN)
    inputs = workload.gen_inputs()
    inputs[-1][0] = -float("inf")
    inputs[-1][1] = float("inf")
    inputs[-1][-1] = 3e38
    TestBase.check(workload, op, *inputs)

    # Give the sink the same mass as all keys together: counting it twice
    # changes the result from 1/2 to 1/3, including with an empty split.
    inputs[0].zero_()
    inputs[1].zero_()
    inputs[2].fill_(1)
    inputs[-1].fill_(math.log(shape[2]))
    TestBase.check(workload, op, *inputs)
