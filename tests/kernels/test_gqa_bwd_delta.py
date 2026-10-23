"""The GQA backward preprocess returns delta scaled by the softmax scale, and every backward
kernel that reads it reproduces the reference gradients from that delta."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.attention import (
    GQABwdMMAKernel,
    GQABwdPreprocessKernel,
    GQABwdWGMMAPipelinedKernel,
    GQABwdWSKernel,
)
from tileops.kernels.kernel_base import Kernel
from workloads.attention.gqa.bwd import GQABwdWorkload
from workloads.numerics import compare_outputs

pytestmark = [pytest.mark.in_tree_kernels, pytest.mark.smoke]


class _GQABwdTest(GQABwdWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    ("kernel_type", "heads_kv", "dim"),
    [
        pytest.param(GQABwdWGMMAPipelinedKernel, 2, 64, id="pipelined", marks=pytest.mark.sm90),
        pytest.param(GQABwdMMAKernel, 2, 64, id="mma", marks=pytest.mark.sm89),
        pytest.param(GQABwdWSKernel, 4, 128, id="warp-specialized", marks=pytest.mark.sm90),
    ],
)
def test_gqa_bwd_kernels_read_delta_scaled_by_the_softmax_scale(
    kernel_type: type[Kernel], heads_kv: int, dim: int
) -> None:
    torch.manual_seed(0)
    batch, heads, seq_len, dtype = 1, 4, 256, torch.bfloat16
    test = _GQABwdTest(batch, heads, heads_kv, seq_len, dim, True, dtype)
    inputs = tuple(t.detach() for t in test.gen_inputs())
    q, k, v, o, do, lse = inputs
    delta, dq_accum = GQABwdPreprocessKernel(batch, heads, seq_len, dim, dtype)(
        o, do, with_dq_accum=not kernel_type.owns_dq_accum
    )
    rowsum = (o.float() * do.float()).sum(-1).transpose(1, 2)
    torch.testing.assert_close(delta, rowsum * dim**-0.5)
    kernel = kernel_type(batch, heads, heads_kv, seq_len, dim, True, dtype)
    compare_outputs(
        kernel(q, k, v, do, lse, delta, dq_accum),
        test.ref_program(*inputs),
        test.verification(*inputs),
    )
