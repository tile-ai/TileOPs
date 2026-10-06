"""Tests for RMSNorm and fused-add RMSNorm."""

import pytest
import torch
import torch.nn.functional as F

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.device import run_device
from workloads.norm import (
    FusedAddRMSNormWorkload,
    RMSNormWorkload,
    norm_verification,
    normalization_verification,
)
from workloads.numerics import compare_outputs

register_compile_contract(RMSNormFwdOp)


class RMSNormTest(RMSNormWorkload, TestBase):
    pass


class RMSNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes (AC required)
                pytest.param(
                    1024,
                    4096,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="norm")],
                ),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-aligned N (AC required)
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m (proves T.copy partial block safety)
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # A short unaligned row: several share a block, with a tail block.
                pytest.param(17, 96, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(normalized_shape=(n,))
    test.check(op, *test.gen_inputs())


class RMSNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@RMSNormNonContigFixture
def test_rms_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device=run_device())

    op = RMSNormFwdOp(normalized_shape=(n,))

    # Reference on contiguous copy
    eps = 1e-6
    x_ref = x.contiguous()
    x_f32 = x_ref.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    compare_outputs(y, y_ref, norm_verification(dtype))


class RMSNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@RMSNorm3DFixture
def test_rms_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())

    op = RMSNormFwdOp(normalized_shape=(hidden,))

    # Reference
    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    compare_outputs(y, y_ref, norm_verification(dtype))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_a_warmed_up_op_can_be_captured_and_replayed() -> None:
    """Building a kernel may compile, so capture only ever sees a memo hit and a launch."""
    op = RMSNormFwdOp(normalized_shape=(4096,))
    x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    weight = torch.randn(4096, dtype=torch.float16, device="cuda")

    expected = op(x, weight)  # warm-up, outside the capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_x = x.clone()
    with torch.cuda.graph(graph):
        static_out = op(static_x, weight)

    static_x.copy_(x)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(static_out, expected)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_cold_op_traces_fullgraph_and_matches_eager() -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op = RMSNormFwdOp(normalized_shape=(4096,))
    x = torch.randn(64, 4096, dtype=torch.float16, device=run_device())
    weight = torch.randn(4096, dtype=torch.float16, device=run_device())

    assert torch.equal(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_traced_graph_holds_only_this_ops_operator() -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op = RMSNormFwdOp(normalized_shape=(256,))
    x = torch.randn(8, 256, dtype=torch.float16, device=run_device())
    weight = torch.randn(256, dtype=torch.float16, device=run_device())

    assert_op_owns_graph_nodes(op, x, weight)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_non_contiguous_input_compiles_to_the_shape_the_fake_promised() -> None:
    """The fake speaks before the body normalizes contiguity, so it promises contiguous."""
    op = RMSNormFwdOp(normalized_shape=(256,))
    x = torch.randn(8, 512, dtype=torch.float16, device=run_device())[:, ::2]
    weight = torch.randn(256, dtype=torch.float16, device=run_device())
    assert not x.is_contiguous()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert output.is_contiguous()
    assert torch.equal(output, op(x, weight))


@pytest.mark.smoke
def test_an_unaligned_row_comes_back_contiguous() -> None:
    """A row width that is not a multiple of the alignment still returns a contiguous output."""
    x = torch.randn(3, 96, dtype=torch.float16, device=run_device())
    assert RMSNormFwdOp(normalized_shape=(96,))(x).is_contiguous()


@pytest.mark.smoke
def test_no_weight_and_no_eps_match_torch() -> None:
    """An absent weight scales by one; ``eps=None`` is torch's float32 machine epsilon."""
    x = torch.full((2, 4), 1e-3, dtype=torch.float16, device=run_device())
    compare_outputs(
        RMSNormFwdOp(normalized_shape=(4,))(x),
        F.rms_norm(x, [4]),
        normalization_verification("RMSNormFwdOp", x.dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rows,n,dtype,has_weight",
    [
        pytest.param(3, 131072, torch.float16, True, id="wide-fp16"),
        pytest.param(3, 262144, torch.bfloat16, True, id="wide-bf16"),
        pytest.param(3, 131073, torch.bfloat16, False, id="tail-no-weight"),
        # More rows than resident CTAs: each CTA walks several rows.
        pytest.param(600, 131072, torch.bfloat16, True, id="rows-per-cta"),
    ],
)
def test_rms_norm_rows_exceeding_shared_memory(rows, n, dtype, has_weight) -> None:
    x = torch.randn(rows, n, device=run_device(), dtype=dtype)
    weight = torch.randn(n, device=x.device, dtype=dtype) if has_weight else None
    expected = F.rms_norm(x.float(), (n,), None if weight is None else weight.float(), eps=1e-6).to(
        dtype
    )
    actual = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6, tune=True)(x, weight)
    compare_outputs(actual, expected, normalization_verification("RMSNormFwdOp", x.dtype))


class FusedAddRMSNormTest(FusedAddRMSNormWorkload, TestBase):
    pass


class FusedAddRMSNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp16
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-aligned N
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # A row long enough for the split path, at the row counts
                # around which it turns on: one row takes it, a few rows do
                # not, and a full grid does not.
                pytest.param(1, 16384, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 16384, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4, 16384, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddRMSNormFixture
def test_fused_add_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRMSNormTest(m, n, dtype)
    op = FusedAddRMSNormFwdOp(tune=tune)
    test.check(op, *test.gen_inputs())


class FusedAddRMSNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddRMSNormNonContigFixture
def test_fused_add_rms_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    r_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype, device=run_device())

    op = FusedAddRMSNormFwdOp()

    # Reference on contiguous copies
    test = FusedAddRMSNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight)

    y, residual_out = op(x, residual, weight)
    compare_outputs(y, y_ref, norm_verification(dtype))
    compare_outputs(residual_out, add_ref, norm_verification(dtype))


class FusedAddRMSNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddRMSNorm3DFixture
def test_fused_add_rms_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())

    M = batch * seq
    op = FusedAddRMSNormFwdOp()

    test = FusedAddRMSNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x, residual, weight)

    y, residual_out = op(x, residual, weight)
    compare_outputs(y, y_ref, norm_verification(dtype))
    compare_outputs(residual_out, add_ref, norm_verification(dtype))
