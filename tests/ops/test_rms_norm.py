import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.normalization import RMSNormWorkload

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
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
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
            ],
        ),
    ]


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(normalized_shape=(n,))
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


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
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(normalized_shape=(n,))

    # Reference on contiguous copy
    eps = 1e-6
    x_ref = x.contiguous()
    x_f32 = x_ref.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), (
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"
    )


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
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(normalized_shape=(hidden,))

    # Reference
    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), (
        f"3D test failed, max err: {(y - y_ref).abs().max()}"
    )


# --------------------------------------------------------------------------------------
# What the seam must not cost: one memo entry per dtype, and a capturable replay
# --------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_op_holds_one_kernel_per_dtype_whatever_the_row_count() -> None:
    """The op keys on dtype: the row count reaches the kernel as an argument.

    The TileLang program is still specialized per row count inside
    ``_rms_norm_kernel``; moving that into the kernel's own cache is kernel-side work.
    """
    from tileops.kernels.norm.rms_norm import _rms_norm_kernel

    op = RMSNormFwdOp(normalized_shape=(4096,))
    weight = torch.randn(4096, dtype=torch.float16, device="cuda")
    programs_before = _rms_norm_kernel.cache_info().currsize

    for rows in (128, 129, 1024):
        op(torch.randn(rows, 4096, dtype=torch.float16, device="cuda"), weight)
    op(torch.randn(2, 8, 4096, dtype=torch.float16, device="cuda"), weight)

    assert list(op.built_kernels("rms_norm")) == [torch.float16], "one kernel object"
    grew = _rms_norm_kernel.cache_info().currsize - programs_before
    assert grew == 3, "one program per distinct row count, held by the kernel not the op"


@pytest.mark.smoke
def test_the_in_tree_kernel_says_it_is_a_cuda_kernel() -> None:
    """The op layer is device-agnostic; the requirement belongs to these kernels."""
    op = RMSNormFwdOp(normalized_shape=(256,))

    with pytest.raises(ValueError, match="is a CUDA kernel"):
        op(torch.randn(4, 256, dtype=torch.float16), torch.randn(256, dtype=torch.float16))


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

    assert torch.allclose(static_out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_cold_op_traces_fullgraph_and_matches_eager() -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op = RMSNormFwdOp(normalized_shape=(4096,))
    x = torch.randn(64, 4096, dtype=torch.float16, device="cuda")
    weight = torch.randn(4096, dtype=torch.float16, device="cuda")

    torch.testing.assert_close(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_traced_graph_holds_only_this_ops_operator() -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op = RMSNormFwdOp(normalized_shape=(256,))
    x = torch.randn(8, 256, dtype=torch.float16, device="cuda")
    weight = torch.randn(256, dtype=torch.float16, device="cuda")

    assert_op_owns_graph_nodes(op, x, weight)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_non_contiguous_input_compiles_to_the_shape_the_fake_promised() -> None:
    """The fake speaks before the body normalizes contiguity, so it promises contiguous."""
    op = RMSNormFwdOp(normalized_shape=(256,))
    x = torch.randn(8, 512, dtype=torch.float16, device="cuda")[:, ::2]
    weight = torch.randn(256, dtype=torch.float16, device="cuda")
    assert not x.is_contiguous()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert output.is_contiguous()
    torch.testing.assert_close(output, op(x, weight))
