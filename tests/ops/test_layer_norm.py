"""Tests for LayerNorm and fused-add LayerNorm."""

import re

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.norm import layer_norm as layer_norm_kernel_module
from tileops._ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops._ops.norm.layer_norm import LayerNormFwdOp
from tileops.backend import BUILTIN
from workloads.device import run_device
from workloads.norm import (
    FusedAddLayerNormWorkload,
    LayerNormLargeOffsetWorkload,
    LayerNormWorkload,
    layer_norm_verification,
    normalization_verification,
)
from workloads.numerics import compare_outputs


class LayerNormTest(LayerNormWorkload, TestBase):
    pass


class LayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                # A DiT row: narrow, unaligned, one warp per row; a tail CTA holds one row
                pytest.param(1025, 1152, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float32, False, marks=pytest.mark.full),
                # Standard aligned shapes -- fp16
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # fp32, one warp per row: 250 vectors, so six lanes hold one fewer
                pytest.param(1025, 1000, torch.float32, False, marks=pytest.mark.full),
                # Tuned over 128 and 256 threads: 256 leaves each thread an odd 95 elements
                # of a row walked from shared memory by a 512-row grid.
                pytest.param(512, 24320, torch.float16, True, marks=pytest.mark.full),
            ],
        ),
    ]


@LayerNormFixture
def test_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormTest(m, n, dtype)
    op = LayerNormFwdOp(normalized_shape=(n,))
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


def _launch_threads(op: LayerNormFwdOp, m: int) -> int:
    """Threads a block of the program the op's in-tree kernel compiled for *m* rows."""
    (kernel,) = op.iter_kernels()
    (program,) = [p for key, p in kernel._generation.programs.items() if key[1] == m]
    (threads,) = set(re.findall(r"__launch_bounds__\((\d+)", program.get_kernel_source()))
    return int(threads)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "row_counts",
    [
        pytest.param((1, 2048, 1), marks=pytest.mark.smoke),
        pytest.param((2048, 1, 2048), marks=pytest.mark.full),
    ],
)
def test_layer_norm_threads_follow_the_row_count(row_counts) -> None:
    # A 16384-wide row: one row fills no SM past one block, so 256 threads; 2048 rows
    # oversubscribe the device, so 128. One op compiles both from one published config.
    n, dtype = 16384, torch.float16
    op = LayerNormFwdOp(normalized_shape=(n,))
    for m in row_counts:
        test = LayerNormTest(m, n, dtype)
        test.check(op, *test.gen_inputs())
    assert (_launch_threads(op, 1), _launch_threads(op, 2048)) == (256, 128)
    assert op.kernel_config()["threads"] is None


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "row_counts, seed_threads",
    [
        pytest.param((1, 2048), 256, marks=pytest.mark.smoke),
        pytest.param((2048, 1), 128, marks=pytest.mark.full),
    ],
)
def test_layer_norm_tunes_from_the_first_row_count(monkeypatch, row_counts, seed_threads) -> None:
    # A request after both row counts ran tunes the first one built, from the config it
    # ran; the checks below run the published configuration.
    n, dtype = 16384, torch.float16
    op = LayerNormFwdOp(normalized_shape=(n,))
    builders = []
    for m in row_counts:
        op(*LayerNormTest(m, n, dtype).gen_inputs())
        (kernel,) = op.iter_kernels()
        builders.append(kernel.kernel)
    assert builders[0] is not builders[1]
    tuned = []
    tune = kernel.tune_jit_kernel

    def recording_tune(jit_kernel, configs, **kwargs):
        tuned.append((jit_kernel, kwargs["seed_config"]))
        return tune(jit_kernel, configs, **kwargs)

    monkeypatch.setattr(kernel, "tune_jit_kernel", recording_tune)
    op.request_tune()
    assert tuned == [(builders[0], {"block_m": 1, "threads": seed_threads})]
    assert op.kernel_config()["threads"] is not None
    for m in row_counts:
        test = LayerNormTest(m, n, dtype)
        test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_layer_norm_published_threads_override_the_row_count_rule() -> None:
    # Call protocol: a configuration published on the kernel the op reached.
    n, dtype = 16384, torch.float16
    op = LayerNormFwdOp(normalized_shape=(n,))
    warm = LayerNormTest(1, n, dtype)
    op(*warm.gen_inputs())
    (kernel,) = op.iter_kernels()
    kernel.config = {"block_m": 1, "threads": 512}
    for m in (1, 2048):
        test = LayerNormTest(m, n, dtype)
        test.check(op, *test.gen_inputs())
    assert (_launch_threads(op, 1), _launch_threads(op, 2048)) == (512, 512)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_layer_norm_tunes_after_a_failed_first_build(monkeypatch) -> None:
    # A program whose first compile raises exposes nothing to tune: a request after it
    # waits for the next build and tunes from that build's config. 2048 rows reach the
    # builder's shared-memory-walk test, which reads the thread count.
    n = 16384
    op = LayerNormFwdOp(normalized_shape=(n,))
    # An fp32 call binds the op first, so the failing fp16 entry leaves its table in place.
    warm = LayerNormTest(1, n, torch.float32)
    op(*warm.gen_inputs())
    build = layer_norm_kernel_module._layer_norm_kernel
    failures = [RuntimeError("injected compile failure")]

    def fails_to_compile(*args):
        jit = build(*args)
        if not failures:
            return jit

        def compile_once(*params):
            raise failures.pop()

        return compile_once

    monkeypatch.setattr(layer_norm_kernel_module, "_layer_norm_kernel", fails_to_compile)
    test = LayerNormTest(2048, n, torch.float16)
    with pytest.raises(RuntimeError, match="injected compile failure"):
        op(*test.gen_inputs())
    op.request_tune()
    op(*test.gen_inputs())
    (kernel,) = [k for k in op.iter_kernels() if k.dtype == torch.float16]
    assert kernel.config["threads"] is not None
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_layer_norm_kernel_handles_unaligned_shape() -> None:
    """The kernel, not the Op layer, owns non-aligned boundary handling."""
    m, n = 16, 3000
    dtype = torch.float16
    test = LayerNormTest(m, n, dtype)
    op = LayerNormFwdOp(normalized_shape=(n,), eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class LayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LayerNormNonContigFixture
def test_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    op = LayerNormFwdOp(normalized_shape=(n,))

    # Reference using torch.nn.functional.layer_norm
    x_ref = x.contiguous()
    y_ref = F.layer_norm(
        x_ref.float(),
        (n,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    compare_outputs(y, y_ref, layer_norm_verification(dtype))


class LayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LayerNorm3DFixture
def test_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())
    bias = torch.randn(hidden, dtype=dtype, device=run_device())

    op = LayerNormFwdOp(normalized_shape=(hidden,))

    # Reference using torch.nn.functional.layer_norm
    y_ref = F.layer_norm(
        x.float(),
        (hidden,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    compare_outputs(y, y_ref, layer_norm_verification(dtype))


class LayerNormLargeOffsetFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(4, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(4, 4096, torch.float16, marks=pytest.mark.smoke),
                # A padded row: the pad columns must not add to the variance, held in
                # registers here and walked from shared memory by a 1024-row grid.
                pytest.param(4, 3000, torch.float32, marks=pytest.mark.smoke),
                # One warp per row: a lane slot past the row must not add to the variance
                pytest.param(4, 1020, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 8300, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@LayerNormLargeOffsetFixture
def test_layer_norm_large_offset(m: int, n: int, dtype: torch.dtype) -> None:
    """Regression: large-mean, low-variance inputs stress the variance formula.

    E[x^2] - mean^2 would suffer catastrophic cancellation here (max_err > 1.0);
    the centered two-pass approach keeps error within a few percent.

    Note: fp32 reduction order differences between TileLang's T.reduce_sum and
    PyTorch's fused CUDA layer_norm cause inherent ~1-2% relative disagreement
    on adversarial large-offset inputs (var ~ 1e-4, mean ~ 10000).  We use
    a relative tolerance of 5% which is tight enough to catch the original
    catastrophic cancellation bug (which produced >100x error) while allowing
    the inherent fp32 parallel reduction precision limits.
    """
    workload = LayerNormLargeOffsetWorkload(m, n, dtype)
    inputs = workload.gen_inputs()
    op = LayerNormFwdOp(normalized_shape=(n,))
    TestBase.check(workload, op, *inputs)


@pytest.mark.smoke
def test_layer_norm_serves_a_changed_leading_dims_product_from_one_kernel() -> None:
    """The row count is a fact of the call, so it does not specialize the op's kernel."""
    n = 4096
    dtype = torch.float16

    op = LayerNormFwdOp(normalized_shape=(n,))
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    x1 = torch.randn(512, n, dtype=dtype, device=run_device())
    y1 = op(x1, weight, bias)
    assert y1.shape == x1.shape

    x2 = torch.randn(1024, n, dtype=dtype, device=run_device())
    y2 = op(x2, weight, bias)
    assert y2.shape == x2.shape

    y_ref = F.layer_norm(
        x2.float(),
        (n,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)
    compare_outputs(y2, y_ref, layer_norm_verification(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("n", [256, 1152])
@pytest.mark.parametrize("give", ["weight", "bias", "neither"])
def test_either_affine_tensor_alone_matches_torch(give: str, n: int) -> None:
    dtype = torch.float16
    x = torch.randn(8, n, dtype=dtype, device=run_device())
    kwargs = {} if give == "neither" else {give: torch.randn(n, dtype=dtype, device=run_device())}
    got = LayerNormFwdOp(normalized_shape=(n,))(x, **kwargs)
    compare_outputs(
        got, F.layer_norm(x, (n,), **kwargs), normalization_verification("LayerNormFwdOp", x.dtype)
    )


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """*t* copied into a contiguous view that starts one element into its storage."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n", [pytest.param(4096, id="aligned-row"), pytest.param(1000, id="warp-row")]
)
def test_layer_norm_reads_inputs_off_the_vector_boundary(n: int) -> None:
    """A contiguous input, weight and bias may start anywhere in their storage."""
    dtype = torch.bfloat16
    x = _misaligned(torch.randn(2, n, dtype=dtype, device=run_device()))
    weight = _misaligned(torch.randn(n, dtype=dtype, device=x.device))
    bias = _misaligned(torch.randn(n, dtype=dtype, device=x.device))
    compare_outputs(
        LayerNormFwdOp(normalized_shape=(n,))(x, weight, bias),
        F.layer_norm(x.float(), (n,), weight.float(), bias.float()).to(dtype),
        normalization_verification("LayerNormFwdOp", dtype),
    )


class FusedAddLayerNormTest(FusedAddLayerNormWorkload, TestBase):
    pass


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                # Standard aligned shapes -- fp16
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddLayerNormFixture
def test_fused_add_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_fused_add_layer_norm_large_offset() -> None:
    """A padded row whose mean far outgrows its spread keeps the pad out of the variance."""
    workload = LayerNormLargeOffsetWorkload(4, 3000, torch.float32)
    x, weight, bias = workload.gen_inputs()
    y, _ = FusedAddLayerNormFwdOp()(x, torch.zeros_like(x), weight, bias)
    compare_outputs(y, workload.ref_program(x, weight, bias), workload.verification(x))


class FusedAddLayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNormNonContigFixture
def test_fused_add_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    r_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    op = FusedAddLayerNormFwdOp()

    # Reference on contiguous copies
    test = FusedAddLayerNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight, bias)

    y, residual_out = op(x, residual, weight, bias)

    compare_outputs(y, y_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype))
    compare_outputs(
        residual_out, add_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype)
    )


class FusedAddLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNorm3DFixture
def test_fused_add_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())
    bias = torch.randn(hidden, dtype=dtype, device=run_device())

    M = batch * seq
    op = FusedAddLayerNormFwdOp()

    test = FusedAddLayerNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x, residual, weight, bias)

    y, residual_out = op(x, residual, weight, bias)

    compare_outputs(y, y_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype))
    compare_outputs(
        residual_out, add_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype)
    )


@pytest.mark.smoke
def test_fused_add_layer_norm_reads_inputs_off_the_vector_boundary() -> None:
    """A contiguous input, residual, weight and bias may start anywhere in their storage."""
    test = FusedAddLayerNormTest(4, 1024, torch.float16)
    test.check(FusedAddLayerNormFwdOp(), *(_misaligned(t) for t in test.gen_inputs()))
