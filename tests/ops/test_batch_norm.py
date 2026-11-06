"""Tests for BatchNormFwdOp and BatchNormBwdOp, against torch.nn.functional.batch_norm and the analytical
gradient from torch.autograd.
"""

import math
import re

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.norm.batch_norm import (
    BatchNormFwdTrainKernel,
    BatchNormFwdTrainSplitKernel,
    BatchNormFwdTrainWholeKernel,
    BatchNormFwdTrainWideKernel,
)
from tileops._kernels.norm.call_spec import (
    BatchNormBwdInterface,
    BatchNormCall,
)
from tileops._ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.device import run_device, run_device_available
from workloads.norm import (
    BatchNormBwdWorkload,
    BatchNormFarFromZeroWorkload,
    BatchNormFwdWorkload,
    batch_norm_backward,
    batch_norm_backward_verification,
    batch_norm_forward_result,
    batch_norm_forward_verification,
    batch_norm_fwd_ref,
    normalization_verification,
)
from workloads.numerics import compare_outputs


class BatchNormBwdTest(BatchNormBwdWorkload, TestBase):
    pass


class BatchNormFwdTest(BatchNormFwdWorkload, TestBase):
    pass


class BatchNormFwdFixture(FixtureBase):
    """(N, C, *spatial, dtype, training)"""

    PARAMS = [
        (
            "N, C, spatial, dtype, training",
            [
                # BatchNorm1d – (N, C)
                pytest.param(32, 64, (), torch.float16, True, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, True, marks=pytest.mark.smoke),
                pytest.param(4, 256, (28, 28), torch.float32, True, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 256, (), torch.bfloat16, True, marks=pytest.mark.full),
                # BatchNorm1d – (N, C, L)
                pytest.param(16, 64, (512,), torch.float16, True, marks=pytest.mark.full),
                # L > 8192, so the tiled kernel this shape falls back to reads global
                # memory twice rather than holding the tile in shared memory.
                pytest.param(4, 64, (64, 64), torch.float16, True, marks=pytest.mark.full),
                # BatchNorm2d – (N, C, H, W)
                pytest.param(8, 64, (1024, 1024), torch.float16, True, marks=pytest.mark.full),
                # N*C*H*W = 2**31, one past INT32_MAX: a flat index over these extents does
                # not fit a signed 32-bit integer. Costs 60 GiB, the largest case in the suite.
                pytest.param(8, 64, (2048, 2048), torch.float16, False, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, True, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, exercises partial-tile path
                pytest.param(8, 64, (30, 30), torch.float16, True, marks=pytest.mark.full),
                pytest.param(8, 64, (30, 30), torch.bfloat16, True, marks=pytest.mark.full),
                # High channel count oversubscribes the SMs, exposing the running-stat update race.
                pytest.param(16, 1024, (512,), torch.float16, True, marks=pytest.mark.full),
                # The streamed path, which no other case reaches: a channel too long to
                # hold in registers (L = 131072) and too many channels to split (C >= 1024).
                pytest.param(2048, 1024, (64,), torch.float16, True, marks=pytest.mark.full),
                # The split path with a chunk no vector step divides: a block must stop at
                # its own chunk rather than sum into the next one's.
                pytest.param(3, 5, (300, 301), torch.float16, True, marks=pytest.mark.full),
            ],
        ),
    ]


class BatchNormBwdFixture(FixtureBase):
    """(N, C, *spatial, dtype)"""

    PARAMS = [
        (
            "N, C, spatial, dtype",
            [
                pytest.param(32, 64, (), torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 256, (28, 28), torch.float32, marks=pytest.mark.smoke),
                pytest.param(8, 64, (32, 32), torch.float16, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, marks=pytest.mark.full),
                # A register-held channel taking several steps per thread.
                pytest.param(4, 64, (64, 64), torch.float16, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, a vector narrower than the dtype allows.
                pytest.param(8, 64, (30, 30), torch.float16, marks=pytest.mark.full),
                pytest.param(8, 64, (30, 30), torch.bfloat16, marks=pytest.mark.full),
                # The split path with a chunk no vector step divides: a block must stop at
                # its own chunk rather than sum into the next one's.
                pytest.param(3, 5, (300, 301), torch.bfloat16, marks=pytest.mark.full),
                # The streamed tiled path: a channel too long for registers (L = 73728, odd
                # spatial run) and too many channels to split (C >= 1024).
                pytest.param(8192, 1024, (9,), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@BatchNormFwdFixture
def test_batch_norm_fwd(N, C, spatial, dtype, training):
    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    inputs = test.gen_inputs()
    op = BatchNormFwdOp(training=training)
    test.check(op, *inputs, runs=lambda *args: batch_norm_forward_result(op, *args))

    if training:
        # Repeat from the same initial statistics to detect update races separately.
        x, mean, var, weight, bias = inputs
        mean2, var2 = mean.clone(), var.clone()
        op(x, mean, var, weight, bias)
        op(x, mean2, var2, weight, bias)
        assert torch.equal(mean, mean2) and torch.equal(var, var2)


@BatchNormBwdFixture
def test_batch_norm_bwd(N, C, spatial, dtype):
    test = BatchNormBwdTest(N, C, spatial, dtype)
    test.check(BatchNormBwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_batch_norm_fwd_infer_returns_no_saved_stats() -> None:
    """Without training the forward returns ``(output, None, None)``: there are no batch
    statistics to save."""
    if not run_device_available():
        pytest.skip("the run device is not available")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=False)
    x = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
    weight = torch.randn(C, device=run_device(), dtype=torch.float32)
    bias = torch.randn(C, device=run_device(), dtype=torch.float32)
    rm = torch.zeros(C, device=run_device(), dtype=torch.float32)
    rv = torch.ones(C, device=run_device(), dtype=torch.float32)

    y, save_mean, save_invstd = op(x, rm, rv, weight, bias)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert save_mean is None and save_invstd is None


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param((32, 64), torch.float32, id="whole"),
        pytest.param((8, 16, 32, 32), torch.float16, id="wide"),
        pytest.param((4, 4, 256, 512), torch.bfloat16, id="split"),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
def test_training_forward_feeds_backward_like_autograd(shape, dtype, affine) -> None:
    """The statistics the training forward returns drive BatchNormBwdOp to torch
    autograd's gradients of F.batch_norm, with a scale and shift or without."""
    c = shape[1]
    x = torch.randn(shape, dtype=dtype, device=run_device())
    weight = torch.rand(c, device=run_device()) + 0.5 if affine else None
    bias = torch.randn(c, device=run_device()) if affine else None
    grad_out = torch.randn_like(x)

    y, save_mean, save_invstd = BatchNormFwdOp(training=True)(x, None, None, weight, bias)
    dinput, dweight, dbias = BatchNormBwdOp()(grad_out, x, save_mean, save_invstd, weight=weight)

    x32 = x.float().requires_grad_(True)
    ones = torch.ones(c, device=run_device())
    w32 = (weight if affine else ones).clone().requires_grad_(True)
    b32 = (bias if affine else ones - 1).clone().requires_grad_(True)
    with torch.enable_grad():
        y32 = torch.nn.functional.batch_norm(x32, None, None, w32, b32, training=True)
    y32.backward(grad_out.float())
    verification = batch_norm_forward_verification(dtype)
    compare_outputs(y, y32.detach().to(dtype), verification)
    compare_outputs(
        (dinput, dweight, dbias),
        (x32.grad.to(dtype), w32.grad, b32.grad),
        batch_norm_backward_verification(dtype),
    )


@pytest.mark.smoke
def test_training_updates_a_non_contiguous_running_stat() -> None:
    """Contiguity normalization must not swallow the write a mutated input promises."""
    if not run_device_available():
        pytest.skip("the run device is not available")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=True)
    x = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
    weight = torch.ones(C, device=run_device(), dtype=torch.float32)
    bias = torch.zeros(C, device=run_device(), dtype=torch.float32)
    # Every other element of a wider buffer: a view the kernel cannot be handed as is.
    rm = torch.zeros(2 * C, device=run_device(), dtype=torch.float32)[::2]
    rv = torch.ones(2 * C, device=run_device(), dtype=torch.float32)[::2]
    assert not rm.is_contiguous()

    workload = BatchNormFwdTest(N, C, (H, W), x.dtype, training=True)
    workload.check(
        op,
        x,
        rm,
        rv,
        weight,
        bias,
        runs=lambda *args: batch_norm_forward_result(op, *args),
    )


@pytest.mark.smoke
def test_training_rejects_one_value_per_channel() -> None:
    """Bessel's correction divides by L - 1; torch refuses the same call."""
    if not run_device_available():
        pytest.skip("the run device is not available")

    C = 4
    x = torch.randn(1, C, 1, 1, device=run_device(), dtype=torch.float32)
    weight = torch.ones(C, device=run_device(), dtype=torch.float32)
    bias = torch.zeros(C, device=run_device(), dtype=torch.float32)
    rm = torch.zeros(C, device=run_device(), dtype=torch.float32)
    rv = torch.ones(C, device=run_device(), dtype=torch.float32)

    torch_message = ""
    try:
        torch.nn.functional.batch_norm(x, rm, rv, weight, bias, training=True)
    except ValueError as exc:
        torch_message = str(exc)
    assert torch_message, "torch accepted L=1 in training; the guard's premise is gone"

    with pytest.raises(ValueError, match=re.escape("B * prod(L) != 1")):
        BatchNormFwdOp(training=True)(x, rm, rv, weight, bias)

    # Inference applies no correction; torch normalizes the same shape.
    infer = BatchNormFwdOp(training=False)
    expected = batch_norm_fwd_ref(x, weight, bias, rm, rv, training=False, eps=infer.eps)
    compare_outputs(
        batch_norm_forward_result(infer, x, rm, rv, weight, bias),
        expected,
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("shape", [(2, 8, 5, 6), (3, 4, 97, 101)], ids=["short", "long"])
def test_a_channel_length_no_tile_divides_matches_torch(shape) -> None:
    """The training forward (no statistics, no affine) and the backward cover a channel
    length no tile divides."""
    x = torch.randn(shape, dtype=torch.float16, device=run_device())
    c = shape[1]
    op = BatchNormFwdOp(training=True)
    compare_outputs(
        batch_norm_forward_result(op, x, None, None, None, None),
        batch_norm_fwd_ref(x, None, None, None, None, training=True),
        batch_norm_forward_verification(x.dtype),
    )
    grad_out, weight = torch.randn_like(x), torch.randn(c, device=run_device())
    var, mean = torch.var_mean(x.float(), dim=[0, 2, 3], correction=0)
    rstd = torch.rsqrt(var + 1e-5)
    inputs = grad_out, x, mean, rstd
    compare_outputs(
        BatchNormBwdOp()(*inputs, weight=weight),
        batch_norm_backward(*inputs, weight),
        batch_norm_backward_verification(x.dtype),
    )


@pytest.mark.parametrize("absent", ["weight", "bias"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 32), marks=pytest.mark.smoke, id="whole"),
        pytest.param((8, 16, 32, 32), marks=pytest.mark.smoke, id="wide"),
        pytest.param((4, 4, 256, 512), marks=pytest.mark.smoke, id="split"),
        # 1024 channels of an odd 65537-element run: too many to split, too long to hold.
        pytest.param((1, 1024, 65537), marks=pytest.mark.full, id="tiled"),
    ],
)
def test_training_forward_built_without_a_tensor_matches_torch(shape, absent) -> None:
    """A training program built without one affine tensor, and without running
    statistics, scales by one or shifts by zero in its place."""
    x = torch.randn(shape, dtype=torch.float16, device=run_device())
    c = shape[1]
    weight = None if absent == "weight" else torch.rand(c, device=run_device()) + 0.5
    bias = None if absent == "bias" else torch.randn(c, device=run_device())
    op = BatchNormFwdOp(training=True)
    compare_outputs(
        batch_norm_forward_result(op, x, None, None, weight, bias),
        batch_norm_fwd_ref(x, weight, bias, None, None, training=True),
        batch_norm_forward_verification(x.dtype),
    )


def _off_a_vector_boundary(shape: tuple) -> torch.Tensor:
    """A contiguous float16 tensor whose storage starts one element past a 16-byte boundary."""
    x = torch.randn(math.prod(shape) + 1, dtype=torch.float16, device=run_device())[1:]
    return x.view(shape)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "training, shape",
    [
        pytest.param(True, (8, 16, 32, 32), id="wide"),
        pytest.param(True, (4, 4, 256, 512), id="split"),
        pytest.param(False, (8, 16, 32, 32), id="infer"),
    ],
)
def test_forward_input_off_a_vector_boundary_matches_torch(training, shape) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    x = _off_a_vector_boundary(shape)
    c = shape[1]
    weight, bias = torch.rand(c, device=run_device()) + 0.5, torch.randn(c, device=run_device())
    rm, rv = torch.randn(c, device=run_device()), torch.rand(c, device=run_device()) + 0.5
    op = BatchNormFwdOp(training=training)
    compare_outputs(
        batch_norm_forward_result(op, x, rm.clone(), rv.clone(), weight, bias),
        batch_norm_fwd_ref(x, weight, bias, rm.clone(), rv.clone(), training=training),
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((8, 16, 32, 32), id="wide"),
        pytest.param((4, 4, 256, 512), id="split"),
    ],
)
def test_backward_inputs_off_a_vector_boundary_match_torch(shape) -> None:
    """A grad_out and an x starting one element past a 16-byte boundary match torch."""
    grad_out, x = _off_a_vector_boundary(shape), _off_a_vector_boundary(shape)
    weight = torch.rand(shape[1], device=run_device()) + 0.5
    var, mean = torch.var_mean(x.float(), dim=[0, 2, 3], correction=0)
    inputs = grad_out, x, mean, torch.rsqrt(var + 1e-5)
    bwd = BatchNormBwdOp()
    compare_outputs(
        bwd(*inputs, weight=weight),
        batch_norm_backward(*inputs, weight),
        batch_norm_backward_verification(x.dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape",
    [
        # A channel a 16-byte vector divides: each piece read a vector at a time.
        pytest.param((4, 4, 256, 512), id="vector"),
        # An odd spatial extent: a vector would straddle two batch items, so one element.
        pytest.param((4, 4, 255, 513), id="element"),
    ],
)
def test_channels_split_across_blocks_match_torch(shape) -> None:
    """The split training forward and backward, a channel summed across blocks, match
    torch whether a piece is read a vector or an element at a time."""
    x = torch.randn(shape, dtype=torch.float16, device=run_device())
    c = shape[1]
    weight, bias = torch.rand(c, device=run_device()) + 0.5, torch.randn(c, device=run_device())
    op = BatchNormFwdOp(training=True)
    compare_outputs(
        batch_norm_forward_result(op, x, None, None, weight, bias),
        batch_norm_fwd_ref(x, weight, bias, None, None, training=True),
        batch_norm_forward_verification(x.dtype),
    )
    grad_out = torch.randn_like(x)
    var, mean = torch.var_mean(x.float(), dim=[0, 2, 3], correction=0)
    inputs = grad_out, x, mean, torch.rsqrt(var + 1e-5)
    bwd = BatchNormBwdOp()
    compare_outputs(
        bwd(*inputs, weight=weight),
        batch_norm_backward(*inputs, weight),
        batch_norm_backward_verification(x.dtype),
    )


class BatchNormFarFromZeroTest(BatchNormFarFromZeroWorkload, TestBase):
    pass


# One channel a thread, a block, several blocks, and one block streaming its channel; a
# block whose threads each hold more than 32 elements (125, one an access), and one whose
# last threads hold fewer than the others (36 or 32 of 33000).
_TRAINING_PATHS = [
    pytest.param((32, 64), marks=pytest.mark.smoke, id="whole"),
    pytest.param((8, 16, 32, 32), marks=pytest.mark.smoke, id="wide"),
    pytest.param((4, 4, 256, 512), marks=pytest.mark.smoke, id="split"),
    pytest.param((1, 1024, 65537), marks=pytest.mark.full, id="tiled"),
    pytest.param((8, 2, 4001), marks=pytest.mark.full, id="wide-long"),
    pytest.param((1, 1, 33000), marks=pytest.mark.full, id="wide-ragged"),
]


_WIDE_MERGE_ROWS = [(torch.float32, "noisy"), (torch.float32, "huge")]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, row",
    [
        pytest.param(
            *path.values, dtype, row, marks=path.marks, id=f"{path.id}-{str(dtype)[6:]}-{row}"
        )
        for path in _TRAINING_PATHS
        for dtype, row in [
            (torch.float32, "constant"),
            (torch.float32, "noisy"),
            (torch.float32, "huge"),
            (torch.bfloat16, "huge"),
        ]
        # A long or ragged wide block's merges: the rows with a spread and past fp32's range.
        if not path.id.startswith("wide-") or (dtype, row) in _WIDE_MERGE_ROWS
    ],
)
def test_training_forward_normalizes_channels_far_from_zero(shape, dtype, row) -> None:
    n, c, *spatial = shape
    test = BatchNormFarFromZeroTest(n, c, tuple(spatial), dtype, row=row)
    op = BatchNormFwdOp(training=True)
    test.check(op, *test.gen_inputs(), runs=lambda *args: batch_norm_forward_result(op, *args))


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape",
    [
        *_TRAINING_PATHS[1:],
        # The issue's channel: 2^25 elements past the outlier.
        pytest.param((1, 1, 2**25 + 1), marks=pytest.mark.full, id="split-2^25"),
    ],
)
def test_training_forward_keeps_the_variance_of_a_channel_led_by_an_outlier(shape) -> None:
    # Every element but the first is zero: a variance taken as the mean square less the
    # squared mean, both less the first element, cancels to nothing. A channel one thread
    # holds is too short to lose it.
    x = torch.zeros(shape, device=run_device())
    x.view(shape[0], shape[1], -1)[0, :, 0] = 1024.0
    mean, var = torch.zeros(shape[1], device=x.device), torch.ones(shape[1], device=x.device)
    expected = batch_norm_fwd_ref(x.double(), None, None, mean.double(), var.double(), True)
    op = BatchNormFwdOp(training=True)
    compare_outputs(
        batch_norm_forward_result(op, x, mean, var, None, None),
        tuple(t.float() for t in expected),
        batch_norm_forward_verification(x.dtype),
    )


def _channel_led_by_a_low_half(shape):
    """-8e18, then 8e18: each difference from the first element squares past fp32's range,
    where each element and the variance do not."""
    x = torch.full(shape, 8e18, device=run_device())
    x.view(-1)[0] = -8e18
    return x


def _thread_of_large_values(shape):
    """28 elements of 1e18 in one thread of the block: their sum squared passes fp32's
    range, their square sum does not."""
    x = torch.zeros(shape, device=run_device())
    x[1:, :, 0, :4] = 1e18
    return x


def _group_then_its_outlier(shape):
    """32 elements of -b a thread apart, then 32b: their means' difference squares past fp32's
    range, the merged square sum does not."""
    b = 5.65e17
    x = torch.zeros(shape, device=run_device())
    x[..., : 256 * 32 : 256] = -b
    x[..., 256 * 32] = 32 * b
    return x


def _pair_of_large_values(shape):
    """-1e19 and 1e19 among zeros, at the start of every channel: an update merging one into
    an empty partial squares their difference past fp32's range."""
    x = torch.zeros(shape, device=run_device())
    x[..., 0] = -1e19
    x[..., 8 if shape[1] == 1 else 1] = 1e19
    return x


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, make",
    [
        pytest.param((8, 16, 32, 32), _thread_of_large_values, marks=pytest.mark.smoke, id="sum"),
        pytest.param(
            (1, 1, 4), _channel_led_by_a_low_half, marks=pytest.mark.smoke, id="difference"
        ),
        pytest.param(
            (1, 1, 8193), _group_then_its_outlier, marks=pytest.mark.smoke, id="merge-wide"
        ),
        pytest.param(
            (1, 1, 262145), _pair_of_large_values, marks=pytest.mark.smoke, id="merge-split"
        ),
        pytest.param(
            (1, 1024, 65537), _pair_of_large_values, marks=pytest.mark.full, id="merge-tiled"
        ),
    ],
)
def test_training_forward_squares_nothing_past_fp32_range(shape, make) -> None:
    x = make(shape)
    y, mean, invstd, *_ = batch_norm_forward_result(
        BatchNormFwdOp(training=True), x, None, None, None, None
    )
    want_y, want_mean, want_invstd, *_ = batch_norm_fwd_ref(
        x.double(), None, None, None, None, True
    )
    compare_outputs(
        (y, mean, invstd),
        (want_y.float(), want_mean.float(), want_invstd.float()),
        batch_norm_forward_verification(x.dtype),
    )


def _lead_positions(lead: str, kernel, length: int) -> list:
    """The channel positions *lead* names, for the kernel that serves the channel: its first
    element, each split piece's first, each tile slot's first (the first tile of 512), or in
    a wide block, where the threads take vectors in turn, each thread's first element or the
    first of each group of 32 it sums."""
    if lead == "channel":
        return [0]
    if lead == "piece":
        return list(range(0, length, -(-length // kernel.config["splits"])))
    if lead == "slot":
        return list(range(512))
    threads, vector = kernel.launch
    steps = -(-length // (threads * vector))
    every = max(1, 32 // vector) if lead == "group" else steps
    heads = ((k * threads + t) * vector for k in range(0, steps, every) for t in range(threads))
    return [head for head in heads if head < length]


# Zeros, or standard normal values, led by a far value at the positions `_lead_positions`
# names. A sum taken less a far element over many terms rounds the mean at its distance from
# the mean every term: zeros show it, each sum a multiple of the lead. A mean formed as a far
# reference plus the spread of the means about it rounds at that distance, past the
# tolerance of a mean as small as the normal values'.
_TRAINING_KERNELS = {
    "whole": BatchNormFwdTrainWholeKernel,
    "wide": BatchNormFwdTrainWideKernel,
    "split": BatchNormFwdTrainSplitKernel,
    "tiled": BatchNormFwdTrainKernel,
}


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "path, lead, shape, scale, value",
    [
        pytest.param(
            "whole", "channel", (32, 64), 1.0, 123456.7, marks=pytest.mark.smoke, id="whole-channel"
        ),
        pytest.param(
            "wide",
            "channel",
            (8, 16, 32, 32),
            1.0,
            123456.7,
            marks=pytest.mark.smoke,
            id="wide-channel",
        ),
        pytest.param(
            "wide", "thread", (1, 1, 8193), 0.0, 10000.1, marks=pytest.mark.smoke, id="wide-thread"
        ),
        pytest.param(
            "wide", "group", (8, 2, 8001), 0.0, 10000.1, marks=pytest.mark.smoke, id="wide-group"
        ),
        pytest.param(
            "split",
            "piece",
            (1, 64, 2**19),
            0.0,
            123456.7,
            marks=pytest.mark.smoke,
            id="split-piece",
        ),
        pytest.param(
            "tiled",
            "slot",
            (1, 1024, 262656),
            0.0,
            10000.1,
            marks=pytest.mark.full,
            id="tiled-slot",
        ),
    ],
)
def test_training_forward_matches_torch_on_channels_led_by_outliers(
    path, lead, shape, scale, value
) -> None:
    op = BatchNormFwdOp(training=True)
    torch.manual_seed(0)
    x = torch.randn(shape, device=run_device()) * scale
    op(x, None, None, None, None)
    (built,) = op.iter_kernels()
    assert isinstance(built, _TRAINING_KERNELS[path])
    length = x[:, 0].numel()
    positions = torch.as_tensor(_lead_positions(lead, built, length), device=x.device)
    spatial = length // shape[0]
    x.view(shape[0], shape[1], -1)[positions // spatial, :, positions % spatial] = value
    if path == "wide":
        # A deviation off zero: the wide block's raw sums hold the variance only within a
        # bound on the ratio of mean square to variance, which this passes, so the
        # statistics take the shifted rule.
        x += x.std()
    y, mean, invstd, *_ = batch_norm_forward_result(op, x, None, None, None, None)
    want_y, want_mean, want_invstd, *_ = batch_norm_fwd_ref(
        x.double(), None, None, None, None, True
    )
    compare_outputs(
        (y, mean, invstd),
        (want_y.float(), want_mean.float(), want_invstd.float()),
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_training_forward_keeps_the_mean_of_a_channel_spread_far_wider_than_it() -> None:
    # A deviation of 4.5e5 about a mean of 3.7e3: any fp32 sum of the channel rounds its mean
    # at the spread's units, far past 1e-5 of the mean itself.
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8192, device=run_device()) * 452227.6 + 3711.6
    y, mean, invstd, *_ = batch_norm_forward_result(
        BatchNormFwdOp(training=True), x, None, None, None, None
    )
    want_y, want_mean, want_invstd, *_ = batch_norm_fwd_ref(
        x.double(), None, None, None, None, True
    )
    compare_outputs(
        (y, mean, invstd),
        (want_y.float(), want_mean.float(), want_invstd.float()),
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape, value, old, momentum",
    [
        # 0.9 * -1 + 0.1 * 9 is 0 in float64 and rounds by 0.9's error in fp32.
        pytest.param((4, 8, 4, 4), 9.0, -1.0, 0.1, id="cancelling"),
        # The old running mean takes no part.
        pytest.param((4, 8, 4, 4), 9.0, -1.0, 1.0, id="momentum-1"),
        # A negative momentum: the batch term's share of the scale is still positive.
        pytest.param((2, 1), None, 0.0, -1.0, id="negative-momentum"),
    ],
)
def test_training_forward_updates_a_running_mean_against_its_operands(
    shape, value, old, momentum
) -> None:
    # The update (1 - m) * old + m * mean rounds at its operands, held against them.
    if value is None:
        x = torch.tensor([[-1.0], [1.0]], device=run_device())
    else:
        x = torch.full(shape, value, device=run_device())
    c = shape[1]
    rm = torch.full((c,), old, device=run_device())
    rv = torch.ones(c, device=run_device())
    op = BatchNormFwdOp(training=True, momentum=momentum)
    reference = batch_norm_fwd_ref(
        x.double(), None, None, rm.double(), rv.double(), True, momentum=momentum
    )
    compare_outputs(
        batch_norm_forward_result(op, x, rm.clone(), rv.clone(), None, None),
        tuple(t.float() for t in reference),
        batch_norm_forward_verification(x.dtype, momentum),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_training_forward_passes_a_channel_with_nan_through() -> None:
    # A NaN in one channel: its mean, deviation and output are NaN as torch's are, and the
    # mean check takes matching NaNs as equal.
    x = torch.randn(4, 2, 4, 4, device=run_device())
    x[1, 0, 2, 3] = float("nan")
    rm, rv = torch.zeros(2, device=run_device()), torch.ones(2, device=run_device())
    op = BatchNormFwdOp(training=True)
    reference = batch_norm_fwd_ref(x.double(), None, None, rm.double(), rv.double(), True)
    compare_outputs(
        batch_norm_forward_result(op, x, rm.clone(), rv.clone(), None, None),
        tuple(t.float() for t in reference),
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_training_forward_weights_a_squared_spread_before_squaring_it() -> None:
    # 1e20 with its first two elements 1e19 either side: the raw squares pass fp32's range, and
    # under the shifted rule a thread's mean lies 2e19 from lane 0's, whose square does too
    # until its share of the warp scales it.
    base = torch.tensor(1e20, dtype=torch.float32)
    x = torch.full((1, 1, 33), base.item(), device=run_device())
    x[0, 0, 0] = (base + 1e19).item()
    x[0, 0, 1] = (base - 1e19).item()
    op = BatchNormFwdOp(training=True)
    y, mean, invstd, *_ = batch_norm_forward_result(op, x, None, None, None, None)
    want_y, want_mean, want_invstd, *_ = batch_norm_fwd_ref(
        x.double(), None, None, None, None, True
    )
    compare_outputs(
        (y, mean, invstd),
        (want_y.float(), want_mean.float(), want_invstd.float()),
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.smoke
def test_the_running_mean_check_takes_the_calls_momentum() -> None:
    # At momentum 0 the running mean stays 0 over a channel of -1e6 and 1e6: a running mean
    # of 0.5 is wrong, and the check rejects it. At the default momentum 0.1 the batch mean's
    # deviation would allow it.
    x = torch.tensor([[-1e6], [1e6]], device=run_device())
    rm, rv = torch.zeros(1, device=run_device()), torch.ones(1, device=run_device())
    params = {"training": True, "momentum": 0.0, "eps": 1e-5}
    expected = tuple(
        t.float()
        for t in batch_norm_fwd_ref(x.double(), None, None, rm.double(), rv.double(), True, 0.0)
    )
    wrong = (*expected[:3], expected[3] + 0.5, expected[4])
    with pytest.raises(AssertionError, match="mean off"):
        compare_outputs(
            wrong, expected, normalization_verification("BatchNormFwdOp", x.dtype, params)
        )


def _alternating_channels(shape, value):
    """*value* with the sign ``(-1)^l`` at element ``l = n * S + s`` of each channel."""
    x = torch.full(shape, torch.tensor(value, dtype=torch.float32).item(), device=run_device())
    items = x.view(shape[0], shape[1], -1)
    if items.shape[2] % 2:
        items[1::2] *= -1
    items[:, :, 1::2] *= -1
    return x


# Channels alternating +-value: at 1e-21 and 1.3e-21 with eps 2^-138 the variance and eps lie
# among fp32's subnormals, so every path takes the squares and rsqrt in a power-of-two unit,
# and the shifted rule must not scale the differences down further; past 1.8e19 the squares
# pass fp32's range, and at 1e25 the variance does too while rstd does not.
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "path, shape, value, eps",
    [
        pytest.param(
            "wide", (1, 1, 8192), 1.3e-21, 2.0**-138, marks=pytest.mark.smoke, id="wide-1.3e-21"
        ),
        pytest.param(
            "split", (1, 2, 2**17), 1.3e-21, 2.0**-138, marks=pytest.mark.smoke, id="split-1.3e-21"
        ),
        pytest.param(
            "wide", (1, 1, 8192), 1e-21, 2.0**-138, marks=pytest.mark.smoke, id="wide-1e-21"
        ),
        pytest.param("whole", (32, 2), 1e-21, 2.0**-138, marks=pytest.mark.smoke, id="whole-1e-21"),
        pytest.param(
            "split", (1, 2, 2**17), 1e-21, 2.0**-138, marks=pytest.mark.smoke, id="split-1e-21"
        ),
        pytest.param("wide", (1, 1, 4001), 1e25, 1e-5, marks=pytest.mark.smoke, id="wide-1e25"),
        pytest.param("split", (1, 2, 2**17), 1e18, 1e-5, marks=pytest.mark.smoke, id="split-1e18"),
        pytest.param(
            "tiled", (1, 1024, 65537), 1e-21, 2.0**-138, marks=pytest.mark.full, id="tiled-1e-21"
        ),
        pytest.param(
            "tiled", (1, 1024, 65537), 1e18, 1e-5, marks=pytest.mark.full, id="tiled-1e18"
        ),
    ],
)
def test_training_forward_keeps_the_precision_of_an_extreme_variance(
    path, shape, value, eps
) -> None:
    x = _alternating_channels(shape, value)
    weight = torch.ones(shape[1], device=run_device())
    bias = torch.zeros(shape[1], device=run_device())
    op = BatchNormFwdOp(training=True, eps=eps)
    produced = batch_norm_forward_result(op, x, None, None, weight, bias)
    (built,) = op.iter_kernels()
    assert isinstance(built, _TRAINING_KERNELS[path])
    reference = batch_norm_fwd_ref(
        x.double(), weight.double(), bias.double(), None, None, True, eps=eps
    )
    compare_outputs(
        produced[:3],
        tuple(t.float() for t in reference[:3]),
        batch_norm_forward_verification(x.dtype),
    )


# Channels of +-value, alternating ("alt") or +value then -value ("half") along each channel's
# elements: at 2e37 and past, a sum less one element passes fp32's range before the variance
# is formed; at 2e19 the variance passes it while the momentum-weighted running variance does
# not; at 1e17 the split pieces keep their raw sums, but the channel's square sum passes it.
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "path, shape, value, pattern, stats",
    [
        pytest.param(
            "whole", (32, 2), 2e37, "alt", False, marks=pytest.mark.smoke, id="whole-mean"
        ),
        pytest.param(
            "wide", (1, 1, 4001), 2e38, "alt", False, marks=pytest.mark.smoke, id="wide-mean"
        ),
        pytest.param(
            "split", (1, 2, 2**17), 2e37, "half", False, marks=pytest.mark.smoke, id="split-mean"
        ),
        pytest.param(
            "whole", (32, 2), 2e19, "alt", True, marks=pytest.mark.smoke, id="whole-running-var"
        ),
        pytest.param(
            "wide", (1, 1, 8192), 2e19, "alt", True, marks=pytest.mark.smoke, id="wide-running-var"
        ),
        pytest.param(
            "split",
            (1, 2, 2**17),
            2e19,
            "alt",
            True,
            marks=pytest.mark.smoke,
            id="split-running-var",
        ),
        pytest.param(
            "split",
            (1, 2, 2**17),
            1e17,
            "alt",
            True,
            marks=pytest.mark.smoke,
            id="split-raw-pieces",
        ),
        pytest.param(
            "tiled", (1, 1024, 65537), 2e37, "alt", False, marks=pytest.mark.full, id="tiled-mean"
        ),
        pytest.param(
            "tiled",
            (1, 1024, 65537),
            2e19,
            "alt",
            True,
            marks=pytest.mark.full,
            id="tiled-running-var",
        ),
    ],
)
def test_training_forward_keeps_a_mean_and_running_variance_near_fp32s_range(
    path, shape, value, pattern, stats
) -> None:
    if pattern == "alt":
        x = _alternating_channels(shape, value)
    else:
        x = torch.full(shape, torch.tensor(value, dtype=torch.float32).item(), device=run_device())
        items = x.view(shape[0], shape[1], -1)
        # The second half of each channel's elements, in batch-then-spatial order, is negative.
        signs = torch.ones(items.shape[0] * items.shape[2], device=run_device())
        signs[signs.numel() // 2 :] = -1
        items *= signs.view(items.shape[0], 1, items.shape[2])
    channels = shape[1]
    weight = torch.ones(channels, device=run_device())
    bias = torch.zeros(channels, device=run_device())
    rm = torch.zeros(channels, device=run_device()) if stats else None
    rv = torch.ones(channels, device=run_device()) if stats else None
    expected = batch_norm_fwd_ref(
        x.double(),
        weight.double(),
        bias.double(),
        None if rm is None else rm.double(),
        None if rv is None else rv.double(),
        True,
        0.1,
    )
    op = BatchNormFwdOp(training=True, momentum=0.1, eps=1e-5)
    produced = batch_norm_forward_result(op, x, rm, rv, weight, bias)
    (built,) = op.iter_kernels()
    assert isinstance(built, _TRAINING_KERNELS[path])
    count = 5 if stats else 3
    compare_outputs(
        produced[:count],
        tuple(t.float() for t in expected[:count]),
        batch_norm_forward_verification(x.dtype, 0.1),
    )


# Input validation and torch.compile.
@pytest.mark.smoke
class TestBatchNormFwdValidation:
    def _make_op(self):
        from tileops._ops.norm.batch_norm import BatchNormFwdOp

        return BatchNormFwdOp()

    def _make_inputs(self, device=None, dtype=torch.float16):
        device = device or run_device()
        x = torch.randn(4, 8, 4, 4, device=device, dtype=dtype)
        weight = torch.randn(8, device=device, dtype=torch.float32)
        bias = torch.randn(8, device=device, dtype=torch.float32)
        rm = torch.zeros(8, device=device, dtype=torch.float32)
        rv = torch.ones(8, device=device, dtype=torch.float32)
        return x, weight, bias, rm, rv

    def test_rejects_wrong_dtype(self):
        from tileops._ops.norm.batch_norm import BatchNormFwdOp

        op = BatchNormFwdOp()
        x_wrong = torch.randn(4, 8, 4, 4, device=run_device(), dtype=torch.float64)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="dtype"):
            op(x_wrong, rm, rv, weight, bias)

    def test_rejects_wrong_shape(self):
        op = self._make_op()
        x_wrong = torch.randn(4, 16, 4, 4, device=run_device(), dtype=torch.float16)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="shape|channel|Channel"):
            op(x_wrong, rm, rv, weight, bias)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
class TestBatchNormCustomOp:
    def test_fwd_torch_compile_smoke(self):
        from tileops._ops.norm.batch_norm import BatchNormFwdOp

        op = BatchNormFwdOp(training=True)
        x = torch.randn(4, 8, 4, 4, device=run_device(), dtype=torch.float16)
        weight = torch.randn(8, device=run_device(), dtype=torch.float32)
        bias = torch.randn(8, device=run_device(), dtype=torch.float32)
        rm = torch.zeros(8, device=run_device(), dtype=torch.float32)
        rv = torch.ones(8, device=run_device(), dtype=torch.float32)

        compiled = torch.compile(op, fullgraph=False)
        # Manifest input order: (x, running_mean, running_var, weight, bias).
        y, save_mean, save_invstd = compiled(x, rm, rv, weight, bias)
        assert y.shape == x.shape
        assert save_mean.shape == save_invstd.shape == (8,)

    def test_bwd_torch_compile_smoke(self):
        from tileops._ops.norm.batch_norm import BatchNormBwdOp

        N, C, H, W = 4, 8, 4, 4
        op = BatchNormBwdOp()
        grad_out = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
        x = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
        weight = torch.randn(C, device=run_device(), dtype=torch.float32)
        x32 = x.float()
        x_cl = x32.permute(1, 0, 2, 3).reshape(C, -1).contiguous()
        mean = x_cl.mean(dim=1)
        rstd = 1.0 / torch.sqrt(x_cl.var(dim=1, unbiased=False) + 1e-5)

        compiled = torch.compile(op, fullgraph=False)
        gx, gw, gb = compiled(grad_out, x, mean, rstd, weight=weight)
        assert gx.shape == x.shape


def _to_cl(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 0, *range(2, x.ndim)).reshape(x.shape[1], -1).contiguous()


def _from_cl(x_cl: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    n = orig_shape[0]
    c = orig_shape[1]
    spatial = orig_shape[2:]
    return x_cl.reshape(c, n, *spatial).permute(1, 0, *range(2, len(orig_shape))).contiguous()


class _FakeKernel(Kernel):
    general = True

    def __init__(self, call: BatchNormCall) -> None:
        super().__init__()
        self.L = call.n * call.spatial
        self.dtype = call.dtype
        self.eps = call.eps
        self.momentum = call.momentum


class _FakeBatchNormBwdKernel(_FakeKernel, BatchNormBwdInterface):
    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_out_cl = _to_cl(grad_out)
        x_cl = _to_cl(x)
        grad_out_f = grad_out_cl.float()
        x_hat = (x_cl.float() - mean[:, None]) * rstd[:, None]
        grad_bias = grad_out_f.sum(dim=1)
        grad_weight = (grad_out_f * x_hat).sum(dim=1)
        grad_x = (
            weight[:, None]
            * rstd[:, None]
            * (self.L * grad_out_f - grad_bias[:, None] - x_hat * grad_weight[:, None])
            / self.L
        )
        return _from_cl(grad_x.to(self.dtype), x.shape), grad_weight, grad_bias


def _batch_norm_infer_ref(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    affine_shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    y = (x.float() - running_mean.reshape(affine_shape)) * torch.rsqrt(
        running_var.reshape(affine_shape) + eps
    )
    y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)
    return y.to(x.dtype)


def _batch_norm_train_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cl = _to_cl(x)
    mean = x_cl.float().mean(dim=1)
    var = x_cl.float().var(dim=1, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    y_cl = (x_cl.float() - mean[:, None]) * rstd[:, None]
    y_cl = y_cl * weight[:, None] + bias[:, None]
    return _from_cl(y_cl.to(x.dtype), x.shape), mean, rstd


def _batch_norm_bwd_ref(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = x.shape[0], x.shape[1]
    call = BatchNormCall(n=n, c=c, spatial=x.numel() // (n * c), dtype=x.dtype)
    kernel = _FakeBatchNormBwdKernel(call)  # lint: test-placement allow: torch reference double
    return kernel(grad_out, x, weight, mean, rstd)
