import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.constants import SM_RESIDENT_BLOCKS
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.pool import (
    AdaptiveAvgPool2dKernel,
    AdaptiveAvgPool2dLanesKernel,
    AdaptiveMaxPool2dKernel,
    AdaptiveMaxPool2dRegisterKernel,
    AvgPool1dKernel,
    AvgPool1dRegisterKernel,
    AvgPool2dFwdInterface,
    MaxPool2dWarpKernel,
)
from tileops._kernels.pool.call_spec import AvgPoolCall, MaxPoolCall
from tileops._utils import device_facts
from tileops.backend import BUILTIN
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from tileops.pool import (
    AdaptiveAvgPool2dFwdOp,
    AdaptiveMaxPool2dFwdOp,
    AdaptiveMaxPool2dIndicesFwdOp,
    AvgPool1dFwdOp,
    AvgPool2dFwdOp,
    AvgPool3dFwdOp,
    MaxPool1dFwdOp,
    MaxPool1dIndicesFwdOp,
    MaxPool2dFwdOp,
    MaxPool2dIndicesFwdOp,
    MaxPool3dFwdOp,
    MaxPool3dIndicesFwdOp,
    MeanPoolingFwdOp,
)
from workloads.device import run_device, run_device_available, run_device_is_cuda
from workloads.numerics import compare_outputs
from workloads.pool import (
    AdaptiveAvgPool2dWorkload,
    AdaptiveMaxPool2dWorkload,
    AvgPoolWorkload,
    MaxPoolWorkload,
    MeanPoolingWorkload,
    max_pool_ref,
    pool_verification,
)

_AVG_POOL_OPS: dict[int, type] = {
    1: AvgPool1dFwdOp,
    2: AvgPool2dFwdOp,
    3: AvgPool3dFwdOp,
}


class AvgPool1dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune",
            [
                pytest.param(
                    2,
                    64,
                    512,
                    3,
                    None,
                    1,
                    False,
                    True,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
                    id="smoke-k3-default-stride-fp16",
                ),
                pytest.param(
                    2,
                    64,
                    512,
                    3,
                    None,
                    1,
                    False,
                    True,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-k3-default-stride-bf16",
                ),
                pytest.param(
                    2,
                    64,
                    512,
                    3,
                    None,
                    1,
                    False,
                    True,
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-k3-default-stride-fp32",
                ),
                pytest.param(
                    2,
                    32,
                    257,
                    5,
                    2,
                    2,
                    False,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-k5-s2-no-pad-count-fp16",
                ),
                pytest.param(
                    1,
                    48,
                    255,
                    4,
                    2,
                    1,
                    True,
                    True,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-ceil-bf16",
                ),
            ],
        ),
    ]


class AvgPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
            [
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (3, 3),
                    None,
                    (1, 1),
                    False,
                    True,
                    None,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
                    id="smoke-3x3-default-stride-fp16",
                ),
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (3, 3),
                    None,
                    (1, 1),
                    False,
                    True,
                    None,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-3x3-default-stride-bf16",
                ),
                pytest.param(
                    1,
                    32,
                    28,
                    28,
                    (3, 3),
                    None,
                    (1, 1),
                    False,
                    True,
                    None,
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-3x3-default-stride-fp32",
                ),
                pytest.param(
                    1,
                    128,
                    55,
                    57,
                    (3, 5),
                    (2, 2),
                    (1, 2),
                    True,
                    False,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-ceil-no-pad-count-fp16",
                ),
                pytest.param(
                    1,
                    96,
                    28,
                    30,
                    (2, 3),
                    (2, 2),
                    (0, 1),
                    False,
                    True,
                    5,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-divisor-override-bf16",
                ),
                pytest.param(
                    1,
                    7,
                    9,
                    10,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    False,
                    False,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-no-ceil-no-pad-count-fp16",
                ),
                pytest.param(
                    1,
                    5,
                    10,
                    11,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    True,
                    True,
                    None,
                    torch.float32,
                    False,
                    marks=pytest.mark.full,
                    id="full-ceil-pad-count-fp32",
                ),
                pytest.param(
                    2,
                    6,
                    9,
                    13,
                    (2, 3),
                    (2, 2),
                    (0, 1),
                    True,
                    True,
                    7,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-ceil-pad-count-divisor-bf16",
                ),
                pytest.param(
                    1,
                    9,
                    11,
                    12,
                    (3, 5),
                    (2, 3),
                    (1, 2),
                    True,
                    False,
                    7,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-ceil-no-pad-count-divisor-fp16",
                ),
                pytest.param(
                    1,
                    8,
                    9,
                    9,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    False,
                    False,
                    7,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-no-ceil-no-pad-count-divisor-fp16",
                ),
            ],
        ),
    ]


class AvgPool3dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
            [
                # Windows past the rows the wide kernel holds, and padded windows past the taps
                # it unrolls: both keep a serial loop.
                pytest.param(
                    1,
                    4,
                    10,
                    10,
                    20,
                    (5, 5, 2),
                    None,
                    (0, 0, 0),
                    False,
                    True,
                    None,
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-5x5x2-wide-serial-fp32",
                ),
                pytest.param(
                    1,
                    4,
                    8,
                    8,
                    8,
                    (4, 4, 3),
                    (2, 2, 2),
                    (1, 1, 1),
                    False,
                    True,
                    5,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
                    id="smoke-4x4x3-padded-serial-fp16",
                ),
                pytest.param(
                    1,
                    24,
                    10,
                    20,
                    22,
                    (2, 2, 3),
                    (2, 2, 2),
                    (0, 1, 1),
                    False,
                    True,
                    7,
                    torch.bfloat16,
                    False,
                    # A padded window with a fixed divisor and at most 32 taps: the clamped taps.
                    marks=pytest.mark.smoke,
                    id="smoke-divisor-override-clamped-bf16",
                ),
                pytest.param(
                    1,
                    32,
                    16,
                    28,
                    28,
                    (2, 2, 2),
                    None,
                    (0, 0, 0),
                    False,
                    True,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-2x2x2-default-stride-fp16",
                ),
                pytest.param(
                    1,
                    48,
                    15,
                    25,
                    27,
                    (2, 3, 3),
                    (2, 2, 2),
                    (1, 1, 1),
                    True,
                    False,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-ceil-no-pad-count-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    8,
                    12,
                    12,
                    (2, 2, 1),
                    (2, 2, 2),
                    (0, 0, 0),
                    False,
                    True,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-window-narrower-than-stride-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    8,
                    12,
                    12,
                    (2, 2, 2),
                    (2, 2, 2),
                    (0, 0, 0),
                    False,
                    True,
                    -3,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-negative-divisor-tiled-w-fp16",
                ),
            ],
        ),
    ]


class AvgPoolTest(AvgPoolWorkload, TestBase):
    """Dim-generic avg-pool reference test (divisor_override is 2d/3d-only)."""


def _run_avg_pool_case(
    ndim: int,
    shape: tuple[int, ...],
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None,
    padding: int | tuple[int, ...],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPoolTest(
        ndim,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    op_kwargs: dict[str, object] = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    if ndim > 1:
        op_kwargs["divisor_override"] = divisor_override
    op = _AVG_POOL_OPS[ndim](**op_kwargs)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs(*shape))


@AvgPool1dFixture
def test_avg_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: int,
    stride: int | None,
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    _run_avg_pool_case(
        1,
        (n, c_in, l_in),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        None,
        dtype,
        tune,
    )


@AvgPool2dFixture
def test_avg_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] | None,
    padding: tuple[int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    _run_avg_pool_case(
        2,
        (n, c_in, h_in, w_in),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
        tune,
    )


@AvgPool3dFixture
def test_avg_pool3d(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    padding: tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    _run_avg_pool_case(
        3,
        (n, c_in, d_in, h_in, w_in),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
        tune,
    )


@pytest.mark.parametrize(
    "l_in, kernel_l, stride_l, pad_l, dtype",
    [
        pytest.param(4096, 3, 2, 1, torch.float16, marks=pytest.mark.smoke),
        pytest.param(2048, 4, 2, 1, torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(32000, 5, 4, 2, torch.float16, marks=pytest.mark.full),
        pytest.param(28000, 12000, 4001, 0, torch.float16, marks=pytest.mark.full),
        pytest.param(127, 7, 3, 3, torch.bfloat16, marks=pytest.mark.full),
        pytest.param(30, 4, 4, 2, torch.bfloat16, marks=pytest.mark.full),
        pytest.param(33, 3, 2, 1, torch.float16, marks=pytest.mark.full),
        pytest.param(64, 1, 1, 0, torch.float16, marks=pytest.mark.full),
    ],
)
def test_avg_pool1d_staged_windows_match_the_reference(
    l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: torch.dtype
) -> None:
    """Rows, strides and padding that move the vectorized staging boundaries, including a
    window too wide to stage at 128 outputs."""
    _run_avg_pool_case(1, (1, 1, l_in), kernel_l, stride_l, pad_l, False, True, None, dtype, False)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "rows, l_in, kernel_l, stride_l, pad_l, ceil_mode, count_include_pad, dtype, offset",
    [
        pytest.param(
            (2, 3),
            4096,
            3,
            2,
            1,
            False,
            True,
            torch.float16,
            0,
            id="s2-pad-before-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3),
            1024,
            3,
            1,
            1,
            False,
            True,
            torch.float32,
            0,
            id="s1-pad-both-sides-fp32",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3),
            512,
            4,
            4,
            0,
            False,
            True,
            torch.bfloat16,
            0,
            id="s4-one-output-a-load-bf16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3),
            2048,
            4,
            2,
            1,
            True,
            False,
            torch.bfloat16,
            0,
            id="uncounted-pad-both-edges-bf16",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 3),
            8,
            3,
            2,
            0,
            True,
            True,
            torch.float32,
            0,
            id="ceil-window-past-the-row-fp32",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 3),
            4096,
            3,
            2,
            1,
            False,
            True,
            torch.float16,
            1,
            id="misaligned-fp16",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 3),
            1024,
            5,
            4,
            0,
            False,
            True,
            torch.float32,
            0,
            id="s4-reach-past-the-load-fp32",
            marks=pytest.mark.full,
        ),
        # Two 16-bit elements a side, each side one 4-byte load.
        pytest.param(
            (2, 3),
            1024,
            5,
            1,
            2,
            False,
            True,
            torch.bfloat16,
            0,
            id="s1-two-each-side-bf16",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 3),
            4096,
            5,
            2,
            2,
            False,
            False,
            torch.float16,
            0,
            id="uncounted-two-before-one-after-fp16",
            marks=pytest.mark.full,
        ),
        # Rows of 256 runs fill a row-grid block, so only the grid's y axis takes the launch
        # off the row grid.
        pytest.param(
            (1, 65536),
            1024,
            3,
            2,
            1,
            False,
            True,
            torch.float32,
            0,
            id="rows-past-the-grid-y-axis-fp32",
            marks=pytest.mark.full,
        ),
        # 2**19 rows of 4096: 2**31 elements, an extent one past INT32_MAX. Costs 8 GiB.
        pytest.param(
            (1, 2**19),
            4096,
            3,
            2,
            1,
            False,
            True,
            torch.float16,
            0,
            id="int32-max-plus-one-elements-fp16",
            marks=pytest.mark.full,
        ),
    ],
)
def test_avg_pool1d_windows_read_into_registers(
    rows: tuple[int, int],
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype,
    offset: int,
) -> None:
    """Windows a 16-byte load covers whole outputs of, read with the elements they reach
    past it on either side and divided as torch divides, so the result matches it bit for
    bit: a window the row's edge cuts divides by the positions it counts. A thread past a
    row's last run stores nothing; more rows than the grid's y axis holds, and an input of
    2**31 elements, are served too, as is an input starting ``offset`` elements into its
    storage."""
    n, c_in = rows
    storage = torch.randn(offset + n * c_in * l_in, device=run_device(), dtype=dtype)
    x = storage[offset:].view(n, c_in, l_in)
    op = AvgPool1dFwdOp(
        kernel_size=kernel_l,
        stride=stride_l,
        padding=pad_l,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        target=BUILTIN,
    )
    # Tuning compiles through TileLang's autotuner, which refuses a builder that closes over
    # anything but scalars.
    op.request_tune()
    out = op(x)
    expected = F.avg_pool1d(x, kernel_l, stride_l, pad_l, ceil_mode, count_include_pad)
    assert [type(k) for k in op.iter_kernels()] == [AvgPool1dRegisterKernel]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, kernel_l, stride_l, pad_l, kernel_type",
    [
        # The fixture's fp16 smoke shape.
        pytest.param(
            (2, 64, 512),
            3,
            2,
            1,
            AvgPool1dRegisterKernel,
            id="fixture-shape-register",
            marks=pytest.mark.smoke,
        ),
        # One row of 2**31 elements: the last window ends at 2**31. Costs 6 GiB.
        pytest.param(
            (1, 1, 2**31),
            3,
            2,
            1,
            AvgPool1dRegisterKernel,
            id="row-of-2-31-elements-register",
            marks=pytest.mark.full,
        ),
        # A row past 2**31 elements: loads, windows and stores start past INT32_MAX. Costs 6 GiB.
        pytest.param(
            (1, 1, 2**31 + 16),
            3,
            2,
            1,
            AvgPool1dRegisterKernel,
            id="row-past-2-31-elements-register",
            marks=pytest.mark.full,
        ),
        # A window reaching three 16-bit elements before a load is staged in shared memory.
        # Costs 6 GiB.
        pytest.param(
            (1, 1, 2**31),
            7,
            2,
            3,
            AvgPool1dKernel,
            id="row-of-2-31-elements-staged",
            marks=pytest.mark.full,
        ),
        # Stride 1 and padding give each row one output more than its input: 2**31 outputs
        # from fewer than 2**31 inputs, so the output alone takes int64 extents. Costs 8 GiB.
        pytest.param(
            (1, 2, 2**30 - 1),
            2,
            1,
            1,
            AvgPool1dKernel,
            id="output-of-2-31-elements-staged",
            marks=pytest.mark.full,
        ),
        # Stride 1 past 2**31 elements: output indices pass INT32_MAX. Costs 12 GiB.
        pytest.param(
            (1, 1, 2**31 + 8),
            7,
            1,
            3,
            AvgPool1dKernel,
            id="row-past-2-31-elements-staged",
            marks=pytest.mark.full,
        ),
    ],
)
def test_avg_pool1d_divides_each_end_of_the_row(
    shape: tuple[int, int, int],
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    kernel_type: type,
) -> None:
    """The windows at each end of a row divide by the positions they count, also where the
    last window's end passes INT32_MAX. torch pools a row of 2**31 elements to zeros, so
    each end is checked against torch on a slice of at most 1024 elements, dropping the
    windows the slice, not the row, cuts."""
    edge = min(shape[-1], 1024)
    x = torch.randn(*shape, device=run_device(), dtype=torch.float16)
    op = AvgPool1dFwdOp(
        kernel_size=kernel_l,
        stride=stride_l,
        padding=pad_l,
        count_include_pad=False,
        target=BUILTIN,
    )
    out = op(x)
    assert [type(k) for k in op.iter_kernels()] == [kernel_type]

    def pool(part: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(part, kernel_l, stride_l, pad_l, False, False)

    # Head windows ending inside the slice; tail windows starting inside it.
    head = pool(x[..., :edge])[..., : (edge + pad_l - kernel_l) // stride_l + 1]
    tail = pool(x[..., -edge:])[..., -(-pad_l // stride_l) :]
    torch.testing.assert_close(out[..., : head.shape[-1]], head, rtol=0, atol=0)
    torch.testing.assert_close(out[..., -tail.shape[-1] :], tail, rtol=0, atol=0)


@pytest.mark.smoke
def test_avg_pool1d_refuses_an_output_row_of_2_31_from_a_shorter_input_row() -> None:
    """Stride 1 and padding make a row of 2**31 - 1 inputs pool to 2**31 outputs, which
    TileLang's int32 literals for the shorter input row cannot lower; no kernel serves it."""
    call = AvgPoolCall(
        n=1, c_in=1, size=(2**31 - 1,), window=(2,), stride=(1,), pad=(1,), dtype=torch.float16
    )
    assert "2**31" in AvgPool1dKernel.refusal(call)
    assert AvgPool1dRegisterKernel.refusal(call) is not None


@pytest.mark.smoke
def test_avg_pool1d_register_refuses_groups_past_its_int32_thread_index() -> None:
    """The register kernel's one-axis grid indexes a thread a group of outputs in int32, so
    a call of more rows than grid y holds whose groups and one more block pass 2**31 - 1
    goes to AvgPool1dKernel."""
    # fp16, kernel 2, stride 2: four outputs a group, 16385 groups a row, 2**31 + 2**17 in all.
    call = AvgPoolCall(
        n=1,
        c_in=131072,
        size=(131080,),
        window=(2,),
        stride=(2,),
        pad=(0,),
        dtype=torch.float16,
    )
    assert "groups of outputs" in AvgPool1dRegisterKernel.refusal(call)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_avg_pool1d_register_serves_one_row_past_int32_groups_one_run_a_thread() -> None:
    """A row of 2**31 groups of 4-byte stores would pair runs on the one-axis grid, whose
    int32 thread index cannot hold them; it takes one run a thread on the row grid."""
    # fp16, kernel 5, stride 4, padding 2: two outputs (4 bytes) a group, 2**31 groups.
    call = AvgPoolCall(
        n=1, c_in=1, size=(2**34,), window=(5,), stride=(4,), pad=(2,), dtype=torch.float16
    )
    assert AvgPool1dRegisterKernel.refusal(call) is None
    # An input of 2**34 elements is too large to run the op on.
    config = AvgPool1dRegisterKernel(call).config  # lint: test-placement allow: 2**34 input
    assert (config["flat"], config["runs"]) == (False, 1)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("sub-wave", id="sub-wave-one-block-an-sm", marks=pytest.mark.smoke),
        pytest.param("widest", id="widest-blocks-one-an-sm", marks=pytest.mark.full),
        pytest.param(
            "past-one-block-an-sm", id="past-one-block-an-sm-row-grid", marks=pytest.mark.full
        ),
        pytest.param("short-rows", id="short-rows-one-axis", marks=pytest.mark.full),
        pytest.param(
            "short-rows-resident", id="short-rows-all-resident-row-grid", marks=pytest.mark.full
        ),
        pytest.param(
            "full-rows-past-residency",
            id="full-rows-past-residency-row-grid",
            marks=pytest.mark.full,
        ),
    ],
)
def test_avg_pool1d_register_launch_fills_each_sm_once(case: str) -> None:
    """Where the row grid puts a second block on some SM, a launch that one block an SM of
    at most 1024 threads covers runs as one block an SM of the fewest whole warps that cover
    it. Another runs 128-thread blocks along each row, unless the row grid has more blocks
    than the SMs hold at once and a row leaves so many of a block's threads idle that the
    blocks an SM holds cannot fill it: then its blocks span rows on one grid axis."""
    facts = device_facts(run_device().index)
    sm_count = facts.sm_count
    # Row-grid blocks the SMs hold at once.
    resident = sm_count * SM_RESIDENT_BLOCKS[facts.arch]
    # fp32 input, kernel 2, stride 2: one thread a 16-byte run of four values, l_in // 4 runs
    # a row.
    shape = {
        # Rows of 64 runs; one block an SM covers both launches, so the row grid's fill does
        # not decide them. Four rows an SM and one more: an SM's share is 8 warps and part of
        # a ninth, which takes a whole warp.
        "sub-wave": (1, sm_count * 4 + 1, 256),
        "widest": (1, sm_count * 16, 256),
        # Rows of 128 runs fill a row-grid block, so the blocks an SM holds fill it on every
        # supported architecture.
        "past-one-block-an-sm": (1, sm_count * 8 + 1, 512),
        "full-rows-past-residency": (1, resident + 1, 512),
        # Rows of 8 runs: past what one block an SM covers and the blocks the SMs hold at
        # once, and as many rows as SMs.
        "short-rows": (1, sm_count * 128 + 8, 32),
        "short-rows-resident": (1, sm_count, 32),
    }[case]
    launch = {
        "sub-wave": {"flat": True, "threads": 288},
        "widest": {"flat": True, "threads": 1024},
        "past-one-block-an-sm": {"flat": False, "threads": 128},
        "short-rows": {"flat": True, "threads": 128},
        "short-rows-resident": {"flat": False, "threads": 128},
        "full-rows-past-residency": {"flat": False, "threads": 128},
    }[case]
    op = AvgPool1dFwdOp(kernel_size=2, stride=2, target=BUILTIN)
    x = torch.randn(*shape, device=run_device(), dtype=torch.float32)
    out = op(x)
    (kernel,) = op.iter_kernels()
    assert type(kernel) is AvgPool1dRegisterKernel
    assert {key: kernel.config[key] for key in launch} == launch
    torch.testing.assert_close(out, F.avg_pool1d(x, 2, 2), rtol=0, atol=0)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_avg_pool1d_register_reads_a_side_in_one_load_of_at_most_4_bytes() -> None:
    """The register kernel reads the elements a window reaches past a load on one side as
    one load of at most 4 bytes, so a side of two fp32 elements goes to AvgPool1dKernel
    while one of two 16-bit elements does not."""
    call = AvgPoolCall(
        n=1, c_in=8, size=(1024,), window=(5,), stride=(2,), pad=(2,), dtype=torch.float32
    )
    assert "at most 4 bytes" in AvgPool1dRegisterKernel.refusal(call)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("paired", id="narrow-store-paired", marks=pytest.mark.smoke),
        pytest.param("covered", id="narrow-store-one-block-an-sm-covers", marks=pytest.mark.full),
        pytest.param("wide-store", id="wide-store-one-run", marks=pytest.mark.full),
    ],
)
def test_avg_pool1d_register_pairs_runs_of_narrow_stores(case: str) -> None:
    """A thread pools two runs half the launch apart where a run's outputs take at most 4
    bytes and one block an SM cannot cover the launch at two runs a thread; otherwise one
    run. An odd number of runs leaves the last thread's second run past the end, which it
    reads and does not store. The paired case spans three rows, so a thread's second run
    lies in another row, and divides by the positions it counts."""
    facts = device_facts(run_device().index)
    # One block an SM covers sm_count * 1024 threads, so 2 * sm_count * 1024 runs at two
    # runs a thread.
    covered_runs = 2 * facts.sm_count * 1024
    # Three rows whose runs total an odd count above covered_runs.
    row_runs = (covered_runs // 3 + 1) | 1
    # (dtype, kernel, stride, padding, runs in the one row): a run is 16 bytes.
    window = {
        # fp16, kernel 5, stride 4, padding 2: two outputs (4 bytes) a run of 8 elements.
        "paired": (torch.float16, 5, 4, 2, row_runs),
        "covered": (torch.float16, 5, 4, 2, covered_runs),
        # fp32, kernel 3, stride 2, padding 1: two outputs (8 bytes) a run of 4 elements.
        "wide-store": (torch.float32, 3, 2, 1, covered_runs + 1),
    }[case]
    dtype, kernel_l, stride_l, pad_l, runs = window
    launch = {
        "paired": {"runs": 2, "flat": True},
        "covered": {"runs": 1},
        "wide-store": {"runs": 1},
    }[case]
    run = 16 // dtype.itemsize
    rows = 3 if case == "paired" else 1
    shape = (1, rows, runs * run)
    counted = case != "paired"
    x = torch.randn(*shape, device=run_device(), dtype=dtype)
    op = AvgPool1dFwdOp(
        kernel_size=kernel_l,
        stride=stride_l,
        padding=pad_l,
        count_include_pad=counted,
        target=BUILTIN,
    )
    out = op(x)
    (kernel,) = op.iter_kernels()
    assert type(kernel) is AvgPool1dRegisterKernel
    assert {key: kernel.config[key] for key in launch} == launch
    ref = F.avg_pool1d(x.float(), kernel_l, stride_l, pad_l, count_include_pad=counted).to(dtype)
    compare_outputs(out, ref, pool_verification(maximum=False))


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "rows, l_in, dtype",
    [
        # 65536 rows: one more than the grid's y axis holds.
        pytest.param(
            (64, 1024),
            64,
            torch.float16,
            id="rows-past-the-grid-y-axis-fp16",
            marks=pytest.mark.smoke,
        ),
        # 32768 rows of 65536: 2**31 elements, an extent one past INT32_MAX, in rows the
        # grid's y axis holds. Costs 8 GiB.
        pytest.param(
            (1, 32768),
            65536,
            torch.float16,
            id="int32-max-plus-one-elements-fp16",
            marks=pytest.mark.full,
        ),
        # 2**31 rows of one element: one block a row, one more than grid x holds. Costs 9 GiB.
        pytest.param(
            (2, 2**30),
            1,
            torch.float16,
            id="blocks-past-the-grid-x-axis-fp16",
            marks=pytest.mark.full,
        ),
    ],
)
def test_avg_pool1d_staged_kernel_launch_limits(
    rows: tuple[int, int], l_in: int, dtype: torch.dtype
) -> None:
    """A window reaching three 16-bit elements before a load is staged in shared memory;
    more rows than the grid's y axis holds, more tiles than grid x holds, and an input of
    2**31 elements, pool like torch. Torch is run a batch entry at a time, under its own
    int32 limit."""
    x = torch.randn(*rows, l_in, device=run_device(), dtype=dtype)
    op = AvgPool1dFwdOp(kernel_size=7, stride=2, padding=3, target=BUILTIN)
    out = op(x)
    assert [type(k) for k in op.iter_kernels()] == [AvgPool1dKernel]
    for n in range(x.shape[0]):
        torch.testing.assert_close(out[n], F.avg_pool1d(x[n], 7, 2, 3), rtol=0, atol=0)


@pytest.mark.parametrize(
    "l_in, kernel_l",
    [
        pytest.param(15, 16, marks=pytest.mark.smoke),
        pytest.param(31, 32, marks=pytest.mark.full),
        pytest.param(100, 128, marks=pytest.mark.full),
        pytest.param(1000, 1024, marks=pytest.mark.full),
    ],
)
def test_max_pool1d_window_wider_than_the_row(l_in: int, kernel_l: int) -> None:
    """Ceil mode admits a window wider than the row; PyTorch pads the missing taps with -inf
    and still emits one output, which reads no tap past the row's end."""
    _run_max_pool_case(
        1, (2, 4, l_in), (kernel_l,), (kernel_l,), (0,), (1,), True, torch.float16, False, True
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    ("ndim", "shape"),
    [
        pytest.param(2, (1, 4, 8, 8), id="avg-pool2d"),
        pytest.param(3, (1, 3, 4, 6, 6), id="avg-pool3d"),
    ],
)
def test_avg_pool_negative_divisor_override_matches_torch(
    ndim: int,
    shape: tuple[int, ...],
) -> None:
    x = torch.randn(*shape, device=run_device(), dtype=torch.float16).contiguous()
    pool_kwargs = {
        "kernel_size": (2,) * ndim,
        "stride": (2,) * ndim,
        "padding": (0,) * ndim,
        "divisor_override": -1,
    }
    op = _AVG_POOL_OPS[ndim](**pool_kwargs)
    out = op(x)
    ref = getattr(F, f"avg_pool{ndim}d")(x.float(), **pool_kwargs).to(x.dtype)
    compare_outputs(out, ref, pool_verification(maximum=False))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_a_pool_param_changed_after_a_call_runs_the_next_call() -> None:
    """A call reads the op's params as they are on that call, and a return to the first
    params runs them again."""
    (x,) = AvgPoolTest(2, 2, 2, 0, False, True, 4, torch.float16).gen_inputs(1, 4, 8, 8)
    op = AvgPool2dFwdOp(kernel_size=2, stride=2, divisor_override=4)
    op(x)
    for divisor in (2, 4):
        op.divisor_override = divisor
        AvgPoolTest(2, 2, 2, 0, False, True, divisor, torch.float16).check(op, x)
    op = MaxPool2dFwdOp(kernel_size=2)
    op(x)
    for kernel_size in (4, 2):
        op.kernel_size = kernel_size
        MaxPoolTest(2, kernel_size, None, 0, 1, False, torch.float16).check(op, x)


_MAX_POOL_OPS: dict[int, type] = {
    1: MaxPool1dFwdOp,
    2: MaxPool2dFwdOp,
    3: MaxPool3dFwdOp,
}

_MAX_POOL_INDICES_OPS: dict[int, type] = {
    1: MaxPool1dIndicesFwdOp,
    2: MaxPool2dIndicesFwdOp,
    3: MaxPool3dIndicesFwdOp,
}


def _max_pool_op_cls(ndim: int, return_indices: bool) -> type:
    return _MAX_POOL_INDICES_OPS[ndim] if return_indices else _MAX_POOL_OPS[ndim]


_MAX_POOL1D_PARAMS = [
    # Smoke: one config across all supported dtypes.
    pytest.param(
        2,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.float16,
        False,
        True,
        marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
        id="smoke-k3-s2-p1-fp16",
    ),
    pytest.param(
        2,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.bfloat16,
        False,
        True,
        marks=pytest.mark.smoke,
        id="smoke-k3-s2-p1-bf16",
    ),
    pytest.param(
        1,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.float32,
        False,
        True,
        marks=pytest.mark.smoke,
        id="smoke-k3-s2-p1-fp32",
    ),
    # Full: distinct setting combinations.
    pytest.param(
        1,
        4,
        63,
        (3,),
        None,
        (1,),
        (2,),
        False,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-default-stride-dilation-fp16",
    ),
    pytest.param(
        1,
        4,
        97,
        (5,),
        (3,),
        (2,),
        (1,),
        True,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-ceil-k5-s3-p2-fp16",
    ),
    pytest.param(
        2,
        8,
        64,
        (3,),
        (2,),
        (1,),
        (1,),
        False,
        torch.float16,
        False,
        False,
        marks=pytest.mark.full,
        id="full-noncontiguous-k3-fp16",
    ),
    pytest.param(
        1,
        4,
        97,
        (5,),
        (3,),
        (2,),
        (1,),
        True,
        torch.bfloat16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-ceil-k5-s3-p2-bf16",
    ),
    # Non-overlapping windows over a short output row: the windowed read.
    pytest.param(
        2,
        32,
        64,
        (8,),
        (8,),
        (0,),
        (1,),
        False,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-windowed-short-output-fp16",
    ),
    # One output a row, at a window width the row-reduce fragment takes.
    pytest.param(
        2,
        8,
        32,
        (32,),
        (32,),
        (0,),
        (1,),
        False,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-rowreduce-one-output-fp16",
    ),
]


class MaxPool1dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune, contiguous",
            _MAX_POOL1D_PARAMS,
        ),
    ]


_MAX_POOL2D_PARAMS = [
    # Smoke: one config across all supported dtypes.
    pytest.param(
        2,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.float16,
        False,
        True,
        marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
        id="smoke-3x3-s2-p1-fp16",
    ),
    pytest.param(
        2,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.bfloat16,
        False,
        True,
        marks=pytest.mark.smoke,
        id="smoke-3x3-s2-p1-bf16",
    ),
    pytest.param(
        1,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.float32,
        False,
        True,
        marks=pytest.mark.smoke,
        id="smoke-3x3-s2-p1-fp32",
    ),
    # Full: distinct setting combinations.
    pytest.param(
        1,
        4,
        16,
        16,
        (2, 2),
        (2, 2),
        (0, 0),
        (1, 1),
        False,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-unpadded-window-inside-fp16",
    ),
    pytest.param(
        1,
        4,
        14,
        14,
        (3, 3),
        None,
        (1, 1),
        (2, 1),
        False,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-default-stride-dilation-fp16",
    ),
    pytest.param(
        1,
        4,
        23,
        27,
        (3, 5),
        (2, 3),
        (1, 2),
        (1, 1),
        True,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-nonsquare-ceil-fp16",
    ),
    pytest.param(
        2,
        8,
        16,
        16,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        False,
        torch.float16,
        False,
        False,
        marks=pytest.mark.full,
        id="full-noncontiguous-3x3-fp16",
    ),
    pytest.param(
        1,
        4,
        23,
        27,
        (3, 5),
        (2, 3),
        (1, 2),
        (1, 1),
        True,
        torch.bfloat16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-nonsquare-ceil-bf16",
    ),
]


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "height,width,dtype",
    [
        pytest.param(33, 33, torch.bfloat16, marks=pytest.mark.smoke, id="bf16-half-warp"),
        pytest.param(55, 55, torch.float16, marks=pytest.mark.smoke, id="fp16-warp-window"),
        pytest.param(47, 63, torch.bfloat16, marks=pytest.mark.full, id="bf16-warp-window"),
    ],
)
def test_warp_pool_odd_rectangle_and_nan(height, width, dtype):
    x = torch.randn(2, 3, height, width, device=run_device(), dtype=dtype)
    x[:, :, -1, -1] = float("nan")
    op = MaxPool2dFwdOp(3, stride=2, ceil_mode=True, target=BUILTIN)
    expected = F.max_pool2d(x, 3, stride=2, ceil_mode=True)
    torch.testing.assert_close(op(x), expected, rtol=0, atol=0, equal_nan=True)
    (kernel,) = op.iter_kernels()
    assert type(kernel) is MaxPool2dWarpKernel


@pytest.mark.smoke
def test_max_pool2d_warp_refuses_mixed_extents() -> None:
    """A 33-wide axis is served only when both axes are 33; 33 by 63 goes to the fallback."""
    call = MaxPoolCall(
        n=2,
        c_in=3,
        size=(33, 63),
        window=(3, 3),
        stride=(2, 2),
        pad=(0, 0),
        dilation=(1, 1),
        ceil_mode=True,
        dtype=torch.bfloat16,
    )
    assert MaxPool2dWarpKernel.refusal(call) is not None


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.sm90
def test_max_pool2d_65x65_bf16_default_config():
    """An unpadded 3x3 stride-2 BF16 pool on 65x65 takes MaxPool2dKernel's tile_h=8 default."""
    from tileops._kernels.pool import MaxPool2dKernel

    x = torch.randn(1, 1, 65, 65, device=run_device(), dtype=torch.bfloat16)
    op = MaxPool2dFwdOp(3, stride=2, target=BUILTIN)
    torch.testing.assert_close(op(x), F.max_pool2d(x, 3, stride=2), rtol=0, atol=0)
    (kernel,) = op.iter_kernels()
    assert type(kernel) is MaxPool2dKernel
    assert kernel.config == {"block_m": 128, "threads": 128, "tile_h": 8, "tile_w": 1}


class MaxPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune, contiguous",
            _MAX_POOL2D_PARAMS,
        ),
    ]


_MAX_POOL3D_PARAMS = [
    # Smoke: one config across all supported dtypes.
    pytest.param(
        2,
        4,
        8,
        16,
        16,
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        False,
        torch.float16,
        False,
        True,
        marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
        id="smoke-k2-s2-fp16",
    ),
    pytest.param(
        2,
        4,
        8,
        16,
        16,
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        False,
        torch.bfloat16,
        False,
        True,
        marks=pytest.mark.smoke,
        id="smoke-k2-s2-bf16",
    ),
    pytest.param(
        1,
        4,
        8,
        16,
        16,
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        False,
        torch.float32,
        False,
        True,
        marks=pytest.mark.smoke,
        id="smoke-k2-s2-fp32",
    ),
    # Full: distinct setting combinations.
    pytest.param(
        1,
        4,
        6,
        14,
        14,
        (3, 3, 3),
        None,
        (1, 1, 1),
        (2, 1, 1),
        False,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-default-stride-dilation-fp16",
    ),
    pytest.param(
        1,
        4,
        7,
        23,
        27,
        (1, 3, 5),
        (1, 2, 3),
        (0, 1, 2),
        (1, 1, 1),
        True,
        torch.float16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-noncube-ceil-fp16",
    ),
    pytest.param(
        2,
        4,
        8,
        16,
        16,
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        False,
        torch.float16,
        False,
        False,
        marks=pytest.mark.full,
        id="full-noncontiguous-k2-fp16",
    ),
    pytest.param(
        1,
        4,
        7,
        23,
        27,
        (1, 3, 5),
        (1, 2, 3),
        (0, 1, 2),
        (1, 1, 1),
        True,
        torch.bfloat16,
        False,
        True,
        marks=pytest.mark.full,
        id="full-noncube-ceil-bf16",
    ),
]


class MaxPool3dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune, contiguous",
            _MAX_POOL3D_PARAMS,
        ),
    ]


class MaxPoolTest(MaxPoolWorkload, TestBase):
    """Dim-generic max-pool reference test."""


def _run_max_pool_case(
    ndim: int,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None,
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
    contiguous: bool,
) -> None:
    # Exercise both the plain and the return_indices op on the same config.
    for return_indices in (False, True):
        test = MaxPoolTest(
            ndim,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            dtype,
            contiguous=contiguous,
            return_indices=return_indices,
        )
        op = _max_pool_op_cls(ndim, return_indices)(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        if tune:
            op.request_tune()
        test.check(op, *test.gen_inputs(*shape))


@MaxPool1dFixture
def test_max_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: tuple[int] | None,
    padding: tuple[int],
    dilation: tuple[int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
    contiguous: bool,
) -> None:
    _run_max_pool_case(
        1,
        (n, c_in, l_in),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        tune,
        contiguous,
    )


@MaxPool2dFixture
def test_max_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] | None,
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
    contiguous: bool,
) -> None:
    _run_max_pool_case(
        2,
        (n, c_in, h_in, w_in),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        tune,
        contiguous,
    )


@MaxPool3dFixture
def test_max_pool3d(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
    contiguous: bool,
) -> None:
    _run_max_pool_case(
        3,
        (n, c_in, d_in, h_in, w_in),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        tune,
        contiguous,
    )


# Curated per-dim special-value inputs (NaN / tied maxima / -inf / padding).
# Smoke cases lead the list: the tier gate requires smoke to collect first.
_MAX_POOL_SPECIAL_VALUE_CASES = [
    # Smoke
    pytest.param(
        1,
        lambda: torch.full((1, 1, 4), float("-inf"), device=run_device(), dtype=torch.float16),
        id="1d-window-all-neg-inf",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        2,
        lambda: torch.tensor(
            [[[[-1.0, -2.0, -3.0, -4.0]]]], device=run_device(), dtype=torch.float16
        ),
        id="2d-all-negative",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        3,
        lambda: torch.full(
            (1, 1, 3, 3, 3), float("-inf"), device=run_device(), dtype=torch.float16
        ),
        id="3d-window-all-neg-inf",
        marks=pytest.mark.smoke,
    ),
    # 1d full
    pytest.param(
        1,
        lambda: torch.tensor(
            [[[1.0, float("nan"), 3.0, 4.0]]], device=run_device(), dtype=torch.float16
        ),
        id="1d-window-with-nan",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        lambda: torch.tensor(
            [[[float("nan"), 1.0, float("nan"), 0.0]]],
            device=run_device(),
            dtype=torch.float16,
        ),
        id="1d-window-with-multiple-nans",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        lambda: torch.tensor([[[5.0, 5.0, 4.0, 3.0]]], device=run_device(), dtype=torch.float16),
        id="1d-window-with-tied-maxima",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        lambda: torch.tensor(
            [[[-1.0, -2.0, -3.0, -4.0]]], device=run_device(), dtype=torch.float16
        ),
        id="1d-all-negative",
        marks=pytest.mark.full,
    ),
    # 2d full
    pytest.param(
        2,
        lambda: torch.full((1, 1, 4, 4), float("-inf"), device=run_device(), dtype=torch.float16),
        id="2d-window-all-neg-inf",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        lambda: torch.tensor(
            [[[[1.0, float("nan"), 3.0, 4.0]]]], device=run_device(), dtype=torch.float16
        ),
        id="2d-window-with-nan",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        lambda: torch.tensor(
            [[[[float("nan"), 1.0, float("nan"), 0.0]]]],
            device=run_device(),
            dtype=torch.float16,
        ),
        id="2d-window-with-multiple-nans",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        lambda: torch.tensor([[[[5.0, 5.0, 4.0, 3.0]]]], device=run_device(), dtype=torch.float16),
        id="2d-window-with-tied-maxima",
        marks=pytest.mark.full,
    ),
    # 3d full
    pytest.param(
        3,
        lambda: torch.tensor(
            [[[[[1.0, float("nan")], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
            device=run_device(),
            dtype=torch.float16,
        ),
        id="3d-window-with-nan",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        lambda: torch.tensor(
            [[[[[float("nan"), 1.0], [float("nan"), 0.0]], [[2.0, 3.0], [4.0, 5.0]]]]],
            device=run_device(),
            dtype=torch.float16,
        ),
        id="3d-window-with-multiple-nans",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        lambda: torch.tensor(
            [[[[[5.0, 5.0], [4.0, 3.0]], [[2.0, 1.0], [0.0, -1.0]]]]],
            device=run_device(),
            dtype=torch.float16,
        ),
        id="3d-window-with-tied-maxima",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        lambda: torch.full((1, 1, 2, 2, 2), -5.0, device=run_device(), dtype=torch.float16),
        id="3d-all-negative",
        marks=pytest.mark.full,
    ),
]


@pytest.mark.parametrize("return_indices", [False, True])
@pytest.mark.parametrize(
    ("ndim", "input_builder"),
    _MAX_POOL_SPECIAL_VALUE_CASES,
)
def test_max_pool_special_values(
    ndim: int,
    input_builder: Callable[[], torch.Tensor],
    return_indices: bool,
) -> None:
    special_kwargs = {
        1: {"kernel_size": 3, "stride": 1, "padding": 1},
        2: {"kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1)},
        3: {"kernel_size": 2, "stride": 1, "padding": 1},
    }
    x = input_builder()
    pool_kwargs = special_kwargs[ndim]
    ref = max_pool_ref(ndim)(x, **pool_kwargs, return_indices=return_indices)
    op = _max_pool_op_cls(ndim, return_indices)(**pool_kwargs)
    if return_indices:
        out, idx = op(x)
        compare_outputs(out, ref[0], pool_verification(maximum=True))
        compare_outputs(idx, ref[1], pool_verification(maximum=True))
    else:
        out = op(x)
        compare_outputs(out, ref, pool_verification(maximum=True))


_MAX_POOL_COMPILE_CASES = [
    pytest.param(MaxPool1dFwdOp, 1, False, (2, 8, 32), id="max-pool1d"),
    pytest.param(MaxPool1dIndicesFwdOp, 1, True, (2, 8, 32), id="max-pool1d-indices"),
    pytest.param(MaxPool2dFwdOp, 2, False, (2, 8, 16, 16), id="max-pool2d"),
    pytest.param(MaxPool2dIndicesFwdOp, 2, True, (2, 8, 16, 16), id="max-pool2d-indices"),
    pytest.param(MaxPool3dFwdOp, 3, False, (1, 4, 8, 16, 16), id="max-pool3d"),
    pytest.param(MaxPool3dIndicesFwdOp, 3, True, (1, 4, 8, 16, 16), id="max-pool3d-indices"),
]
for _case in _MAX_POOL_COMPILE_CASES:
    register_compile_contract(_case.values[0])


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize(
    ("op_cls", "ndim", "return_indices", "x_shape"),
    _MAX_POOL_COMPILE_CASES,
)
def test_max_pool_compile_fullgraph(
    op_cls: type,
    ndim: int,
    return_indices: bool,
    x_shape: tuple[int, ...],
) -> None:
    ctor_kwargs = {
        1: {"kernel_size": 3, "stride": 2, "padding": 1},
        2: {"kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1)},
        3: {"kernel_size": 3, "stride": 2, "padding": 1},
    }
    op = op_cls(**ctor_kwargs[ndim])
    x = torch.randn(*x_shape, device=run_device(), dtype=torch.float16)
    compiled = torch.compile(op, fullgraph=True)
    out = compiled(x)
    ref = max_pool_ref(ndim)(
        x,
        **ctor_kwargs[ndim],
        return_indices=return_indices,
    )
    if return_indices:
        compare_outputs(out[0], ref[0], pool_verification(maximum=True))
        compare_outputs(out[1], ref[1], pool_verification(maximum=True))
    else:
        compare_outputs(out, ref, pool_verification(maximum=True))
    assert_op_owns_graph_nodes(op, x)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.usefixtures("isolated_dynamo")
def test_pool_compile_two_instances_one_frame() -> None:
    """Second instance through the same frame must not degrade the dispatch key.

    Dynamo generalizes non-static scalar arguments across recompilations;
    an int instance key becomes an unhashable SymInt on the second cold
    compile of the same class.
    """
    x = torch.randn(2, 8, 32, device=run_device(), dtype=torch.float16)
    a = MaxPool1dFwdOp(kernel_size=3, stride=2, padding=1)
    b = MaxPool1dFwdOp(kernel_size=3, stride=1, padding=1)
    compare_outputs(
        torch.compile(a, fullgraph=True)(x),
        F.max_pool1d(x, 3, 2, 1),
        pool_verification(maximum=True),
    )
    compare_outputs(
        torch.compile(b, fullgraph=True)(x),
        F.max_pool1d(x, 3, 1, 1),
        pool_verification(maximum=True),
    )


_AVG_POOL_COMPILE_CASES = [
    pytest.param(AvgPool1dFwdOp, (2, 8, 32), id="avg-pool1d"),
    pytest.param(AvgPool2dFwdOp, (2, 4, 16, 16), id="avg-pool2d"),
    pytest.param(AvgPool3dFwdOp, (2, 4, 8, 16, 16), id="avg-pool3d"),
]
for _case in _AVG_POOL_COMPILE_CASES:
    register_compile_contract(_case.values[0])


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize(("op_cls", "x_shape"), _AVG_POOL_COMPILE_CASES)
def test_avg_pool_compile_fullgraph(op_cls: type, x_shape: tuple) -> None:
    dims = len(x_shape) - 2
    op = op_cls(kernel_size=2, stride=2, padding=0)
    x = torch.randn(*x_shape, device=run_device(), dtype=torch.float16)
    compiled = torch.compile(op, fullgraph=True)
    out = compiled(x)
    ref = getattr(F, f"avg_pool{dims}d")(x, 2, 2, 0)
    compare_outputs(out, ref, pool_verification(maximum=False))
    assert_op_owns_graph_nodes(op, x)


@pytest.mark.parametrize(
    "size, kernel, stride, padding, dtype, offset",
    [
        pytest.param(
            (112, 112),
            (3, 3),
            (2, 2),
            (1, 1),
            torch.float16,
            0,
            id="pad-before-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (32, 32),
            (2, 2),
            (2, 2),
            (0, 0),
            torch.float32,
            0,
            id="no-reach-fp32",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (15, 16),
            (3, 2),
            (2, 2),
            (1, 0),
            torch.bfloat16,
            0,
            id="pad-rows-bf16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (112, 112),
            (3, 3),
            (2, 2),
            (1, 1),
            torch.float16,
            1,
            id="misaligned-fp16",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (16, 32),
            (3, 3),
            (1, 1),
            (1, 1),
            torch.float32,
            0,
            id="reach-both-sides-fp32",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (17, 16),
            (4, 5),
            (3, 4),
            (0, 0),
            torch.float32,
            0,
            id="reach-after-fp32",
            marks=pytest.mark.full,
        ),
    ],
)
def test_avg_pool2d_windows_read_into_registers(
    size, kernel, stride, padding, dtype, offset: int
) -> None:
    """Windows a 16-byte load of an input row covers whole outputs of, summed in torch's
    order and divided as torch divides, so the result matches it bit for bit; an input
    starting ``offset`` elements into its storage is served too."""
    numel = 2 * 3 * size[0] * size[1]
    storage = torch.randn(offset + numel, device=run_device(), dtype=dtype)
    x = storage[offset:].view(2, 3, *size)
    out = AvgPool2dFwdOp(kernel_size=kernel, stride=stride, padding=padding)(x)
    torch.testing.assert_close(out, F.avg_pool2d(x, kernel, stride, padding), rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "make_op, shape",
    [
        pytest.param(lambda: AvgPool1dFwdOp(7, 1, 3), (2, 64, 4096), id="avg-pool1d"),
        pytest.param(lambda: MaxPool1dFwdOp(3, 2, 1), (2, 64, 4096), id="max-pool1d"),
        pytest.param(lambda: AvgPool3dFwdOp(2, 2, 0), (2, 16, 16, 64, 64), id="avg-pool3d"),
        pytest.param(lambda: MaxPool2dFwdOp((1, 8), (1, 8), 0), (2, 64, 32, 256), id="max-pool2d"),
    ],
)
def test_pool_serves_an_input_off_a_vector_boundary(make_op, shape) -> None:
    """A contiguous input starting one element into its storage, which 16-byte loads
    cannot read in place, gives the result an aligned copy of it gives."""
    numel = 1
    for extent in shape:
        numel *= extent
    storage = torch.randn(1 + numel, device=run_device(), dtype=torch.float16)
    x = storage[1:].view(shape)
    op = make_op()
    torch.testing.assert_close(op(x), op(x.clone()), rtol=0, atol=0)


class _PassthroughGenericKernel(Kernel, AvgPool2dFwdInterface):
    supported_archs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every other element per spatial axis: the shape a 2-wide, 2-stride window gives.
        # clone(): custom-op outputs must not alias custom-op inputs.
        return x[:, :, ::2, ::2].clone()


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_avg_pool_runs_a_registered_implementation_preferred_over_the_general_one() -> None:
    """An added implementation inheriting the key's interface runs where it is preferred."""

    class _PreferredPassthrough(_PassthroughGenericKernel):
        preferred_over = frozenset({"avg_pool2d_kernel"})

    state = registry.snapshot()
    try:
        register_kernel_type("AvgPool2dFwdOp", "avg_pool2d_passthrough", _PreferredPassthrough)
        op = AvgPool2dFwdOp(kernel_size=2, target=BUILTIN)
    finally:
        registry.restore(state)

    # A width no 16-byte load divides, so no implementation preferred over the key serves it.
    op(torch.randn(1, 2, 8, 10, device=run_device(), dtype=torch.float16))

    assert [type(k) for k in op.iter_kernels()] == [_PreferredPassthrough]


class AdaptiveAvgPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, output_size, dtype, tune",
            [
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (6, 6),
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
                    id="smoke-spp-6x6-fp16",
                ),
                pytest.param(
                    1,
                    96,
                    28,
                    30,
                    (7, 5),
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-asymmetric-bf16",
                ),
                pytest.param(
                    2,
                    16,
                    56,
                    56,
                    (1, 1),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-global-vector-fp16",
                ),
                pytest.param(
                    1,
                    8,
                    11,
                    7,
                    (3, 1),
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-row-ragged-bins-bf16",
                ),
                pytest.param(
                    1,
                    128,
                    55,
                    57,
                    (7, 7),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-nondiv-fp16",
                ),
                pytest.param(
                    1,
                    4,
                    5,
                    40,
                    (8, 1),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-expanding-rows-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    11,
                    13,
                    (None, 5),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-partial-none-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    10,
                    10,
                    4,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-scalar-output-size-bf16",
                ),
                pytest.param(
                    1,
                    8,
                    9,
                    11,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-all-none-identity-fp16",
                ),
            ],
        ),
    ]


class AdaptiveMaxPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, output_size, dtype, tune",
            [
                pytest.param(
                    2,
                    64,
                    56,
                    56,
                    (6, 6),
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="pool")],
                    id="smoke-spp-6x6-fp16",
                ),
                pytest.param(
                    1,
                    96,
                    28,
                    30,
                    (7, 5),
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-asymmetric-bf16",
                ),
                pytest.param(
                    1,
                    128,
                    55,
                    57,
                    (7, 7),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-nondiv-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    11,
                    13,
                    (5, None),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-partial-none-fp16",
                ),
                # Shape coverage: an output wider than the input gives bins of 1 and 2,
                # so the widest bin exceeds ceil(in/out), and the other axis is ragged.
                pytest.param(
                    1,
                    8,
                    8,
                    8,
                    (12, 3),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-expanding-bins-fp16",
                ),
                # Five planes of sixteen threads in a block of 128: 48 threads idle.
                pytest.param(
                    1,
                    5,
                    9,
                    9,
                    (1, 1),
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-global-partial-warp-bf16",
                ),
                # Shape coverage: a 28x28 plane's lane runs pass 32 bytes, so the lanes read a
                # staged tile (AdaptiveMaxPool2dKernel), not registers loaded from global.
                pytest.param(
                    1,
                    4,
                    28,
                    28,
                    (1, 1),
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-global-staged-lanes-fp16",
                ),
            ],
        ),
    ]


class AdaptiveAvgPool2dTest(AdaptiveAvgPool2dWorkload, TestBase):
    """Adaptive avg-pool reference test."""


class AdaptiveMaxPool2dTest(AdaptiveMaxPool2dWorkload, TestBase):
    """Adaptive max-pool reference test, with or without indices."""


@AdaptiveAvgPool2dFixture
def test_adaptive_avg_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: int | None | tuple[int | None, int | None],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveAvgPool2dTest(n, c_in, h_in, w_in, output_size, dtype)
    op = AdaptiveAvgPool2dFwdOp(output_size)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@AdaptiveMaxPool2dFixture
def test_adaptive_max_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: int | None | tuple[int | None, int | None],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    # Exercise both the plain and the return_indices op on the same config.
    for return_indices in (False, True):
        test = AdaptiveMaxPool2dTest(
            n, c_in, h_in, w_in, output_size, dtype, return_indices=return_indices
        )
        op_cls = AdaptiveMaxPool2dIndicesFwdOp if return_indices else AdaptiveMaxPool2dFwdOp
        op = op_cls(output_size)
        if tune:
            op.request_tune()
        test.check(op, *test.gen_inputs())


_ADAPTIVE_POOL_COMPILE_CASES = [
    pytest.param(AdaptiveAvgPool2dFwdOp, (2, 4, 16, 16), False, id="adaptive-avg-pool2d"),
    pytest.param(AdaptiveMaxPool2dFwdOp, (2, 4, 16, 16), False, id="adaptive-max-pool2d"),
    pytest.param(
        AdaptiveMaxPool2dIndicesFwdOp, (2, 4, 16, 16), True, id="adaptive-max-pool2d-indices"
    ),
]
for _case in _ADAPTIVE_POOL_COMPILE_CASES:
    register_compile_contract(_case.values[0])


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize(
    ("op_cls", "x_shape", "return_indices"),
    _ADAPTIVE_POOL_COMPILE_CASES,
)
def test_adaptive_pool_compile_fullgraph(
    op_cls: type,
    x_shape: tuple[int, ...],
    return_indices: bool,
) -> None:
    op = op_cls(output_size=(4, 4))
    x = torch.randn(*x_shape, device=run_device(), dtype=torch.float16)
    out = torch.compile(op, fullgraph=True)(x)  # cold start: no eager warmup
    if op_cls is AdaptiveAvgPool2dFwdOp:
        ref = F.adaptive_avg_pool2d(x, (4, 4))
        compare_outputs(out, ref, pool_verification(maximum=False))
    elif return_indices:
        ref_out, ref_idx = F.adaptive_max_pool2d(x, (4, 4), return_indices=True)
        compare_outputs(out[0], ref_out, pool_verification(maximum=True))
        compare_outputs(out[1], ref_idx, pool_verification(maximum=True))
    else:
        ref = F.adaptive_max_pool2d(x, (4, 4))
        compare_outputs(out, ref, pool_verification(maximum=True))
    assert_op_owns_graph_nodes(op, x)


@pytest.mark.smoke
def test_adaptive_pool2d_accepts_chw() -> None:
    """A 3-D input keeps its rank on the way out; NCHW cases never exercise that."""
    x = torch.randn(16, 11, 13, device=run_device(), dtype=torch.float16)
    avg = AdaptiveAvgPool2dFwdOp(output_size=(5, 4))(x)
    assert avg.shape == (16, 5, 4)
    compare_outputs(
        avg, F.adaptive_avg_pool2d(x.float(), (5, 4)).to(x.dtype), pool_verification(maximum=False)
    )

    val, idx = AdaptiveMaxPool2dIndicesFwdOp(output_size=(5, 4))(x)
    ref_val, ref_idx = F.adaptive_max_pool2d(x.float(), (5, 4), return_indices=True)
    assert val.shape == (16, 5, 4) and idx.shape == (16, 5, 4)
    compare_outputs(val.float(), ref_val, pool_verification(maximum=True))
    compare_outputs(idx, ref_idx, pool_verification(maximum=True))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_adaptive_avg_pool2d_lanes_follow_window_and_device() -> None:
    """A bin splits over lanes while a lane reads past one full-width access and the grid does not fill the device."""
    facts = device_facts(run_device().index)
    resident = facts.sm_count * facts.threads_per_sm
    cases = [
        ((1, 4, 7, 7), (1, 1), AdaptiveAvgPool2dLanesKernel, 8),
        ((1, resident // 2, 7, 7), (1, 1), AdaptiveAvgPool2dLanesKernel, 2),
        ((1, 4, 8, 8), (8, 1), AdaptiveAvgPool2dKernel, 1),
        ((1, resident, 7, 7), (1, 1), AdaptiveAvgPool2dKernel, 1),
    ]
    for shape, size, implementation, lanes in cases:
        op = AdaptiveAvgPool2dFwdOp(output_size=size, target=BUILTIN)
        op(torch.randn(shape, device=run_device(), dtype=torch.float16))
        (kernel,) = op.iter_kernels()
        assert type(kernel) is implementation, (shape, size)
        assert kernel.config.get("lanes", 1) == lanes, (shape, size)


@pytest.mark.smoke
def test_adaptive_avg_pool2d_misaligned_view() -> None:
    """A contiguous view two bytes past a 16-byte boundary pools like an aligned tensor: the
    lane kernel's 16-byte loads of this global window read an aligned copy."""
    test = AdaptiveAvgPool2dTest(2, 64, 56, 56, (1, 1), torch.float16)
    buf = torch.randn(1 + 2 * 64 * 56 * 56, device=run_device(), dtype=torch.float16)
    x = buf[1:].view(2, 64, 56, 56)
    assert x.data_ptr() % 16 != 0
    out = AdaptiveAvgPool2dFwdOp(output_size=test.output_size)(x)
    compare_outputs(out, test.ref_program(x), test.verification(x))


@pytest.mark.parametrize(
    "op_cls",
    [
        pytest.param(AdaptiveAvgPool2dFwdOp, marks=pytest.mark.smoke),
        pytest.param(AdaptiveMaxPool2dFwdOp, marks=pytest.mark.full),
        pytest.param(AdaptiveMaxPool2dIndicesFwdOp, marks=pytest.mark.full),
    ],
)
def test_adaptive_pool2d_bin_products_past_int32(op_cls) -> None:
    """Bins whose bound products pass int32 before the division: 99999 * 150001.

    torch's CUDA kernel overflows on this call too, so the reference runs on the CPU.
    """
    args = (1, 1, 150001, 16, (100000, 1), torch.float16)
    if op_cls is AdaptiveAvgPool2dFwdOp:
        test = AdaptiveAvgPool2dTest(*args)
    else:
        test = AdaptiveMaxPool2dTest(*args, return_indices=op_cls is AdaptiveMaxPool2dIndicesFwdOp)
    (x,) = test.gen_inputs()
    out = op_cls(output_size=test.output_size)(x)
    ref = test.ref_program(x.cpu())
    for got, expected in zip(
        out if isinstance(out, tuple) else (out,),
        ref if isinstance(ref, tuple) else (ref,),
        strict=True,
    ):
        compare_outputs(got.cpu(), expected, test.verification(x))


@pytest.mark.smoke
def test_adaptive_max_pool2d_indices_nan_window() -> None:
    """A NaN in the window wins, and the recorded index is the last one seen."""
    x = torch.randn(1, 1, 4, 4, device=run_device(), dtype=torch.float16)
    x[0, 0, 1, 1] = float("nan")
    x[0, 0, 3, 3] = float("nan")

    val, idx = AdaptiveMaxPool2dIndicesFwdOp(output_size=(1, 1))(x)
    ref_val, ref_idx = F.adaptive_max_pool2d(x.float(), (1, 1), return_indices=True)
    assert torch.isnan(val.float()).all()
    compare_outputs(idx, ref_idx, pool_verification(maximum=True))


@pytest.mark.smoke
def test_adaptive_max_pool2d_global_nan_plane() -> None:
    """A NaN anywhere in a plane makes its global max NaN, whichever thread reads it."""
    x = torch.randn(2, 64, 7, 7, device=run_device(), dtype=torch.float16)
    x[0, 3, 6, 6] = float("nan")
    x[1, 10, 0, 0] = float("nan")
    out = AdaptiveMaxPool2dFwdOp(output_size=(1, 1))(x)
    torch.testing.assert_close(out, F.adaptive_max_pool2d(x, (1, 1)), equal_nan=True)


@pytest.mark.smoke
@pytest.mark.parametrize("first_sign", [1.0, -1.0], ids=["positive-first", "negative-first"])
def test_adaptive_max_pool2d_global_zero_keeps_first_sign(first_sign: float) -> None:
    """Of +0.0 and -0.0, the one torch meets first is the max, across the threads of a plane."""
    x = torch.full((1, 1, 33), -1.0, device=run_device(), dtype=torch.float16)
    x[0, 0, 1] = 0.0 * first_sign
    x[0, 0, 32] = -0.0 * first_sign
    out = AdaptiveMaxPool2dFwdOp(output_size=(1, 1))(x)
    expected = F.adaptive_max_pool2d(x, (1, 1))
    assert torch.equal(torch.signbit(out), torch.signbit(expected))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    "h_in, implementation",
    [
        pytest.param(7, AdaptiveMaxPool2dRegisterKernel, id="7x7-registers"),
        pytest.param(28, AdaptiveMaxPool2dKernel, id="28x28-staged"),
    ],
)
def test_adaptive_max_pool2d_global_lane_run_selects_kernel(h_in: int, implementation) -> None:
    """A global pool reads planes into registers while a lane's run is at most 32 bytes."""
    op = AdaptiveMaxPool2dFwdOp(output_size=(1, 1), target=BUILTIN)
    op(torch.randn(1, 4, h_in, h_in, device=run_device(), dtype=torch.float16))
    assert [type(k) for k in op.iter_kernels()] == [implementation]


@pytest.mark.parametrize(
    "l_in, kernel_l, stride_l, pad_l, dtype",
    [
        pytest.param(4096, 3, 3, 0, torch.float32, id="in-row-fp32", marks=pytest.mark.smoke),
        pytest.param(4096, 3, 3, 0, torch.bfloat16, id="in-row-bf16", marks=pytest.mark.smoke),
        pytest.param(1000, 2, 3, 1, torch.float32, id="padded-fp32", marks=pytest.mark.full),
        pytest.param(1001, 3, 4, 1, torch.bfloat16, id="padded-bf16", marks=pytest.mark.full),
    ],
)
def test_max_pool1d_windowed_taps_match_torch(
    l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: torch.dtype
) -> None:
    """Windows no wider than their stride read each tap where it lies, with the 4- and
    2-byte element loads; the maxima match torch exactly, a NaN winning."""
    x = torch.randn(4, 8, l_in, device=run_device(), dtype=dtype)
    x.view(-1)[::313] = float("nan")
    out = MaxPool1dFwdOp(kernel_size=kernel_l, stride=stride_l, padding=pad_l)(x)
    expected = F.max_pool1d(x, kernel_l, stride_l, pad_l)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "l_in, kernel_l, stride_l, pad_l, dtype",
    [
        pytest.param(3072, 3, 3, 0, torch.float32, id="in-row-fp32", marks=pytest.mark.smoke),
        pytest.param(3072, 3, 3, 0, torch.bfloat16, id="in-row-bf16", marks=pytest.mark.smoke),
        pytest.param(3070, 2, 3, 1, torch.float32, id="padded-fp32", marks=pytest.mark.full),
        pytest.param(4093, 3, 4, 1, torch.bfloat16, id="padded-bf16", marks=pytest.mark.full),
    ],
)
def test_max_pool1d_indices_windowed_taps_match_torch(
    l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: torch.dtype
) -> None:
    """Windows no wider than their stride, 1024 outputs a row so that a block divides the
    output and each thread stores adjacent outputs as vectors, give torch's values and
    positions with the 4- and 2-byte element loads, the last NaN in a window winning."""
    x = torch.randn(4, 8, l_in, device=run_device(), dtype=dtype)
    # Adjacent pairs put two NaNs in some windows.
    x.view(-1)[::313] = float("nan")
    x.view(-1)[1::313] = float("nan")
    op = MaxPool1dIndicesFwdOp(kernel_size=kernel_l, stride=stride_l, padding=pad_l)
    expected = max_pool_ref(1)(x, kernel_l, stride_l, pad_l, return_indices=True)
    compare_outputs(op(x), expected, pool_verification(maximum=True))


@pytest.mark.parametrize(
    "shape, kernel, stride, padding, dtype, offset, nan",
    [
        pytest.param(
            (2, 3, 112, 112),
            3,
            2,
            1,
            torch.float16,
            0,
            False,
            id="pad-before-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3, 32, 32), 2, 2, 0, torch.float32, 0, True, id="nan-fp32", marks=pytest.mark.smoke
        ),
        pytest.param(
            (2, 3, 15, 16),
            (3, 2),
            2,
            (1, 0),
            torch.bfloat16,
            0,
            False,
            id="pad-rows-bf16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3, 112, 112),
            3,
            2,
            1,
            torch.float16,
            1,
            False,
            id="misaligned-fp16",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 3, 16, 32),
            3,
            1,
            1,
            torch.float32,
            0,
            False,
            id="reach-both-sides-fp32",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 3, 17, 16),
            (4, 5),
            (3, 4),
            0,
            torch.float32,
            0,
            False,
            id="reach-after-fp32",
            marks=pytest.mark.full,
        ),
        # Past 128 MiB the run is read without the evict-first hint.
        pytest.param(
            (1, 1344, 224, 224),
            2,
            2,
            0,
            torch.float16,
            0,
            True,
            id="large-fp16",
            marks=pytest.mark.full,
        ),
    ],
)
def test_max_pool2d_windows_read_into_registers(
    shape, kernel, stride, padding, dtype, offset: int, nan: bool
) -> None:
    """Windows a 16-byte load of an input row covers whole outputs of match torch bit for
    bit, a NaN in a window winning; an input starting ``offset`` elements into its storage
    is served too."""
    numel = math.prod(shape)
    storage = torch.randn(offset + numel, device=run_device(), dtype=dtype)
    x = storage[offset:].view(shape)
    if nan:
        x.view(-1)[::997] = float("nan")
    out = MaxPool2dFwdOp(kernel_size=kernel, stride=stride, padding=padding)(x)
    expected = F.max_pool2d(x, kernel, stride, padding)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "shape, kernel, stride, padding, dtype, offset, fill",
    [
        pytest.param(
            (2, 3, 112, 112),
            3,
            2,
            1,
            torch.float16,
            0,
            "ties",
            id="pad-ties-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3, 112, 112),
            3,
            2,
            1,
            torch.bfloat16,
            1,
            "randn",
            id="misaligned-bf16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3, 32, 32), 2, 2, 0, torch.float32, 0, "nan", id="nan-fp32", marks=pytest.mark.smoke
        ),
        pytest.param(
            (2, 3, 16, 32),
            3,
            1,
            1,
            torch.float32,
            0,
            "ties",
            id="reach-both-ties-fp32",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3, 16, 32),
            3,
            2,
            1,
            torch.float32,
            0,
            "-inf",
            id="all-neg-inf-fp32",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 3, 17, 16),
            (4, 5),
            (3, 4),
            0,
            torch.float32,
            0,
            "nan",
            id="reach-after-nan-fp32",
            marks=pytest.mark.smoke,
        ),
        # Three threads of eight positions: two sectors each, and the last lane pair
        # has one valid owner.
        pytest.param(
            (1, 1, 3, 8),
            3,
            1,
            1,
            torch.float16,
            0,
            "ties",
            id="odd-lanes-two-sectors-fp16",
            marks=pytest.mark.full,
        ),
    ],
)
def test_max_pool2d_indices_windows_read_into_registers(
    shape, kernel, stride, padding, dtype, offset: int, fill: str
) -> None:
    """Windows a 16-byte load of an input row covers whole outputs of give torch's values
    and positions: a tie keeps the first position, the last NaN in a window wins, a window
    of nothing but -inf reports its first tap, and an edge read in place of padding
    reports the edge. An input starting ``offset`` elements into its storage is served
    too."""
    numel = math.prod(shape)
    if fill == "ties":
        storage = torch.randint(0, 3, (offset + numel,), device=run_device()).to(dtype)
    elif fill == "-inf":
        storage = torch.full((offset + numel,), float("-inf"), device=run_device(), dtype=dtype)
    else:
        storage = torch.randn(offset + numel, device=run_device(), dtype=dtype)
    x = storage[offset:].view(shape)
    if fill == "nan":
        x.view(-1)[::97] = float("nan")
        x.view(-1)[1::97] = float("nan")
    out, indices = MaxPool2dIndicesFwdOp(kernel_size=kernel, stride=stride, padding=padding)(x)
    expected, expected_indices = F.max_pool2d(x, kernel, stride, padding, return_indices=True)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(indices, expected_indices)


class MeanPoolingTest(MeanPoolingWorkload, TestBase):
    pass


class MeanPoolingFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, chunk_size, dtype, tune, seq_lens",
            [
                pytest.param(
                    1, 8192, 64, 128, 64, torch.float16, False, None, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 8192, 64, 128, 64, torch.float16, True, None, marks=pytest.mark.full
                ),
                pytest.param(
                    2, 2048, 64, 128, 64, torch.float16, False, None, marks=pytest.mark.full
                ),
                # Ragged, every sequence ending mid-chunk: the case that divides by a short
                # count.
                pytest.param(
                    1,
                    1000,
                    64,
                    128,
                    32,
                    torch.float16,
                    False,
                    [100, 200, 300, 400],
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 2048, 64, 128, 64, torch.float16, False, [1024, 1024], marks=pytest.mark.full
                ),
            ],
        ),
    ]


@MeanPoolingFixture
def test_mean_pooling_op(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
    seq_lens: list[int] | None,
) -> None:
    test = MeanPoolingTest(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        dtype=dtype,
        seq_lens=seq_lens,
    )
    op = MeanPoolingFwdOp(chunk_size=chunk_size, accum_dtype=torch.float32)
    if tune:
        op.request_tune()
    inputs = list(test.gen_inputs())
    if seq_lens is not None:
        # Output rows follow the caller's chunk order, not the canonical one.
        inputs[2] = inputs[2].flip(0).contiguous()
    if seq_lens is not None and run_device_is_cuda():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(*inputs)
        graph.replay()
        test.check(op, *inputs, runs=lambda *args: output)
    else:
        test.check(op, *inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [100, 256])
def test_mean_pooling_dim_not_one_full_tile(dim: int) -> None:
    """`dim` values the manifest allows but no workload row carries.

    A block takes a power-of-two share of `heads * dim`: `dim` 100 gives a width of 200
    that no share divides, so the last block is bounds-tested; 256 gives 512 that they
    divide. The range the manifest rules out between them is a contract bound, not a shape
    the kernel cannot read.
    """
    test = MeanPoolingTest(
        batch=1,
        seq_len=64,
        heads=2,
        dim=dim,
        chunk_size=32,
        dtype=torch.float16,
    )
    op = MeanPoolingFwdOp(chunk_size=32, accum_dtype=torch.float32)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_mean_pooling_reads_its_shapes_off_the_call() -> None:
    """One op, two shapes, two chunk counts — what taking shapes from the call buys."""
    op = MeanPoolingFwdOp(chunk_size=32, accum_dtype=torch.float32, validate_inputs=True)
    short = torch.randn(1, 64, 2, 64, device=run_device(), dtype=torch.float16)
    assert tuple(op(short).shape) == (1, 2, 2, 64)
    tall = torch.randn(3, 128, 2, 64, device=run_device(), dtype=torch.float16)
    assert tuple(op(tall).shape) == (3, 4, 2, 64)
