from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import (
    AvgPool1dKernel,
    AvgPool1dSpatialKernel,
    AvgPool2dSpatialKernel,
    AvgPool3dKernel,
    AvgPool3dSpatialKernel,
    MaxPool2dKernel,
)
from tileops.ops import AvgPool1dFwdOp, AvgPool2dFwdOp, AvgPool3dFwdOp, MaxPool2dFwdOp


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
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


class AvgPool1dTest(TestBase):
    def __init__(
        self,
        kernel_size: int,
        stride: int | None,
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def gen_inputs(self, n: int, c_in: int, l_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, c_in, l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )


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
    test = AvgPool1dTest(kernel_size, stride, padding, ceil_mode, count_include_pad, dtype)
    op = AvgPool1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        tune=tune,
    )
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(n, c_in, l_in), atol=atol, rtol=rtol)
    expected_kernel = (
        AvgPool1dSpatialKernel if not ceil_mode and count_include_pad else AvgPool1dKernel
    )
    assert isinstance(op.kernel, expected_kernel)


@pytest.mark.smoke
def test_avg_pool1d_rejects_wrong_tuple_arity() -> None:
    with pytest.raises(ValueError, match="kernel_size must be an int or a tuple of 1 ints"):
        AvgPool1dFwdOp(kernel_size=(3, 4))


@pytest.mark.smoke
def test_avg_pool1d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool1dFwdOp(kernel_size=3, stride=0)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kernel_size": True}, "kernel_size must be an int or a tuple of 1 ints"),
        ({"stride": True}, "stride must be an int or a tuple of 1 ints"),
        ({"padding": True}, "padding must be an int or a tuple of 1 ints"),
    ],
)
def test_avg_pool1d_rejects_bool_pool_params(kwargs: dict[str, object], match: str) -> None:
    base_kwargs = {"kernel_size": 3}
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool1dFwdOp(**base_kwargs)


@pytest.mark.smoke
def test_avg_pool1d_rejects_non_3d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool1dFwdOp(
        kernel_size=3,
        stride=1,
        padding=1,
        kernel_map={"avg_pool1d_kernel": _DummyKernel},
    )
    x = torch.randn(2, 8, 16, 4)
    with pytest.raises(ValueError, match="expects input to be a 3D NCL tensor"):
        op(x)


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
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
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


class AvgPool2dTest(TestBase):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self, n: int, c_in: int, h_in: int, w_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, c_in, h_in, w_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


@AvgPool2dFixture
def test_avg_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool2dTest(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    op = AvgPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        tune=tune,
    )
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(n, c_in, h_in, w_in), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_avg_pool2d_dispatches_kernel() -> None:
    op = AvgPool2dFwdOp(
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
    )
    x = torch.randn(1, 32, 28, 28, device="cuda", dtype=torch.float16).contiguous()
    op(x)
    assert isinstance(op.kernel, AvgPool2dSpatialKernel)


@pytest.mark.smoke
def test_avg_pool2d_rejects_non_positive_output_size() -> None:
    op = AvgPool2dFwdOp(
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=True,
    )
    x = torch.randn(1, 1, 2, 2, device="cuda", dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match="output size must be greater than zero"):
        op(x)


@pytest.mark.smoke
def test_avg_pool2d_rejects_zero_divisor_override() -> None:
    with pytest.raises(ValueError, match="divisor_override must not be zero"):
        AvgPool2dFwdOp(
            kernel_size=(3, 3),
            divisor_override=0,
        )


@pytest.mark.smoke
def test_avg_pool2d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool2dFwdOp(
            kernel_size=(3, 3),
            stride=(1, 0),
        )


@pytest.mark.smoke
def test_avg_pool2d_rejects_invalid_padding() -> None:
    with pytest.raises(ValueError, match="padding must be at most half"):
        AvgPool2dFwdOp(
            kernel_size=(3, 3),
            padding=(2, 1),
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kernel_size": True}, "kernel_size must be an int or a tuple of 2 ints"),
        ({"stride": True}, "stride must be an int or a tuple of 2 ints"),
        ({"padding": True}, "padding must be an int or a tuple of 2 ints"),
        ({"kernel_size": (3, True)}, "kernel_size must contain only ints"),
        ({"divisor_override": True}, "divisor_override must be an int or None"),
        ({"divisor_override": 1.5}, "divisor_override must be an int or None"),
    ],
)
def test_avg_pool2d_rejects_invalid_param_types(kwargs: dict[str, object], match: str) -> None:
    base_kwargs = {"kernel_size": (3, 3)}
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool2dFwdOp(**base_kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool2d_negative_divisor_override_matches_torch() -> None:
    x = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float16).contiguous()
    op = AvgPool2dFwdOp(
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        divisor_override=-1,
    )
    out = op(x)
    ref = F.avg_pool2d(
        x,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        divisor_override=-1,
    )
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_avg_pool2d_rejects_non_4d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool2dFwdOp(
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        kernel_map={"avg_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(2, 8, 16)
    with pytest.raises(ValueError, match="expects input to be a 4D NCHW tensor"):
        op(x)


class AvgPool3dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
            [
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
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-2x2x2-default-stride-fp16",
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
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-2x2x2-default-stride-bf16",
                ),
                pytest.param(
                    1,
                    16,
                    8,
                    14,
                    14,
                    (2, 2, 2),
                    None,
                    (0, 0, 0),
                    False,
                    True,
                    None,
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-2x2x2-default-stride-fp32",
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
                    marks=pytest.mark.full,
                    id="full-divisor-override-bf16",
                ),
            ],
        ),
    ]


class AvgPool3dTest(TestBase):
    def __init__(
        self,
        kernel_size: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]],
        padding: tuple[int, int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
    ) -> tuple[torch.Tensor]:
        x = torch.randn(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


@AvgPool3dFixture
def test_avg_pool3d(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool3dTest(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    op = AvgPool3dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        tune=tune,
    )
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(n, c_in, d_in, h_in, w_in), atol=atol, rtol=rtol)
    expected_kernel = (
        AvgPool3dSpatialKernel
        if not ceil_mode and count_include_pad and divisor_override is None
        else AvgPool3dKernel
    )
    assert isinstance(op.kernel, expected_kernel)


@pytest.mark.smoke
def test_avg_pool3d_rejects_zero_divisor_override() -> None:
    with pytest.raises(ValueError, match="divisor_override must not be zero"):
        AvgPool3dFwdOp(
            kernel_size=(2, 2, 2),
            divisor_override=0,
        )


@pytest.mark.smoke
def test_avg_pool3d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool3dFwdOp(
            kernel_size=(2, 2, 2),
            stride=(2, 0, 2),
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kernel_size": True}, "kernel_size must be an int or a tuple of 3 ints"),
        ({"stride": True}, "stride must be an int or a tuple of 3 ints"),
        ({"padding": True}, "padding must be an int or a tuple of 3 ints"),
        ({"kernel_size": (2, 2, True)}, "kernel_size must contain only ints"),
        ({"divisor_override": True}, "divisor_override must be an int or None"),
        ({"divisor_override": 1.5}, "divisor_override must be an int or None"),
    ],
)
def test_avg_pool3d_rejects_invalid_param_types(kwargs: dict[str, object], match: str) -> None:
    base_kwargs = {"kernel_size": (2, 2, 2)}
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool3dFwdOp(**base_kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool3d_negative_divisor_override_matches_torch() -> None:
    x = torch.randn(1, 3, 4, 6, 6, device="cuda", dtype=torch.float16).contiguous()
    op = AvgPool3dFwdOp(
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        divisor_override=-1,
    )
    out = op(x)
    ref = F.avg_pool3d(
        x,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        divisor_override=-1,
    )
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_avg_pool3d_rejects_non_5d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool3dFwdOp(
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        kernel_map={"avg_pool3d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 4, 8, 8)
    with pytest.raises(ValueError, match="expects input to be a 5D NCDHW tensor"):
        op(x)


@pytest.mark.smoke
def test_avg_pool2d_dynamic_shape_kernel_cache_and_roofline() -> None:
    op = AvgPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x1 = torch.randn(1, 4, 16, 16, dtype=torch.float16, device="cuda")
    x2 = torch.randn(2, 4, 16, 16, dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="requires a prior forward"):
        op.eval_roofline()

    op(x1)
    assert len(op._kernel_cache) == 1
    flops, nbytes = op.eval_roofline()
    assert flops > 0
    assert nbytes > 0

    op(x1)
    assert len(op._kernel_cache) == 1

    op(x2)
    assert len(op._kernel_cache) == 2


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [AvgPool1dFwdOp, AvgPool2dFwdOp, AvgPool3dFwdOp])
def test_avg_pool_dynamic_dtype_ignores_last_runtime_dtype(
    op_cls: type[AvgPool1dFwdOp | AvgPool2dFwdOp | AvgPool3dFwdOp],
) -> None:
    op = op_cls(kernel_size=2)
    op.dtype = torch.float16

    op._validate_dtypes(torch.empty((), dtype=torch.bfloat16))


class MaxPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune, contiguous",
            [
                # Smoke representative: one config across all supported dtypes.
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
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
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
                    2,
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
                # Full: cover distinct setting combinations (one dtype each).
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
                # ResNet stem manifest workload across dtypes.
                pytest.param(
                    32,
                    64,
                    112,
                    112,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    False,
                    torch.float16,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-resnet-stem-fp16",
                ),
                pytest.param(
                    32,
                    64,
                    112,
                    112,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    False,
                    torch.bfloat16,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-resnet-stem-bf16",
                ),
                pytest.param(
                    32,
                    64,
                    112,
                    112,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    False,
                    torch.float32,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-resnet-stem-fp32",
                ),
                # VGG block manifest workload across dtypes.
                pytest.param(
                    16,
                    128,
                    56,
                    56,
                    (2, 2),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    False,
                    torch.float16,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-vgg-block-fp16",
                ),
                pytest.param(
                    16,
                    128,
                    56,
                    56,
                    (2, 2),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    False,
                    torch.bfloat16,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-vgg-block-bf16",
                ),
                pytest.param(
                    16,
                    128,
                    56,
                    56,
                    (2, 2),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    False,
                    torch.float32,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-vgg-block-fp32",
                ),
                # AlexNet-style ceil manifest workload across dtypes.
                pytest.param(
                    32,
                    64,
                    55,
                    55,
                    (3, 3),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    True,
                    torch.float16,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-alexnet-ceil-fp16",
                ),
                pytest.param(
                    32,
                    64,
                    55,
                    55,
                    (3, 3),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    True,
                    torch.bfloat16,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-alexnet-ceil-bf16",
                ),
                pytest.param(
                    32,
                    64,
                    55,
                    55,
                    (3, 3),
                    (2, 2),
                    (0, 0),
                    (1, 1),
                    True,
                    torch.float32,
                    False,
                    True,
                    marks=pytest.mark.full,
                    id="full-alexnet-ceil-fp32",
                ),
            ],
        ),
    ]


class MaxPool2dTest(TestBase):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        ceil_mode: bool,
        dtype: torch.dtype,
        contiguous: bool = True,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.contiguous = contiguous

    def gen_inputs(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
    ) -> tuple[torch.Tensor]:
        x = torch.randn(n, c_in, h_in, w_in, device="cuda", dtype=self.dtype)
        if self.contiguous:
            x = x.contiguous()
        else:
            # Non-contiguous NCHW view: transpose twice so strides differ but
            # shape semantics stay NCHW.
            x = x.transpose(2, 3).contiguous().transpose(2, 3)
            assert not x.is_contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=False,
        )


@MaxPool2dFixture
def test_max_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
    contiguous: bool,
) -> None:
    test = MaxPool2dTest(
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        contiguous=contiguous,
    )
    op = MaxPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(n, c_in, h_in, w_in), atol=0, rtol=0)


@pytest.mark.parametrize(
    ("case_name", "input_builder"),
    [
        pytest.param(
            "all_negative",
            lambda: torch.tensor(
                [[[[-1.0, -2.0, -3.0, -4.0]]]], device="cuda", dtype=torch.float16
            ),
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            "window_all_neg_inf",
            lambda: torch.full((1, 1, 4, 4), float("-inf"), device="cuda", dtype=torch.float16),
            marks=pytest.mark.full,
        ),
        pytest.param(
            "window_with_nan",
            lambda: torch.tensor(
                [[[[1.0, float("nan"), 3.0, 4.0]]]], device="cuda", dtype=torch.float16
            ),
            marks=pytest.mark.full,
        ),
        pytest.param(
            "padding_does_not_win_over_negative",
            lambda: torch.full((1, 1, 4, 4), -5.0, device="cuda", dtype=torch.float16),
            marks=pytest.mark.full,
        ),
    ],
)
def test_max_pool2d_special_values(case_name: str, input_builder: callable) -> None:
    op = MaxPool2dFwdOp(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    x = input_builder()
    out = op(x)
    ref = F.max_pool2d(x, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), return_indices=False)
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "exc_type", "match"),
    [
        ({"kernel_size": True}, TypeError, "kernel_size must be an int or a tuple of 2 ints"),
        ({"stride": True}, TypeError, "stride must be an int or a tuple of 2 ints"),
        ({"padding": True}, TypeError, "padding must be an int or a tuple of 2 ints"),
        ({"dilation": True}, TypeError, "dilation must be an int or a tuple of 2 ints"),
        ({"kernel_size": (3, True)}, TypeError, "kernel_size must contain only ints"),
        ({"kernel_size": (3, 3), "stride": (1, 0)}, ValueError, "stride must be greater than zero"),
        (
            {"kernel_size": (3, 3), "dilation": (0, 1)},
            ValueError,
            "dilation must be greater than zero",
        ),
        ({"kernel_size": (3, 3), "padding": (2, 1)}, ValueError, "padding must be at most half"),
        ({"kernel_size": (3, 3), "padding": (-1, 0)}, ValueError, "padding must be non-negative"),
        ({"kernel_size": (3, 3), "ceil_mode": "true"}, TypeError, "ceil_mode must be a bool"),
    ],
)
def test_max_pool2d_rejects_invalid_params(
    kwargs: dict[str, object], exc_type: type[Exception], match: str
) -> None:
    base_kwargs = {"kernel_size": (3, 3)}
    base_kwargs.update(kwargs)
    with pytest.raises(exc_type, match=match):
        MaxPool2dFwdOp(**base_kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("case_name", "extra_kwargs", "input_spec", "expected_match", "needs_dummy_kernel"),
    [
        pytest.param(
            "non_4d_input",
            {},
            ((2, 8, 16), None),
            "expects input to be a 4D NCHW tensor",
            True,
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            "cpu_input",
            {},
            ((1, 1, 8, 8), None),
            "input must be a CUDA tensor",
            False,
            marks=pytest.mark.full,
        ),
        pytest.param(
            "unsupported_dtype",
            {},
            ((1, 1, 8, 8), torch.float64),
            "input.dtype must be float16, bfloat16, or float32",
            False,
            marks=pytest.mark.full,
        ),
        pytest.param(
            "non_positive_output_size",
            {"kernel_size": (5, 5), "stride": (1, 1), "padding": (0, 0)},
            ((1, 1, 2, 2), torch.float16),
            "output size must be greater than zero",
            False,
            marks=pytest.mark.full,
        ),
    ],
)
def test_max_pool2d_rejects_invalid_input(
    case_name: str,
    extra_kwargs: dict[str, object],
    input_spec: tuple[tuple[int, ...], torch.dtype | None],
    expected_match: str,
    needs_dummy_kernel: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {"kernel_size": (3, 3)}
    kwargs.update(extra_kwargs)
    if needs_dummy_kernel:
        monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
        kwargs["kernel_map"] = {"max_pool2d_kernel": MaxPool2dKernel}
    op = MaxPool2dFwdOp(**kwargs)

    shape, dtype = input_spec
    x = torch.randn(*shape) if dtype is None else torch.randn(*shape, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=expected_match):
        op(x)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d_dynamic_shape_kernel_cache_and_roofline() -> None:
    op = MaxPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x1 = torch.randn(1, 4, 16, 16, dtype=torch.float16, device="cuda")
    x2 = torch.randn(2, 4, 16, 16, dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="requires a prior forward"):
        op.eval_roofline()

    op(x1)
    assert len(op._kernel_cache) == 1
    flops, nbytes = op.eval_roofline()
    assert flops > 0
    assert nbytes > 0

    op(x1)
    assert len(op._kernel_cache) == 1

    op(x2)
    assert len(op._kernel_cache) == 2


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d_compile_fullgraph() -> None:
    op = MaxPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x = torch.randn(2, 8, 16, 16, device="cuda", dtype=torch.float16)
    # Warm up the kernel cache so torch.compile traces only the custom-op call.
    op(x)
    compiled = torch.compile(op, fullgraph=True)
    out = compiled(x)
    ref = F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), return_indices=False)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("input_size", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "expected"),
    [
        pytest.param(7, 3, 2, 1, 1, False, 4, marks=pytest.mark.smoke),
        pytest.param(7, 3, 2, 1, 2, False, 3, marks=pytest.mark.full),
        pytest.param(7, 3, 2, 1, 1, True, 4, marks=pytest.mark.full),
        pytest.param(55, 3, 2, 0, 1, True, 27, marks=pytest.mark.full),
        pytest.param(56, 2, 2, 0, 1, False, 28, marks=pytest.mark.full),
        # Default dilation regression: omitting dilation must equal explicit dilation=1.
        pytest.param(56, 3, 2, 1, 1, False, "default_matches_explicit", marks=pytest.mark.full),
    ],
)
def test_pool_output_dim_with_dilation(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
    expected: int | str,
) -> None:
    from tileops.kernels.pool.common import pool_output_dim

    if expected == "default_matches_explicit":
        default = pool_output_dim(input_size, kernel_size, stride, padding, ceil_mode)
        explicit = pool_output_dim(
            input_size, kernel_size, stride, padding, ceil_mode, dilation=dilation
        )
        assert default == explicit
    else:
        assert (
            pool_output_dim(input_size, kernel_size, stride, padding, ceil_mode, dilation)
            == expected
        )


@pytest.mark.parametrize(
    ("dilation", "valid"),
    [
        pytest.param((1, 1), True, marks=pytest.mark.smoke),
        pytest.param((0, 1), False, marks=pytest.mark.full),
    ],
)
def test_validate_pool_params_with_dilation(dilation: tuple[int, int], valid: bool) -> None:
    from tileops.kernels.pool.common import validate_pool_params

    if valid:
        validate_pool_params(
            ndim=2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=dilation,
        )
    else:
        with pytest.raises(ValueError, match="dilation must be greater than zero"):
            validate_pool_params(
                ndim=2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dilation=dilation,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
