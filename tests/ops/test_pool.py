from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import AvgPool2dSpatialKernel, AvgPool3dKernel
from tileops.ops import AvgPool1dFwdOp, AvgPool2dFwdOp, AvgPool3dFwdOp


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AvgPool1dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune", [
            pytest.param(
                2, 64, 512, 3, None, 1, False, True, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-k3-default-stride-fp16",
            ),
            pytest.param(
                2, 64, 512, 3, None, 1, False, True, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-k3-default-stride-bf16",
            ),
            pytest.param(
                2, 64, 512, 3, None, 1, False, True, torch.float32, False,
                marks=pytest.mark.smoke,
                id="smoke-k3-default-stride-fp32",
            ),
            pytest.param(
                2, 32, 257, 5, 2, 2, False, False, torch.float16, False,
                marks=pytest.mark.full,
                id="full-k5-s2-no-pad-count-fp16",
            ),
            pytest.param(
                1, 48, 255, 4, 2, 1, True, True, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-ceil-bf16",
            ),
        ]),
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
        n=n,
        c_in=c_in,
        l_in=l_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(n, c_in, l_in), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_avg_pool1d_rejects_wrong_tuple_arity() -> None:
    with pytest.raises(ValueError, match="kernel_size must be an int or a tuple of 1 ints"):
        AvgPool1dFwdOp(n=1, c_in=32, l_in=128, kernel_size=(3, 4))


@pytest.mark.smoke
def test_avg_pool1d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool1dFwdOp(n=1, c_in=32, l_in=128, kernel_size=3, stride=0)


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
    base_kwargs = {"n": 1, "c_in": 32, "l_in": 128, "kernel_size": 3}
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool1dFwdOp(**base_kwargs)


@pytest.mark.smoke
def test_avg_pool1d_rejects_non_3d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool1dFwdOp(
        n=2,
        c_in=8,
        l_in=16,
        kernel_size=3,
        stride=1,
        padding=1,
        dtype=torch.float32,
        kernel_map={"avg_pool1d_kernel": _DummyKernel},
    )
    x = torch.randn(2, 8, 16, 4)
    with pytest.raises(ValueError, match="expects input to be a 3D NCL tensor"):
        op(x)


class AvgPool2dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune", [
            pytest.param(
                2, 64, 56, 56, (3, 3), None, (1, 1), False, True, None, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-3x3-default-stride-fp16",
            ),
            pytest.param(
                2, 64, 56, 56, (3, 3), None, (1, 1), False, True, None, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-3x3-default-stride-bf16",
            ),
            pytest.param(
                1, 32, 28, 28, (3, 3), None, (1, 1), False, True, None, torch.float32, False,
                marks=pytest.mark.smoke,
                id="smoke-3x3-default-stride-fp32",
            ),
            pytest.param(
                1, 128, 55, 57, (3, 5), (2, 2), (1, 2), True, False, None, torch.float16, False,
                marks=pytest.mark.full,
                id="full-ceil-no-pad-count-fp16",
            ),
            pytest.param(
                1, 96, 28, 30, (2, 3), (2, 2), (0, 1), False, True, 5, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-divisor-override-bf16",
            ),
            pytest.param(
                1, 7, 9, 10, (3, 3), (2, 2), (1, 1), False, False, None, torch.float16, False,
                marks=pytest.mark.full,
                id="full-no-ceil-no-pad-count-fp16",
            ),
            pytest.param(
                1, 5, 10, 11, (3, 3), (2, 2), (1, 1), True, True, None, torch.float32, False,
                marks=pytest.mark.full,
                id="full-ceil-pad-count-fp32",
            ),
            pytest.param(
                2, 6, 9, 13, (2, 3), (2, 2), (0, 1), True, True, 7, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-ceil-pad-count-divisor-bf16",
            ),
            pytest.param(
                1, 9, 11, 12, (3, 5), (2, 3), (1, 2), True, False, 7, torch.float16, False,
                marks=pytest.mark.full,
                id="full-ceil-no-pad-count-divisor-fp16",
            ),
            pytest.param(
                1, 8, 9, 9, (3, 3), (2, 2), (1, 1), False, False, 7, torch.float16, False,
                marks=pytest.mark.full,
                id="full-no-ceil-no-pad-count-divisor-fp16",
            ),
        ]),
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
        n=n,
        c_in=c_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(n, c_in, h_in, w_in), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_avg_pool2d_dispatches_kernel() -> None:
    op = AvgPool2dFwdOp(
        n=1,
        c_in=32,
        h_in=28,
        w_in=28,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
    )
    assert isinstance(op.kernel, AvgPool2dSpatialKernel)


@pytest.mark.smoke
def test_avg_pool2d_rejects_non_positive_output_size() -> None:
    with pytest.raises(ValueError, match="output size must be greater than zero"):
        AvgPool2dFwdOp(
            n=1,
            c_in=1,
            h_in=2,
            w_in=2,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            ceil_mode=False,
            count_include_pad=True,
        )


@pytest.mark.smoke
def test_avg_pool2d_rejects_zero_divisor_override() -> None:
    with pytest.raises(ValueError, match="divisor_override must not be zero"):
        AvgPool2dFwdOp(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(3, 3),
            divisor_override=0,
        )


@pytest.mark.smoke
def test_avg_pool2d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool2dFwdOp(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(3, 3),
            stride=(1, 0),
        )


@pytest.mark.smoke
def test_avg_pool2d_rejects_invalid_padding() -> None:
    with pytest.raises(ValueError, match="padding must be at most half"):
        AvgPool2dFwdOp(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
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
    base_kwargs = {
        "n": 1,
        "c_in": 8,
        "h_in": 16,
        "w_in": 16,
        "kernel_size": (3, 3),
    }
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool2dFwdOp(**base_kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool2d_negative_divisor_override_matches_torch() -> None:
    x = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float16).contiguous()
    op = AvgPool2dFwdOp(
        n=1,
        c_in=4,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        divisor_override=-1,
        dtype=torch.float16,
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
        n=2,
        c_in=8,
        h_in=16,
        w_in=16,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dtype=torch.float32,
        kernel_map={"avg_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(2, 8, 16)
    with pytest.raises(ValueError, match="expects input to be a 4D NCHW tensor"):
        op(x)


@pytest.mark.smoke
def test_avg_pool2d_rejects_wrong_nchw_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool2dFwdOp(
        n=2,
        c_in=8,
        h_in=16,
        w_in=16,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dtype=torch.float32,
        kernel_map={"avg_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(2, 16, 16, 8)
    with pytest.raises(ValueError, match=r"expects input shape \(2, 8, 16, 16\)"):
        op(x)


class AvgPool3dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune", [
            pytest.param(
                1, 32, 16, 28, 28, (2, 2, 2), None, (0, 0, 0), False, True, None, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-2x2x2-default-stride-fp16",
            ),
            pytest.param(
                1, 32, 16, 28, 28, (2, 2, 2), None, (0, 0, 0), False, True, None, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-2x2x2-default-stride-bf16",
            ),
            pytest.param(
                1, 16, 8, 14, 14, (2, 2, 2), None, (0, 0, 0), False, True, None, torch.float32, False,
                marks=pytest.mark.smoke,
                id="smoke-2x2x2-default-stride-fp32",
            ),
            pytest.param(
                1, 48, 15, 25, 27, (2, 3, 3), (2, 2, 2), (1, 1, 1), True, False, None, torch.float16, False,
                marks=pytest.mark.full,
                id="full-ceil-no-pad-count-fp16",
            ),
            pytest.param(
                1, 24, 10, 20, 22, (2, 2, 3), (2, 2, 2), (0, 1, 1), False, True, 7, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-divisor-override-bf16",
            ),
        ]),
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
        n=n,
        c_in=c_in,
        d_in=d_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(n, c_in, d_in, h_in, w_in), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_avg_pool3d_dispatches_kernel() -> None:
    op = AvgPool3dFwdOp(
        n=1,
        c_in=16,
        d_in=8,
        h_in=16,
        w_in=16,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
    )
    assert isinstance(op.kernel, AvgPool3dKernel)


@pytest.mark.smoke
def test_avg_pool3d_rejects_zero_divisor_override() -> None:
    with pytest.raises(ValueError, match="divisor_override must not be zero"):
        AvgPool3dFwdOp(
            n=1,
            c_in=8,
            d_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(2, 2, 2),
            divisor_override=0,
        )


@pytest.mark.smoke
def test_avg_pool3d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool3dFwdOp(
            n=1,
            c_in=8,
            d_in=8,
            h_in=16,
            w_in=16,
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
    base_kwargs = {
        "n": 1,
        "c_in": 8,
        "d_in": 8,
        "h_in": 16,
        "w_in": 16,
        "kernel_size": (2, 2, 2),
    }
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool3dFwdOp(**base_kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool3d_negative_divisor_override_matches_torch() -> None:
    x = torch.randn(1, 3, 4, 6, 6, device="cuda", dtype=torch.float16).contiguous()
    op = AvgPool3dFwdOp(
        n=1,
        c_in=3,
        d_in=4,
        h_in=6,
        w_in=6,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        divisor_override=-1,
        dtype=torch.float16,
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
        n=1,
        c_in=4,
        d_in=8,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        dtype=torch.float32,
        kernel_map={"avg_pool3d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 4, 8, 8)
    with pytest.raises(ValueError, match="expects input to be a 5D NCDHW tensor"):
        op(x)


@pytest.mark.smoke
def test_avg_pool3d_rejects_wrong_ncdhw_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool3dFwdOp(
        n=1,
        c_in=4,
        d_in=8,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        dtype=torch.float32,
        kernel_map={"avg_pool3d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 8, 8, 8, 4)
    with pytest.raises(ValueError, match=r"expects input shape \(1, 4, 8, 8, 8\)"):
        op(x)


@pytest.mark.smoke
def test_avg_pool1d_preferred_api_infers_shape_and_dtype() -> None:
    x = torch.randn(2, 4, 16, dtype=torch.float16, device="cuda")
    op = AvgPool1dFwdOp(kernel_size=3, stride=2, padding=1)

    y = op(x)
    ref = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1)

    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_avg_pool2d_preferred_api_infers_shape_and_dtype() -> None:
    x = torch.randn(2, 4, 16, 18, dtype=torch.float16, device="cuda")
    op = AvgPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    y = op(x)
    ref = F.avg_pool2d(x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_avg_pool3d_preferred_api_infers_shape_and_dtype() -> None:
    x = torch.randn(1, 4, 8, 10, 12, dtype=torch.float16, device="cuda")
    op = AvgPool3dFwdOp(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

    y = op(x)
    ref = F.avg_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_avg_pool2d_committed_input_shape_mismatch_raises() -> None:
    x = torch.randn(2, 4, 16, 18, dtype=torch.float16, device="cuda")
    op = AvgPool2dFwdOp(
        n=2,
        c_in=4,
        h_in=15,
        w_in=18,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="Expected input shape"):
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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
