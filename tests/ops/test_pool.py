from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import AvgPool1dKernel, AvgPool2dKernel, AvgPool3dKernel
from tileops.ops import AvgPool1dOp, AvgPool2dOp, AvgPool3dOp


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
        x = torch.randn(n, l_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool1d(
            x.permute(0, 2, 1).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )
        return out.permute(0, 2, 1).contiguous()


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
    op = AvgPool1dOp(
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
def test_avg_pool1d_dispatches_kernel() -> None:
    op = AvgPool1dOp(n=1, c_in=32, l_in=128, kernel_size=3, stride=2, padding=1)
    assert isinstance(op.kernel, AvgPool1dKernel)


@pytest.mark.smoke
def test_avg_pool1d_rejects_wrong_tuple_arity() -> None:
    with pytest.raises(ValueError, match="kernel_size must be an int or a tuple of 1 ints"):
        AvgPool1dOp(n=1, c_in=8, l_in=32, kernel_size=(3, 4))


@pytest.mark.smoke
def test_avg_pool1d_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be greater than zero"):
        AvgPool1dOp(n=1, c_in=8, l_in=32, kernel_size=3, stride=0)


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
    base_kwargs = {
        "n": 1,
        "c_in": 8,
        "l_in": 32,
        "kernel_size": 3,
    }
    base_kwargs.update(kwargs)
    with pytest.raises(TypeError, match=match):
        AvgPool1dOp(**base_kwargs)


@pytest.mark.smoke
def test_avg_pool1d_forward_warns_on_ambiguous_nlc_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool1dOp(
        n=1,
        c_in=8,
        l_in=8,
        kernel_size=2,
        stride=2,
        kernel_map={"avg_pool1d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 8, 8)
    with pytest.warns(UserWarning, match="ambiguous NLC shape"):
        out = op(x)
    assert out is x


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
                1, 128, 55, 57, (3, 5), (2, 2), (1, 2), True, False, None, torch.float16, False,
                marks=pytest.mark.full,
                id="full-ceil-no-pad-count-fp16",
            ),
            pytest.param(
                1, 96, 28, 30, (2, 3), (2, 2), (0, 1), False, True, 5, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-divisor-override-bf16",
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
        x = torch.randn(n, h_in, w_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool2d(
            x.permute(0, 3, 1, 2).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
        return out.permute(0, 2, 3, 1).contiguous()


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
    op = AvgPool2dOp(
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
    op = AvgPool2dOp(n=1, c_in=32, h_in=28, w_in=28, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    assert isinstance(op.kernel, AvgPool2dKernel)


@pytest.mark.smoke
def test_avg_pool2d_rejects_zero_divisor_override() -> None:
    with pytest.raises(ValueError, match="divisor_override must not be zero"):
        AvgPool2dOp(
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
        AvgPool2dOp(
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
        AvgPool2dOp(
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
        AvgPool2dOp(**base_kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool2d_negative_divisor_override_matches_torch() -> None:
    x = torch.randn(1, 8, 8, 4, device="cuda", dtype=torch.float16).contiguous()
    op = AvgPool2dOp(
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
        x.permute(0, 3, 1, 2).contiguous(),
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        divisor_override=-1,
    ).permute(0, 2, 3, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_avg_pool2d_forward_rejects_nchw_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool2dOp(
        n=1,
        c_in=4,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        kernel_map={"avg_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 4, 8, 8)
    with pytest.raises(ValueError, match="NHWC"):
        op(x)


@pytest.mark.smoke
def test_avg_pool2d_forward_warns_on_ambiguous_nhwc_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool2dOp(
        n=1,
        c_in=8,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        kernel_map={"avg_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 8, 8, 8)
    with pytest.warns(UserWarning, match="ambiguous NHWC shape"):
        out = op(x)
    assert out is x


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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

    def gen_inputs(self, n: int, c_in: int, d_in: int, h_in: int, w_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, d_in, h_in, w_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool3d(
            x.permute(0, 4, 1, 2, 3).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
        return out.permute(0, 2, 3, 4, 1).contiguous()


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
    op = AvgPool3dOp(
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
    op = AvgPool3dOp(
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
        AvgPool3dOp(
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
        AvgPool3dOp(
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
        AvgPool3dOp(**base_kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool3d_negative_divisor_override_matches_torch() -> None:
    x = torch.randn(1, 4, 6, 6, 3, device="cuda", dtype=torch.float16).contiguous()
    op = AvgPool3dOp(
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
        x.permute(0, 4, 1, 2, 3).contiguous(),
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        divisor_override=-1,
    ).permute(0, 2, 3, 4, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_avg_pool3d_forward_rejects_ncdhw_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool3dOp(
        n=1,
        c_in=3,
        d_in=4,
        h_in=6,
        w_in=6,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        kernel_map={"avg_pool3d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 3, 4, 6, 6)
    with pytest.raises(ValueError, match="NDHWC"):
        op(x)


@pytest.mark.smoke
def test_avg_pool3d_forward_warns_on_ambiguous_ndhwc_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = AvgPool3dOp(
        n=1,
        c_in=4,
        d_in=4,
        h_in=4,
        w_in=4,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        kernel_map={"avg_pool3d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 4, 4, 4, 4)
    with pytest.warns(UserWarning, match="ambiguous NDHWC shape"):
        out = op(x)
    assert out is x


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
