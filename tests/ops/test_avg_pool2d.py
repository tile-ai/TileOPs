from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import AvgPool2dKernel
from tileops.ops import AvgPool2dOp


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
        kernel_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]],
        padding: Tuple[int, int],
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
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    padding: Tuple[int, int],
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
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
