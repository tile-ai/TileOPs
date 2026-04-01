import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.pool import AvgPool1dKernel
from tileops.ops import AvgPool1dOp


class AvgPool1dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune", [
            pytest.param(
                2, 64, 512, 3, None, 1, False, True, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-k3-default-stride-fp16",
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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
