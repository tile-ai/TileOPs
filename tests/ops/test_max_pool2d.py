from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import MaxPool2dKernel
from tileops.ops import MaxPool2dOp


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros_like(x, dtype=torch.int64)


class MaxPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, return_indices, ceil_mode, dtype, tune",
            [
                pytest.param(
                    2, 64, 56, 56, (3, 3), None, (1, 1), (1, 1), False, False, torch.float16, False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-3x3-default-stride-fp16",
                ),
                pytest.param(
                    1, 96, 29, 31, (3, 5), (2, 2), (1, 2), (1, 1), False, True, torch.float16, False,
                    marks=pytest.mark.full,
                    id="full-ceil-nonpow2-fp16",
                ),
                pytest.param(
                    1, 80, 28, 30, (3, 3), (2, 2), (1, 1), (2, 1), False, False, torch.bfloat16, False,
                    marks=pytest.mark.full,
                    id="full-dilation-bf16",
                ),
                pytest.param(
                    1, 32, 16, 18, (2, 3), (2, 2), (0, 1), (1, 1), True, False, torch.float16, False,
                    marks=pytest.mark.full,
                    id="full-return-indices-fp16",
                ),
            ],
        ),
    ]


class MaxPool2dTest(TestBase):
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        return_indices: bool,
        ceil_mode: bool,
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.dtype = dtype

    def gen_inputs(self, n: int, c_in: int, h_in: int, w_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, h_in, w_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = F.max_pool2d(
            x.permute(0, 3, 1, 2).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices,
            ceil_mode=self.ceil_mode,
        )
        if self.return_indices:
            values, indices = out
            return (
                values.permute(0, 2, 3, 1).contiguous(),
                indices.permute(0, 2, 3, 1).contiguous(),
            )
        return out.permute(0, 2, 3, 1).contiguous()


@MaxPool2dFixture
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    return_indices: bool,
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool2dTest(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        ceil_mode=ceil_mode,
        dtype=dtype,
    )
    op = MaxPool2dOp(
        n=n,
        c_in=c_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        ceil_mode=ceil_mode,
        dtype=dtype,
        tune=tune,
    )
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3 if dtype == torch.float16 else 1.6e-2
    test.check(op, *test.gen_inputs(n, c_in, h_in, w_in), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_max_pool2d_dispatches_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
    op = MaxPool2dOp(
        n=1,
        c_in=32,
        h_in=28,
        w_in=28,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
    )
    assert isinstance(op.kernel, MaxPool2dKernel)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d_returns_indices_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
    op = MaxPool2dOp(
        n=1,
        c_in=4,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        return_indices=True,
        kernel_map={"max_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 8, 8, 4, device="cuda", dtype=torch.float16)
    values, indices = op(x)
    assert values is x
    assert indices.dtype == torch.int64
    assert indices.shape == x.shape


@pytest.mark.smoke
def test_max_pool2d_rejects_non_positive_dilation() -> None:
    with pytest.raises(ValueError, match="dilation must be greater than zero"):
        MaxPool2dOp(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(3, 3),
            dilation=(1, 0),
        )


@pytest.mark.smoke
def test_max_pool2d_rejects_invalid_padding_for_effective_kernel() -> None:
    with pytest.raises(ValueError, match="padding must be at most half"):
        MaxPool2dOp(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(3, 3),
            padding=(3, 1),
            dilation=(2, 1),
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dilation": True}, "dilation must be an int or a tuple of 2 ints"),
        ({"dilation": (1, True)}, "dilation must contain only ints"),
        ({"kernel_size": True}, "kernel_size must be an int or a tuple of 2 ints"),
        ({"stride": True}, "stride must be an int or a tuple of 2 ints"),
        ({"padding": True}, "padding must be an int or a tuple of 2 ints"),
    ],
)
def test_max_pool2d_rejects_invalid_param_types(kwargs: dict[str, object], match: str) -> None:
    base_kwargs = {
        "n": 1,
        "c_in": 8,
        "h_in": 16,
        "w_in": 16,
        "kernel_size": (3, 3),
    }
    base_kwargs.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        MaxPool2dOp(**base_kwargs)


@pytest.mark.smoke
def test_max_pool2d_rejects_unsupported_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
    with pytest.raises(ValueError, match="only supports dtypes"):
        MaxPool2dOp(
            n=1,
            c_in=8,
            h_in=16,
            w_in=16,
            kernel_size=(3, 3),
            dtype=torch.float32,
        )


@pytest.mark.smoke
def test_max_pool2d_forward_rejects_non_cuda_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
    op = MaxPool2dOp(
        n=1,
        c_in=4,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        kernel_map={"max_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 8, 8, 4)
    with pytest.raises(ValueError, match="CUDA"):
        op(x)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d_forward_rejects_nchw_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
    op = MaxPool2dOp(
        n=1,
        c_in=4,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        kernel_map={"max_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="NHWC"):
        op(x)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d_forward_warns_on_ambiguous_nhwc_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tileops.ops.op.get_sm_version", lambda: 80)
    op = MaxPool2dOp(
        n=1,
        c_in=8,
        h_in=8,
        w_in=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        kernel_map={"max_pool2d_kernel": _DummyKernel},
    )
    x = torch.randn(1, 8, 8, 8, device="cuda", dtype=torch.float16)
    with pytest.warns(UserWarning, match="ambiguous NHWC shape"):
        out = op(x)
    assert out is x


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
