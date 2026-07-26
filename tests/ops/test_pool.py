import inspect
from typing import Callable, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.compile_contract import register_compile_contract
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import (
    AvgPool1dKernel,
    AvgPool1dSpatialKernel,
    AvgPool2dSpatialKernel,
    AvgPool3dKernel,
    AvgPool3dSpatialKernel,
    MaxPool1dKernel,
    MaxPool1dWithIndicesKernel,
    MaxPool2dKernel,
    MaxPool2dWithIndicesKernel,
    MaxPool3dKernel,
    MaxPool3dWithIndicesKernel,
)
from tileops.ops import (
    AvgPool1dFwdOp,
    AvgPool2dFwdOp,
    AvgPool3dFwdOp,
    MaxPool1dFwdOp,
    MaxPool1dIndicesFwdOp,
    MaxPool2dFwdOp,
    MaxPool2dIndicesFwdOp,
    MaxPool3dFwdOp,
    MaxPool3dIndicesFwdOp,
)


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# AvgPool family
# ---------------------------------------------------------------------------

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
                    2, 64, 512, 3, None, 1, False, True, torch.float16, False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-k3-default-stride-fp16"),
                pytest.param(
                    2, 64, 512, 3, None, 1, False, True, torch.bfloat16, False,
                    marks=pytest.mark.smoke, id="smoke-k3-default-stride-bf16"),
                pytest.param(
                    2, 64, 512, 3, None, 1, False, True, torch.float32, False,
                    marks=pytest.mark.smoke, id="smoke-k3-default-stride-fp32"),
                pytest.param(
                    2, 32, 257, 5, 2, 2, False, False, torch.float16, False,
                    marks=pytest.mark.full, id="full-k5-s2-no-pad-count-fp16"),
                pytest.param(
                    1, 48, 255, 4, 2, 1, True, True, torch.bfloat16, False,
                    marks=pytest.mark.full, id="full-ceil-bf16"),
            ],
        ),
    ]


class AvgPool2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
            [
                pytest.param(
                    2, 64, 56, 56, (3, 3), None, (1, 1), False, True, None,
                    torch.float16, False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-3x3-default-stride-fp16"),
                pytest.param(
                    2, 64, 56, 56, (3, 3), None, (1, 1), False, True, None,
                    torch.bfloat16, False,
                    marks=pytest.mark.smoke, id="smoke-3x3-default-stride-bf16"),
                pytest.param(
                    1, 32, 28, 28, (3, 3), None, (1, 1), False, True, None,
                    torch.float32, False,
                    marks=pytest.mark.smoke, id="smoke-3x3-default-stride-fp32"),
                pytest.param(
                    1, 128, 55, 57, (3, 5), (2, 2), (1, 2), True, False, None,
                    torch.float16, False,
                    marks=pytest.mark.full, id="full-ceil-no-pad-count-fp16"),
                pytest.param(
                    1, 96, 28, 30, (2, 3), (2, 2), (0, 1), False, True, 5,
                    torch.bfloat16, False,
                    marks=pytest.mark.full, id="full-divisor-override-bf16"),
                pytest.param(
                    1, 7, 9, 10, (3, 3), (2, 2), (1, 1), False, False, None,
                    torch.float16, False,
                    marks=pytest.mark.full, id="full-no-ceil-no-pad-count-fp16"),
                pytest.param(
                    1, 5, 10, 11, (3, 3), (2, 2), (1, 1), True, True, None,
                    torch.float32, False,
                    marks=pytest.mark.full, id="full-ceil-pad-count-fp32"),
                pytest.param(
                    2, 6, 9, 13, (2, 3), (2, 2), (0, 1), True, True, 7,
                    torch.bfloat16, False,
                    marks=pytest.mark.full, id="full-ceil-pad-count-divisor-bf16"),
                pytest.param(
                    1, 9, 11, 12, (3, 5), (2, 3), (1, 2), True, False, 7,
                    torch.float16, False,
                    marks=pytest.mark.full, id="full-ceil-no-pad-count-divisor-fp16"),
                pytest.param(
                    1, 8, 9, 9, (3, 3), (2, 2), (1, 1), False, False, 7,
                    torch.float16, False,
                    marks=pytest.mark.full, id="full-no-ceil-no-pad-count-divisor-fp16"),
            ],
        ),
    ]


class AvgPool3dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
            [
                pytest.param(
                    1, 32, 16, 28, 28, (2, 2, 2), None, (0, 0, 0), False, True,
                    None, torch.float16, False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-2x2x2-default-stride-fp16"),
                pytest.param(
                    1, 32, 16, 28, 28, (2, 2, 2), None, (0, 0, 0), False, True,
                    None, torch.bfloat16, False,
                    marks=pytest.mark.smoke, id="smoke-2x2x2-default-stride-bf16"),
                pytest.param(
                    1, 16, 8, 14, 14, (2, 2, 2), None, (0, 0, 0), False, True,
                    None, torch.float32, False,
                    marks=pytest.mark.smoke, id="smoke-2x2x2-default-stride-fp32"),
                pytest.param(
                    1, 48, 15, 25, 27, (2, 3, 3), (2, 2, 2), (1, 1, 1), True, False,
                    None, torch.float16, False,
                    marks=pytest.mark.full, id="full-ceil-no-pad-count-fp16"),
                pytest.param(
                    1, 24, 10, 20, 22, (2, 2, 3), (2, 2, 2), (0, 1, 1), False, True,
                    7, torch.bfloat16, False,
                    marks=pytest.mark.full, id="full-divisor-override-bf16"),
            ],
        ),
    ]


class AvgPoolTest(TestBase):
    """Dim-generic avg-pool reference harness (divisor_override is 2d/3d-only)."""

    def __init__(
        self,
        ndim: int,
        kernel_size: int | tuple[int, ...],
        stride: Optional[int | tuple[int, ...]],
        padding: int | tuple[int, ...],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, object] = {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "ceil_mode": self.ceil_mode,
            "count_include_pad": self.count_include_pad,
        }
        if self.ndim > 1:
            kwargs["divisor_override"] = self.divisor_override
        return getattr(F, f"avg_pool{self.ndim}d")(input, **kwargs)


def _avg_pool_expected_kernel(
    ndim: int,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> Optional[type[Kernel]]:
    """Per-dim kernel-dispatch expectation for the main correctness test.

    2d dispatch is covered by test_avg_pool2d_dispatches_kernel instead.
    """
    if ndim == 1:
        return AvgPool1dSpatialKernel if not ceil_mode and count_include_pad else AvgPool1dKernel
    if ndim == 3:
        return (
            AvgPool3dSpatialKernel
            if not ceil_mode and count_include_pad and divisor_override is None
            else AvgPool3dKernel
        )
    return None


def _run_avg_pool_case(
    ndim: int,
    shape: tuple[int, ...],
    kernel_size: int | tuple[int, ...],
    stride: Optional[int | tuple[int, ...]],
    padding: int | tuple[int, ...],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
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
        "tune": tune,
    }
    if ndim > 1:
        op_kwargs["divisor_override"] = divisor_override
    op = _AVG_POOL_OPS[ndim](**op_kwargs)
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(*shape), atol=atol, rtol=rtol)
    expected_kernel = _avg_pool_expected_kernel(
        ndim, ceil_mode, count_include_pad, divisor_override
    )
    if expected_kernel is not None:
        assert isinstance(op.kernel, expected_kernel)


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
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
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
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
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
@pytest.mark.parametrize(
    ("op_cls", "kwargs", "exc_type", "match"),
    [
        # 1d
        pytest.param(
            AvgPool1dFwdOp, {"kernel_size": (3, 4)}, ValueError,
            "kernel_size must be an int or a tuple of 1 ints", id="1d-wrong-tuple-arity"),
        pytest.param(
            AvgPool1dFwdOp, {"kernel_size": 3, "stride": 0}, ValueError,
            "stride must be greater than zero", id="1d-zero-stride"),
        pytest.param(
            AvgPool1dFwdOp, {"kernel_size": True}, TypeError,
            "kernel_size must be an int or a tuple of 1 ints", id="1d-bool-kernel-size"),
        pytest.param(
            AvgPool1dFwdOp, {"kernel_size": 3, "stride": True}, TypeError,
            "stride must be an int or a tuple of 1 ints", id="1d-bool-stride"),
        pytest.param(
            AvgPool1dFwdOp, {"kernel_size": 3, "padding": True}, TypeError,
            "padding must be an int or a tuple of 1 ints", id="1d-bool-padding"),
        # 2d
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "divisor_override": 0}, ValueError,
            "divisor_override must not be zero", id="2d-zero-divisor-override"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "stride": (1, 0)}, ValueError,
            "stride must be greater than zero", id="2d-zero-stride"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "padding": (2, 1)}, ValueError,
            "padding must be at most half", id="2d-padding-too-large"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": True}, TypeError,
            "kernel_size must be an int or a tuple of 2 ints", id="2d-bool-kernel-size"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "stride": True}, TypeError,
            "stride must be an int or a tuple of 2 ints", id="2d-bool-stride"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "padding": True}, TypeError,
            "padding must be an int or a tuple of 2 ints", id="2d-bool-padding"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, True)}, TypeError,
            "kernel_size must contain only ints", id="2d-kernel-size-contents"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "divisor_override": True}, TypeError,
            "divisor_override must be an int or None", id="2d-bool-divisor-override"),
        pytest.param(
            AvgPool2dFwdOp, {"kernel_size": (3, 3), "divisor_override": 1.5}, TypeError,
            "divisor_override must be an int or None", id="2d-float-divisor-override"),
        # 3d
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, 2), "divisor_override": 0}, ValueError,
            "divisor_override must not be zero", id="3d-zero-divisor-override"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, 2), "stride": (2, 0, 2)}, ValueError,
            "stride must be greater than zero", id="3d-zero-stride"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": True}, TypeError,
            "kernel_size must be an int or a tuple of 3 ints", id="3d-bool-kernel-size"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, 2), "stride": True}, TypeError,
            "stride must be an int or a tuple of 3 ints", id="3d-bool-stride"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, 2), "padding": True}, TypeError,
            "padding must be an int or a tuple of 3 ints", id="3d-bool-padding"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, True)}, TypeError,
            "kernel_size must contain only ints", id="3d-kernel-size-contents"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, 2), "divisor_override": True}, TypeError,
            "divisor_override must be an int or None", id="3d-bool-divisor-override"),
        pytest.param(
            AvgPool3dFwdOp, {"kernel_size": (2, 2, 2), "divisor_override": 1.5}, TypeError,
            "divisor_override must be an int or None", id="3d-float-divisor-override"),
    ],
)
def test_avg_pool_rejects_invalid_params(
    op_cls: type,
    kwargs: dict[str, object],
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        op_cls(**kwargs)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
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
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    pool_kwargs = {
        "kernel_size": (2,) * ndim,
        "stride": (2,) * ndim,
        "padding": (0,) * ndim,
        "divisor_override": -1,
    }
    op = _AVG_POOL_OPS[ndim](**pool_kwargs)
    out = op(x)
    ref = getattr(F, f"avg_pool{ndim}d")(x, **pool_kwargs)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ndim", "ctor_kwargs", "kernel_slot", "bad_shape", "match"),
    [
        pytest.param(
            1,
            {"kernel_size": 3, "stride": 1, "padding": 1},
            "avg_pool1d_kernel",
            (2, 8, 16, 4),
            "expects input to be a 3D NCL tensor",
            id="avg-pool1d",
        ),
        pytest.param(
            2,
            {"kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1)},
            "avg_pool2d_kernel",
            (2, 8, 16),
            "expects input to be a 4D NCHW tensor",
            id="avg-pool2d",
        ),
        pytest.param(
            3,
            {"kernel_size": (2, 2, 2), "stride": (2, 2, 2), "padding": (0, 0, 0)},
            "avg_pool3d_kernel",
            (1, 4, 8, 8),
            "expects input to be a 5D NCDHW tensor",
            id="avg-pool3d",
        ),
    ],
)
def test_avg_pool_rejects_wrong_rank_input(
    ndim: int,
    ctor_kwargs: dict[str, object],
    kernel_slot: str,
    bad_shape: tuple[int, ...],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
    op = _AVG_POOL_OPS[ndim](kernel_map={kernel_slot: _DummyKernel}, **ctor_kwargs)
    x = torch.randn(*bad_shape)
    with pytest.raises(ValueError, match=match):
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


# ---------------------------------------------------------------------------
# MaxPool family
# ---------------------------------------------------------------------------

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

_MAX_POOL_KERNEL_SLOTS: dict[type, str] = {
    MaxPool1dFwdOp: "max_pool1d_kernel",
    MaxPool1dIndicesFwdOp: "max_pool1d_with_indices_kernel",
    MaxPool2dFwdOp: "max_pool2d_kernel",
    MaxPool2dIndicesFwdOp: "max_pool2d_with_indices_kernel",
    MaxPool3dFwdOp: "max_pool3d_kernel",
    MaxPool3dIndicesFwdOp: "max_pool3d_with_indices_kernel",
}

_MAX_POOL_DUMMY_KERNELS: dict[type, type[Kernel]] = {
    MaxPool1dFwdOp: MaxPool1dKernel,
    MaxPool1dIndicesFwdOp: MaxPool1dWithIndicesKernel,
    MaxPool2dFwdOp: MaxPool2dKernel,
    MaxPool2dIndicesFwdOp: MaxPool2dWithIndicesKernel,
    MaxPool3dFwdOp: MaxPool3dKernel,
    MaxPool3dIndicesFwdOp: MaxPool3dWithIndicesKernel,
}


def _max_pool_ref(ndim: int) -> Callable:
    return getattr(F, f"max_pool{ndim}d")


def _max_pool_op_cls(ndim: int, return_indices: bool) -> type:
    return _MAX_POOL_INDICES_OPS[ndim] if return_indices else _MAX_POOL_OPS[ndim]


_MAX_POOL1D_PARAMS = [
    # Smoke: one config across all supported dtypes.
    pytest.param(
        2, 8, 64, (3,), (2,), (1,), (1,), False, torch.float16, False, True,
        marks=[pytest.mark.smoke, pytest.mark.packaging],
        id="smoke-k3-s2-p1-fp16"),
    pytest.param(
        2, 8, 64, (3,), (2,), (1,), (1,), False, torch.bfloat16, False, True,
        marks=pytest.mark.smoke, id="smoke-k3-s2-p1-bf16"),
    pytest.param(
        1, 8, 64, (3,), (2,), (1,), (1,), False, torch.float32, False, True,
        marks=pytest.mark.smoke, id="smoke-k3-s2-p1-fp32"),
    # Full: distinct setting combinations.
    pytest.param(
        1, 4, 63, (3,), None, (1,), (2,), False, torch.float16, False, True,
        marks=pytest.mark.full, id="full-default-stride-dilation-fp16"),
    pytest.param(
        1, 4, 97, (5,), (3,), (2,), (1,), True, torch.float16, False, True,
        marks=pytest.mark.full, id="full-ceil-k5-s3-p2-fp16"),
    pytest.param(
        2, 8, 64, (3,), (2,), (1,), (1,), False, torch.float16, False, False,
        marks=pytest.mark.full, id="full-noncontiguous-k3-fp16"),
    pytest.param(
        1, 4, 97, (5,), (3,), (2,), (1,), True, torch.bfloat16, False, True,
        marks=pytest.mark.full, id="full-ceil-k5-s3-p2-bf16"),
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
        2, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False,
        torch.float16, False, True,
        marks=[pytest.mark.smoke, pytest.mark.packaging],
        id="smoke-3x3-s2-p1-fp16"),
    pytest.param(
        2, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False,
        torch.bfloat16, False, True,
        marks=pytest.mark.smoke, id="smoke-3x3-s2-p1-bf16"),
    pytest.param(
        1, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False,
        torch.float32, False, True,
        marks=pytest.mark.smoke, id="smoke-3x3-s2-p1-fp32"),
    # Full: distinct setting combinations.
    pytest.param(
        1, 4, 14, 14, (3, 3), None, (1, 1), (2, 1), False,
        torch.float16, False, True,
        marks=pytest.mark.full, id="full-default-stride-dilation-fp16"),
    pytest.param(
        1, 4, 23, 27, (3, 5), (2, 3), (1, 2), (1, 1), True,
        torch.float16, False, True,
        marks=pytest.mark.full, id="full-nonsquare-ceil-fp16"),
    pytest.param(
        2, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False,
        torch.float16, False, False,
        marks=pytest.mark.full, id="full-noncontiguous-3x3-fp16"),
    pytest.param(
        1, 4, 23, 27, (3, 5), (2, 3), (1, 2), (1, 1), True,
        torch.bfloat16, False, True,
        marks=pytest.mark.full, id="full-nonsquare-ceil-bf16"),
]


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
        2, 4, 8, 16, 16, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), False,
        torch.float16, False, True,
        marks=[pytest.mark.smoke, pytest.mark.packaging],
        id="smoke-k2-s2-fp16"),
    pytest.param(
        2, 4, 8, 16, 16, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), False,
        torch.bfloat16, False, True,
        marks=pytest.mark.smoke, id="smoke-k2-s2-bf16"),
    pytest.param(
        1, 4, 8, 16, 16, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), False,
        torch.float32, False, True,
        marks=pytest.mark.smoke, id="smoke-k2-s2-fp32"),
    # Full: distinct setting combinations.
    pytest.param(
        1, 4, 6, 14, 14, (3, 3, 3), None, (1, 1, 1), (2, 1, 1), False,
        torch.float16, False, True,
        marks=pytest.mark.full, id="full-default-stride-dilation-fp16"),
    pytest.param(
        1, 4, 7, 23, 27, (1, 3, 5), (1, 2, 3), (0, 1, 2), (1, 1, 1), True,
        torch.float16, False, True,
        marks=pytest.mark.full, id="full-noncube-ceil-fp16"),
    pytest.param(
        2, 4, 8, 16, 16, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), False,
        torch.float16, False, False,
        marks=pytest.mark.full, id="full-noncontiguous-k2-fp16"),
    pytest.param(
        1, 4, 7, 23, 27, (1, 3, 5), (1, 2, 3), (0, 1, 2), (1, 1, 1), True,
        torch.bfloat16, False, True,
        marks=pytest.mark.full, id="full-noncube-ceil-bf16"),
]


class MaxPool3dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune, contiguous",
            _MAX_POOL3D_PARAMS,
        ),
    ]


class MaxPoolTest(TestBase):
    """Dim-generic max-pool reference harness."""

    def __init__(
        self,
        ndim: int,
        kernel_size: tuple[int, ...],
        stride: Optional[tuple[int, ...]],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        ceil_mode: bool,
        dtype: torch.dtype,
        contiguous: bool = True,
        return_indices: bool = False,
    ) -> None:
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.contiguous = contiguous
        self.return_indices = return_indices

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device="cuda", dtype=self.dtype)
        if self.contiguous:
            x = x.contiguous()
        else:
            # Non-contiguous view: transpose the last two dims twice so strides
            # differ but shape semantics stay N,C,<spatial dims>.
            x = x.transpose(-2, -1).contiguous().transpose(-2, -1)
            assert not x.is_contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return _max_pool_ref(self.ndim)(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


def _run_max_pool_case(
    ndim: int,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    stride: Optional[tuple[int, ...]],
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
            tune=tune,
        )
        test.check(op, *test.gen_inputs(*shape), atol=0, rtol=0)


@MaxPool1dFixture
def test_max_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
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
    stride: Optional[tuple[int, int]],
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
    stride: Optional[tuple[int, int, int]],
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


# Per-dim pool config for the special-values tests.
_MAX_POOL_SPECIAL_KWARGS: dict[int, dict[str, object]] = {
    1: {"kernel_size": 3, "stride": 1, "padding": 1},
    2: {"kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1)},
    3: {"kernel_size": 2, "stride": 1, "padding": 1},
}

# Curated per-dim special-value inputs (NaN / tied maxima / -inf / padding).
# Smoke cases lead the list: the tier gate requires smoke to collect first.
_MAX_POOL_SPECIAL_VALUE_CASES = [
    # Smoke
    pytest.param(
        1,
        "window_all_neg_inf",
        lambda: torch.full((1, 1, 4), float("-inf"), device="cuda", dtype=torch.float16),
        id="1d-window-all-neg-inf",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        2,
        "all_negative",
        lambda: torch.tensor([[[[-1.0, -2.0, -3.0, -4.0]]]], device="cuda", dtype=torch.float16),
        id="2d-all-negative",
        marks=pytest.mark.smoke,
    ),
    pytest.param(
        3,
        "window_all_neg_inf",
        lambda: torch.full((1, 1, 3, 3, 3), float("-inf"), device="cuda", dtype=torch.float16),
        id="3d-window-all-neg-inf",
        marks=pytest.mark.smoke,
    ),
    # 1d full
    pytest.param(
        1,
        "window_with_nan",
        lambda: torch.tensor([[[1.0, float("nan"), 3.0, 4.0]]], device="cuda", dtype=torch.float16),
        id="1d-window-with-nan",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        "window_with_multiple_nans",
        lambda: torch.tensor(
            [[[float("nan"), 1.0, float("nan"), 0.0]]],
            device="cuda",
            dtype=torch.float16,
        ),
        id="1d-window-with-multiple-nans",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        "window_with_tied_maxima",
        lambda: torch.tensor([[[5.0, 5.0, 4.0, 3.0]]], device="cuda", dtype=torch.float16),
        id="1d-window-with-tied-maxima",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        "all_negative",
        lambda: torch.tensor([[[-1.0, -2.0, -3.0, -4.0]]], device="cuda", dtype=torch.float16),
        id="1d-all-negative",
        marks=pytest.mark.full,
    ),
    pytest.param(
        1,
        "padding_does_not_win_over_negative",
        lambda: torch.full((1, 1, 4), -5.0, device="cuda", dtype=torch.float16),
        id="1d-padding-does-not-win",
        marks=pytest.mark.full,
    ),
    # 2d full
    pytest.param(
        2,
        "window_all_neg_inf",
        lambda: torch.full((1, 1, 4, 4), float("-inf"), device="cuda", dtype=torch.float16),
        id="2d-window-all-neg-inf",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        "window_with_nan",
        lambda: torch.tensor(
            [[[[1.0, float("nan"), 3.0, 4.0]]]], device="cuda", dtype=torch.float16
        ),
        id="2d-window-with-nan",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        "window_with_multiple_nans",
        lambda: torch.tensor(
            [[[[float("nan"), 1.0, float("nan"), 0.0]]]],
            device="cuda",
            dtype=torch.float16,
        ),
        id="2d-window-with-multiple-nans",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        "window_with_tied_maxima",
        lambda: torch.tensor([[[[5.0, 5.0, 4.0, 3.0]]]], device="cuda", dtype=torch.float16),
        id="2d-window-with-tied-maxima",
        marks=pytest.mark.full,
    ),
    pytest.param(
        2,
        "padding_does_not_win_over_negative",
        lambda: torch.full((1, 1, 4, 4), -5.0, device="cuda", dtype=torch.float16),
        id="2d-padding-does-not-win",
        marks=pytest.mark.full,
    ),
    # 3d full
    pytest.param(
        3,
        "window_with_nan",
        lambda: torch.tensor(
            [[[[[1.0, float("nan")], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
            device="cuda",
            dtype=torch.float16,
        ),
        id="3d-window-with-nan",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        "window_with_multiple_nans",
        lambda: torch.tensor(
            [[[[[float("nan"), 1.0], [float("nan"), 0.0]], [[2.0, 3.0], [4.0, 5.0]]]]],
            device="cuda",
            dtype=torch.float16,
        ),
        id="3d-window-with-multiple-nans",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        "window_with_tied_maxima",
        lambda: torch.tensor(
            [[[[[5.0, 5.0], [4.0, 3.0]], [[2.0, 1.0], [0.0, -1.0]]]]],
            device="cuda",
            dtype=torch.float16,
        ),
        id="3d-window-with-tied-maxima",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        "all_negative",
        lambda: torch.full((1, 1, 2, 2, 2), -5.0, device="cuda", dtype=torch.float16),
        id="3d-all-negative",
        marks=pytest.mark.full,
    ),
    pytest.param(
        3,
        "padding_does_not_win_over_negative",
        lambda: torch.full((1, 1, 3, 3, 3), -5.0, device="cuda", dtype=torch.float16),
        id="3d-padding-does-not-win",
        marks=pytest.mark.full,
    ),
]


@pytest.mark.parametrize("return_indices", [False, True])
@pytest.mark.parametrize(
    ("ndim", "case_name", "input_builder"),
    _MAX_POOL_SPECIAL_VALUE_CASES,
)
def test_max_pool_special_values(
    ndim: int,
    case_name: str,
    input_builder: Callable[[], torch.Tensor],
    return_indices: bool,
) -> None:
    _ = case_name
    x = input_builder()
    pool_kwargs = _MAX_POOL_SPECIAL_KWARGS[ndim]
    ref = _max_pool_ref(ndim)(x, **pool_kwargs, return_indices=return_indices)
    op = _max_pool_op_cls(ndim, return_indices)(**pool_kwargs)
    if return_indices:
        out, idx = op(x)
        torch.testing.assert_close(out, ref[0], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(idx, ref[1], rtol=0, atol=0)
    else:
        out = op(x)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)


# Curated per-dim constructor rejection cases; 2d carries the exhaustive
# type-validation coverage, 1d/3d keep the dim-specific spot checks.
_MAX_POOL_INVALID_PARAM_CASES = [
    # Smoke cases lead the list: the tier gate requires smoke to collect first.
    pytest.param(
        1, {"kernel_size": True}, TypeError,
        "kernel_size must be an int or a tuple of 1 ints",
        id="1d-kernel-size-type", marks=pytest.mark.smoke),
    pytest.param(
        2, {"kernel_size": True}, TypeError,
        "kernel_size must be an int or a tuple of 2 ints",
        id="2d-kernel-size-type", marks=pytest.mark.smoke),
    pytest.param(
        3, {"kernel_size": True}, TypeError,
        "kernel_size must be an int or a tuple of 3 ints",
        id="3d-kernel-size-type", marks=pytest.mark.smoke),
    # 1d full
    pytest.param(
        1, {"kernel_size": 3, "stride": 0}, ValueError,
        "stride must be greater than zero", id="1d-zero-stride", marks=pytest.mark.full),
    pytest.param(
        1, {"kernel_size": 3, "padding": 2}, ValueError,
        "padding must be at most half", id="1d-padding-too-large", marks=pytest.mark.full),
    pytest.param(
        1, {"kernel_size": 3, "ceil_mode": "true"}, TypeError,
        "ceil_mode must be a bool", id="1d-ceil-mode-type", marks=pytest.mark.full),
    # 2d full
    pytest.param(
        2, {"kernel_size": (3, 3), "stride": True}, TypeError,
        "stride must be an int or a tuple of 2 ints",
        id="2d-stride-type", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "padding": True}, TypeError,
        "padding must be an int or a tuple of 2 ints",
        id="2d-padding-type", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "dilation": True}, TypeError,
        "dilation must be an int or a tuple of 2 ints",
        id="2d-dilation-type", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, True)}, TypeError,
        "kernel_size must contain only ints",
        id="2d-kernel-size-contents", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "stride": (1, 0)}, ValueError,
        "stride must be greater than zero", id="2d-zero-stride", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "dilation": (0, 1)}, ValueError,
        "dilation must be greater than zero", id="2d-zero-dilation", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "padding": (2, 1)}, ValueError,
        "padding must be at most half", id="2d-padding-too-large", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "padding": (-1, 0)}, ValueError,
        "padding must be non-negative", id="2d-padding-negative", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3), "ceil_mode": "true"}, TypeError,
        "ceil_mode must be a bool", id="2d-ceil-mode-type", marks=pytest.mark.full),
    # 3d full
    pytest.param(
        3, {"kernel_size": 3, "stride": (1, 0, 1)}, ValueError,
        "stride must be greater than zero", id="3d-zero-stride", marks=pytest.mark.full),
    pytest.param(
        3, {"kernel_size": 3, "padding": (0, 2, 0)}, ValueError,
        "padding must be at most half", id="3d-padding-too-large", marks=pytest.mark.full),
    pytest.param(
        3, {"kernel_size": 3, "ceil_mode": "true"}, TypeError,
        "ceil_mode must be a bool", id="3d-ceil-mode-type", marks=pytest.mark.full),
]


@pytest.mark.parametrize("return_indices", [False, True], ids=["plain", "indices"])
@pytest.mark.parametrize(
    ("ndim", "kwargs", "exc_type", "match"),
    _MAX_POOL_INVALID_PARAM_CASES,
)
def test_max_pool_rejects_invalid_params(
    ndim: int,
    kwargs: dict[str, object],
    exc_type: type[Exception],
    match: str,
    return_indices: bool,
) -> None:
    op_cls = _max_pool_op_cls(ndim, return_indices)
    with pytest.raises(exc_type, match=match):
        op_cls(**kwargs)


# Curated per-dim runtime-input rejection cases:
# (ndim, ctor_kwargs, (input shape, input dtype), expected match, needs dummy kernel).
_MAX_POOL_INVALID_INPUT_CASES = [
    # Smoke cases lead the list: the tier gate requires smoke to collect first.
    pytest.param(
        1, {"kernel_size": 3}, ((2, 8, 16, 16), None),
        "expects input to be a 3D NCL tensor", True,
        id="1d-wrong-rank-input", marks=pytest.mark.smoke),
    pytest.param(
        2, {"kernel_size": (3, 3)}, ((2, 8, 16), None),
        "expects input to be a 4D NCHW tensor", True,
        id="2d-wrong-rank-input", marks=pytest.mark.smoke),
    pytest.param(
        3, {"kernel_size": 3}, ((2, 8, 16, 16), None),
        "expects input to be a 5D NCDHW tensor", True,
        id="3d-wrong-rank-input", marks=pytest.mark.smoke),
    # 1d full
    pytest.param(
        1, {"kernel_size": 3}, ((1, 1, 8), None),
        "input must be a CUDA tensor", False,
        id="1d-cpu-input", marks=pytest.mark.full),
    pytest.param(
        1, {"kernel_size": 3}, ((1, 1, 8), torch.float64),
        "input.dtype must be float16, bfloat16, or float32", False,
        id="1d-unsupported-dtype", marks=pytest.mark.full),
    pytest.param(
        1, {"kernel_size": 5, "stride": 1, "padding": 0}, ((1, 1, 2), torch.float16),
        "output size must be greater than zero", False,
        id="1d-non-positive-output-size", marks=pytest.mark.full),
    # 2d full
    pytest.param(
        2, {"kernel_size": (3, 3)}, ((1, 1, 8, 8), None),
        "input must be a CUDA tensor", False,
        id="2d-cpu-input", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (3, 3)}, ((1, 1, 8, 8), torch.float64),
        "input.dtype must be float16, bfloat16, or float32", False,
        id="2d-unsupported-dtype", marks=pytest.mark.full),
    pytest.param(
        2, {"kernel_size": (5, 5), "stride": (1, 1), "padding": (0, 0)},
        ((1, 1, 2, 2), torch.float16),
        "output size must be greater than zero", False,
        id="2d-non-positive-output-size", marks=pytest.mark.full),
    # 3d full
    pytest.param(
        3, {"kernel_size": 3}, ((1, 1, 4, 8, 8), None),
        "input must be a CUDA tensor", False,
        id="3d-cpu-input", marks=pytest.mark.full),
    pytest.param(
        3, {"kernel_size": 3}, ((1, 1, 4, 8, 8), torch.float64),
        "input.dtype must be float16, bfloat16, or float32", False,
        id="3d-unsupported-dtype", marks=pytest.mark.full),
    pytest.param(
        3, {"kernel_size": 5, "stride": 1, "padding": 0}, ((1, 1, 2, 8, 8), torch.float16),
        "output size must be greater than zero", False,
        id="3d-non-positive-output-size", marks=pytest.mark.full),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("return_indices", [False, True], ids=["plain", "indices"])
@pytest.mark.parametrize(
    ("ndim", "ctor_kwargs", "input_spec", "expected_match", "needs_dummy_kernel"),
    _MAX_POOL_INVALID_INPUT_CASES,
)
def test_max_pool_rejects_invalid_input(
    ndim: int,
    ctor_kwargs: dict[str, object],
    input_spec: tuple[tuple[int, ...], torch.dtype | None],
    expected_match: str,
    needs_dummy_kernel: bool,
    return_indices: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op_cls = _max_pool_op_cls(ndim, return_indices)
    kwargs = dict(ctor_kwargs)
    if needs_dummy_kernel:
        monkeypatch.setattr("tileops.ops.op_base.get_sm_version", lambda: 80)
        kwargs["kernel_map"] = {_MAX_POOL_KERNEL_SLOTS[op_cls]: _MAX_POOL_DUMMY_KERNELS[op_cls]}
    op = op_cls(**kwargs)

    shape, dtype = input_spec
    x = torch.randn(*shape) if dtype is None else torch.randn(*shape, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=expected_match):
        op(x)


# Per-dim constructor config shared by the dynamic-shape and compile tests.
_MAX_POOL_CTOR_KWARGS: dict[int, dict[str, object]] = {
    1: {"kernel_size": 3, "stride": 2, "padding": 1},
    2: {"kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1)},
    3: {"kernel_size": 3, "stride": 2, "padding": 1},
}

# Per-dim (first, second) input shapes for the dynamic-shape cache test.
_MAX_POOL_DYNAMIC_SHAPES: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
    1: ((1, 4, 32), (2, 4, 32)),
    2: ((1, 4, 16, 16), (2, 4, 16, 16)),
    3: ((1, 4, 8, 16, 16), (2, 4, 8, 16, 16)),
}


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("return_indices", [False, True], ids=["plain", "indices"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1d", "2d", "3d"])
def test_max_pool_dynamic_shape_kernel_cache_and_roofline(
    ndim: int,
    return_indices: bool,
) -> None:
    op = _max_pool_op_cls(ndim, return_indices)(**_MAX_POOL_CTOR_KWARGS[ndim])
    shape1, shape2 = _MAX_POOL_DYNAMIC_SHAPES[ndim]
    x1 = torch.randn(*shape1, dtype=torch.float16, device="cuda")
    x2 = torch.randn(*shape2, dtype=torch.float16, device="cuda")

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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
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
    op = op_cls(**_MAX_POOL_CTOR_KWARGS[ndim])
    x = torch.randn(*x_shape, device="cuda", dtype=torch.float16)
    compiled = torch.compile(op, fullgraph=True)
    out = compiled(x)
    ref = _max_pool_ref(ndim)(
        x,
        **_MAX_POOL_CTOR_KWARGS[ndim],
        return_indices=return_indices,
    )
    if return_indices:
        torch.testing.assert_close(out[0], ref[0], atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(out[1], ref[1], atol=0, rtol=0)
    else:
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Kernel helpers and cross-family compile tests
# ---------------------------------------------------------------------------


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


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("isolated_dynamo")
def test_pool_compile_two_instances_one_frame() -> None:
    """Second instance through the same frame must not degrade the dispatch key.

    Dynamo generalizes non-static scalar arguments across recompilations;
    an int instance key becomes an unhashable SymInt on the second cold
    compile of the same class.
    """
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.float16)
    a = MaxPool1dFwdOp(kernel_size=3, stride=2, padding=1)
    b = MaxPool1dFwdOp(kernel_size=3, stride=1, padding=1)
    torch.testing.assert_close(
        torch.compile(a, fullgraph=True)(x), F.max_pool1d(x, 3, 2, 1), atol=0, rtol=0)
    torch.testing.assert_close(
        torch.compile(b, fullgraph=True)(x), F.max_pool1d(x, 3, 1, 1), atol=0, rtol=0)


_AVG_POOL_COMPILE_CASES = [
    pytest.param(AvgPool1dFwdOp, (2, 8, 32), id="avg-pool1d"),
    pytest.param(AvgPool2dFwdOp, (2, 4, 16, 16), id="avg-pool2d"),
    pytest.param(AvgPool3dFwdOp, (2, 4, 8, 16, 16), id="avg-pool3d"),
]
for _case in _AVG_POOL_COMPILE_CASES:
    register_compile_contract(_case.values[0])


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize(("op_cls", "x_shape"), _AVG_POOL_COMPILE_CASES)
def test_avg_pool_compile_fullgraph(op_cls: type, x_shape: tuple) -> None:
    dims = len(x_shape) - 2
    op = op_cls(kernel_size=2, stride=2, padding=0)
    x = torch.randn(*x_shape, device="cuda", dtype=torch.float16)
    compiled = torch.compile(op, fullgraph=True)
    out = compiled(x)
    ref = getattr(F, f"avg_pool{dims}d")(x, 2, 2, 0)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Cross-family contract snapshots
# ---------------------------------------------------------------------------

_EMPTY = inspect.Parameter.empty

_AVG_POOL_CTOR_PARAMS_1D = (
    ("kernel_size", _EMPTY),
    ("stride", None),
    ("padding", 0),
    ("ceil_mode", False),
    ("count_include_pad", True),
    ("kernel_map", None),
    ("tune", False),
)

_AVG_POOL_CTOR_PARAMS_ND = (
    ("kernel_size", _EMPTY),
    ("stride", None),
    ("padding", 0),
    ("ceil_mode", False),
    ("count_include_pad", True),
    ("divisor_override", None),
    ("kernel_map", None),
    ("tune", False),
)

_MAX_POOL_CTOR_PARAMS = (
    ("kernel_size", _EMPTY),
    ("stride", None),
    ("padding", 0),
    ("dilation", 1),
    ("ceil_mode", False),
    ("kernel_map", None),
    ("tune", False),
)

# Manifest-pinned constructor contract: parameter names, order, and defaults.
_POOL_CTOR_SNAPSHOTS: list = [
    pytest.param(AvgPool1dFwdOp, _AVG_POOL_CTOR_PARAMS_1D, id="avg-pool1d"),
    pytest.param(AvgPool2dFwdOp, _AVG_POOL_CTOR_PARAMS_ND, id="avg-pool2d"),
    pytest.param(AvgPool3dFwdOp, _AVG_POOL_CTOR_PARAMS_ND, id="avg-pool3d"),
    pytest.param(MaxPool1dFwdOp, _MAX_POOL_CTOR_PARAMS, id="max-pool1d"),
    pytest.param(MaxPool1dIndicesFwdOp, _MAX_POOL_CTOR_PARAMS, id="max-pool1d-indices"),
    pytest.param(MaxPool2dFwdOp, _MAX_POOL_CTOR_PARAMS, id="max-pool2d"),
    pytest.param(MaxPool2dIndicesFwdOp, _MAX_POOL_CTOR_PARAMS, id="max-pool2d-indices"),
    pytest.param(MaxPool3dFwdOp, _MAX_POOL_CTOR_PARAMS, id="max-pool3d"),
    pytest.param(MaxPool3dIndicesFwdOp, _MAX_POOL_CTOR_PARAMS, id="max-pool3d-indices"),
]


@pytest.mark.smoke
@pytest.mark.parametrize(("op_cls", "expected"), _POOL_CTOR_SNAPSHOTS)
def test_pool_ctor_signature_snapshot(
    op_cls: type,
    expected: tuple[tuple[str, object], ...],
) -> None:
    params = inspect.signature(op_cls.__init__).parameters
    got = tuple((name, p.default) for name, p in params.items() if name != "self")
    assert got == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_cls", "ndim"),
    [
        pytest.param(AvgPool1dFwdOp, 1, id="avg-pool1d"),
        pytest.param(AvgPool2dFwdOp, 2, id="avg-pool2d"),
        pytest.param(AvgPool3dFwdOp, 3, id="avg-pool3d"),
        pytest.param(MaxPool1dFwdOp, 1, id="max-pool1d"),
        pytest.param(MaxPool1dIndicesFwdOp, 1, id="max-pool1d-indices"),
        pytest.param(MaxPool2dFwdOp, 2, id="max-pool2d"),
        pytest.param(MaxPool2dIndicesFwdOp, 2, id="max-pool2d-indices"),
        pytest.param(MaxPool3dFwdOp, 3, id="max-pool3d"),
        pytest.param(MaxPool3dIndicesFwdOp, 3, id="max-pool3d-indices"),
    ],
)
def test_pool_ctor_rank_annotations_snapshot(op_cls: type, ndim: int) -> None:
    """Public ctor annotations stay rank-specific; ``Tuple[int, ...]`` is a regression."""
    rank_tuple = "typing.Tuple[" + ", ".join(["int"] * ndim) + "]"
    params = inspect.signature(op_cls.__init__).parameters
    pool_params = ["kernel_size", "stride", "padding"]
    if "dilation" in params:
        pool_params.append("dilation")
    for name in pool_params:
        ann = str(params[name].annotation)
        assert "Tuple[int, ...]" not in ann, f"{op_cls.__name__}.{name} widened to variadic: {ann}"
        assert rank_tuple in ann, f"{op_cls.__name__}.{name} lost rank annotation: {ann}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_cls", "expected_return"),
    [
        pytest.param(MaxPool2dFwdOp, torch.Tensor, id="max-pool2d"),
        pytest.param(
            MaxPool2dIndicesFwdOp, Tuple[torch.Tensor, torch.Tensor],
            id="max-pool2d-indices"),
    ],
)
def test_max_pool_forward_return_annotation_snapshot(op_cls: type, expected_return) -> None:
    """forward return annotations match manifest outputs per concrete class."""
    assert "forward" in op_cls.__dict__
    ann = inspect.signature(op_cls.forward).return_annotation
    assert ann == expected_return, f"{op_cls.__name__}.forward -> {ann}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls",
    [
        AvgPool1dFwdOp, AvgPool2dFwdOp, AvgPool3dFwdOp,
        MaxPool1dFwdOp, MaxPool1dIndicesFwdOp,
        MaxPool2dFwdOp, MaxPool2dIndicesFwdOp,
        MaxPool3dFwdOp, MaxPool3dIndicesFwdOp,
    ],
)
def test_pool_codegen_slots_are_class_local(op_cls: type) -> None:
    """eval_roofline / _validate_dtypes must live in each concrete class __dict__.

    Manifest codegen (``maybe_install_validator`` / ``maybe_install_eval_roofline``)
    keys off the concrete class definition; a definition inherited only from an
    intermediate base either gets silently shadowed by generated code or
    silently bypasses per-op generation.
    """
    assert "eval_roofline" in op_cls.__dict__
    assert "_validate_dtypes" in op_cls.__dict__


class _PassthroughGenericKernel(Kernel):
    supported_archs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clone(): custom-op outputs must not alias custom-op inputs.
        return x.clone()


class _PassthroughSpatialKernel(Kernel):
    supported_archs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clone(): custom-op outputs must not alias custom-op inputs.
        return x.clone()


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ndim", [1, 3], ids=["1d", "3d"])
def test_avg_pool_explicit_generic_kernel_map_disables_fast_path(ndim: int) -> None:
    """1d/3d policy: an explicit generic override alone opts out of the fast path."""
    generic_slot = f"avg_pool{ndim}d_kernel"
    spatial_slot = f"avg_pool{ndim}d_spatial_kernel"
    shape = (1, 2) + (8,) * ndim
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)

    op = _AVG_POOL_OPS[ndim](kernel_size=2, kernel_map={generic_slot: _PassthroughGenericKernel})
    op(x)
    assert isinstance(op.kernel, _PassthroughGenericKernel)

    op_both = _AVG_POOL_OPS[ndim](
        kernel_size=2,
        kernel_map={
            generic_slot: _PassthroughGenericKernel,
            spatial_slot: _PassthroughSpatialKernel,
        },
    )
    op_both(x)
    assert isinstance(op_both.kernel, _PassthroughSpatialKernel)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool2d_explicit_generic_kernel_map_keeps_fast_path() -> None:
    """2d policy asymmetry: an explicit generic override does NOT opt out."""
    op = AvgPool2dFwdOp(kernel_size=2, kernel_map={"avg_pool2d_kernel": _PassthroughGenericKernel})
    x = torch.randn(1, 2, 8, 8, device="cuda", dtype=torch.float16)
    op(x)
    assert isinstance(op.kernel, AvgPool2dSpatialKernel)


# _infer_output_shapes snapshot: (op_cls, ctor kwargs, input shape, expected shapes).
_POOL_INFER_SHAPE_SNAPSHOTS: list = [
    pytest.param(
        AvgPool1dFwdOp, {"kernel_size": 3, "stride": 2, "padding": 1},
        (2, 4, 32), {"output": (2, 4, 16)}, id="avg-pool1d"),
    pytest.param(
        AvgPool2dFwdOp, {"kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1)},
        (2, 4, 16, 16), {"output": (2, 4, 8, 8)}, id="avg-pool2d"),
    pytest.param(
        AvgPool3dFwdOp, {"kernel_size": (3, 3, 3), "stride": (2, 2, 2), "padding": (1, 1, 1)},
        (2, 4, 8, 16, 16), {"output": (2, 4, 4, 8, 8)}, id="avg-pool3d"),
    pytest.param(
        MaxPool1dFwdOp, {"kernel_size": 3, "stride": 2, "padding": 1},
        (2, 4, 32), {"output": (2, 4, 16)}, id="max-pool1d"),
    pytest.param(
        MaxPool1dIndicesFwdOp, {"kernel_size": 3, "stride": 2, "padding": 1},
        (2, 4, 32), {"output": (2, 4, 16), "indices": (2, 4, 16)}, id="max-pool1d-indices"),
    pytest.param(
        MaxPool2dFwdOp, {"kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1)},
        (2, 4, 16, 16), {"output": (2, 4, 8, 8)}, id="max-pool2d"),
    pytest.param(
        MaxPool2dIndicesFwdOp, {"kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1)},
        (2, 4, 16, 16), {"output": (2, 4, 8, 8), "indices": (2, 4, 8, 8)},
        id="max-pool2d-indices"),
    pytest.param(
        MaxPool3dFwdOp, {"kernel_size": 3, "stride": 2, "padding": 1},
        (2, 4, 8, 16, 16), {"output": (2, 4, 4, 8, 8)}, id="max-pool3d"),
    pytest.param(
        MaxPool3dIndicesFwdOp, {"kernel_size": 3, "stride": 2, "padding": 1},
        (2, 4, 8, 16, 16), {"output": (2, 4, 4, 8, 8), "indices": (2, 4, 4, 8, 8)},
        id="max-pool3d-indices"),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_cls", "ctor_kwargs", "input_shape", "expected"),
    _POOL_INFER_SHAPE_SNAPSHOTS,
)
def test_pool_infer_output_shapes_snapshot(
    op_cls: type,
    ctor_kwargs: dict[str, object],
    input_shape: tuple[int, ...],
    expected: dict[str, tuple[int, ...]],
) -> None:
    op = op_cls(**ctor_kwargs)
    assert op._infer_output_shapes(input_shape) == expected


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pool_fake_indices_shapes_and_int64_dtype() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = MaxPool2dIndicesFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    with FakeTensorMode():
        x = torch.empty(2, 4, 16, 16, device="cuda", dtype=torch.float16)
        out, idx = op(x)
    assert tuple(out.shape) == (2, 4, 8, 8)
    assert out.dtype == torch.float16
    assert tuple(idx.shape) == (2, 4, 8, 8)
    assert idx.dtype == torch.int64


# eval_roofline snapshot over n=1, c=2, 8-per-spatial-dim, k=2, s=2, p=0, fp16.
# Expected values are the hand-expanded flops/bytes formulas:
#   flops = n*c*prod(out)*prod(k); bytes = (n*c*prod(in) + n*c*prod(out))*2 [+ n*c*prod(out)*8].
_POOL_ROOFLINE_SNAPSHOTS: list = [
    pytest.param(AvgPool1dFwdOp, (1, 2, 8), 16, 48, id="avg-pool1d"),
    pytest.param(AvgPool2dFwdOp, (1, 2, 8, 8), 128, 320, id="avg-pool2d"),
    pytest.param(AvgPool3dFwdOp, (1, 2, 8, 8, 8), 1024, 2304, id="avg-pool3d"),
    pytest.param(MaxPool1dFwdOp, (1, 2, 8), 16, 48, id="max-pool1d"),
    pytest.param(MaxPool1dIndicesFwdOp, (1, 2, 8), 16, 112, id="max-pool1d-indices"),
    pytest.param(MaxPool2dFwdOp, (1, 2, 8, 8), 128, 320, id="max-pool2d"),
    pytest.param(MaxPool2dIndicesFwdOp, (1, 2, 8, 8), 128, 576, id="max-pool2d-indices"),
    pytest.param(MaxPool3dFwdOp, (1, 2, 8, 8, 8), 1024, 2304, id="max-pool3d"),
    pytest.param(MaxPool3dIndicesFwdOp, (1, 2, 8, 8, 8), 1024, 3328, id="max-pool3d-indices"),
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("op_cls", "shape", "expected_flops", "expected_bytes"),
    _POOL_ROOFLINE_SNAPSHOTS,
)
def test_pool_eval_roofline_snapshot(
    op_cls: type,
    shape: tuple[int, ...],
    expected_flops: int,
    expected_bytes: int,
) -> None:
    op = op_cls(kernel_size=2, stride=2, padding=0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    op(x)
    assert op.eval_roofline() == (expected_flops, expected_bytes)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_cls", "ctor", "in_dims", "spatial", "expected"),
    [
        pytest.param(
            AvgPool1dFwdOp, dict(kernel_size=2), (16,), True,
            ("avg_pool1d_spatial_kernel", 2, 4, 16, 2, 2, 0, False, True,
             torch.float16, 0, False),
            id="avg1d-spatial"),
        pytest.param(
            AvgPool1dFwdOp, dict(kernel_size=2, ceil_mode=True), (16,), False,
            ("avg_pool1d_kernel", 2, 4, 16, 2, 2, 0, True, True,
             torch.float16, 0, False),
            id="avg1d-general"),
        pytest.param(
            AvgPool2dFwdOp, dict(kernel_size=2), (8, 8), True,
            ("spatial", 2, 4, 8, 8, (2, 2), (2, 2), (0, 0), False, True, None,
             torch.float16, 0, False),
            id="avg2d-spatial"),
        pytest.param(
            AvgPool2dFwdOp, dict(kernel_size=2, ceil_mode=True), (8, 8), False,
            ("general", 2, 4, 8, 8, (2, 2), (2, 2), (0, 0), True, True, None,
             torch.float16, 0, False),
            id="avg2d-general"),
        pytest.param(
            AvgPool3dFwdOp, dict(kernel_size=2), (4, 8, 8), True,
            ("avg_pool3d_spatial_kernel", 2, 4, 4, 8, 8, (2, 2, 2), (2, 2, 2),
             (0, 0, 0), False, True, None, torch.float16, 0, False),
            id="avg3d-spatial"),
        pytest.param(
            AvgPool3dFwdOp, dict(kernel_size=2, ceil_mode=True), (4, 8, 8), False,
            ("avg_pool3d_kernel", 2, 4, 4, 8, 8, (2, 2, 2), (2, 2, 2),
             (0, 0, 0), True, True, None, torch.float16, 0, False),
            id="avg3d-general"),
    ],
)
def test_avg_pool_kernel_cache_key_snapshot(
    op_cls: type, ctor: dict, in_dims: tuple, spatial: bool, expected: tuple,
) -> None:
    """Cache-key tuples stay byte-identical to their per-rank pre-collapse form."""
    op = op_cls(**ctor)
    kernel_name = op._spatial_slot if spatial else op._generic_slot
    key = op._kernel_cache_key(
        kernel_name, spatial, 2, 4, in_dims, torch.float16, 0,
    )
    assert key == expected
    assert op._use_spatial_fast_path() == spatial



@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_avg_pool2d_kernel_cache_separates_dtypes() -> None:
    op = AvgPool2dFwdOp(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    shape = (1, 4, 16, 16)
    op(torch.randn(*shape, dtype=torch.float16, device="cuda"))
    op(torch.randn(*shape, dtype=torch.float32, device="cuda"))
    assert len(op._kernel_cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
