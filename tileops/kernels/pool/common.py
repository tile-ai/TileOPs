import warnings
from collections.abc import Sequence
from typing import Tuple


def _normalize_pool_dims(name: str, value: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or a tuple of {ndim} ints")

    if isinstance(value, int):
        return (value,) * ndim

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an int or a tuple of {ndim} ints")

    if len(value) != ndim:
        raise ValueError(f"{name} must be an int or a tuple of {ndim} ints")

    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise TypeError(f"{name} must contain only ints")

    return tuple(value)


def normalize_1d(value: int | Tuple[int]) -> int:
    return _normalize_pool_dims("value", value, 1)[0]


def normalize_2d(value: int | Tuple[int, int]) -> Tuple[int, int]:
    return _normalize_pool_dims("value", value, 2)


def normalize_3d(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _normalize_pool_dims("value", value, 3)


def validate_pool_params(
    *,
    ndim: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...] | None = None,
    divisor_override: int | None = None,
) -> None:
    if len(kernel_size) != ndim or len(stride) != ndim or len(padding) != ndim:
        raise ValueError("kernel_size, stride, and padding must match pooling dimensionality")
    if dilation is not None and len(dilation) != ndim:
        raise ValueError("dilation must match pooling dimensionality")

    for name, values in (
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
    ):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"{name} must contain only ints")

    if any(v <= 0 for v in kernel_size):
        raise ValueError("kernel_size must be greater than zero")

    if any(v <= 0 for v in stride):
        raise ValueError("stride must be greater than zero")

    if any(v < 0 for v in padding):
        raise ValueError("padding must be non-negative")

    if dilation is not None:
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in dilation):
            raise TypeError("dilation must contain only ints")
        if any(v <= 0 for v in dilation):
            raise ValueError("dilation must be greater than zero")
        effective_kernel = tuple(
            (kernel - 1) * step + 1 for kernel, step in zip(kernel_size, dilation, strict=True)
        )
    else:
        effective_kernel = kernel_size

    for pad, kernel in zip(padding, effective_kernel, strict=True):
        if pad > kernel // 2:
            raise ValueError("padding must be at most half of the effective kernel size")

    if divisor_override is not None and (not isinstance(divisor_override, int) or isinstance(divisor_override, bool)):
        raise TypeError("divisor_override must be an int or None")

    if divisor_override == 0:
        raise ValueError("divisor_override must not be zero")


def validate_channels_last_input(
    *,
    op_name: str,
    x_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    layout: str,
    ambiguous_layout_shape: tuple[int, ...] | None = None,
) -> None:
    if x_shape != expected_shape:
        raise ValueError(
            f"{op_name} expects a {layout} input tensor with shape {expected_shape}, "
            f"but got {x_shape}"
        )

    if ambiguous_layout_shape is not None and ambiguous_layout_shape == expected_shape:
        warnings.warn(
            (
                f"{op_name} received an ambiguous {layout} shape {x_shape}; "
                "shape alone cannot distinguish channels-last from channels-first layout"
            ),
            stacklevel=2,
        )


def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    effective_kernel = (kernel_size - 1) * dilation + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)
