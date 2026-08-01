import itertools
from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Optional

import torch

from tileops.kernels.kernel_base import Kernel


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

    if dilation is None:
        dilation = (1,) * ndim
    if len(dilation) != ndim:
        raise ValueError("dilation must match pooling dimensionality")

    for name, values in (
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"{name} must contain only ints")

    if any(v <= 0 for v in kernel_size):
        raise ValueError("kernel_size must be greater than zero")

    if any(v <= 0 for v in stride):
        raise ValueError("stride must be greater than zero")

    if any(v <= 0 for v in dilation):
        raise ValueError("dilation must be greater than zero")

    if any(v < 0 for v in padding):
        raise ValueError("padding must be non-negative")

    for pad, kernel in zip(padding, kernel_size, strict=True):
        if pad > kernel // 2:
            raise ValueError("padding must be at most half of the effective kernel size")

    if divisor_override is not None and (
        not isinstance(divisor_override, int) or isinstance(divisor_override, bool)
    ):
        raise TypeError("divisor_override must be an int or None")

    if divisor_override == 0:
        raise ValueError("divisor_override must not be zero")


def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)


def adaptive_bin(o, size_in: int, size_out: int):
    """Half-open input range `[start, end)` feeding output index ``o``.

    PyTorch partitions each spatial axis as
    ``[floor(o*in/out), ceil((o+1)*in/out))``. Bins are never empty, including
    when ``size_out > size_in``. Evaluated while tracing, so the emitted
    expression is the same as writing the arithmetic inline.
    """
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def max_adaptive_bin_extent(size_in: int, size_out: int) -> int:
    """Widest bin on this axis — a compile-time bound for a `T.serial` loop.

    `ceil(in/out)` is not a valid bound: in=55/out=7 has a bin of 9 against
    ceil = 8, and expanding in=8/out=12 gives bins of 2 against 1.
    """
    extents = (adaptive_bin(o, size_in, size_out) for o in range(size_out))
    return max(end - start for start, end in extents)


class AdaptivePool2dKernelBase(Kernel):
    """Shared scaffold for the adaptive 2D pool kernels.

    The variants differ only in their reduction and in whether they emit
    indices. Everything around that -- the dtype gate, the config space, the
    argument marshalling -- is the same, so a subclass binds two callables and
    declares nothing else:

    - ``_build`` the cached builder that traces the prim_func.
    - ``_dispatch`` the registered custom op the forward call goes through.

    Each variant keeps its own ``torch.library`` registration; those need
    distinct names and are what ``torch.compile`` resolves against.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        out_h: int,
        out_w: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                f"{type(self).__name__} supports float16 and bfloat16, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self.kernel = type(self)._build(
            n, c_in, h_in, w_in, out_h, out_w, self.dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 256, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512], [128, 256, 512])
        ]

    def forward(self, x: torch.Tensor):
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            self.out_h,
            self.out_w,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
