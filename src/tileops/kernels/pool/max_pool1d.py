import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool1dKernel", "MaxPool1dWithIndicesKernel"]

# A window shorter than this has too few positions to fill even one lane group,
# and the cross-lane reduction costs more than the serial run it replaces.
_MIN_SPLIT_WINDOW = 8

# Output positions below which a launch cannot fill the device from the output
# axis alone: 132 SMs at 2048 resident threads is the widest device this kernel
# is built for.
_SMALL_LAUNCH = 132 * 2048


@functools.lru_cache(maxsize=32)
def _max_pool1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    total = rows * out_l
    # With no padding and no ceil overshoot every window lies inside its row, so
    # the per-tap bounds test is a compile-time truth and drops out.
    window_inside = pad_w == 0 and (out_l - 1) * stride_w + (kernel_w - 1) * dilation_w < l_in

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_func(block_m: int, block_k: int, threads: int):
        rounds = -(-kernel_w // block_k)
        lanes_cover_window = rounds * block_k == kernel_w
        tile_full = total % block_m == 0

        def safe(col):
            """*col* as an index that is always in range, clamping only when it
            can leave the row. The clamped value is read under a predicate that
            already discarded it, so which position it names does not matter."""
            return col if window_inside else T.max(0, T.min(col, l_in - 1))

        @T.prim_func
        def _max_pool1d_main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                if block_k == 1:
                    for i in T.Parallel(block_m):
                        idx = tile * block_m + i
                        if tile_full or idx < total:
                            ow = idx % out_l
                            row = idx // out_l
                            first = ow * stride_w - pad_w
                            run = T.alloc_var(T.float32)
                            run = -T.infinity(accum_dtype)
                            for k in T.serial(kernel_w):
                                col = first + k * dilation_w
                                if window_inside or ((col >= 0) and (col < l_in)):
                                    v = T.cast(x[row, safe(col)], accum_dtype)
                                    # NaN enters `run` and never leaves: a later
                                    # value fails `v > NaN`, which is how PyTorch
                                    # propagates it.
                                    run = T.if_then_else(T.isnan(v) or (v > run), v, run)
                            out[row, ow] = T.cast(run, dtype)
                else:
                    # Window positions run across lanes and finish with a cross-lane
                    # max. Cross-lane max drops NaN, so a lane's NaN travels beside
                    # its value as a flag and is put back after the reduction.
                    part = T.alloc_fragment((block_m, block_k), accum_dtype)
                    nan_part = T.alloc_fragment((block_m, block_k), accum_dtype)
                    best = T.alloc_fragment((block_m,), accum_dtype)
                    nan_seen = T.alloc_fragment((block_m,), accum_dtype)
                    for i, lane in T.Parallel(block_m, block_k):
                        idx = tile * block_m + i
                        run = T.alloc_var(T.float32)
                        flag = T.alloc_var(T.float32)
                        run = -T.infinity(accum_dtype)
                        flag = 0.0
                        if tile_full or idx < total:
                            ow = idx % out_l
                            row = idx // out_l
                            first = ow * stride_w - pad_w
                            for r in T.serial(rounds):
                                # A lane takes one contiguous run of window
                                # positions, so a warp's taps stay a contiguous
                                # stretch of the row on every round.
                                k = lane * rounds + r
                                if lanes_cover_window or k < kernel_w:
                                    col = first + k * dilation_w
                                    if window_inside or ((col >= 0) and (col < l_in)):
                                        v = T.cast(x[row, safe(col)], accum_dtype)
                                        flag = T.max(flag, T.if_then_else(T.isnan(v), 1.0, 0.0))
                                        run = T.if_then_else(v > run, v, run)
                        part[i, lane] = run
                        nan_part[i, lane] = flag
                    T.reduce_max(part, best, dim=1, clear=True)
                    T.reduce_max(nan_part, nan_seen, dim=1, clear=True)
                    for i in T.Parallel(block_m):
                        idx = tile * block_m + i
                        if tile_full or idx < total:
                            out[idx // out_l, idx % out_l] = T.cast(
                                T.if_then_else(
                                    nan_seen[i] > 0.0,
                                    T.cast(float("nan"), accum_dtype),
                                    best[i],
                                ),
                                dtype,
                            )

        return _max_pool1d_main

    return _max_pool1d_func


def _launch_max_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    config: dict,
    x: torch.Tensor,
) -> torch.Tensor:
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    kernel = _max_pool1d_kernel(
        n,
        c_in,
        l_in,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
        ceil_mode,
        dtype,
    )(**config)
    # The kernel addresses one row per (batch, channel) pair, so the two leading
    # extents are folded away on the way in and restored on the way out.
    return kernel(x.reshape(n * c_in, l_in)).view(n, c_in, out_l)


class _MaxPool1dKernelBase(Kernel):
    """Shared construction and dispatch for the 1d max-pool kernels.

    Concrete kernels supply ``_build`` and ``_dispatch``; everything else —
    parameter capture, output extents, and launch — is identical between the
    value-only and with-indices variants. Each declares its own config space,
    because the two schedule the window differently.
    """

    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_w: int,
        stride_w: int,
        pad_w: int,
        dilation_w: int,
        ceil_mode: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"{type(self).__name__} supports float16, bfloat16, and float32, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_w = kernel_w
        self.stride_w = stride_w
        self.pad_w = pad_w
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            ceil_mode,
            self.dtype_str,
        )
        self.init_config(config, tune)

    def forward(self, x: torch.Tensor) -> Any:
        self._require_cuda(x=x)
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_w,
            self.stride_w,
            self.pad_w,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            dict(self.config),
            x,
        )


class MaxPool1dKernel(_MaxPool1dKernelBase):
    """Max pooling forward kernel (return_indices=False).

    One thread owns one output position and folds its window into a register,
    so an output is written once and the window never leaves the thread.

    ``block_k`` spreads the window itself across lanes and finishes with a
    cross-lane max. Global pooling — a window as long as the row, one output
    per row — has no output positions to spread over threads, and is the shape
    that needs it.
    """

    _build = staticmethod(_max_pool1d_kernel)

    def _splits_window(self) -> bool:
        """Whether the default config should spread the window over lanes.

        A split trades window positions for output positions, so it wants a
        window long enough to fill lanes and a launch with too few output
        positions to fill the device without them. Both families stay in the
        tuning space whenever the window is long enough to divide, so this
        decides only where an untuned build starts.
        """
        return self.kernel_w >= _MIN_SPLIT_WINDOW and (
            self.n * self.c_in * self.out_l < _SMALL_LAUNCH
        )

    @property
    def default_config(self) -> dict:
        if self._splits_window():
            return {
                "block_m": 32,
                "block_k": min(16, self.kernel_w),
                "threads": 512,
            }
        return {"block_m": 512, "block_k": 1, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        space = list(itertools.product((256, 512, 1024), (1,), (128, 256)))
        if self.kernel_w >= _MIN_SPLIT_WINDOW:
            space += itertools.product((16, 32, 64), (8, 16, 32), (256, 512))
        configs = [
            {"block_m": block_m, "block_k": block_k, "threads": threads}
            for block_m, block_k, threads in space
            if block_k <= self.kernel_w and threads <= block_m * block_k <= 8192
        ]
        return configs or [self.default_config]

    _dispatch = staticmethod(_launch_max_pool1d)


@functools.lru_cache(maxsize=32)
def _max_pool1d_with_indices_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    total_output = n * c_in * out_l
    # Static specialization: with zero padding and no ceil overshoot every window
    # lies fully inside the input, so the per-element bounds check can be dropped.
    always_in_bounds = pad_w == 0 and (out_l - 1) * stride_w + (kernel_w - 1) * dilation_w < l_in

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_with_indices_func(block_m: int, threads: int):
        @T.prim_func
        def _max_pool1d_with_indices_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
            indices: T.Tensor((n, c_in, out_l), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_l
                        channel_batch_idx = out_idx // out_l
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in
                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_idx = T.alloc_var(T.int32)
                        nan_idx = T.alloc_var(T.int32)
                        first_valid = T.alloc_var(T.bool)
                        # Loop-invariant window corner, materialized so it is not
                        # re-inlined into every window element.
                        iw0 = T.alloc_var(T.int32)
                        max_val = T.cast(float("-inf"), accum_dtype)
                        has_nan = False
                        first_valid = True
                        iw0 = ow * stride_w - pad_w
                        if always_in_bounds:
                            # Window element 0 is in bounds here, so its flat index is
                            # the correct seed: an all--inf window reports the first
                            # position, matching PyTorch, and first_valid is unneeded.
                            max_idx = iw0
                            nan_idx = iw0
                        else:
                            max_idx = 0
                            nan_idx = 0
                        for kw in T.serial(kernel_w):
                            iw = iw0 + kw * dilation_w
                            if always_in_bounds:
                                val = T.cast(x[batch, c_idx, iw], accum_dtype)
                                is_nan = T.isnan(val)
                                # Branch-free update. Strict > keeps the first
                                # maximum; NaN never touches max_val/max_idx and
                                # records the last NaN visited, matching PyTorch.
                                take = (not is_nan) and (val > max_val)
                                max_val = T.if_then_else(take, val, max_val)
                                max_idx = T.if_then_else(take, iw, max_idx)
                                nan_idx = T.if_then_else(is_nan, iw, nan_idx)
                                has_nan = has_nan or is_nan
                            elif iw >= 0 and iw < l_in:
                                val = T.cast(x[batch, c_idx, iw], accum_dtype)
                                is_nan = T.isnan(val)
                                take = (not is_nan) and (first_valid or (val > max_val))
                                max_val = T.if_then_else(take, val, max_val)
                                max_idx = T.if_then_else(take, iw, max_idx)
                                first_valid = first_valid and is_nan
                                nan_idx = T.if_then_else(is_nan, iw, nan_idx)
                                has_nan = has_nan or is_nan

                        result = T.if_then_else(
                            has_nan,
                            T.cast(float("nan"), accum_dtype),
                            max_val,
                        )
                        out[batch, c_idx, ow] = T.cast(result, dtype)
                        indices[batch, c_idx, ow] = T.cast(
                            T.if_then_else(
                                has_nan,
                                nan_idx,
                                max_idx,
                            ),
                            "int64",
                        )

        return _max_pool1d_with_indices_main

    return _max_pool1d_with_indices_func


def _launch_max_pool1d_with_indices(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    config: dict,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _max_pool1d_with_indices_kernel(
        n,
        c_in,
        l_in,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
        ceil_mode,
        dtype,
    )(**config)(x)


class MaxPool1dWithIndicesKernel(_MaxPool1dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool1d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool1d_with_indices)

    @property
    def default_config(self) -> dict:
        return {"block_m": 256, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512], [128, 256, 512])
        ]
