"""1-D convolution kernels: dense, grouped, and the pointwise (kernel size 1) form."""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.convolution._common import (
    CONV_SWIZZLE_PANEL,
    conv_autotune_configs,
    conv_num_stages,
    launch,
)
from tileops.kernels.convolution.call_spec import Conv1dCall, Conv1dFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = [
    "Conv1dKernel",
    "Conv1dPointwiseKernel",
    "DepthwiseConv1dKernel",
    "GroupConv1dKernel",
]


@functools.lru_cache(maxsize=64)
def _conv1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    k_total = c_in * kernel_l

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_body(x, weight_flat, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                tile_ol_start = bx * block_n
                tile_ol_end = tile_ol_start + block_n - 1
                tile_input_start = tile_ol_start * stride_l - pad_left
                tile_input_end = tile_ol_end * stride_l + (kernel_l - 1) * dilation_l - pad_left
                tile_spatial_full = (
                    (tile_ol_end < out_l) & (tile_input_start >= 0) & (tile_input_end < l_in)
                )

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)

                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        ol = bx * block_n + j
                        # k runs over (kernel_l, c_in): one k tile then covers a single
                        # tap across every input channel, which is one rectangle of x.
                        # The other order splits each tap across k tiles.
                        kw = k_idx // c_in
                        ci = k_idx % c_in
                        il = ol * stride_l + kw * dilation_l - pad_left
                        if tile_spatial_full & ((k_iter + 1) * block_k <= k_total):
                            data_shared[i, j] = x[bz, ci, il]
                        else:
                            in_bound = (k_idx < k_total) & (ol < out_l) & (il >= 0) & (il < l_in)
                            data_shared[i, j] = T.if_then_else(
                                in_bound,
                                x[bz, ci, il],
                                T.cast(0.0, dtype),
                            )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (ol < out_l),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (ol < out_l),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv1d_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_body(x, weight_flat, out, bias)

            return _conv1d_bias_main

        @T.prim_func
        def _conv1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_body(x, weight_flat, out, None)

        return _conv1d_main

    return _conv1d_func


@functools.lru_cache(maxsize=32)
def _conv1d_direct_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_direct_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_direct_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for kw in T.serial(kernel_l):
                    for i, j in T.Parallel(block_m, block_n):
                        oc = by * block_m + i
                        ol = bx * block_n + j
                        il = ol * stride_l + kw * dilation_l - pad_left
                        valid = (oc < c_out) & (ol < out_l) & (il >= 0) & (il < l_in)
                        out_local[i, j] += T.if_then_else(
                            valid,
                            T.cast(x[bz, oc, il], accum_dtype)
                            * T.cast(weight[oc, 0, kw], accum_dtype),
                            T.cast(0.0, accum_dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if oc < c_out and ol < out_l:
                        if has_bias:
                            out[bz, oc, ol] = T.cast(
                                out_local[i, j] + T.cast(bias[oc], accum_dtype),
                                dtype,
                            )
                        else:
                            out[bz, oc, ol] = T.cast(out_local[i, j], dtype)

        if has_bias:

            @T.prim_func
            def _conv1d_direct_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, 1, kernel_l), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_direct_body(x, weight, out, bias)

            return _conv1d_direct_bias_main

        @T.prim_func
        def _conv1d_direct_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, 1, kernel_l), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_direct_body(x, weight, out, None)

        return _conv1d_direct_main

    return _conv1d_direct_func


@functools.lru_cache(maxsize=64)
def _conv1d_group_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
    groups: int = 1,
    c_in_g: int = 0,
    c_out_g: int = 0,
):
    accum_dtype = "float"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    c_in_g = c_in_g if c_in_g > 0 else c_in // groups
    c_out_g = c_out_g if c_out_g > 0 else c_out // groups
    k_total = c_in_g * kernel_l

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_group_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_group_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out_g, block_m),
                n * groups,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                batch_id = bz // groups
                group_id = bz % groups
                oc_base = group_id * c_out_g + by * block_m

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, k in T.Parallel(block_m, block_k):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        k_idx = k_iter * block_k + k
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        weight_shared[i, k] = T.if_then_else(
                            (oc_g < c_out_g) & (k_idx < k_total),
                            weight[oc, ci_g, kw],
                            T.cast(0.0, dtype),
                        )

                    for k, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + k
                        ol = bx * block_n + j
                        # k runs over (kernel_l, c_in_g), matching the general Conv1d
                        # kernel; the weight is staged by gather because that order is not
                        # the one it is stored in.
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        il = ol * stride_l + kw * dilation_l - pad_left
                        data_shared[k, j] = T.if_then_else(
                            (k_idx < k_total) & (ol < out_l) & (il >= 0) & (il < l_in),
                            x[batch_id, group_id * c_in_g + ci_g, il],
                            T.cast(0.0, dtype),
                        )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (ol < out_l),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (ol < out_l),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                if c_out_g % block_m == 0:
                    # The tile ends on this group's last channel, so the copy cannot spill
                    # into the next group's rows.
                    T.copy(out_shared, out[batch_id, oc_base, bx * block_n])
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        ol = bx * block_n + j
                        if oc_g < c_out_g and ol < out_l:
                            out[batch_id, oc, ol] = out_shared[i, j]

        if has_bias:

            @T.prim_func
            def _conv1d_group_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_group_body(x, weight, out, bias)

            return _conv1d_group_bias_main

        @T.prim_func
        def _conv1d_group_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_group_body(x, weight, out, None)

        return _conv1d_group_main

    return _conv1d_group_func


@functools.lru_cache(maxsize=32)
def _conv1d_pointwise_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_pointwise_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_pointwise_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(l_in, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                tile_l_end = bx * block_n + block_n - 1
                tile_spatial_full = tile_l_end < l_in
                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)

                    if tile_spatial_full & ((k_iter + 1) * block_k <= c_in):
                        T.copy(x[bz, k_iter * block_k, bx * block_n], data_shared)
                    else:
                        for i, j in T.Parallel(block_k, block_n):
                            ci = k_iter * block_k + i
                            l_idx = bx * block_n + j
                            data_shared[i, j] = T.if_then_else(
                                (ci < c_in) & (l_idx < l_in),
                                x[bz, ci, l_idx],
                                T.cast(0.0, dtype),
                            )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    l_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (l_idx < l_in),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (l_idx < l_in),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv1d_pointwise_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
                out: T.Tensor((n, c_out, l_in), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_pointwise_body(x, weight, out, bias)

            return _conv1d_pointwise_bias_main

        @T.prim_func
        def _conv1d_pointwise_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
            out: T.Tensor((n, c_out, l_in), dtype),  # type: ignore
        ):
            _conv1d_pointwise_body(x, weight, out, None)

        return _conv1d_pointwise_main

    return _conv1d_pointwise_func


class Conv1dPointwiseKernel(Kernel, Conv1dFwdInterface):
    """Dense 1x1 Conv1d, which lowers to a pointwise GEMM over the channel axis."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: Conv1dCall) -> bool:
        return (
            call.groups == 1
            and call.kernel_l == 1
            and call.stride_l == 1
            and call.pad_left == 0
            and call.pad_right == 0
            and call.dilation_l == 1
        )

    @classmethod
    def entry_for(cls, call: Conv1dCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = dict(n=call.n, c_in=call.c_in, l_in=call.l_in, c_out=call.c_out, dtype=call.dtype)
        identity = (*args.values(), call.has_bias, index)
        return identity, lambda: cls(**args, has_bias=call.has_bias, device_index=index)

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_l = l_in
        self.k_total = c_in
        self.kernel = _conv1d_pointwise_kernel(
            n,
            c_in,
            l_in,
            c_out,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_m, block_n, block_k = 64, 128, 128
        # Fewer stages where shared memory cannot hold them, down to one.
        num_stages = conv_num_stages(self.device_index)
        cap = get_shared_memory_optin(self.device_index)
        while num_stages > 1 and self._shared_bytes(block_m, block_n, block_k, num_stages) > cap:
            num_stages -= 1
        return {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_stages": num_stages,
            "threads": 128,
            "enable_rasterization": True,
        }

    def _shared_bytes(self, block_m: int, block_n: int, block_k: int, num_stages: int) -> int:
        """Upper bound on the program's shared memory. The pipeline buffers each weight
        tile per stage, and each x tile too where the tiles divide ``c_in`` and ``l_in``,
        since the x load is then a plain copy; otherwise the masked gather holds one. The
        output tile is counted once."""
        per_stage = block_m * block_k
        once = block_m * block_n
        if self.c_in % block_k == 0 and self.l_in % block_n == 0:
            per_stage += block_k * block_n
        else:
            once += block_k * block_n
        return (num_stages * per_stage + once) * self.dtype.itemsize

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(self.dtype, self.device_index)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_2d = weight[:, :, 0].contiguous()
        return launch(self, x, weight_2d, bias=bias)


class Conv1dKernel(Kernel, Conv1dFwdInterface):
    """Dense Conv1d over the full kernel window; serves every ungrouped call."""

    general = True
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: Conv1dCall) -> bool:
        return call.groups == 1

    @classmethod
    def entry_for(cls, call: Conv1dCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = dict(
            n=call.n,
            c_in=call.c_in,
            l_in=call.l_in,
            c_out=call.c_out,
            dtype=call.dtype,
            kernel_l=call.kernel_l,
            stride_l=call.stride_l,
            pad_l=(call.pad_left, call.pad_right),
            dilation_l=call.dilation_l,
        )
        identity = (*args.values(), call.has_bias, index)
        return identity, lambda: cls(**args, has_bias=call.has_bias, device_index=index)

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: Tuple[int, int],
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.pad_left, self.pad_right = pad_l
        self.dilation_l = dilation_l
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_l = (l_in + sum(pad_l) - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
        self.m = n * self.out_l
        self.k_total = c_in * kernel_l
        self._weight_flat_cache_source: Optional[torch.Tensor] = None
        self._weight_flat_cache_version: Optional[int] = None
        self._weight_flat_cache: Optional[torch.Tensor] = None
        self.kernel = _conv1d_kernel(
            n,
            c_in,
            l_in,
            c_out,
            kernel_l,
            stride_l,
            self.pad_left,
            self.pad_right,
            dilation_l,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "num_stages": conv_num_stages(self.device_index),
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(self.dtype, self.device_index, block_n=[64, 128])

    def _get_weight_flat(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the weight laid out as the prim_func's ``(c_out, k_total)``.

        k runs over ``(kernel_l, c_in)``, so this is a permute the caller's
        ``(c_out, c_in, kernel_l)`` cannot alias. Held per weight identity and version:
        a weight is a parameter, so it repeats across calls.
        """
        weight_version = weight._version
        if (
            self._weight_flat_cache_source is weight
            and self._weight_flat_cache_version == weight_version
            and self._weight_flat_cache is not None
        ):
            return self._weight_flat_cache

        weight_flat = weight.permute(0, 2, 1).contiguous().view(self.c_out, self.k_total)
        self._weight_flat_cache_source = weight
        self._weight_flat_cache_version = weight_version
        self._weight_flat_cache = weight_flat
        return weight_flat

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return launch(self, x, self._get_weight_flat(weight), bias=bias)


class GroupConv1dKernel(Kernel, Conv1dFwdInterface):
    """Grouped Conv1d: one implicit GEMM per group over that group's channels."""

    supported_archs: list[int] = [80, 86, 89, 90]

    # The m tiles this kernel builds. The default configuration and the autotune filter
    # read the same tuple, so widening it cannot leave the two disagreeing.
    block_m_candidates: tuple[int, ...] = (16, 32, 64, 128)

    @classmethod
    def applies(cls, call: Conv1dCall) -> bool:
        return call.groups > 1

    @classmethod
    def entry_for(cls, call: Conv1dCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = dict(
            n=call.n,
            c_in=call.c_in,
            l_in=call.l_in,
            c_out=call.c_out,
            dtype=call.dtype,
            kernel_l=call.kernel_l,
            stride_l=call.stride_l,
            pad_l=(call.pad_left, call.pad_right),
            dilation_l=call.dilation_l,
            groups=call.groups,
            c_in_g=call.c_in_g,
            c_out_g=call.c_out_g,
        )
        identity = (*args.values(), call.has_bias, index)
        return identity, lambda: cls(**args, has_bias=call.has_bias, device_index=index)

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: Tuple[int, int],
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        groups: int = 1,
        c_in_g: Optional[int] = None,
        c_out_g: Optional[int] = None,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.pad_left, self.pad_right = pad_l
        self.dilation_l = dilation_l
        self.groups = groups
        self.c_in_g = c_in_g if c_in_g is not None else c_in // groups
        self.c_out_g = c_out_g if c_out_g is not None else c_out // groups
        self.dtype = dtype
        self.has_bias = has_bias
        if self.groups <= 1:
            raise ValueError(f"{type(self).__name__} requires groups > 1")
        self._build_program()
        self.init_config(config, tune)
        self._check_config()

    def _build_program(self) -> None:
        """Compile the grouped implicit GEMM over this call's group shape."""
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError(
                f"{type(self).__name__} requires c_in and c_out divisible by groups; "
                f"got c_in={self.c_in}, c_out={self.c_out}, groups={self.groups}"
            )
        self.kernel = _conv1d_group_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_left,
            self.pad_right,
            self.dilation_l,
            self.has_bias,
            self.dtype_str,
            self.groups,
            self.c_in_g,
            self.c_out_g,
        )

    def _check_config(self) -> None:
        """Reject a tile the implicit GEMM's MMA step cannot take."""
        for key in ("block_m", "block_k"):
            if self.config[key] % 16:
                raise ValueError(
                    f"{type(self).__name__} requires {key} to be a multiple of 16; "
                    f"got {key}={self.config[key]}"
                )

    @property
    def default_config(self) -> dict:
        block_m = next(
            (choice for choice in self.block_m_candidates if choice >= self.c_out_g),
            max(self.block_m_candidates),
        )
        return {
            "block_m": block_m,
            "block_n": 128,
            "block_k": 128,
            "num_stages": conv_num_stages(self.device_index),
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(
            self.dtype,
            self.device_index,
            block_m=list(self.block_m_candidates),
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return launch(self, x, weight, bias=bias)


class DepthwiseConv1dKernel(GroupConv1dKernel):
    """Depthwise Conv1d: one input channel per group, one output channel per group.

    A different program from :class:`GroupConv1dKernel`: with a single channel per group
    there is no contraction to tile, so each thread accumulates one output position over
    the kernel window instead of running an implicit GEMM.
    """

    preferred_over = frozenset({"group_conv1d"})

    @classmethod
    def applies(cls, call: Conv1dCall) -> bool:
        return call.groups > 1 and call.c_in_g == 1 and call.c_out_g == 1

    def _build_program(self) -> None:
        self.kernel = _conv1d_direct_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_left,
            self.pad_right,
            self.dilation_l,
            self.has_bias,
            self.dtype_str,
        )

    def _check_config(self) -> None:
        """One output position per thread takes any tile."""

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 1,
            "block_n": 128,
            "block_k": 1,
            "num_stages": 1,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]
