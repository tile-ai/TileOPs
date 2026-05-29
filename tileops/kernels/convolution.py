import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

__all__ = [
    "Conv1dKernel",
    "Conv1dPointwiseKernel",
    "Conv2d1x1Kernel",
    "Conv2dKernel",
    "Conv3dKernel",
    "GroupConv1dKernel",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_shared_memory_limit_bytes() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin


def conv_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    out_shared_bytes = block_m * block_n * dtype_bytes
    return per_stage_bytes * max(1, num_stages) + out_shared_bytes


def _group_conv1d_block_m_choices(c_out_g: int) -> list[int]:
    del c_out_g
    return [16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Conv1d
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _conv1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
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
        @T.prim_func
        def _conv1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
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

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                tile_ol_start = bx * block_n
                tile_ol_end = tile_ol_start + block_n - 1
                tile_input_start = tile_ol_start * stride_l - pad_l
                tile_input_end = tile_ol_end * stride_l + (kernel_l - 1) * dilation_l - pad_l
                tile_spatial_full = (
                    (tile_ol_end < out_l)
                    & (tile_input_start >= 0)
                    & (tile_input_end < l_in)
                )

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)

                    tile_full = tile_spatial_full & ((k_iter + 1) * block_k <= k_total)
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        ol = bx * block_n + j
                        kw = k_idx // c_in
                        ci = k_idx % c_in
                        il = ol * stride_l + kw * dilation_l - pad_l
                        if tile_full:
                            data_shared[i, j] = x[bz, ci, il]
                        else:
                            in_bound = (
                                (k_idx < k_total)
                                & (ol < out_l)
                                & (il >= 0)
                                & (il < l_in)
                            )
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

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if oc < c_out and ol < out_l:
                        out[bz, oc, ol] = out_shared[i, j]

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
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_direct_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.prim_func
        def _conv1d_direct_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, 1, kernel_l), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for kw in T.serial(kernel_l):
                    for i, j in T.Parallel(block_m, block_n):
                        oc = by * block_m + i
                        ol = bx * block_n + j
                        il = ol * stride_l + kw * dilation_l - pad_l
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
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
    groups: int = 1,
    c_in_g: int = 0,
    c_out_g: int = 0,
):
    accum_dtype = "float"
    out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
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
        @T.prim_func
        def _conv1d_group_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
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

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                batch_id = bz // groups
                group_id = bz % groups

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
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        il = ol * stride_l + kw * dilation_l - pad_l
                        data_shared[k, j] = T.if_then_else(
                            (k_idx < k_total)
                            & (ol < out_l)
                            & (il >= 0)
                            & (il < l_in),
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

                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    ol = bx * block_n + j
                    if oc_g < c_out_g and ol < out_l:
                        out[batch_id, oc, ol] = out_shared[i, j]

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
        @T.prim_func
        def _conv1d_pointwise_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
            out: T.Tensor((n, c_out, l_in), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
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

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                tile_l_end = bx * block_n + block_n - 1
                tile_spatial_full = tile_l_end < l_in
                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)

                    tile_full = tile_spatial_full & ((k_iter + 1) * block_k <= c_in)
                    if tile_full:
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

        return _conv1d_pointwise_main

    return _conv1d_pointwise_func


@torch.library.custom_op("top::conv1d_wrapped_kernel", mutates_args=())
def _conv1d_wrapped_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv1d_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_l, dilation_l, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@torch.library.custom_op("top::conv1d_direct_wrapped_kernel", mutates_args=())
def _conv1d_direct_wrapped_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv1d_direct_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_l, dilation_l, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@torch.library.custom_op("top::conv1d_group_wrapped_kernel", mutates_args=())
def _conv1d_group_wrapped_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str,
    groups: int,
    c_in_g: int,
    c_out_g: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv1d_group_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_l, dilation_l, has_bias, dtype, groups, c_in_g, c_out_g
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@torch.library.custom_op("top::conv1d_pointwise_wrapped_kernel", mutates_args=())
def _conv1d_pointwise_wrapped_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv1d_pointwise_kernel(
        n, c_in, l_in, c_out, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@_conv1d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    return torch.empty((n, c_out, out_l), dtype=inputs[0].dtype, device=inputs[0].device)


@_conv1d_direct_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    return torch.empty((n, c_out, out_l), dtype=inputs[0].dtype, device=inputs[0].device)


@_conv1d_group_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str,
    groups: int,
    c_in_g: int,
    c_out_g: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    return torch.empty((n, c_out, out_l), dtype=inputs[0].dtype, device=inputs[0].device)


@_conv1d_pointwise_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((n, c_out, l_in), dtype=inputs[0].dtype, device=inputs[0].device)


class Conv1dPointwiseKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

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
    ) -> None:
        super().__init__()
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
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 2,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        shared_memory_limit_bytes = get_shared_memory_limit_bytes()
        configs = itertools.product(
            [32, 64, 128],
            [64, 128, 256],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasterization in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > shared_memory_limit_bytes:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        weight_2d = weight[:, :, 0].contiguous()
        return _conv1d_pointwise_wrapped_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.has_bias,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight_2d,
            bias,
        )


class Conv1dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.dilation_l = dilation_l
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_l = (l_in + 2 * pad_l - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
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
            pad_l,
            dilation_l,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 2,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        shared_memory_limit_bytes = get_shared_memory_limit_bytes()
        configs = itertools.product(
            [32, 64, 128],
            [64, 128, 256],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasterization in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > shared_memory_limit_bytes:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            })
        return valid_configs

    def _get_weight_flat(self, weight: torch.Tensor) -> torch.Tensor:
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
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        weight_flat = self._get_weight_flat(weight)
        return _conv1d_wrapped_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dilation_l,
            self.has_bias,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight_flat,
            bias,
        )


class GroupConv1dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        dtype: torch.dtype,
        dilation_l: int = 1,
        has_bias: bool = False,
        groups: int = 1,
        c_in_g: Optional[int] = None,
        c_out_g: Optional[int] = None,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.dilation_l = dilation_l
        self.groups = groups
        self.c_in_g = c_in_g if c_in_g is not None else c_in // groups
        self.c_out_g = c_out_g if c_out_g is not None else c_out // groups
        self.dtype = dtype
        self.has_bias = has_bias
        self.use_direct = self.c_in_g == 1 and self.c_out_g == 1
        self._validate_group_shape()
        if self.use_direct:
            self.kernel = _conv1d_direct_kernel(
                n,
                c_in,
                l_in,
                c_out,
                kernel_l,
                stride_l,
                pad_l,
                dilation_l,
                has_bias,
                self.dtype_str,
            )
        else:
            self.kernel = _conv1d_group_kernel(
                n,
                c_in,
                l_in,
                c_out,
                kernel_l,
                stride_l,
                pad_l,
                dilation_l,
                has_bias,
                self.dtype_str,
                groups,
                self.c_in_g,
                self.c_out_g,
            )
        self.init_config(config, tune)
        if not self.use_direct and self.config["block_m"] % 16 != 0:
            raise ValueError(
                f"GroupConv1dKernel requires block_m to be a multiple of 16; "
                f"got block_m={self.config['block_m']}"
            )
        if not self.use_direct and self.config["block_k"] % 16 != 0:
            raise ValueError(
                f"GroupConv1dKernel requires block_k to be a multiple of 16; "
                f"got block_k={self.config['block_k']}"
            )

    def _validate_group_shape(self) -> None:
        if self.groups <= 1:
            raise ValueError("GroupConv1dKernel requires groups > 1")
        if self.use_direct:
            return
        if self.c_in % self.groups != 0 or self.c_out % self.groups != 0:
            raise ValueError(
                f"GroupConv1dKernel requires c_in and c_out divisible by groups; "
                f"got c_in={self.c_in}, c_out={self.c_out}, groups={self.groups}"
            )

    @property
    def _block_m_choices(self) -> list[int]:
        if self.use_direct:
            return [1]
        return _group_conv1d_block_m_choices(self.c_out_g)

    @property
    def default_config(self) -> dict:
        if self.use_direct:
            return {
                "block_m": 1,
                "block_n": 128,
                "block_k": 1,
                "num_stages": 1,
                "threads": 128,
                "enable_rasterization": True,
            }
        block_m = next(
            (choice for choice in self._block_m_choices if choice >= self.c_out_g),
            max(self._block_m_choices),
        )
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": block_m,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": block_m,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 2,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        if self.use_direct:
            return [self.default_config]
        shared_memory_limit_bytes = get_shared_memory_limit_bytes()
        configs = itertools.product(
            self._block_m_choices,
            [64, 128, 256],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasterization in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > shared_memory_limit_bytes:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        if self.use_direct:
            return _conv1d_direct_wrapped_kernel(
                self.n,
                self.c_in,
                self.l_in,
                self.c_out,
                self.kernel_l,
                self.stride_l,
                self.pad_l,
                self.dilation_l,
                self.has_bias,
                self.dtype_str,
                self.config["block_m"],
                self.config["block_n"],
                self.config["block_k"],
                self.config["num_stages"],
                self.config["threads"],
                self.config["enable_rasterization"],
                x,
                weight,
                bias,
            )
        return _conv1d_group_wrapped_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dilation_l,
            self.has_bias,
            self.dtype_str,
            self.groups,
            self.c_in_g,
            self.c_out_g,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight,
            bias,
        )


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _conv2d_1x1_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    if stride_h != 1 or stride_w != 1 or pad_h != 0 or pad_w != 0:
        raise ValueError("Conv2d1x1Kernel requires stride=1 and padding=0")
    hw = h * w

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_1x1_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.prim_func
        def _conv2d_1x1_main(
            x: T.Tensor((n, h, w, c_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
            out: T.Tensor((n, h, w, c_out), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            x_flat = T.Tensor((n, hw, c_in), dtype, x.data)
            out_flat = T.Tensor((n, hw, c_out), dtype, out.data)
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(hw, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(x_flat[bz, by * block_m, k_iter * block_k], data_shared)
                    T.copy(weight[bx * block_n, k_iter * block_k], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local, transpose_B=True)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < hw) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < hw) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])

        return _conv2d_1x1_main

    return _conv2d_1x1_func


@functools.lru_cache(maxsize=32)
def _conv2d_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = (h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w + 2 * pad_w - kernel_w) // stride_w + 1
    k_total = kernel_h * kernel_w * c_in

    # Re-enable automatic async copy once TileLang lowers scalar cp.async
    # widening for vectorized manual data loads. Keep weight T.copy eligible for TMA.
    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv2d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.prim_func
        def _conv2d_main(
            x: T.Tensor((n, h, w, c_in), dtype),  # type: ignore
            weight: T.Tensor((kernel_h, kernel_w, c_in, c_out), dtype),  # type: ignore
            out: T.Tensor((n, out_h, out_w, c_out), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            use_hopper_im2col = (
                get_sm_version() == 90
                and stride_h == stride_w
                and pad_h == pad_w
                and kernel_h == kernel_w
                and c_in % block_k == 0
            )
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((k_total, c_out), dtype, weight.data)
                out_flat = T.Tensor((n * out_h * out_w, c_out), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    if use_hopper_im2col:
                        T.c2d_im2col(x, data_shared, by, k_iter, kernel_h, stride_h, 1, pad_h)
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            m_idx = by * block_m + i
                            k_idx = k_iter * block_k + j
                            kh = k_idx // (kernel_w * c_in)
                            kw = (k_idx // c_in) % kernel_w
                            ci = k_idx % c_in
                            out_idx = m_idx % (out_h * out_w)
                            batch = m_idx // (out_h * out_w)
                            oh = out_idx // out_w
                            ow = out_idx % out_w
                            ih = oh * stride_h + kh - pad_h
                            iw = ow * stride_w + kw - pad_w
                            in_bound = (
                                (m_idx < n * out_h * out_w)
                                & (k_idx < k_total)
                                & (ih >= 0)
                                & (iw >= 0)
                                & (ih < h)
                                & (iw < w)
                            )
                            data_shared[i, j] = T.if_then_else(
                                in_bound,
                                x[batch, ih, iw, ci],
                                T.cast(0.0, dtype),
                            )

                    T.copy(weight_flat[k_iter * block_k, bx * block_n], weight_shared)

                    T.gemm(data_shared, weight_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_h * out_w) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_h * out_w) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                if use_hopper_im2col:
                    T.copy(out_shared, out_flat[by * block_m, bx * block_n])
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        m_idx = by * block_m + i
                        oc = bx * block_n + j
                        if m_idx < n * out_h * out_w and oc < c_out:
                            out_flat[m_idx, oc] = out_shared[i, j]

        return _conv2d_main

    return _conv2d_func


@torch.library.custom_op("top::conv2d_1x1_wrapped_kernel", mutates_args=())
def _conv2d_1x1_wrapped_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv2d_1x1_kernel(
        n, c_in, h, w, c_out, 1, 1, 0, 0, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@_conv2d_1x1_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((n, h, w, c_out), dtype=inputs[0].dtype, device=inputs[0].device)


@torch.library.custom_op("top::conv2d_wrapped_kernel", mutates_args=())
def _conv2d_wrapped_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv2d_kernel(
        n, c_in, h, w, c_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@_conv2d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out_h = (h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w + 2 * pad_w - kernel_w) // stride_w + 1
    return torch.empty((n, out_h, out_w, c_out), dtype=inputs[0].dtype, device=inputs[0].device)

class Conv2dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_h = (h + 2 * pad_h - kernel_h) // stride_h + 1
        self.out_w = (w + 2 * pad_w - kernel_w) // stride_w + 1
        self.m = n * self.out_h * self.out_w
        self.k_total = c_in * kernel_h * kernel_w

        self.kernel = _conv2d_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": False,
            }
        if sm_version in {80}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "threads": 128,
                "num_stages": 2,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "threads": 128,
            "num_stages": 2,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        shared_memory_limit_bytes = get_shared_memory_limit_bytes()
        configs = itertools.product(
            [64, 128],
            [64, 128, 256],
            [64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasterization in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > shared_memory_limit_bytes:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        # OIHW -> HWIO to match the kernel layout expected by the implicit GEMM path.
        weight_hwcf = weight.permute(2, 3, 1, 0).contiguous()
        return _conv2d_wrapped_kernel(
            self.n,
            self.c_in,
            self.h,
            self.w,
            self.c_out,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.has_bias,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight_hwcf,
            bias,
        )


class Conv2d1x1Kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.has_bias = has_bias

        self.kernel = _conv2d_1x1_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {80}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 1,
                "threads": 128,
                "enable_rasterization": True,
            }
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 2,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 1,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        shared_memory_limit_bytes = get_shared_memory_limit_bytes()
        configs = itertools.product(
            [64, 128, 256],
            [64, 128, 256],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasterization in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > shared_memory_limit_bytes:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        # OIHW -> OC,IC since the 1x1 kernel consumes a dense [C_out, C_in] weight matrix.
        weight_oc_ci = weight.view(self.c_out, self.c_in).contiguous()
        return _conv2d_1x1_wrapped_kernel(
            self.n,
            self.c_in,
            self.h,
            self.w,
            self.c_out,
            self.has_bias,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight_oc_ci,
            bias,
        )


# ---------------------------------------------------------------------------
# Conv3d
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _conv3d_kernel(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_d = (d_in + 2 * pad_d - kernel_d) // stride_d + 1
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    k_total = kernel_d * kernel_h * kernel_w * c_in

    # Re-enable automatic async copy once TileLang lowers scalar cp.async
    # widening for vectorized manual data loads. Keep weight T.copy eligible for TMA.
    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv3d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.prim_func
        def _conv3d_main(
            x: T.Tensor((n, d_in, h_in, w_in, c_in), dtype),  # type: ignore
            weight: T.Tensor((kernel_d, kernel_h, kernel_w, c_in, c_out), dtype),  # type: ignore
            out: T.Tensor((n, out_d, out_h, out_w, c_out), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_d * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((k_total, c_out), dtype, weight.data)
                out_flat = T.Tensor((n * out_d * out_h * out_w, c_out), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        m_idx = by * block_m + i
                        k_idx = k_iter * block_k + j
                        kd = k_idx // (kernel_h * kernel_w * c_in)
                        kh = (k_idx // (kernel_w * c_in)) % kernel_h
                        kw = (k_idx // c_in) % kernel_w
                        ci = k_idx % c_in
                        out_idx = m_idx % (out_d * out_h * out_w)
                        batch = m_idx // (out_d * out_h * out_w)
                        od = out_idx // (out_h * out_w)
                        oh = (out_idx // out_w) % out_h
                        ow = out_idx % out_w
                        id_ = od * stride_d + kd - pad_d
                        ih = oh * stride_h + kh - pad_h
                        iw = ow * stride_w + kw - pad_w
                        in_bound = (
                            (m_idx < n * out_d * out_h * out_w)
                            & (k_idx < k_total)
                            & (id_ >= 0)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (id_ < d_in)
                            & (ih < h_in)
                            & (iw < w_in)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x[batch, id_, ih, iw, ci],
                            T.cast(0.0, dtype),
                        )

                    T.copy(weight_flat[k_iter * block_k, bx * block_n], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_d * out_h * out_w) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_d * out_h * out_w) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if m_idx < n * out_d * out_h * out_w and oc < c_out:
                        out_flat[m_idx, oc] = out_shared[i, j]

        return _conv3d_main

    return _conv3d_func


@torch.library.custom_op("top::conv3d_wrapped_kernel", mutates_args=())
def _conv3d_wrapped_kernel(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv3d_kernel(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        c_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        has_bias,
        dtype,
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@_conv3d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out_d = (d_in + 2 * pad_d - kernel_d) // stride_d + 1
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    return torch.empty((n, out_d, out_h, out_w, c_out), dtype=inputs[0].dtype, device=inputs[0].device)


class Conv3dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_d = kernel_d
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_d = stride_d
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_d = (d_in + 2 * pad_d - kernel_d) // stride_d + 1
        self.out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
        self.out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
        self.m = n * self.out_d * self.out_h * self.out_w
        self.k_total = c_in * kernel_d * kernel_h * kernel_w

        self.kernel = _conv3d_kernel(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            c_out,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        shared_memory_limit_bytes = get_shared_memory_limit_bytes()
        configs = itertools.product(
            [32, 64, 128],
            [32, 64, 128],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasterization in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > shared_memory_limit_bytes:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasterization": enable_rasterization,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        # OIDHW -> DHWIO so the kernel can flatten weights into [K_total, C_out].
        weight_kdhwio = weight.permute(2, 3, 4, 1, 0).contiguous()
        return _conv3d_wrapped_kernel(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            self.c_out,
            self.kernel_d,
            self.kernel_h,
            self.kernel_w,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.pad_d,
            self.pad_h,
            self.pad_w,
            self.has_bias,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight_kdhwio,
            bias,
        )
