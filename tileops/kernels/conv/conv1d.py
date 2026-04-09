import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.conv.common import conv_shared_memory_bytes, get_shared_memory_limit_bytes
from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv1dKernel"]


def _conv1d_out_length(
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
) -> int:
    return (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1


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
    out_l = _conv1d_out_length(l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l)
    k_total = kernel_l * c_in

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
            x: T.Tensor((n, l_in, c_in), dtype),  # type: ignore
            weight: T.Tensor((kernel_l, c_in, c_out), dtype),  # type: ignore
            out: T.Tensor((n, out_l, c_out), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_l, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((k_total, c_out), dtype, weight.data)
                out_flat = T.Tensor((n * out_l, c_out), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        m_idx = by * block_m + i
                        k_idx = k_iter * block_k + j
                        kw = k_idx // c_in
                        ci = k_idx % c_in
                        batch = m_idx // out_l
                        ol = m_idx % out_l
                        il = ol * stride_l + kw * dilation_l - pad_left
                        in_bound = (
                            (m_idx < n * out_l)
                            & (k_idx < k_total)
                            & (il >= 0)
                            & (il < l_in)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x[batch, il, ci],
                            T.cast(0.0, dtype),
                        )

                    T.copy(weight_flat[k_iter * block_k, bx * block_n], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_l) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_l) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if m_idx < n * out_l and oc < c_out:
                        out_flat[m_idx, oc] = out_shared[i, j]

        return _conv1d_main

    return _conv1d_func


@functools.lru_cache(maxsize=64)
def _grouped_conv1d_kernel(
    n: int,
    groups: int,
    c_in_per_group: int,
    l_in: int,
    c_out_per_group: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = _conv1d_out_length(l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l)
    k_total = kernel_l * c_in_per_group

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _grouped_conv1d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.prim_func
        def _grouped_conv1d_main(
            x: T.Tensor((n, l_in, groups * c_in_per_group), dtype),  # type: ignore
            weight: T.Tensor((groups, kernel_l, c_in_per_group, c_out_per_group), dtype),  # type: ignore
            out: T.Tensor((n, out_l, groups * c_out_per_group), dtype),  # type: ignore
            bias: T.Tensor((groups, c_out_per_group), dtype),  # type: ignore
        ):
            weight_flat = T.Tensor((groups, k_total, c_out_per_group), dtype, weight.data)
            with T.Kernel(
                T.ceildiv(c_out_per_group, block_n),
                T.ceildiv(n * out_l, block_m),
                groups,
                threads=threads,
            ) as (bx, by, bz):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        m_idx = by * block_m + i
                        k_idx = k_iter * block_k + j
                        kw = k_idx // c_in_per_group
                        ci = k_idx % c_in_per_group
                        batch = m_idx // out_l
                        ol = m_idx % out_l
                        il = ol * stride_l + kw * dilation_l - pad_left
                        in_bound = (
                            (m_idx < n * out_l)
                            & (k_idx < k_total)
                            & (il >= 0)
                            & (il < l_in)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x[batch, il, bz * c_in_per_group + ci],
                            T.cast(0.0, dtype),
                        )
                    T.copy(weight_flat[bz, k_iter * block_k, bx * block_n], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_l) & (oc < c_out_per_group),
                            T.cast(out_local[i, j] + T.cast(bias[bz, oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < n * out_l) & (oc < c_out_per_group),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if m_idx < n * out_l and oc < c_out_per_group:
                        batch = m_idx // out_l
                        ol = m_idx % out_l
                        out[batch, ol, bz * c_out_per_group + oc] = out_shared[i, j]

        return _grouped_conv1d_main

    return _grouped_conv1d_func


@torch.library.custom_op("top::conv1d_wrapped_kernel", mutates_args=())
def _conv1d_wrapped_kernel(
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
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right, dilation_l, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@_conv1d_wrapped_kernel.register_fake
def _(
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
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out_l = _conv1d_out_length(l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l)
    return torch.empty((n, out_l, c_out), dtype=inputs[0].dtype, device=inputs[0].device)


@torch.library.custom_op("top::grouped_conv1d_wrapped_kernel", mutates_args=())
def _grouped_conv1d_wrapped_kernel(
    n: int,
    groups: int,
    c_in_per_group: int,
    l_in: int,
    c_out_per_group: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
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
    return _grouped_conv1d_kernel(
        n,
        groups,
        c_in_per_group,
        l_in,
        c_out_per_group,
        kernel_l,
        stride_l,
        pad_left,
        pad_right,
        dilation_l,
        has_bias,
        dtype,
    )(block_m, block_n, block_k, num_stages, threads, enable_rasterization)(x, weight, bias)


@_grouped_conv1d_wrapped_kernel.register_fake
def _(
    n: int,
    groups: int,
    c_in_per_group: int,
    l_in: int,
    c_out_per_group: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
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
    out_l = _conv1d_out_length(l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l)
    return torch.empty((n, out_l, groups * c_out_per_group), dtype=inputs[0].dtype, device=inputs[0].device)


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
        pad_left: int,
        pad_right: int,
        dilation_l: int,
        dtype: torch.dtype,
        groups: int = 1,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.l_in = l_in
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.dilation_l = dilation_l
        self.dtype = dtype
        self.groups = groups
        self.has_bias = has_bias
        self.group_c_in = c_in // groups
        self.group_c_out = c_out // groups

        self.kernel = (
            _conv1d_kernel(
                n,
                self.group_c_in,
                l_in,
                self.group_c_out,
                kernel_l,
                stride_l,
                pad_left,
                pad_right,
                dilation_l,
                has_bias,
                self.dtype_str,
            )
            if groups == 1
            else _grouped_conv1d_kernel(
                n,
                groups,
                self.group_c_in,
                l_in,
                self.group_c_out,
                kernel_l,
                stride_l,
                pad_left,
                pad_right,
                dilation_l,
                has_bias,
                self.dtype_str,
            )
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 128,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 128,
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
            [64, 128, 256],
            [64, 128],
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
        if self.groups == 1:
            bias_tensor = bias
            if bias_tensor is None:
                bias_tensor = torch.zeros(self.group_c_out, device=x.device, dtype=x.dtype)
            weight_tensor = weight.permute(2, 1, 0).contiguous()
            return _conv1d_wrapped_kernel(
                self.n,
                self.group_c_in,
                self.l_in,
                self.group_c_out,
                self.kernel_l,
                self.stride_l,
                self.pad_left,
                self.pad_right,
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
                weight_tensor,
                bias_tensor,
            )

        bias_tensor = (
            torch.zeros((self.groups, self.group_c_out), device=x.device, dtype=x.dtype)
            if bias is None
            else bias.view(self.groups, self.group_c_out).contiguous()
        )
        weight_tensor = (
            weight.view(self.groups, self.group_c_out, self.group_c_in, self.kernel_l)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        return _grouped_conv1d_wrapped_kernel(
            self.n,
            self.groups,
            self.group_c_in,
            self.l_in,
            self.group_c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_left,
            self.pad_right,
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
            weight_tensor,
            bias_tensor,
        )
