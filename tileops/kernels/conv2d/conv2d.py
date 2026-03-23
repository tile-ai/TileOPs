import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm.gemm import GemmKernel
from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv2d1x1Kernel", "Conv2d3x3Kernel", "Conv2dKernel"]

_HOPPER_SHARED_MEMORY_LIMIT_BYTES = 227 * 1024


def _conv2d_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    return per_stage_bytes * max(1, num_stages)


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
    out_h = h
    out_w = w
    hw = h * w

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_1x1_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasteration: bool,
    ):
        @T.prim_func
        def _conv2d_1x1_main(
            x: T.Tensor((n, c_in, hw), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
            out: T.Tensor((n, c_out, hw), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(hw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.use_swizzle(10, enable=enable_rasteration)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)
                    T.copy(x[bz, k_iter * block_k, bx * block_n], data_shared)
                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    if has_bias:
                        out_shared[i, j] = T.cast(
                            out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype)
                    else:
                        out_shared[i, j] = T.cast(out_local[i, j], dtype)

                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

        return _conv2d_1x1_main

    return _conv2d_1x1_func


def _conv2d_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    k_h: int,
    k_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = (h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (w + 2 * pad_w - k_w) // stride_w + 1
    k_total = k_h * k_w * c_in

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasteration: bool,
    ):
        @T.prim_func
        def _conv2d_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in, k_h, k_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.use_swizzle(10, enable=enable_rasteration)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        m_idx = by * block_m + i
                        k_idx = k_iter * block_k + j
                        out_hw = m_idx % (out_h * out_w)
                        oh = out_hw // out_w
                        ow = out_hw % out_w
                        kh = k_idx // (k_w * c_in)
                        kw = (k_idx // c_in) % k_w
                        ci = k_idx % c_in
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
                            x[m_idx // (out_h * out_w), ci, ih, iw],
                            T.cast(0.0, dtype),
                        )

                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        oc = bx * block_n + j
                        kh = k_idx // (k_w * c_in)
                        kw = (k_idx // c_in) % k_w
                        ci = k_idx % c_in
                        weight_shared[i, j] = T.if_then_else(
                            (k_idx < k_total) & (oc < c_out),
                            weight[oc, ci, kh, kw],
                            T.cast(0.0, dtype),
                        )

                    T.gemm(data_shared, weight_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if m_idx < n * out_h * out_w and oc < c_out:
                        out_hw = m_idx % (out_h * out_w)
                        oh = out_hw // out_w
                        ow = out_hw % out_w
                        if has_bias:
                            out[m_idx // (out_h * out_w), oc, oh, ow] = T.cast(
                                out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype)
                        else:
                            out[m_idx // (out_h * out_w), oc, oh, ow] = T.cast(
                                out_local[i, j], dtype)

        return _conv2d_main

    return _conv2d_func


def _conv2d_3x3_im2col_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str = "float16",
):
    out_h = (h + 2 * pad_h - 3) // stride_h + 1
    out_w = (w + 2 * pad_w - 3) // stride_w + 1
    m = n * out_h * out_w
    k_total = c_in * 9

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_3x3_im2col_func(
        block_m: int,
        block_k: int,
        threads: int,
    ):
        @T.prim_func
        def _conv2d_3x3_im2col_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            col: T.Tensor((m, k_total), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(k_total, block_k),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                for i, j in T.Parallel(block_m, block_k):
                    m_idx = by * block_m + i
                    k_idx = bx * block_k + j
                    filter_idx = k_idx % 9
                    ci = k_idx // 9
                    out_idx = m_idx % (out_h * out_w)
                    batch = m_idx // (out_h * out_w)
                    oh = out_idx // out_w
                    ow = out_idx % out_w
                    kh = filter_idx // 3
                    kw = filter_idx % 3
                    ih = oh * stride_h + kh - pad_h
                    iw = ow * stride_w + kw - pad_w
                    in_bound = (
                        (m_idx < m)
                        & (k_idx < k_total)
                        & (ih >= 0)
                        & (iw >= 0)
                        & (ih < h)
                        & (iw < w)
                    )
                    if m_idx < m and k_idx < k_total:
                        col[m_idx, k_idx] = T.if_then_else(
                            in_bound,
                            x[batch, ci, ih, iw],
                            T.cast(0.0, dtype),
                        )

        return _conv2d_3x3_im2col_main

    return _conv2d_3x3_im2col_func


def _conv2d_3x3_shared_memory_bytes(
    block_m: int,
    block_k: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    return block_m * block_k * dtype_bytes


class Conv2dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        k_h: int,
        k_w: int,
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
        self.k_h = k_h
        self.k_w = k_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.has_bias = has_bias

        self.kernel = _conv2d_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            k_h,
            k_w,
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
                "block_m": 128,
                "block_n": 256,
                "block_k": 32,
                "num_stages": 2,
                "threads": 128,
                "enable_rasteration": True,
            }
        if sm_version in {90}:
            return {
                "block_m": 128,
                "block_n": 128,
                "block_k": 64,
                "num_stages": 0,
                "threads": 128,
                "enable_rasteration": True,
            }
        return {
            "block_m": 128,
            "block_n": 256,
            "block_k": 32,
            "num_stages": 0,
            "threads": 128,
            "enable_rasteration": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product(
            [64, 128, 256],
            [64, 128, 256],
            [32, 64],
            [1, 2, 3],
            [128, 256],
            [True],
        )
        return [
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasteration": enable_rasteration,
            }
            for block_m, block_n, block_k, num_stages, threads, enable_rasteration in configs
        ]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # if bias is None:
        #     bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        return self.kernel(
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasteration"],
        )(x, weight, bias)


class Conv2d3x3Kernel(Kernel):
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
        self.out_h = (h + 2 * pad_h - 3) // stride_h + 1
        self.out_w = (w + 2 * pad_w - 3) // stride_w + 1
        self.m = n * self.out_h * self.out_w
        self.k_total = c_in * 9

        self.im2col_kernel = _conv2d_3x3_im2col_kernel(
            n,
            c_in,
            h,
            w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            self.dtype_str,
        )
        self.gemm_kernel = GemmKernel(self.m, c_out, self.k_total, dtype, tune=tune)
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 128,
                "block_k": 128,
                "threads": 256,
            }
        if sm_version in {80}:
            return {
                "block_m": 64,
                "block_k": 64,
                "threads": 128,
            }
        return {
            "block_m": 64,
            "block_k": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product(
            [64, 128, 256],
            [64, 128],
            [128, 256],
        )
        valid_configs = []
        for block_m, block_k, threads in configs:
            shared_memory_bytes = _conv2d_3x3_shared_memory_bytes(block_m, block_k, self.dtype)
            if shared_memory_bytes > _HOPPER_SHARED_MEMORY_LIMIT_BYTES:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_k": block_k,
                "threads": threads,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        col = self.im2col_kernel(
            self.config["block_m"],
            self.config["block_k"],
            self.config["threads"],
        )(x)
        weight_matrix = weight.permute(1, 2, 3, 0).contiguous().view(self.k_total, self.c_out)
        out_matrix = self.gemm_kernel(col, weight_matrix)
        out = out_matrix.view(self.n, self.out_h, self.out_w, self.c_out).permute(0, 3, 1, 2).contiguous()
        if self.has_bias and bias is not None:
            out = out + bias.view(1, self.c_out, 1, 1)
        return out


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
                "enable_rasteration": True,
            }
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 2,
                "threads": 128,
                "enable_rasteration": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 1,
            "threads": 128,
            "enable_rasteration": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product(
            [64, 128, 256],
            [64, 128, 256],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasteration in configs:
            shared_memory_bytes = _conv2d_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, self.dtype)
            if shared_memory_bytes > _HOPPER_SHARED_MEMORY_LIMIT_BYTES:
                continue
            valid_configs.append({
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": threads,
                "enable_rasteration": enable_rasteration,
            })
        return valid_configs

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.kernel(
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasteration"],
        )(
            x.view(self.n, self.c_in, self.h * self.w),
            weight.view(self.c_out, self.c_in),
            bias,
        )
        return out.view(self.n, self.c_out, self.h, self.w)
