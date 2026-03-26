import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.conv.implicit_gemm import conv_shared_memory_bytes, make_conv_nd_implicit_gemm_kernel
from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv2d1x1Kernel", "Conv2dKernel"]

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
        enable_rasteration: bool,
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

                T.use_swizzle(10, enable=enable_rasteration)
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
    return make_conv_nd_implicit_gemm_kernel(
        batch=n,
        c_in=c_in,
        c_out=c_out,
        in_spatial_shape=(h, w),
        kernel_shape=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        has_bias=has_bias,
        dtype=dtype,
        enable_hopper_im2col=(get_sm_version() == 90),
    )


@torch.library.custom_op("top::conv2d_1x1_wrapped_kernel", mutates_args=())
def _conv2d_1x1_wrapped_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasteration: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv2d_1x1_kernel(
        n, c_in, h, w, c_out, 1, 1, 0, 0, True, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasteration)(x, weight, bias)


@_conv2d_1x1_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasteration: bool,
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
    enable_rasteration: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv2d_kernel(
        n, c_in, h, w, c_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasteration)(x, weight, bias)


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
    enable_rasteration: bool,
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
                "enable_rasteration": True,
            }
        if sm_version in {80}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "threads": 128,
                "num_stages": 2,
                "enable_rasteration": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "threads": 128,
            "num_stages": 2,
            "enable_rasteration": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product(
            [64, 128],
            [64, 128, 256],
            [64, 128],
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
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        # OIHW -> HWIO so the shared implicit-GEMM path can flatten weights to [K_total, C_out].
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
            self.config["enable_rasteration"],
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
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        # 1x1 weights are consumed as [C_out, C_in], matching the GEMM-specialized fast path.
        weight_oc_ci = weight.view(self.c_out, self.c_in).contiguous()
        return _conv2d_1x1_wrapped_kernel(
            self.n,
            self.c_in,
            self.h,
            self.w,
            self.c_out,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasteration"],
            x,
            weight_oc_ci,
            bias,
        )
