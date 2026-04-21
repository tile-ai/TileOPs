import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv1dKernel", "Conv2d1x1Kernel", "Conv2dKernel", "Conv3dKernel"]


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
    return per_stage_bytes * max(1, num_stages)


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
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
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
                        il = ol * stride_l + kw - pad_l
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


@torch.library.custom_op("top::conv1d_wrapped_kernel", mutates_args=())
def _conv1d_wrapped_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
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
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_l, has_bias, dtype
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
    out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
    return torch.empty((n, out_l, c_out), dtype=inputs[0].dtype, device=inputs[0].device)


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
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
        self.m = n * self.out_l
        self.k_total = c_in * kernel_l

        self.kernel = _conv1d_kernel(
            n,
            c_in,
            l_in,
            c_out,
            kernel_l,
            stride_l,
            pad_l,
            has_bias,
            self.dtype_str,
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
        if bias is None:
            bias = torch.zeros(self.c_out, device=x.device, dtype=x.dtype)
        # OIK -> KIO so the kernel can flatten weights into [K_total, C_out].
        weight_kio = weight.permute(2, 1, 0).contiguous()
        return _conv1d_wrapped_kernel(
            self.n,
            self.c_in,
            self.l_in,
            self.c_out,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.has_bias,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["enable_rasterization"],
            x,
            weight_kio,
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

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
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

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
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
