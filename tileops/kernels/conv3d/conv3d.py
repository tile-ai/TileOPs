import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv3dKernel"]

_HOPPER_SHARED_MEMORY_LIMIT_BYTES = 227 * 1024


def _conv3d_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    return per_stage_bytes * max(1, num_stages)


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
        enable_rasteration: bool,
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

                T.use_swizzle(10, enable=enable_rasteration)
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
    enable_rasteration: bool,
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
    )(block_m, block_n, block_k, num_stages, threads, enable_rasteration)(x, weight, bias)


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
    enable_rasteration: bool,
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
                "enable_rasteration": False,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
            "enable_rasteration": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product(
            [32, 64, 128],
            [32, 64, 128],
            [32, 64, 128],
            [2, 3],
            [128, 256],
            [True, False],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasteration in configs:
            shared_memory_bytes = _conv3d_shared_memory_bytes(
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
            self.config["enable_rasteration"],
            x,
            weight_kdhwio,
            bias,
        )
