import functools
import itertools
from typing import Optional

import torch

from tileops.kernels.conv.implicit_gemm import conv_shared_memory_bytes, make_conv_nd_implicit_gemm_kernel
from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv3dKernel"]

_HOPPER_SHARED_MEMORY_LIMIT_BYTES = 227 * 1024


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
    return make_conv_nd_implicit_gemm_kernel(
        batch=n,
        c_in=c_in,
        c_out=c_out,
        in_spatial_shape=(d_in, h_in, w_in),
        kernel_shape=(kernel_d, kernel_h, kernel_w),
        stride=(stride_d, stride_h, stride_w),
        padding=(pad_d, pad_h, pad_w),
        has_bias=has_bias,
        dtype=dtype,
    )


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
                "enable_rasteration": True,
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
            [True],
        )
        valid_configs = []
        dtype_bytes = torch.tensor([], dtype=self.dtype).element_size()
        for block_m, block_n, block_k, num_stages, threads, enable_rasteration in configs:
            shared_memory_bytes = conv_shared_memory_bytes(
                block_m, block_n, block_k, num_stages, dtype_bytes)
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
        # OIDHW -> KDHWIO so the shared implicit-GEMM path can flatten weights to [K_total, C_out].
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
