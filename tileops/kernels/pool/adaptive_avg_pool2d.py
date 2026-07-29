import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["AdaptiveAvgPool2dKernel"]


@functools.lru_cache(maxsize=64)
def _adaptive_avg_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str = "float16",
):
    accum_dtype = "float"
    total_output = n * c_in * out_h * out_w
    # Static upper bounds for the adaptive bin extents (compile-time constants;
    # TileLang rejects dynamic T.serial bounds). Note ceil(in/out) alone is NOT
    # a valid bound: e.g. in=55/out=7 has a bin of 9 > ceil(55/7) = 8, and
    # expansion in=8/out=12 has bins of 2 > 1.
    max_kh = max(
        ((o + 1) * h_in + out_h - 1) // out_h - (o * h_in) // out_h
        for o in range(out_h)
    )
    max_kw = max(
        ((o + 1) * w_in + out_w - 1) // out_w - (o * w_in) // out_w
        for o in range(out_w)
    )

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _adaptive_avg_pool2d_func(block_m: int, threads: int):
        @T.prim_func
        def _adaptive_avg_pool2d_main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_w
                        spatial_idx = out_idx // out_w
                        oh = spatial_idx % out_h
                        channel_batch_idx = spatial_idx // out_h
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in

                        # PyTorch adaptive bins partition each spatial axis as
                        # [floor(o*in/out), ceil((o+1)*in/out)); bins are always
                        # non-empty, including output_size > input_size.
                        ih_start = (oh * h_in) // out_h
                        ih_end = ((oh + 1) * h_in + out_h - 1) // out_h
                        iw_start = (ow * w_in) // out_w
                        iw_end = ((ow + 1) * w_in + out_w - 1) // out_w

                        sum_val = T.alloc_var(T.float32)
                        sum_val = T.cast(0.0, accum_dtype)
                        # Static-bound loops (TileLang rejects dynamic T.serial
                        # bounds); guard skips lanes outside this output's bin.
                        for kh in T.serial(max_kh):
                            for kw in T.serial(max_kw):
                                if ih_start + kh < ih_end and iw_start + kw < iw_end:
                                    sum_val += T.cast(
                                        x[batch, c_idx, ih_start + kh, iw_start + kw],
                                        accum_dtype,
                                    )

                        bin_size = (ih_end - ih_start) * (iw_end - iw_start)
                        out[batch, c_idx, oh, ow] = T.cast(
                            sum_val / T.cast(bin_size, accum_dtype),
                            dtype,
                        )

        return _adaptive_avg_pool2d_main

    return _adaptive_avg_pool2d_func


@torch.library.custom_op("top::adaptive_avg_pool2d_wrapped_kernel", mutates_args=())
def _adaptive_avg_pool2d_wrapped_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _adaptive_avg_pool2d_kernel(
        n, c_in, h_in, w_in, out_h, out_w, dtype
    )(block_m, threads)(x)


@_adaptive_avg_pool2d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (dtype, block_m, threads)
    return torch.empty((n, c_in, out_h, out_w), dtype=x.dtype, device=x.device)


class AdaptiveAvgPool2dKernel(Kernel):
    """Adaptive average pooling forward kernel for NCHW inputs."""

    supported_archs: list[int] = [80, 86, 89, 90]

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
                f"AdaptiveAvgPool2dKernel supports float16 and bfloat16, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self.kernel = _adaptive_avg_pool2d_kernel(
            n, c_in, h_in, w_in, out_h, out_w, self.dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 256,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512], [128, 256, 512])
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _adaptive_avg_pool2d_wrapped_kernel(
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
