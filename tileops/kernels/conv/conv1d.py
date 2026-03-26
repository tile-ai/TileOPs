import functools
import itertools
from typing import Optional

import torch

from tileops.kernels.conv.implicit_gemm import conv_shared_memory_bytes, make_conv_nd_implicit_gemm_kernel
from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv1dKernel"]

_HOPPER_SHARED_MEMORY_LIMIT_BYTES = 227 * 1024


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
    return make_conv_nd_implicit_gemm_kernel(
        batch=n,
        c_in=c_in,
        c_out=c_out,
        in_spatial_shape=(l_in,),
        kernel_shape=(kernel_l,),
        stride=(stride_l,),
        padding=(pad_l,),
        has_bias=has_bias,
        dtype=dtype,
    )


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
    enable_rasteration: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _conv1d_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_l, has_bias, dtype
    )(block_m, block_n, block_k, num_stages, threads, enable_rasteration)(x, weight, bias)


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
    enable_rasteration: bool,
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
                "enable_rasteration": True,
            }
        return {
            "block_m": 128,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
            "enable_rasteration": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product(
            [64, 128, 256],
            [64, 128],
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
        # OC, CI, K -> K, CI, OC so the shared implicit-GEMM path can flatten weights to [K_total, C_out].
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
            self.config["enable_rasteration"],
            x,
            weight_kio,
            bias,
        )
