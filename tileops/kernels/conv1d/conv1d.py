import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version

__all__ = ["Conv1dKernel"]

_HOPPER_SHARED_MEMORY_LIMIT_BYTES = 227 * 1024


def _conv1d_shared_memory_bytes(
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
        enable_rasteration: bool,
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

                T.use_swizzle(10, enable=enable_rasteration)
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
                "enable_rasteration": False,
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
            [True, False],
        )
        valid_configs = []
        for block_m, block_n, block_k, num_stages, threads, enable_rasteration in configs:
            shared_memory_bytes = _conv1d_shared_memory_bytes(
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
