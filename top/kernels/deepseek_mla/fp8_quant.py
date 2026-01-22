import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["Fp8QuantKernel"]


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


def fast_log2_ceil(x):
    bits_x = T.reinterpret(T.uint32, x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast(T.int32, exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret(T.float32, bits_x)


def _fp8_quant_kernel(seq_len_kv, index_dim):

    @tilelang.jit
    def _fp8_quant_fwd_func(num_stages, block_m, group_size, round_scale=False):
        #for fp8 indexer, group_size = index_dim
        in_dtype = T.bfloat16
        out_dtype = T.float8
        scale_dtype = T.float32
        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1 / fp8_max

        @T.prim_func
        def _fp8_quant_fwd_main(input_tensor: T.Tensor[(seq_len_kv, index_dim), in_dtype],
                                scale_tensor: T.Tensor[(seq_len_kv), scale_dtype],
                                output_tensor: T.Tensor[(seq_len_kv, index_dim), out_dtype]):
            with T.Kernel(
                    T.ceildiv(seq_len_kv, block_m), T.ceildiv(index_dim, group_size),
                    threads=128) as (
                        pid_m,
                        pid_n,
                    ):
                input_shared = T.alloc_shared((block_m, group_size), in_dtype)
                input_local = T.alloc_fragment((block_m, group_size), in_dtype)
                amax_local = T.alloc_fragment((block_m,), scale_dtype)
                scale_local = T.alloc_fragment((block_m,), scale_dtype)
                output_local = T.alloc_fragment((block_m, group_size), out_dtype)
                output_shared = T.alloc_shared((block_m, group_size), out_dtype)

                for _ in T.Pipelined(1, num_stages=num_stages):
                    T.copy(input_tensor[pid_m * block_m, pid_n * group_size], input_shared)
                    T.copy(input_shared, input_local)
                    T.reduce_absmax(input_local, amax_local, dim=1)
                    for i in T.Parallel(block_m):
                        amax_local[i] = T.max(amax_local[i], 1e-4)
                        if round_scale:
                            scale_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                        else:
                            scale_local[i] = amax_local[i] * fp8_max_inv
                    for i, j in T.Parallel(block_m, group_size):
                        output_local[i, j] = T.clamp(input_local[i, j] / scale_local[i], fp8_min,
                                                     fp8_max)
                    for i in T.Parallel(block_m):
                        scale_tensor[pid_m * block_m + i, pid_n] = scale_local[i]
                    T.copy(output_local, output_shared)
                    T.copy(output_shared, output_tensor[pid_m * block_m, pid_n * group_size])

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


@torch.library.custom_op("top::fp8_quant_wrapped_kernel", mutates_args=())
def _fp8_quant_wrapped_kernel(seq_len_kv, index_dim, num_stages, block_m, group_size, round_scale,
                              input_tensor, scale_tensor, output_tensor) -> torch.Tensor:
    return _fp8_quant_kernel(
        seq_len_kv,
        index_dim,
    )(num_stages, block_m, group_size, round_scale)(input_tensor, scale_tensor, output_tensor)


@_fp8_quant_wrapped_kernel.register_fake
def _(seq_len_kv, index_dim,
      num_stages, block_m, group_size, round_scale,
      input_tensor):
    return torch.empty(
        (seq_len_kv, index_dim), dtype=torch.bfloat16, device=input_tensor.device), torch.empty(
            (seq_len_kv), dtype=torch.float32, device=input_tensor.device), torch.empty(
                (seq_len_kv, index_dim), dtype=torch.float8, device=input_tensor.device)


class Fp8QuantKernel(Kernel):
    """Minimal FP8 quantization kernel class placeholder.

    The original tilelang implementation was partially edited and caused
    a syntax error. Keep a minimal class here so importing this module
    doesn't fail. The actual kernel logic can be restored or expanded
    later as needed.
    """

    supported_archs: list[int] = [90]

    def __init__(self,
                 seq_len_kv: int,
                 index_dim: int,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.config = config or {}
        self.kernel = _fp8_quant_wrapped_kernel(self.seq_len_kv, self.index_dim)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 0, "block_m": 32, "group_size": 128, "round_scale": False}

    @property
    def autotune_configs(self) -> list[dict]:
        num_stages = [0, 2]
        block_m = [32]
        group_size = [128]
        round_scale = [False, True]
        _configs = list(itertools.product(num_stages, block_m, group_size, round_scale))

        return [{
            'num_stages': c[0],
            'block_m': c[1],
            "group_size": c[2],
            "round_scale": c[3]
        } for c in _configs]

    def forward(self, input_tensor: torch.Tensor):
        return _fp8_quant_wrapped_kernel(self.seq_len_kv, self.index_dim, self.config["num_stages"],
                                         self.config["block_m"], self.config["group_size"],
                                         self.config["round_scale"], input_tensor)
