import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["Fp8QuantKernel"]


def _fp8_quant_kernel(seq_len_kv, index_dim, in_dtype):

    @tilelang.jit(out_idx=[1, 2])
    def _fp8_quant_fwd_func(num_stages, block_m):
        out_dtype = T.float8
        scale_dtype = T.float32
        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1 / fp8_max

        @T.prim_func
        def _fp8_quant_fwd_main(input_tensor: T.Tensor[(seq_len_kv, index_dim), in_dtype],
                                scale_tensor: T.Tensor[(seq_len_kv), scale_dtype],
                                output_tensor: T.Tensor[(seq_len_kv, index_dim), out_dtype]):
            with T.Kernel(T.ceildiv(seq_len_kv, block_m), threads=128) as (pid_m):
                input_shared = T.alloc_shared((block_m, index_dim), in_dtype)
                input_local = T.alloc_fragment((block_m, index_dim), in_dtype)
                amax_local = T.alloc_fragment((block_m,), scale_dtype)
                scale_local = T.alloc_fragment((block_m,), scale_dtype)
                output_local = T.alloc_fragment((block_m, index_dim), out_dtype)
                output_shared = T.alloc_shared((block_m, index_dim), out_dtype)

                for _ in T.Pipelined(1, num_stages=num_stages):
                    T.copy(input_tensor[pid_m * block_m, index_dim], input_shared)
                    T.copy(input_shared, input_local)
                    T.reduce_absmax(input_local, amax_local, dim=1)
                    for i in T.Parallel(block_m):
                        amax_local[i] = T.max(amax_local[i], 1e-4)
                        scale_local[i] = amax_local[i] * fp8_max_inv
                    for i, j in T.Parallel(block_m, index_dim):
                        output_local[i, j] = T.clamp(input_local[i, j] / scale_local[i], fp8_min,
                                                     fp8_max)
                    for i in T.Parallel(block_m):
                        scale_tensor[pid_m * block_m + i] = scale_local[i]
                    T.copy(output_local, output_shared)
                    T.copy(output_shared, output_tensor[pid_m * block_m, index_dim])

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


@torch.library.custom_op("top::fp8_quant_wrapped_kernel", mutates_args=())
def _fp8_quant_wrapped_kernel(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                              num_stages: int, block_m: int, input_tensor: torch.Tensor,
                              scale_tensor: torch.Tensor,
                              output_tensor: torch.Tensor) -> torch.Tensor:
    return _fp8_quant_kernel(seq_len_kv, index_dim, in_dtype)(num_stages,
                                                              block_m)(input_tensor, scale_tensor,
                                                                       output_tensor)


@_fp8_quant_wrapped_kernel.register_fake
def _(seq_len_kv, index_dim, in_dtype,
      num_stages, block_m,
      input_tensor, *inputs):
    return torch.empty((seq_len_kv), dtype=torch.float32, device=input_tensor.device), torch.empty(
        (seq_len_kv, index_dim), dtype=torch.float8, device=input_tensor.device)


class Fp8QuantKernel(Kernel):

    supported_archs: list[int] = [90]

    def __init__(self,
                 seq_len_kv: int,
                 index_dim: int,
                 in_dtype: torch.dtype,
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
        return {"num_stages": 0, "block_m": 32, "group_size": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        num_stages = [0, 2]
        block_m = [32]
        group_size = [128]
        _configs = list(itertools.product(num_stages, block_m, group_size))

        return [{'num_stages': c[0], 'block_m': c[1], "group_size": c[2]} for c in _configs]

    def forward(self, input_tensor: torch.Tensor):
        return _fp8_quant_wrapped_kernel(self.seq_len_kv, self.index_dim, self.config["num_stages"],
                                         self.config["block_m"], input_tensor)
