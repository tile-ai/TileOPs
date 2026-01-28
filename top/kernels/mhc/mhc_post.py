import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["mhc_post_kernel"]


def _mhc_post_kernel(batch: int, n_expand: int, c_x: int, x_dtype: str = 'bfloat16'):

    dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mhc_func(block_x_b, block_C, num_stages, threads=128):

        @T.macro
        def _get_output_x(
                x_layer_out: T.Tensor([batch, c_x], x_dtype),
                h_post: T.Tensor([batch, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_out: T.Tensor([batch, n_expand * c_x], x_dtype),
        ):
            with T.Kernel(batch, c_x // block_C, threads=threads) as (bx, by):

                # copy the h_post matrix into fragment
                h_post_shared = T.alloc_shared([n_expand], dtype)
                x_layer_out_shared = T.alloc_shared([block_C], dtype)
                x_res_shared = T.alloc_shared([n_expand, block_C], dtype)
                x_out_shared = T.alloc_shared([n_expand * block_C], dtype)

                for i in T.Parallel(n_expand):
                    h_post_shared[i] = h_post[bx, i]

                for i in T.Parallel(block_C):
                    x_layer_out_shared[i] = x_layer_out[bx, by * block_C + i]

                for i, j in T.Parallel(n_expand, block_C):
                    x_res_shared[i, j] = x_res[bx, i * c_x + by * block_C + j]

                for i, j in T.Parallel(n_expand, block_C):
                    x_out_shared[i * n_expand +
                                 j] = h_post_shared[i] * x_layer_out_shared[j] + x_res_shared[i, j]
                    x_out[bx, i * c_x + block_C * by + j] = x_out_shared[i * n_expand + j]

        @T.prim_func
        def mhc_post(
                x_layer_out: T.Tensor([batch, c_x], x_dtype),
                h_post: T.Tensor([batch, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_out: T.Tensor([batch, n_expand * c_x], x_dtype),
        ):
            _get_output_x(x_layer_out, h_post, x_res, x_out)

        return mhc_post

    return _mhc_func


@torch.library.custom_op("top::mhc_post_wrapped_kernel", mutates_args=())
def _mhc_post_wrapped_kernel(batch: int, n_expand: int, c_x: int, dtype: str, block_x_b: int,
                             block_C: int, num_stages: int, threads: int, x_layer_out: torch.Tensor,
                             h_post: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
    return _mhc_post_kernel(batch, n_expand, c_x, dtype)(block_x_b, block_C, num_stages,
                                                         threads)(x_layer_out, h_post, x_res)


@_mhc_post_wrapped_kernel.register_fake
def _(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: str,
    num_stages: int,
    threads: int,
    *input,
) -> torch.Tensor:
    return torch.empty_like(input[0], dtype=input[0].dtype, device=input[0].device)


class mhc_post_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 n_expand,
                 c_x,
                 dtype: str = 'float32',
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32
        self.kernel = _mhc_post_kernel(self.batch, self.n_expand, self.c_x, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {'block_x_b': 1, 'block_C': 64, "num_stages": 2, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        num_stages = [2, 3]
        threads = [128, 256]
        block_x_b = [1, 8, 64]
        block_C = [64, 128]
        _configs = list(itertools.product(block_x_b, block_C, num_stages, threads))

        configs = [{
            'block_x_b': c[0],
            'block_C': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, x_layer_out, h_post, x_res):

        result = _mhc_post_wrapped_kernel(self.batch, self.n_expand, self.c_x, self.dtype_str,
                                          self.config["block_x_b"], self.config["block_C"],
                                          self.config["num_stages"], self.config["threads"],
                                          x_layer_out, h_post, x_res)
        return result
