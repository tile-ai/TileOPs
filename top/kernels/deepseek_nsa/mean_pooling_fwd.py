import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["mean_pooling_fwd_kernel"]


def _mean_pooling_kernel(
    batch_size: int,
    total_seqlen: int,
    total_chunks: int,
    heads: int,
    dim: int,
    chunk_size: int,
):
    dtype = "float16"
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        # pass_configs={
        #     tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        #     tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        #     tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # },
        compile_flags=[
            "--use_fast_math", "-O3", "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG",
            "-DENABLE_BF16", "-DCUDA_FP8_ENABLED=0"
        ],
    )
    def _mean_pooling_func(block_D, threads):

        ND = T.ceildiv(dim, block_D)

        x_shape = [total_seqlen, heads, dim]
        cu_seqlens_shape = [batch_size + 1]
        chunk_indices_shape = [total_chunks, 2]
        output_shape = [total_chunks, heads, dim]

        @T.prim_func
        def _mean_pooling_main(
                X_unpad: T.Tensor(x_shape, dtype),
                cu_seqlens: T.Tensor(cu_seqlens_shape, "int32"),
                chunk_indices: T.Tensor(chunk_indices_shape, "int32"),
                Output: T.Tensor(output_shape, dtype),
        ):
            with T.Kernel(ND, total_chunks, heads, threads=threads) as (i_d, i_t, i_h):
                accum = T.alloc_fragment([block_D], accum_dtype)
                d_start = i_d * block_D

                seq_id = chunk_indices[i_t, 0]
                local_chunk_id = chunk_indices[i_t, 1]
                start = cu_seqlens[seq_id]
                end = cu_seqlens[seq_id + 1]
                seqlen = end - start

                chunk_start = local_chunk_id * chunk_size
                chunk_end = T.min(chunk_start + chunk_size, seqlen)
                actual_bt = chunk_end - chunk_start

                for d in T.Parallel(block_D):
                    accum[d] = T.cast(0, accum_dtype)
                for t_rel in T.serial(actual_bt):
                    t_abs = start + chunk_start + t_rel
                    for d in T.Parallel(block_D):
                        if d_start + d < dim:
                            accum[d] += T.cast(X_unpad[t_abs, i_h, d_start + d], accum_dtype)
                for d in T.Parallel(block_D):
                    if d_start + d < dim:
                        Output[i_t, i_h,
                               d_start + d] = T.cast(accum[d] / T.cast(actual_bt, accum_dtype),
                                                     dtype)

        return _mean_pooling_main

    return _mean_pooling_func


@torch.library.custom_op("top::mean_pooling_fwd_wrapped_kernel", mutates_args=())
def _mean_pooling_wrapped_kernel(
    batch_size: int,
    total_seqlen: int,
    total_chunks: int,
    heads: int,
    dim: int,
    chunk_size: int,
    block_D: int,
    threads: int,
    x_unpad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    return _mean_pooling_kernel(
        batch_size,
        total_seqlen,
        total_chunks,
        heads,
        dim,
        chunk_size,
    )(block_D, threads)(x_unpad, cu_seqlens, chunk_indices)


@_mean_pooling_wrapped_kernel.register_fake
def _(
        batch_size: int,
        total_seqlen: int,
        total_chunks: int,
        heads: int,
        dim: int,
        chunk_size: int,
        block_D: int,
        threads: int,
        *inputs
) -> torch.Tensor:
    fake_o = torch.empty_like(inputs[0])
    return fake_o


class mean_pooling_fwd_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch_size: int,
                 total_seqlen: int,
                 total_chunks: int,
                 heads: int,
                 dim: int,
                 chunk_size: int,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch_size = batch_size
        self.total_seqlen = total_seqlen
        self.total_chunks = total_chunks
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size

        self.kernel = _mean_pooling_kernel(
            self.batch_size,
            self.total_seqlen,
            self.total_chunks,
            self.heads,
            self.dim,
            self.chunk_size,
        )

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_D": min(64, self.dim),
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_D = [32, 64, 128]
        threads = [32, 64, 128]
        _configs = list(itertools.product(block_D, threads))
        configs = [{"block_D": c[0], "threads": c[1]} for c in _configs]
        return configs

    def forward(self, x_unpad: torch.Tensor, cu_seqlens: torch.Tensor, chunk_indices: torch.Tensor):
        return _mean_pooling_wrapped_kernel(self.batch_size, self.total_seqlen, self.total_chunks,
                                            self.heads, self.dim, self.chunk_size,
                                            self.config["block_D"], self.config["threads"], x_unpad,
                                            cu_seqlens, chunk_indices)
