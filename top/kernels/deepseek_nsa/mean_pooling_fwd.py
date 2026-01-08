import itertools
from typing import Any, Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["MeanPoolingFwdKernel"]


def _mean_pooling_kernel(
    batch_size: int,
    total_seqlen: int,
    total_chunks: int,
    heads: int,
    dim: int,
    chunk_size: int,
) -> None:
    dtype = "float16"
    accum_dtype = "float32"
    int_dtype = "int32"

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
            "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG", "-DENABLE_BF16",
            "-DCUDA_FP8_ENABLED=0"
        ],
    )
    def _mean_pooling_func(block_d: int, threads: int) -> None:

        nd = T.ceildiv(dim, block_d)

        x_shape = [total_seqlen, heads, dim]
        cu_seqlens_shape = [batch_size + 1]
        chunk_indices_shape = [total_chunks, 2]
        output_shape = [total_chunks, heads, dim]

        @T.prim_func
        def _mean_pooling_main(
                x_unpad: T.Tensor(x_shape, dtype),
                cu_seqlens: T.Tensor(cu_seqlens_shape, int_dtype),
                chunk_indices: T.Tensor(chunk_indices_shape, int_dtype),
                output: T.Tensor(output_shape, dtype),
        ) -> None:  # noqa: U100
            with T.Kernel(nd, total_chunks, heads, threads=threads) as (i_d, i_t, i_h):
                accum = T.alloc_fragment([block_d], accum_dtype)
                d_start = i_d * block_d

                seq_id = chunk_indices[i_t, 0]
                local_chunk_id = chunk_indices[i_t, 1]
                start = cu_seqlens[seq_id]
                end = cu_seqlens[seq_id + 1]
                seqlen = end - start

                chunk_start = local_chunk_id * chunk_size
                chunk_end = T.min(chunk_start + chunk_size, seqlen)
                actual_bt = chunk_end - chunk_start

                for d in T.Parallel(block_d):
                    accum[d] = T.cast(0, accum_dtype)
                for t_rel in T.serial(actual_bt):
                    t_abs = start + chunk_start + t_rel
                    for d in T.Parallel(block_d):
                        if d_start + d < dim:
                            accum[d] += T.cast(x_unpad[t_abs, i_h, d_start + d], accum_dtype)
                for d in T.Parallel(block_d):
                    if d_start + d < dim:
                        if actual_bt > 0:
                            output[i_t, i_h,
                                   d_start + d] = T.cast(accum[d] / T.cast(actual_bt, accum_dtype),
                                                         dtype)
                        else:
                            output[i_t, i_h, d_start + d] = T.cast(0, dtype)

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
    block_d: int,
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
    )(block_d, threads)(x_unpad, cu_seqlens, chunk_indices)


@_mean_pooling_wrapped_kernel.register_fake
def _(
        batch_size: int,
        total_seqlen: int,
        total_chunks: int,
        heads: int,
        dim: int,
        chunk_size: int,
        block_d: int,
        threads: int,
        *inputs: tuple[Any],  # noqa: U100
) -> torch.Tensor:
    # Output shape is [total_chunks, heads, dim]
    _ = (batch_size, total_seqlen, total_chunks, heads, dim, chunk_size, block_d, threads)
    x = inputs[0]
    return torch.empty(
        (total_chunks, heads, dim),
        device=x.device,
        dtype=x.dtype,
    )


class MeanPoolingFwdKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch_size: int,
                 total_seqlen: int,
                 total_chunks: int,
                 heads: int,
                 dim: int,
                 chunk_size: int,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
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
            "block_d": min(64, self.dim),
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_d = [32, 64, 128]
        threads = [32, 64, 128]
        return [{"block_d": b, "threads": t} for b, t in itertools.product(block_d, threads)]

    def forward(self, x_unpad: torch.Tensor, cu_seqlens: torch.Tensor,
                chunk_indices: torch.Tensor) -> torch.Tensor:
        return _mean_pooling_wrapped_kernel(self.batch_size, self.total_seqlen, self.total_chunks,
                                            self.heads, self.dim, self.chunk_size,
                                            self.config["block_d"], self.config["threads"], x_unpad,
                                            cu_seqlens, chunk_indices)
