from typing import Any, Callable, Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["MeanPoolingFwdKernel"]


def _mean_pooling_kernel(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_bacth: int,
    seq_num: int,
    use_offsets: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:

    @tilelang.jit(out_idx=[1], pass_configs={"tl.disable_tma_lower": True})
    def _mean_pooling_func(bdim: int, threads: int) -> None:

        @T.prim_func
        def _mean_pooling_main(
                x: T.Tensor((batch_size, seq_len, heads, dim), dtype),
                o: T.Tensor((batch_size, chunks_per_bacth, heads, dim), dtype),
                offsets: T.Tensor((seq_num + 1,), T.int32),
                indices: T.Tensor((chunks_per_bacth, 2),
                                  T.int32),  # [chunks_per_bacth, 2] (seq_id, chunk_id in sequence)
        ) -> None:
            with T.Kernel(
                    T.ceildiv(dim, bdim), chunks_per_bacth, batch_size * heads,
                    threads=threads) as (i_d, i_t, i_bh):
                i_b = i_bh // heads
                i_h = i_bh % heads
                # load data [chunk_size, D]
                x_shared = T.alloc_shared((chunk_size, bdim), dtype)
                x_local = T.alloc_fragment((chunk_size, bdim), dtype)
                output_local = T.alloc_fragment((bdim,), accum_dtype)

                start_token = T.alloc_var(T.int32)
                end_token = T.alloc_var(T.int32)

                if use_offsets == 0:
                    start_token = i_t * chunk_size
                    end_token = T.min(start_token + chunk_size, seq_len)
                else:
                    seq_id = indices[i_t, 0]
                    local_chunk_id = indices[i_t, 1]
                    start_token = offsets[seq_id] + local_chunk_id * chunk_size
                    end_token = T.min(start_token + chunk_size, offsets[seq_id + 1])

                start_dim = i_d * bdim
                end_dim = T.min(start_dim + bdim, dim)

                T.clear(x_shared)
                T.copy(x[i_b, start_token:end_token, i_h, start_dim:end_dim],
                       x_shared[0:end_token - start_token, :end_dim - start_dim])
                T.copy(x_shared, x_local)
                T.reduce_sum(x_local, output_local, dim=0)
                for d_idx in T.Parallel(bdim):
                    o[i_b, i_t, i_h, start_dim + d_idx] = T.cast(
                        output_local[d_idx] / T.cast(end_token - start_token, accum_dtype), dtype)

        return _mean_pooling_main

    return _mean_pooling_func


@torch.library.custom_op("top::mean_pooling_fwd_wrapped_kernel", mutates_args=())
def _mean_pooling_wrapped_kernel(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_bacth: int,
    seq_num: int,
    use_offsets: int,
    dtype: str,
    accum_dtype: str,
    bdim: int,
    threads: int,
    x: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return _mean_pooling_kernel(
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        chunks_per_bacth=chunks_per_bacth,
        seq_num=seq_num,
        use_offsets=use_offsets,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )(bdim, threads)(x, offsets, indices)


@_mean_pooling_wrapped_kernel.register_fake
def _(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_bacth: int,
    seq_num: int,
    use_offsets: int,
    dtype: str,
    accum_dtype: str,
    bdim: int,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    # Output shape is [batch_size, chunks_per_bacth, heads, dim]
    _ = (seq_len, chunk_size, seq_num, bdim, use_offsets, dtype, accum_dtype, threads)
    x = inputs[0]
    return torch.empty(
        (batch_size, chunks_per_bacth, heads, dim),
        device=x.device,
        dtype=x.dtype,
    )


class MeanPoolingFwdKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 heads: int,
                 dim: int,
                 chunk_size: int,
                 chunks_per_bacth: int,
                 seq_num: int,
                 use_offsets: int,
                 dtype: torch.dtype,
                 accum_dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunks_per_bacth = chunks_per_bacth
        self.seq_num = seq_num
        self.use_offsets = use_offsets
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.dtype_name = str(dtype).split('.')[-1]
        self.accum_dtype_name = str(accum_dtype).split('.')[-1]

        self.kernel = _mean_pooling_kernel(self.batch_size, self.seq_len, self.heads, self.dim,
                                           self.chunk_size, self.chunks_per_bacth, self.seq_num,
                                           self.use_offsets, self.dtype_name, self.accum_dtype_name)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "bdim": 128,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        threads = [32, 64, 128, 256]
        bdim = [16, 32, 64, 128]
        return [{"bdim": b, "threads": t} for b in bdim for t in threads]

    def forward(self, x: torch.Tensor, offsets: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        return _mean_pooling_wrapped_kernel(self.batch_size, self.seq_len, self.heads, self.dim,
                                            self.chunk_size, self.chunks_per_bacth, self.seq_num,
                                            self.use_offsets, self.dtype_name,
                                            self.accum_dtype_name, self.config["bdim"],
                                            self.config["threads"], x, offsets, indices)
