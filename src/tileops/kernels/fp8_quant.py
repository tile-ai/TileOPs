import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import FP8_E4M3_MAX, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel

__all__ = ["FP8QuantKernel"]

# A row whose absolute maximum falls below this quantizes against the floor, which
# keeps an all-zero row's scale finite. Matches the reference's ``clamp(min=1e-4)``.
_AMAX_FLOOR = 1e-4


def _block_rows(seq_len_kv: int, row_bytes: int, threads: int) -> int:
    """Rows per block: as many as give each thread one vector access, bounded two ways.

    A power of two keeps a thread's share of the tile a whole number of elements. Dividing
    ``seq_len_kv`` is what lets the body read and write whole blocks with no bounds test.
    """
    widest = threads * VECTOR_ACCESS_BYTES // row_bytes
    block_m = 1
    while block_m * 2 <= widest and seq_len_kv % (block_m * 2) == 0:
        block_m *= 2
    return block_m


@functools.lru_cache(maxsize=32)
def _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype: str):
    @tilelang.jit(out_idx=[1, 2])
    def _fp8_quant_fwd_func(block_m, threads):
        if seq_len_kv % block_m:
            raise ValueError(
                f"block_m={block_m} does not divide seq_len_kv={seq_len_kv}; the body reads "
                "and writes whole blocks, so a partial one would run past the tensor"
            )
        out_dtype = T.float8_e4m3fn
        scale_dtype = T.float32
        fp8_min = -FP8_E4M3_MAX
        fp8_max = FP8_E4M3_MAX

        @T.prim_func
        def _fp8_quant_fwd_main(
            input_tensor: T.Tensor[(batch, seq_len_kv, kv_group, index_dim), in_dtype],
            scale_tensor: T.Tensor[(batch, seq_len_kv, kv_group), scale_dtype],
            output_tensor: T.Tensor[(batch, seq_len_kv, kv_group, index_dim), out_dtype],
        ):
            with T.Kernel(batch, T.ceildiv(seq_len_kv, block_m), kv_group, threads=threads) as (
                bx,
                pid_m,
                g,
            ):
                input_local = T.alloc_fragment((block_m, index_dim), in_dtype)
                output_local = T.alloc_fragment((block_m, index_dim), out_dtype)
                amax_local = T.alloc_fragment((block_m,), scale_dtype)
                scale_local = T.alloc_fragment((block_m,), scale_dtype)
                recip_local = T.alloc_fragment((block_m,), scale_dtype)

                # The row is read once and quantized out of registers, and nothing is
                # written before the reduction has settled.
                for i, j in T.Parallel(block_m, index_dim):
                    input_local[i, j] = input_tensor[bx, pid_m * block_m + i, g, j]

                T.reduce_absmax(input_local, amax_local, dim=1)
                for i in T.Parallel(block_m):
                    amax_local[i] = T.max(amax_local[i], _AMAX_FLOOR)
                    scale_local[i] = amax_local[i] / fp8_max
                    # One reciprocal per row, so the elementwise step is a multiply.
                    recip_local[i] = 1.0 / scale_local[i]

                for i, j in T.Parallel(block_m, index_dim):
                    output_local[i, j] = T.clamp(
                        input_local[i, j] * recip_local[i], fp8_min, fp8_max
                    )

                for i in T.Parallel(block_m):
                    scale_tensor[bx, pid_m * block_m + i, g] = scale_local[i]
                for i, j in T.Parallel(block_m, index_dim):
                    output_tensor[bx, pid_m * block_m + i, g, j] = output_local[i, j]

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


@torch.library.custom_op("tileops::fp8_quant_wrapped_kernel", mutates_args=())
def _fp8_quant_wrapped_kernel(
    batch: int,
    seq_len_kv: int,
    kv_group: int,
    index_dim: int,
    in_dtype: str,
    block_m: int,
    threads: int,
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype)(block_m, threads)(
        input_tensor
    )


@_fp8_quant_wrapped_kernel.register_fake
def _(batch, seq_len_kv, kv_group, index_dim, in_dtype, block_m, threads, *inputs):
    return torch.empty(
        (batch, seq_len_kv, kv_group), dtype=torch.float32, device=inputs[0].device
    ), torch.empty(
        (batch, seq_len_kv, kv_group, index_dim), dtype=torch.float8_e4m3fn, device=inputs[0].device
    )


class FP8QuantKernel(Kernel):
    """Per-group fp8 quantization of a $[B \\times S\\_kv \\times G \\times D]$ index tensor.

    A block owns whole rows and reduces each of them across the threads holding it, so
    the launch shape follows from the row width rather than a search: ``block_m`` is the
    number of rows whose bytes give each of ``threads`` threads one vector access.

    Args:
        batch: Batch size.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        index_dim: Index (head) dimension.
        dtype: Torch dtype of the tensor being quantized. The output element
            type is fixed at ``torch.float8_e4m3fn`` and the scale at
            ``torch.float32``, so this is the kernel's only free dtype.
        config: Optional dict with "block_m" and "threads".
        tune: Whether to autotune.

    Raises:
        ValueError: ``block_m`` does not divide ``seq_len_kv``, raised where the kernel is
            built. The body reads and writes whole blocks, so a partial one would run past
            the tensor.
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.dtype = dtype
        self.kernel = _fp8_quant_kernel(
            self.batch, self.seq_len_kv, self.kv_group, self.index_dim, self.dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        threads = 128
        row_bytes = self.index_dim * self.dtype.itemsize
        return {"block_m": _block_rows(self.seq_len_kv, row_bytes, threads), "threads": threads}

    @property
    def autotune_configs(self) -> list[dict]:
        # ``_block_rows`` determines the launch from the row width, leaving one candidate.
        return [self.default_config]

    def forward(self, input_tensor: torch.Tensor):
        return _fp8_quant_wrapped_kernel(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            input_tensor,
        )
