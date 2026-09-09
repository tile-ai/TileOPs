import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import FP8_E4M3_MAX, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel

__all__ = ["FP8QuantKernel"]

# Lower bound on a row's absolute maximum, so an all-zero row still has a finite scale.
_AMAX_FLOOR = 1e-4
# Threads a block runs. ``_block_rows`` reads it, so the two are one launch rule.
_THREADS = 128


def _block_rows(seq_len_kv: int, row_bytes: int) -> int:
    """Rows per block: the most whose bytes fit one vector access per thread.

    The result is a power of two that divides ``seq_len_kv``, which is what makes every
    block the launch covers a full one. A row width that leaves the tile indivisible by
    ``_THREADS`` still gets a valid block; it gets one whose threads read unequal shares.
    """
    widest = _THREADS * VECTOR_ACCESS_BYTES // row_bytes
    block_m = 1
    while block_m * 2 <= widest and seq_len_kv % (block_m * 2) == 0:
        block_m *= 2
    return block_m


@functools.lru_cache(maxsize=32)
def _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype: str):
    @tilelang.jit(out_idx=[1, 2])
    def _fp8_quant_fwd_func(block_m):
        if block_m < 1 or seq_len_kv % block_m:
            raise ValueError(
                f"block_m={block_m} must be a positive divisor of seq_len_kv={seq_len_kv}: "
                "the body reads and writes whole blocks, so any other width would run past "
                "the tensor"
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
            with T.Kernel(batch, T.ceildiv(seq_len_kv, block_m), kv_group, threads=_THREADS) as (
                bx,
                pid_m,
                g,
            ):
                input_local = T.alloc_fragment((block_m, index_dim), in_dtype)
                output_local = T.alloc_fragment((block_m, index_dim), out_dtype)
                amax_local = T.alloc_fragment((block_m,), scale_dtype)
                scale_local = T.alloc_fragment((block_m,), scale_dtype)
                recip_local = T.alloc_fragment((block_m,), scale_dtype)

                # Every read the block makes happens before any of its writes: the row is
                # read once here, and no output is stored until the scale has settled.
                for i, j in T.Parallel(block_m, index_dim):
                    input_local[i, j] = input_tensor[bx, pid_m * block_m + i, g, j]

                T.reduce_absmax(input_local, amax_local, dim=1)
                for i in T.Parallel(block_m):
                    amax_local[i] = T.max(amax_local[i], _AMAX_FLOOR)
                    scale_local[i] = amax_local[i] / fp8_max
                    recip_local[i] = 1.0 / scale_local[i]

                # Multiplying by the row's reciprocal is not bitwise the reference's divide
                # by the scale: an element on an fp8 rounding boundary can land one code
                # away. ``FP8QuantFwdOp`` states the bound this holds to.
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
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype)(block_m)(
        input_tensor
    )


@_fp8_quant_wrapped_kernel.register_fake
def _(batch, seq_len_kv, kv_group, index_dim, in_dtype, block_m, *inputs):
    return torch.empty(
        (batch, seq_len_kv, kv_group), dtype=torch.float32, device=inputs[0].device
    ), torch.empty(
        (batch, seq_len_kv, kv_group, index_dim), dtype=torch.float8_e4m3fn, device=inputs[0].device
    )


class FP8QuantKernel(Kernel):
    """Per-group fp8 quantization of a $[B \\times S\\_kv \\times G \\times D]$ index tensor.

    A block owns whole rows and reduces each of them across the threads holding it. The
    default ``block_m`` follows from the row width: it is the number of rows whose bytes fit
    one vector access per thread. A caller may override it with any positive divisor of
    ``seq_len_kv``.

    Args:
        batch: Batch size.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        index_dim: Index (head) dimension.
        dtype: Torch dtype of the tensor being quantized. The output element
            type is fixed at ``torch.float8_e4m3fn`` and the scale at
            ``torch.float32``, so this is the kernel's only free dtype.
        config: Optional dict with "block_m".
        tune: Whether to autotune.

    Raises:
        ValueError: ``block_m`` is not a positive divisor of ``seq_len_kv``, raised where
            the kernel is built. The body reads and writes whole blocks, so any other width
            would run past the tensor.
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
        return {"block_m": _block_rows(self.seq_len_kv, self.index_dim * self.dtype.itemsize)}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, input_tensor: torch.Tensor):
        return _fp8_quant_wrapped_kernel(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            self.dtype_str,
            self.config["block_m"],
            input_tensor,
        )
