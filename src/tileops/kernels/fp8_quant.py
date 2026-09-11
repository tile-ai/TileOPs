import functools
from typing import ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import FP8_E4M3_MAX, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel

__all__ = ["FP8QuantKernel"]

# workloads/fp8_quant.py clamps a row's absolute maximum to this before dividing.
_AMAX_FLOOR = 1e-4


def _pow2_floor(n: int) -> int:
    """The largest power of two at most *n*, and at least one."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=32)
def _fp8_quant_kernel(rows: int, index_dim: int, in_dtype: str, threads: int):
    @tilelang.jit(out_idx=[1, 2])
    def _fp8_quant_fwd_func(block_m):
        if block_m < 1:
            raise ValueError(f"block_m={block_m} must be positive")
        out_dtype = T.float8_e4m3fn
        scale_dtype = T.float32
        fp8_min = -FP8_E4M3_MAX
        fp8_max = FP8_E4M3_MAX

        @T.prim_func
        def _fp8_quant_fwd_main(
            input_tensor: T.Tensor[(rows, index_dim), in_dtype],
            scale_tensor: T.Tensor[(rows,), scale_dtype],
            output_tensor: T.Tensor[(rows, index_dim), out_dtype],
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as pid_m:
                input_local = T.alloc_fragment((block_m, index_dim), in_dtype)
                output_local = T.alloc_fragment((block_m, index_dim), out_dtype)
                amax_local = T.alloc_fragment((block_m,), scale_dtype)
                scale_local = T.alloc_fragment((block_m,), scale_dtype)
                recip_local = T.alloc_fragment((block_m,), scale_dtype)

                # A block past the axis re-reads its last row; the stores drop those rows.
                for i, j in T.Parallel(block_m, index_dim):
                    input_local[i, j] = input_tensor[T.min(pid_m * block_m + i, rows - 1), j]

                T.reduce_absmax(input_local, amax_local, dim=1)
                for i in T.Parallel(block_m):
                    amax_local[i] = T.max(amax_local[i], _AMAX_FLOOR)
                    scale_local[i] = amax_local[i] / fp8_max
                    recip_local[i] = 1.0 / scale_local[i]

                # Not the reference's divide by the scale; ``FP8QuantFwdOp`` bounds the gap.
                for i, j in T.Parallel(block_m, index_dim):
                    output_local[i, j] = T.clamp(
                        input_local[i, j] * recip_local[i], fp8_min, fp8_max
                    )

                for i in T.Parallel(block_m):
                    if pid_m * block_m + i < rows:
                        scale_tensor[pid_m * block_m + i] = scale_local[i]
                for i, j in T.Parallel(block_m, index_dim):
                    if pid_m * block_m + i < rows:
                        output_tensor[pid_m * block_m + i, j] = output_local[i, j]

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


@torch.library.custom_op("tileops::fp8_quant_wrapped_kernel", mutates_args=())
def _fp8_quant_wrapped_kernel(
    batch: int,
    seq_len_kv: int,
    kv_group: int,
    index_dim: int,
    in_dtype: str,
    threads: int,
    block_m: int,
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = batch * seq_len_kv * kv_group
    scale, quant = _fp8_quant_kernel(rows, index_dim, in_dtype, threads)(block_m)(
        input_tensor.view(rows, index_dim)
    )
    return scale.view(batch, seq_len_kv, kv_group), quant.view(
        batch, seq_len_kv, kv_group, index_dim
    )


@_fp8_quant_wrapped_kernel.register_fake
def _(batch, seq_len_kv, kv_group, index_dim, in_dtype, threads, block_m, *inputs):
    return torch.empty(
        (batch, seq_len_kv, kv_group), dtype=torch.float32, device=inputs[0].device
    ), torch.empty(
        (batch, seq_len_kv, kv_group, index_dim), dtype=torch.float8_e4m3fn, device=inputs[0].device
    )


class FP8QuantKernel(Kernel):
    """Per-group fp8 quantization of a $[B \\times S\\_kv \\times G \\times D]$ index tensor.

    A block owns whole rows, taken along the flattened $[B \\times S\\_kv \\times G]$ row
    axis so that its rows are adjacent in memory, and reduces each of them across the
    threads holding it, or inside one thread where the row width leaves the reduction no
    power-of-two group of threads. Any positive ``block_m`` serves any row count.

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
        ValueError: ``block_m`` is not positive, raised where the kernel is built.
    """

    supported_archs: list[int] = [90]

    # This kernel's launch, not the device's. A block reducing a row across its threads
    # runs _LANE_THREADS of them; a block reducing a row inside one thread runs _ROW_THREADS.
    _LANE_THREADS: ClassVar[int] = 128
    _ROW_THREADS: ClassVar[int] = 64

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
        self._threads, self._block_m = self._launch()
        self.kernel = _fp8_quant_kernel(
            batch * seq_len_kv * kv_group, self.index_dim, self.dtype_str, self._threads
        )
        self.init_config(config, tune)

    def _launch(self) -> tuple[int, int]:
        """Threads a block runs, and rows it owns.

        A row width that is a power of two times an odd number above one cannot both give
        each thread a whole number of vector accesses and leave a power-of-two group of
        threads on each row, which is what the cross-lane reduction takes. Such a row goes
        to one thread whole, so the reduction stays in registers and no group exists:
        ``index_dim=96`` float16 measures 2.783 us that way against 7.264 with the row
        split across threads. Every other width keeps the split, faster where the two
        demands agree: 2.016 us against 2.369 at ``index_dim=64``.
        """
        rows = self.batch * self.seq_len_kv * self.kv_group
        odd = self.index_dim
        while odd % 2 == 0:
            odd //= 2
        if odd > 1:
            return self._ROW_THREADS, min(self._ROW_THREADS, _pow2_floor(rows))
        row_bytes = self.index_dim * self.dtype.itemsize
        widest = self._LANE_THREADS * VECTOR_ACCESS_BYTES // row_bytes
        return self._LANE_THREADS, _pow2_floor(min(widest, rows))

    @property
    def default_config(self) -> dict:
        return {"block_m": self._block_m}

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
            self._threads,
            self.config["block_m"],
            input_tensor,
        )
