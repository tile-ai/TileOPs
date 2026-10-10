import dataclasses
import functools
from abc import abstractmethod
from typing import ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.constants import FP8_E4M3_MAX, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["FP8QuantCall", "FP8QuantFwdInterface", "FP8QuantKernel"]


@dataclasses.dataclass(frozen=True)
class FP8QuantCall(CallSpec):
    """One per-row FP8 quantization of a $[B \\times S \\times G \\times D]$ index tensor."""

    batch: int = 0
    seq_len_kv: int = 0
    kv_group: int = 0
    index_dim: int = 0
    dtype: Optional[torch.dtype] = None


class FP8QuantFwdInterface(KernelInterface):
    """Row-wise quantization of an index tensor to ``float8_e4m3fn``."""

    request = FP8QuantCall

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale each row by its own maximum; nothing is written in place.

        A row is the trailing ``call.index_dim`` axis. Its scale is the absolute maximum
        over the row, floored at ``1e-4``, divided by 448; each element is multiplied by
        the reciprocal of that scale and clamped to the ``float8_e4m3fn`` range. A row
        holding an infinity or a NaN has no defined result: whether the maximum and the
        clamp carry the non-finite value is left to the implementation. Nothing is written
        in place, and neither output aliases the input.

        Args:
            input_tensor: The input,
                ``(call.batch, call.seq_len_kv, call.kv_group, call.index_dim)`` in
                ``call.dtype`` on ``call.device``, contiguous in that axis order. The op
                makes it contiguous before the call.

        Returns:
            New contiguous ``(scale_tensor, output_tensor)`` on ``call.device``:
            ``float32`` ``(call.batch, call.seq_len_kv, call.kv_group)`` scales, and the
            rows divided by them in ``float8_e4m3fn``, shaped as the input.
        """


# workloads/quantization/fp8_quant.py clamps a row's absolute maximum to this before dividing.
_AMAX_FLOOR = 1e-4


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


class FP8QuantKernel(Kernel, FP8QuantFwdInterface):
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

    Raises:
        ValueError: ``block_m`` is not positive, raised where the kernel is built.
    """

    @staticmethod
    def _fp8_quant_run(
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

    supported_archs: list[int] = [89, 90]

    # This kernel's launch, not the device's. A block reducing a row across its threads
    # runs _LANE_THREADS of them; a block reducing a row inside one thread runs _ROW_THREADS.
    _LANE_THREADS: ClassVar[int] = 128
    _ROW_THREADS: ClassVar[int] = 64

    @staticmethod
    def _pow2_floor(n: int) -> int:
        """The largest power of two at most *n*, and at least one."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @classmethod
    def entry_for(cls, call: FP8QuantCall) -> Entry:
        return call, lambda: cls(
            call.batch, call.seq_len_kv, call.kv_group, call.index_dim, call.dtype
        )

    def __init__(
        self,
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
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
        self.init_config(config)

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
            return self._ROW_THREADS, min(self._ROW_THREADS, FP8QuantKernel._pow2_floor(rows))
        row_bytes = self.index_dim * self.dtype.itemsize
        widest = self._LANE_THREADS * VECTOR_ACCESS_BYTES // row_bytes
        return self._LANE_THREADS, FP8QuantKernel._pow2_floor(min(widest, rows))

    @property
    def default_config(self) -> dict:
        return {"block_m": self._block_m}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, input_tensor: torch.Tensor):
        return self._fp8_quant_run(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            self.dtype_str,
            self._threads,
            self.config["block_m"],
            input_tensor,
        )
