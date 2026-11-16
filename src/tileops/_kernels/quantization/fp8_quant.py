import dataclasses
from abc import abstractmethod
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import FP8_E4M3_MAX, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["FP8QuantCall", "FP8QuantFwdInterface", "FP8QuantKernel"]


@dataclasses.dataclass(frozen=True)
class FP8QuantCall(CallSpec):
    """One per-group FP8 quantization of a ``[*S, k]`` tensor, groups of ``group_size`` columns.

    The leading axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    k: int = 0
    group_size: int = 0
    eps: float = 1e-10
    dtype: torch.dtype | None = None
    column_major_scales: bool = False
    tma_aligned_scales: bool = False
    use_ue8m0: bool | None = None


class FP8QuantFwdInterface(KernelInterface):
    """Group-wise quantization to ``float8_e4m3fn``."""

    request = FP8QuantCall

    @abstractmethod
    def forward(
        self, input_tensor: torch.Tensor, out_q: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale each group by its own maximum, writing the codes into *out_q* when given.

        A group is ``call.group_size`` adjacent columns of the last axis. Its scale is the
        absolute maximum over the group, floored at ``call.eps``, divided by 448; each element
        is multiplied by the reciprocal of that scale and clamped to the ``float8_e4m3fn``
        range. A group holding an infinity or a NaN, or an all-zero group under ``eps`` 0,
        has no defined result: whether the maximum and the clamp carry the non-finite value
        is left to the implementation. Nothing but *out_q* is written in place, and no output
        aliases the input.

        Args:
            input_tensor: ``[*S, call.k]`` in ``call.dtype`` on ``call.device``, contiguous.
            out_q: ``[*S, call.k]`` ``float8_e4m3fn`` on ``call.device``, contiguous, the
                codes are stored into; or None.

        Returns:
            ``(q, scale)`` on ``call.device``: the codes, *out_q* where given and otherwise a
            new contiguous ``float8_e4m3fn`` tensor shaped as the input, and new contiguous
            ``float32`` ``[*S, call.k // call.group_size]`` scales.
        """


@target_cache(maxsize=32)
def _fp8_quant_kernel(group_size: int, in_dtype: str, threads: int, eps: float):
    @tilelang.jit
    def _fp8_quant_fwd_func(block_m):
        if block_m < 1:
            raise ValueError(f"block_m={block_m} must be positive")
        out_dtype = T.float8_e4m3fn
        scale_dtype = T.float32
        fp8_min = -FP8_E4M3_MAX
        fp8_max = FP8_E4M3_MAX
        amax_floor = eps
        # The group count is a runtime extent, so a new token count builds no program.
        rows = T.dynamic("rows")

        @T.prim_func
        def _fp8_quant_fwd_main(
            input_tensor: T.Tensor[(rows, group_size), in_dtype],
            scale_tensor: T.Tensor[(rows,), scale_dtype],
            output_tensor: T.Tensor[(rows, group_size), out_dtype],
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as pid_m:
                input_local = T.alloc_fragment((block_m, group_size), in_dtype)
                output_local = T.alloc_fragment((block_m, group_size), out_dtype)
                amax_local = T.alloc_fragment((block_m,), scale_dtype)
                scale_local = T.alloc_fragment((block_m,), scale_dtype)
                recip_local = T.alloc_fragment((block_m,), scale_dtype)

                first = pid_m * block_m
                # Only the last block can reach past the axis; every other block loads and
                # stores without the bound, which a runtime row count would otherwise cost
                # each access.
                if first + block_m <= rows:
                    for i, j in T.Parallel(block_m, group_size):
                        input_local[i, j] = input_tensor[first + i, j]
                else:
                    # The last block re-reads its last group; the stores drop those groups.
                    for i, j in T.Parallel(block_m, group_size):
                        input_local[i, j] = input_tensor[T.min(first + i, rows - 1), j]

                T.reduce_absmax(input_local, amax_local, dim=1)
                for i in T.Parallel(block_m):
                    amax_local[i] = T.max(amax_local[i], amax_floor)
                    scale_local[i] = amax_local[i] / fp8_max
                    recip_local[i] = 1.0 / scale_local[i]

                # Not the reference's divide by the scale; ``FP8QuantPerTokenGroupFwdOp`` bounds the gap.
                for i, j in T.Parallel(block_m, group_size):
                    output_local[i, j] = T.clamp(
                        input_local[i, j] * recip_local[i], fp8_min, fp8_max
                    )

                if first + block_m <= rows:
                    for i in T.Parallel(block_m):
                        scale_tensor[first + i] = scale_local[i]
                    for i, j in T.Parallel(block_m, group_size):
                        output_tensor[first + i, j] = output_local[i, j]
                else:
                    for i in T.Parallel(block_m):
                        if first + i < rows:
                            scale_tensor[first + i] = scale_local[i]
                    for i, j in T.Parallel(block_m, group_size):
                        if first + i < rows:
                            output_tensor[first + i, j] = output_local[i, j]

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


class FP8QuantKernel(Kernel, FP8QuantFwdInterface):
    """Per-group fp8 quantization; one program per regime serves any number of groups.

    A block owns whole groups, adjacent in memory, and reduces each of them across the
    threads holding it, or inside one thread where the group width leaves the reduction no
    power-of-two group of threads. Any positive ``block_m`` serves any group count.

    Args:
        k: Columns of the last axis.
        group_size: Elements per group; divides ``k``.
        eps: Floor of each group's absolute maximum.
        dtype: Torch dtype of the tensor being quantized. The output element
            type is fixed at ``torch.float8_e4m3fn`` and the scale at
            ``torch.float32``, so this is the kernel's only free dtype.
        config: Optional dict with the threads and 16-byte vectors a thread of the lane
            layout runs and loads, for a launch below and at least ``_WIDE_BYTES``:
            ``narrow_threads``, ``narrow_vectors``, ``wide_threads``, ``wide_vectors``.
    """

    supported_archs: list[int] = [89, 90]

    # The launch picks its program from the input size, so tuning has no candidate to time.
    autotune_configs = None

    # A block reducing a group inside one thread runs this many threads, one group each.
    _ROW_THREADS: ClassVar[int] = 64
    # An input of at least this many bytes is bound by DRAM throughput, which two vectors a
    # thread bring closer to peak (dsv3-act-prefill: 24.96 us against 27.09 with one); a
    # smaller one is bound by one load-reduce-store chain, which one vector a thread over
    # 256-thread blocks shortens (kv-index-flat-8k float16: 1.79 us against 1.86 with two).
    _WIDE_BYTES: ClassVar[int] = 8 << 20

    @staticmethod
    def _pow2_floor(n: int) -> int:
        """The largest power of two at most *n*, and at least one."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @classmethod
    def _launch(
        cls, group_size: int, dtype: torch.dtype, threads: int, vectors: int
    ) -> tuple[int, int]:
        """Threads a block runs, and groups it owns, for a lane layout of *threads* loading
        *vectors* 16-byte vectors each.

        A group width that is a power of two times an odd number above one cannot both give
        each thread a whole number of vector accesses and leave a power-of-two group of
        threads on each group, which is what the cross-lane reduction takes. Such a group
        goes to one thread whole, so the reduction stays in registers:
        ``group_size=96`` float16 measures 2.783 us that way against 7.264 with the group
        split across threads.
        """
        odd = group_size
        while odd % 2 == 0:
            odd //= 2
        if odd > 1:
            return cls._ROW_THREADS, cls._ROW_THREADS
        widest = vectors * threads * VECTOR_ACCESS_BYTES // (group_size * dtype.itemsize)
        return threads, cls._pow2_floor(widest)

    @classmethod
    def refusal(cls, call: FP8QuantCall) -> str | None:
        # vLLM reads ``tma_aligned_scales`` only for column-major scales.
        if call.column_major_scales:
            return "does not serve column-major scales"
        if call.use_ue8m0:
            return "does not serve UE8M0 scales"
        if call.dtype is not None:
            # The wide layout gives a thread the most elements: 128 threads, two vectors.
            threads, block_m = cls._launch(call.group_size, call.dtype, 128, 2)
            # A thread holds its share of the block's groups in registers, once in the input
            # dtype and once as codes; past this many elements the fragments spill.
            per_thread = block_m * call.group_size // threads
            if per_thread > 128:
                return (
                    f"does not serve group_size={call.group_size}: {per_thread} elements a "
                    "thread exceed the register budget of 128"
                )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FP8QuantCall) -> Entry:
        return call, lambda: cls(call.k, call.group_size, call.eps, call.dtype)

    def __init__(
        self,
        k: int,
        group_size: int,
        eps: float,
        dtype: torch.dtype,
        config: dict | None = None,
    ):
        super().__init__()
        self.k = k
        self.group_size = group_size
        self.eps = eps
        self.dtype = dtype
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"narrow_threads": 256, "narrow_vectors": 1, "wide_threads": 128, "wide_vectors": 2}

    def forward(
        self, input_tensor: torch.Tensor, out_q: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = input_tensor.numel() // self.group_size
        q = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn) if out_q is None else out_q
        scale = torch.empty(
            (*input_tensor.shape[:-1], self.k // self.group_size),
            dtype=torch.float32,
            device=input_tensor.device,
        )
        wide = input_tensor.numel() * input_tensor.element_size() >= self._WIDE_BYTES
        self.program(self._compile, wide)(
            input_tensor.view(groups, self.group_size),
            scale.view(groups),
            q.view(groups, self.group_size),
        )
        return q, scale

    def _compile(self, cfg: dict, wide: bool):
        """The program of config *cfg* for a launch in the *wide* regime or not."""
        regime = "wide" if wide else "narrow"
        threads, block_m = self._launch(
            self.group_size, self.dtype, cfg[f"{regime}_threads"], cfg[f"{regime}_vectors"]
        )
        return _fp8_quant_kernel(self.group_size, self.dtype_str, threads, self.eps)(block_m)
