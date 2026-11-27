import dataclasses
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._kernels.quantization.group_quant import (
    LAUNCH_CONFIG,
    exponent_scale,
    float32_constant,
    launch_regime,
    launch_shape,
    pow2_ceil_exponent,
    register_refusal,
    store_groups,
    write_in_place,
)

__all__ = ["FP4QuantCall", "FP4QuantFwdInterface", "FP4QuantKernel"]


@dataclasses.dataclass(frozen=True)
class FP4QuantCall(CallSpec):
    """One per-block e2m1 quantization of a ``[*S, n]`` tensor, blocks of ``block_size``
    columns.

    The leading axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    n: int = 0
    block_size: int = 32
    dtype: torch.dtype = torch.bfloat16
    inplace: bool = False


class FP4QuantFwdInterface(KernelInterface):
    """Block-wise quantization to e2m1 under power-of-two scales."""

    request = FP4QuantCall

    @abstractmethod
    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Scale each block by the power of two its maximum rounds up to, and round to e2m1.

        A block is ``call.block_size`` adjacent columns of the last axis. Its scale is
        ``2 ** ceil(log2(m * r))``, ``m`` the block's absolute maximum floored at
        ``6 * 2 ** -126`` and ``r`` the float32 nearest ``1 / 6``; each element's code is
        ``x / scale`` clamped to ``[-6, 6]`` and rounded to the nearest e2m1 value, ties to
        the even code, its sign kept for a zero. A block holding an infinity or a NaN has no
        defined result. Nothing but *input_tensor* under ``call.inplace`` is written in
        place, and no output aliases the input.

        Args:
            input_tensor: ``[*S, call.n]`` in ``call.dtype`` on ``call.device``, contiguous;
                with ``call.inplace``, of any layout.

        Returns:
            ``(q, scale)`` on ``call.device``: new contiguous ``uint8`` ``[*S, call.n // 2]``
            codes, two a byte with the even column in the low nibble, and new contiguous
            ``float8_e8m0fnu`` ``[*S, call.n // call.block_size]`` scales. With
            ``call.inplace``, *input_tensor* is overwritten with each element's e2m1 value
            times its scale and ``(None, None)`` returns.
        """


def _e2m1_step(bits):
    """The biased float32 exponent of the e2m1 spacing at the magnitude whose float32 bits are
    *bits*: 0.5 below 2, and half the binade at and above it."""
    return T.max((bits >> T.uint32(23)) & T.uint32(0xFF), T.uint32(127)) - T.uint32(1)


def _e2m1_code(y):
    """The e2m1 code of a float32 *y* in ``[-6, 6]``: its magnitude rounded half to even to
    the e2m1 values, and its sign in bit 3.

    The magnitude over its spacing is exact, as the spacing is a power of two; adding
    ``2 ** 23`` rounds it half to even to ``k``, which the sum's low mantissa bits hold. An
    even ``k`` is an even code, as a magnitude ``k`` spacings of biased exponent ``step`` has
    code ``k + 2 * (step - 126)``.
    """
    rounder = T.float32(2.0**23)
    bits = T.reinterpret(y, "uint32")
    step = _e2m1_step(bits)
    spacings = T.reinterpret(bits & T.uint32(0x7FFFFFFF), "float32") * exponent_scale(
        T.uint32(254) - step
    )
    k = T.reinterpret(T.ieee_add(spacings, rounder), "uint32") & T.uint32(7)
    return (k + T.uint32(2) * (step - T.uint32(126))) | ((bits >> T.uint32(31)) << T.uint32(3))


def _e2m1_value(y):
    """A float32 *y* in ``[-6, 6]`` rounded half to even to the nearest e2m1 value, its sign
    kept for a zero.

    Adding ``2 ** 23`` spacings rounds the magnitude to a whole number of spacings, and
    subtracting them back is exact; ``ieee_add`` and ``ieee_sub`` keep any compiler from
    folding the pair.
    """
    bits = T.reinterpret(y, "uint32")
    rounder = exponent_scale(_e2m1_step(bits) + T.uint32(23))
    magnitude = T.reinterpret(bits & T.uint32(0x7FFFFFFF), "float32")
    rounded = T.ieee_sub(T.ieee_add(magnitude, rounder), rounder)
    return T.reinterpret(
        T.reinterpret(rounded, "uint32") | (bits & T.uint32(0x80000000)), "float32"
    )


def _load_blocks(block_m: int, block_size: int, vector: int):
    """A macro loading the blocks at column ``col`` of rows ``first`` onward of ``rows`` into a
    ``[block_m, block_size // vector, vector]`` fragment, one thread's run on its last axis.

    Only a block of rows reaching past ``rows`` bounds its loads, re-reading the last row in
    place of the missing ones."""
    runs = block_size // vector

    @T.macro
    def load_blocks(input_tensor, first, rows, col, input_local):
        if first + block_m <= rows:
            for i, j, v in T.Parallel(block_m, runs, vector):
                input_local[i, j, v] = input_tensor[first + i, col + j * vector + v]
        else:
            for i, j, v in T.Parallel(block_m, runs, vector):
                input_local[i, j, v] = input_tensor[
                    T.min(first + i, rows - 1), col + j * vector + v
                ]

    return load_blocks


def _block_scales(block_m: int, block_size: int, vector: int):
    """A macro writing each loaded block's scale exponent and the reciprocal of its scale.

    A run is reduced in the thread holding it, then across the threads holding the block.
    The scale is at least the block maximum times the float32 nearest ``1 / 6``, as
    ``fp4_act_quant`` multiplies rather than divides, so a bfloat16 element times the
    reciprocal lies in ``[-6, 6]`` and needs no clamp.
    """
    runs = block_size // vector
    e2m1_max = 6.0
    amax_floor = e2m1_max * 2.0**-126
    inv_e2m1_max = float32_constant(1.0 / e2m1_max)

    @T.macro
    def block_scales(input_local, exponent_local, recip_local):
        run_max = T.alloc_fragment((block_m, runs), "float32")
        amax_local = T.alloc_fragment((block_m,), "float32")
        for i, j in T.Parallel(block_m, runs):
            run_max[i, j] = 0.0
            for v in T.unroll(vector):
                run_max[i, j] = T.max(run_max[i, j], T.abs(T.cast(input_local[i, j, v], "float32")))
        T.reduce_max(run_max, amax_local, dim=1)
        for i in T.Parallel(block_m):
            exponent_local[i] = pow2_ceil_exponent(T.max(amax_local[i], amax_floor) * inv_e2m1_max)
            recip_local[i] = exponent_scale(T.uint32(254) - exponent_local[i])

    return block_scales


# The word a run of 8, 4 or 2 four-bit codes packs into; the program stores it and forward
# views the codes as it.
_PACKED_DTYPES = {8: "uint32", 4: "uint16", 2: "uint8"}


@target_cache(maxsize=32)
def _fp4_quant_kernel(block_size: int, in_dtype: str, threads: int, vector: int):
    @tilelang.jit
    def _fp4_quant_fwd_func(block_m):
        runs = block_size // vector
        packed_dtype = _PACKED_DTYPES[vector]
        load_blocks = _load_blocks(block_m, block_size, vector)
        block_scales = _block_scales(block_m, block_size, vector)
        # An e8m0 scale is its biased exponent.
        store = store_groups(block_m, runs, lambda exponent, i: T.cast(exponent[i], "uint8"))
        rows = T.dynamic("rows")

        @T.prim_func
        def _fp4_quant_fwd_main(
            input_tensor: T.Tensor[(rows, block_size), in_dtype],
            scale_tensor: T.Tensor[(rows,), T.uint8],
            output_tensor: T.Tensor[(rows, runs), packed_dtype],
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as pid_m:
                input_local = T.alloc_fragment((block_m, runs, vector), in_dtype)
                output_local = T.alloc_fragment((block_m, runs), packed_dtype)
                exponent_local = T.alloc_fragment((block_m,), "uint32")
                recip_local = T.alloc_fragment((block_m,), "float32")

                first = pid_m * block_m
                load_blocks(input_tensor, first, rows, 0, input_local)
                block_scales(input_local, exponent_local, recip_local)
                for i, j in T.Parallel(block_m, runs):
                    packed = T.alloc_var("uint32")
                    packed = T.uint32(0)
                    for v in T.unroll(vector):
                        y = T.cast(input_local[i, j, v], "float32") * recip_local[i]
                        packed = packed + (_e2m1_code(y) << T.cast(4 * v, "uint32"))
                    output_local[i, j] = T.cast(packed, packed_dtype)

                store(scale_tensor, output_tensor, first, rows, exponent_local, output_local)

        return _fp4_quant_fwd_main

    return _fp4_quant_fwd_func


@target_cache(maxsize=32)
def _fp4_fake_quant_kernel(
    block_size: int, in_dtype: str, threads: int, vector: int, row_blocks: int, row_stride: int
):
    @tilelang.jit
    def _fp4_fake_quant_fwd_func(block_m):
        runs = block_size // vector
        load_blocks = _load_blocks(block_m, block_size, vector)
        block_scales = _block_scales(block_m, block_size, vector)
        width = row_blocks * block_size
        rows = T.dynamic("rows")

        @T.prim_func
        def _fp4_fake_quant_fwd_main(
            input_tensor: T.StridedTensor[(rows, width), (row_stride, 1), in_dtype],
        ):
            with T.Kernel(T.ceildiv(rows, block_m), row_blocks, threads=threads) as (
                pid_m,
                pid_n,
            ):
                input_local = T.alloc_fragment((block_m, runs, vector), in_dtype)
                output_local = T.alloc_fragment((block_m, runs, vector), in_dtype)
                exponent_local = T.alloc_fragment((block_m,), "uint32")
                recip_local = T.alloc_fragment((block_m,), "float32")

                first = pid_m * block_m
                col = pid_n * block_size
                load_blocks(input_tensor, first, rows, col, input_local)
                block_scales(input_local, exponent_local, recip_local)
                for i, j, v in T.Parallel(block_m, runs, vector):
                    y = T.cast(input_local[i, j, v], "float32") * recip_local[i]
                    output_local[i, j, v] = T.cast(
                        _e2m1_value(y) * exponent_scale(exponent_local[i]), in_dtype
                    )

                if first + block_m <= rows:
                    for i, j, v in T.Parallel(block_m, runs, vector):
                        input_tensor[first + i, col + j * vector + v] = output_local[i, j, v]
                else:
                    for i, j, v in T.Parallel(block_m, runs, vector):
                        if first + i < rows:
                            input_tensor[first + i, col + j * vector + v] = output_local[i, j, v]

        return _fp4_fake_quant_fwd_main

    return _fp4_fake_quant_fwd_func


class FP4QuantKernel(Kernel, FP4QuantFwdInterface):
    """Per-block e2m1 quantization; one program per regime serves any number of blocks.
    Rounding to e2m1 is integer arithmetic on the float32 bits, so it needs no FP4
    conversion instruction.

    Args:
        n: Columns of the last axis.
        block_size: Elements per block; divides ``n``.
        dtype: Torch dtype of the tensor being quantized.
        inplace: Overwrite the input with its fake-quantized values.
        config: Optional dict with the threads and 16-byte vectors a thread of the lane
            layout runs and loads in each launch regime, ``LAUNCH_CONFIG``'s keys.
    """

    supported_archs: list[int] = [89, 90]

    @classmethod
    def refusal(cls, call: FP4QuantCall) -> str | None:
        if call.block_size % 2 and not call.inplace:
            return (
                f"does not quantize block_size={call.block_size}: a byte would hold the codes "
                "of two blocks"
            )
        reason = register_refusal("block_size", call.block_size, call.dtype.itemsize)
        if reason is not None:
            return reason
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FP4QuantCall) -> Entry:
        return call, lambda: cls(call.n, call.block_size, call.dtype, call.inplace)

    def __init__(
        self,
        n: int,
        block_size: int,
        dtype: torch.dtype,
        inplace: bool = False,
        config: dict | None = None,
    ):
        super().__init__()
        self.n = n
        self.block_size = block_size
        self.dtype = dtype
        self.inplace = inplace
        # Elements of the run a thread holds: one 16-byte vector, or the widest power of two
        # dividing the block.
        self.vector = math.gcd(block_size, VECTOR_ACCESS_BYTES // dtype.itemsize)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return dict(LAUNCH_CONFIG)

    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        regime = launch_regime(input_tensor)
        if self.inplace:
            write_in_place(
                input_tensor,
                self.n,
                self.block_size,
                lambda rows, row_blocks, row_stride: self.program(
                    self._compile_fake, regime, row_blocks, row_stride
                )(rows),
            )
            return None, None
        blocks = input_tensor.numel() // self.block_size
        q = torch.empty(
            (*input_tensor.shape[:-1], self.n // 2), dtype=torch.uint8, device=input_tensor.device
        )
        scale = torch.empty(
            (*input_tensor.shape[:-1], self.n // self.block_size),
            dtype=torch.float8_e8m0fnu,
            device=input_tensor.device,
        )
        self.program(self._compile, regime)(
            input_tensor.view(blocks, self.block_size),
            scale.view(torch.uint8).view(blocks),
            q.view(blocks, self.block_size // 2).view(getattr(torch, _PACKED_DTYPES[self.vector])),
        )
        return q, scale

    def _compile(self, cfg: dict, regime: str):
        """The program of config *cfg* for a launch in *regime*."""
        threads, block_m = launch_shape(cfg, regime, self.block_size, self.dtype.itemsize)
        return _fp4_quant_kernel(self.block_size, self.dtype_str, threads, self.vector)(block_m)

    def _compile_fake(self, cfg: dict, regime: str, row_blocks: int, row_stride: int):
        """The in-place program of config *cfg* in *regime* over rows of *row_blocks* blocks,
        *row_stride* elements apart."""
        threads, block_m = launch_shape(cfg, regime, self.block_size, self.dtype.itemsize)
        return _fp4_fake_quant_kernel(
            self.block_size, self.dtype_str, threads, self.vector, row_blocks, row_stride
        )(block_m)
