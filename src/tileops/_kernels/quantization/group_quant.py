"""Pieces the per-group quantize kernels share: launch geometry, power-of-two scales, and the
rows an in-place call writes.

The power-of-two scales are DeepSeek-V4-Flash ``inference/kernel.py``'s ``fast_round_scale``,
so a kernel's scales equal its ``act_quant`` and ``fp4_act_quant`` bit for bit.
"""

import struct
from collections.abc import Callable

import tilelang.language as T
import torch

from tileops._kernels.constants import MAX_GRID_YZ_BLOCKS, VECTOR_ACCESS_BYTES

__all__ = [
    "LAUNCH_CONFIG",
    "exponent_scale",
    "float32_constant",
    "launch_regime",
    "launch_shape",
    "pow2_ceil_exponent",
    "register_refusal",
    "store_groups",
    "write_in_place",
]

# The threads and 16-byte vectors a thread of the lane layout runs and loads in each
# launch_regime; the kernels' default config and register_refusal read it.
LAUNCH_CONFIG = {"narrow_threads": 256, "narrow_vectors": 1, "wide_threads": 128, "wide_vectors": 2}


def _pow2_floor(n: int) -> int:
    """The largest power of two at most *n*, and at least one."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _group_launch(group_size: int, itemsize: int, threads: int, vectors: int) -> tuple[int, int]:
    """Threads a block runs, and groups it owns, for a lane layout of *threads* loading
    *vectors* 16-byte vectors each of *itemsize*-byte elements.

    A group width that is a power of two times an odd number above one cannot both give
    each thread a whole number of vector accesses and leave a power-of-two group of
    threads on each group, which is what the cross-lane reduction takes. Such a group
    goes to one thread whole, so the reduction stays in registers.
    """
    odd = group_size
    while odd % 2 == 0:
        odd //= 2
    if odd > 1:
        # A block reducing a group inside one thread runs this many threads, one group each.
        row_threads = 64
        return row_threads, row_threads
    widest = vectors * threads * VECTOR_ACCESS_BYTES // (group_size * itemsize)
    return threads, _pow2_floor(widest)


def float32_constant(value: float):
    """*value* rounded to float32, as a float32 immediate whose literal is that float exactly.

    A reference multiplying a float32 tensor by a Python scalar multiplies by the scalar
    rounded to float32; a literal holding the unrounded double would leave the rounding to
    the CUDA compiler.
    """
    return T.float32(struct.unpack("<f", struct.pack("<f", value))[0])


def pow2_ceil_exponent(value):
    """The biased float32 exponent of ``2 ** ceil(log2(value))`` of a positive float32, as
    uint32: the exponent field, plus one where a mantissa bit is set.

    A subnormal *value* gives 2 ** -126, as DeepSeek-V4-Flash's ``fast_log2_ceil`` does.
    """
    bits = T.reinterpret(value, "uint32")
    exponent = (bits >> T.uint32(23)) & T.uint32(0xFF)
    return exponent + T.if_then_else((bits & T.uint32(0x7FFFFF)) != 0, T.uint32(1), T.uint32(0))


def exponent_scale(biased):
    """The float32 ``2 ** (biased - 127)`` of a uint32 biased exponent in ``[1, 254]``.

    Built from its bits, not divided, so ``exponent_scale(254 - e)`` is the exact reciprocal
    of ``exponent_scale(e)``: a value times it equals the value divided by the scale.
    """
    return T.reinterpret(biased << T.uint32(23), "float32")


def launch_regime(x: torch.Tensor) -> str:
    """The ``LAUNCH_CONFIG`` regime of a launch over *x*: ``"wide"`` or ``"narrow"``."""
    # An input of at least this many bytes is bound by DRAM throughput, which two vectors a
    # thread bring closer to peak; a smaller one is bound by one load-reduce-store chain,
    # which one vector a thread over 256-thread blocks shortens.
    wide_bytes = 8 << 20
    return "wide" if x.numel() * x.element_size() >= wide_bytes else "narrow"


def launch_shape(cfg: dict, regime: str, group_size: int, itemsize: int) -> tuple[int, int]:
    """Threads a block runs, and groups it owns, under config *cfg* in *regime*."""
    return _group_launch(group_size, itemsize, cfg[f"{regime}_threads"], cfg[f"{regime}_vectors"])


def register_refusal(name: str, group_size: int, itemsize: int) -> str | None:
    """Why a group of *group_size* elements is refused, or None.

    A thread holds its share of a block's groups in registers, once loaded and once
    converted; past 128 elements the fragments spill. The default wide layout gives a thread
    the most.
    """
    threads, block_m = launch_shape(LAUNCH_CONFIG, "wide", group_size, itemsize)
    per_thread = block_m * group_size // threads
    if per_thread > 128:
        return (
            f"does not serve {name}={group_size}: {per_thread} elements a thread exceed the "
            "register budget of 128"
        )
    return None


def store_groups(block_m: int, width: int, scale_value: Callable):
    """A macro storing the ``block_m`` groups from row ``first``: ``scale_value(scale_local,
    i)`` to ``scale_tensor[first + i]`` and the ``width`` codes of ``output_local[i]`` to
    ``output_tensor[first + i]``.

    Only a block reaching past ``rows`` bounds its stores, dropping the rows past it.
    """

    @T.macro
    def store(scale_tensor, output_tensor, first, rows, scale_local, output_local):
        if first + block_m <= rows:
            for i in T.Parallel(block_m):
                scale_tensor[first + i] = scale_value(scale_local, i)
            for i, j in T.Parallel(block_m, width):
                output_tensor[first + i, j] = output_local[i, j]
        else:
            for i in T.Parallel(block_m):
                if first + i < rows:
                    scale_tensor[first + i] = scale_value(scale_local, i)
            for i, j in T.Parallel(block_m, width):
                if first + i < rows:
                    output_tensor[first + i, j] = output_local[i, j]

    return store


def write_in_place(
    x: torch.Tensor, k: int, group_size: int, launch: Callable[[torch.Tensor, int, int], None]
) -> None:
    """Run ``launch(rows, row_groups, row_stride)`` over *x*'s groups where they lie;
    ``launch`` puts ``row_groups`` on the grid's y extent.

    *x* is ``[*S, k]``. Rows of unit column stride evenly spaced are written without a copy:
    rows ``k`` apart as one run of groups, ``[groups, group_size]`` with ``row_groups`` 1,
    and the columns of wider rows, the leading 448 of a ``[*S, 512]`` cache row for one, as
    ``[rows, k]`` of ``k // group_size`` groups where that is at most ``MAX_GRID_YZ_BLOCKS``.
    Other layouts, such as a slice along a middle axis, go through a contiguous copy written
    back.
    """
    try:
        rows = x.view(-1, k)
    except RuntimeError:
        rows = None
    if rows is not None and (k <= 1 or rows.stride(1) == 1):
        if rows.is_contiguous():
            launch(rows.view(-1, group_size), 1, group_size)
            return
        if k // group_size <= MAX_GRID_YZ_BLOCKS:
            launch(rows, k // group_size, rows.stride(0))
            return
    dense = x.contiguous()
    launch(dense.view(-1, group_size), 1, group_size)
    x.copy_(dense)
