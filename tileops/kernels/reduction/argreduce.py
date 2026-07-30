"""Streaming arg-reduction kernels (argmax and argmin) using TileLang.

The implementation follows the same broad decomposition as the CUDA
reduction kernels used by PyTorch:

* values and indices are reduced together in one pass;
* input elements are streamed directly from global memory (there is no
  materialized input tile);
* contiguous reduction axes use input-parallel warp/CTA kernels, while
  strided reduction axes use an output-parallel kernel;
* every thread maintains four independent value/index accumulators;
* long rows are reduced register -> warp shuffle -> shared memory -> global;
* launch geometry is selected from the reduction and output sizes.

The Op layer passes a flattened contiguous input plus ``inner_stride``.
For a contiguous tensor reduced along dimension ``d``, ``inner_stride`` is
the product of the dimensions after ``d``.  This lets the kernel address a
non-last reduction axis directly instead of materializing ``movedim(...).
contiguous()``.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["ArgreduceKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}
_WARP_SIZE = 32
_NUM_ACCUMULATORS = 4


def _lanes_per_row(n: int) -> int:
    """Return a power-of-two subgroup size for an input-parallel row."""
    lanes = 1
    while lanes < min(n, _WARP_SIZE):
        lanes *= 2
    return lanes


@functools.lru_cache(maxsize=64)
def _argreduce_warp_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build the input-parallel kernel used for the common contiguous case."""
    lanes = _lanes_per_row(N)
    num_accumulators = _NUM_ACCUMULATORS
    items_per_iteration = lanes * num_accumulators
    iterations = (N + items_per_iteration - 1) // items_per_iteration
    log_lanes = lanes.bit_length() - 1

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        @T.prim_func
        def main(
            x: T.Tensor[(M * N,), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid:
                tx = T.get_thread_binding()
                row_in_block = tx // lanes
                lane = tx % lanes
                row = pid * block_m + row_in_block

                best_values = T.alloc_local((num_accumulators,), "float32")
                best_indices = T.alloc_local((num_accumulators,), "int32")

                for accumulator in T.serial(num_accumulators):
                    if op_kind == "argmax":
                        best_values[accumulator] = -T.infinity("float32")
                    else:
                        best_values[accumulator] = T.infinity("float32")
                    best_indices[accumulator] = T.int32(N)

                # Four independent dependency chains hide comparison latency.
                for iteration in T.serial(iterations):
                    for accumulator in T.serial(num_accumulators):
                        index = (
                            iteration * items_per_iteration
                            + accumulator * lanes
                            + lane
                        )
                        if row < M and index < N:
                            value = T.cast(x[row * N + index], "float32")
                            value_nan = T.isnan(value)
                            best_nan = T.isnan(best_values[accumulator])
                            if op_kind == "argmax":
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value > best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            else:
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value < best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            if better:
                                best_values[accumulator] = value
                                best_indices[accumulator] = T.cast(index, "int32")

                # Merge the four thread-local pairs.
                best_value = T.alloc_var(T.float32)
                best_index = T.alloc_var(T.int32)
                best_value = best_values[0]
                best_index = best_indices[0]
                for accumulator in T.serial(1, num_accumulators):
                    other_value = best_values[accumulator]
                    other_index = best_indices[accumulator]
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                # Pair reduction within each power-of-two subgroup.
                for stage in T.serial(log_lanes):
                    mask = T.int32(lanes // 2) >> stage
                    other_value = T.shfl_xor(best_value, mask, width=lanes)
                    other_index = T.shfl_xor(best_index, mask, width=lanes)
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                if lane == 0 and row < M:
                    out[row] = T.cast(best_index, "int64")

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _argreduce_output_kernel(
    M: int,
    N: int,
    inner_stride: int,
    op_kind: str,
    dtype: str,
):
    """Build an output-parallel kernel for contiguous non-last-axis reductions."""
    num_accumulators = _NUM_ACCUMULATORS
    iterations = (N + num_accumulators - 1) // num_accumulators

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        @T.prim_func
        def main(
            x: T.Tensor[(M * N,), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * block_m + tx
                outer = row // inner_stride
                inner = row % inner_stride

                best_values = T.alloc_local((num_accumulators,), "float32")
                best_indices = T.alloc_local((num_accumulators,), "int32")
                for accumulator in T.serial(num_accumulators):
                    if op_kind == "argmax":
                        best_values[accumulator] = -T.infinity("float32")
                    else:
                        best_values[accumulator] = T.infinity("float32")
                    best_indices[accumulator] = T.int32(N)

                # Threads expand along output, so each iteration performs
                # coalesced reads across adjacent output elements.
                for iteration in T.serial(iterations):
                    for accumulator in T.serial(num_accumulators):
                        index = iteration * num_accumulators + accumulator
                        if row < M and index < N:
                            offset = (
                                outer * N * inner_stride
                                + index * inner_stride
                                + inner
                            )
                            value = T.cast(x[offset], "float32")
                            value_nan = T.isnan(value)
                            best_nan = T.isnan(best_values[accumulator])
                            if op_kind == "argmax":
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value > best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            else:
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value < best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            if better:
                                best_values[accumulator] = value
                                best_indices[accumulator] = T.cast(index, "int32")

                best_value = T.alloc_var(T.float32)
                best_index = T.alloc_var(T.int32)
                best_value = best_values[0]
                best_index = best_indices[0]
                for accumulator in T.serial(1, num_accumulators):
                    other_value = best_values[accumulator]
                    other_index = best_indices[accumulator]
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                if row < M:
                    out[row] = T.cast(best_index, "int64")

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _argreduce_cta_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a block-per-row hierarchical kernel for long contiguous rows."""
    num_accumulators = _NUM_ACCUMULATORS

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        num_warps = threads // _WARP_SIZE
        iterations = (N + threads * num_accumulators - 1) // (
            threads * num_accumulators
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M * N,), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                lane = tx % _WARP_SIZE
                warp = tx // _WARP_SIZE
                warp_values = T.alloc_shared((num_warps,), "float32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                best_values = T.alloc_local((num_accumulators,), "float32")
                best_indices = T.alloc_local((num_accumulators,), "int32")

                for accumulator in T.serial(num_accumulators):
                    if op_kind == "argmax":
                        best_values[accumulator] = -T.infinity("float32")
                    else:
                        best_values[accumulator] = T.infinity("float32")
                    best_indices[accumulator] = T.int32(N)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(num_accumulators):
                        index = (
                            iteration * threads * num_accumulators
                            + accumulator * threads
                            + tx
                        )
                        if index < N:
                            value = T.cast(x[row * N + index], "float32")
                            value_nan = T.isnan(value)
                            best_nan = T.isnan(best_values[accumulator])
                            if op_kind == "argmax":
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value > best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            else:
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value < best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            if better:
                                best_values[accumulator] = value
                                best_indices[accumulator] = T.cast(index, "int32")

                best_value = T.alloc_var(T.float32)
                best_index = T.alloc_var(T.int32)
                best_value = best_values[0]
                best_index = best_indices[0]
                for accumulator in T.serial(1, num_accumulators):
                    other_value = best_values[accumulator]
                    other_index = best_indices[accumulator]
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                # register -> warp shuffle
                for stage in T.serial(5):
                    mask = T.int32(16) >> stage
                    other_value = T.shfl_xor(best_value, mask)
                    other_index = T.shfl_xor(best_index, mask)
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                # warp -> shared
                if lane == 0:
                    warp_values[warp] = best_value
                    warp_indices[warp] = best_index
                T.sync_threads()

                # shared -> final warp -> global
                if lane < num_warps:
                    best_value = warp_values[lane]
                    best_index = warp_indices[lane]
                else:
                    if op_kind == "argmax":
                        best_value = -T.infinity("float32")
                    else:
                        best_value = T.infinity("float32")
                    best_index = T.int32(N)

                if warp == 0:
                    for stage in T.serial(5):
                        mask = T.int32(16) >> stage
                        other_value = T.shfl_xor(best_value, mask)
                        other_index = T.shfl_xor(best_index, mask)
                        other_nan = T.isnan(other_value)
                        best_nan = T.isnan(best_value)
                        if op_kind == "argmax":
                            better = (
                                (other_nan and not best_nan)
                                or (
                                    other_nan == best_nan
                                    and (
                                        other_value > best_value
                                        or (
                                            other_value == best_value
                                            and other_index < best_index
                                        )
                                    )
                                )
                            )
                        else:
                            better = (
                                (other_nan and not best_nan)
                                or (
                                    other_nan == best_nan
                                    and (
                                        other_value < best_value
                                        or (
                                            other_value == best_value
                                            and other_index < best_index
                                        )
                                    )
                                )
                            )
                        if better:
                            best_value = other_value
                            best_index = other_index
                    if lane == 0:
                        out[row] = T.cast(best_index, "int64")

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _argreduce_multicta_partial_kernel(
    M: int,
    N: int,
    op_kind: str,
    dtype: str,
    ctas_per_row: int,
):
    """Build stage one of the multi-CTA long-row reduction."""
    num_accumulators = _NUM_ACCUMULATORS
    chunk_size = (N + ctas_per_row - 1) // ctas_per_row
    num_partials = M * ctas_per_row

    @tilelang.jit(out_idx=[1, 2])
    def _func(threads: int):
        num_warps = threads // _WARP_SIZE
        iterations = (
            chunk_size + threads * num_accumulators - 1
        ) // (threads * num_accumulators)

        @T.prim_func
        def main(
            x: T.Tensor[(M * N,), dtype],
            partial_values: T.Tensor[(num_partials,), "float32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
        ):
            with T.Kernel(num_partials, threads=threads) as partial:
                tx = T.get_thread_binding()
                lane = tx % _WARP_SIZE
                warp = tx // _WARP_SIZE
                row = partial // ctas_per_row
                split = partial % ctas_per_row
                chunk_start = split * chunk_size
                chunk_end = T.min(chunk_start + chunk_size, N)

                warp_values = T.alloc_shared((num_warps,), "float32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                best_values = T.alloc_local((num_accumulators,), "float32")
                best_indices = T.alloc_local((num_accumulators,), "int32")

                for accumulator in T.serial(num_accumulators):
                    if op_kind == "argmax":
                        best_values[accumulator] = -T.infinity("float32")
                    else:
                        best_values[accumulator] = T.infinity("float32")
                    best_indices[accumulator] = T.int32(N)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(num_accumulators):
                        index = (
                            chunk_start
                            + iteration * threads * num_accumulators
                            + accumulator * threads
                            + tx
                        )
                        if index < chunk_end:
                            value = T.cast(x[row * N + index], "float32")
                            value_nan = T.isnan(value)
                            best_nan = T.isnan(best_values[accumulator])
                            if op_kind == "argmax":
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value > best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            else:
                                better = (
                                    (value_nan and not best_nan)
                                    or (
                                        value_nan == best_nan
                                        and (
                                            value < best_values[accumulator]
                                            or (
                                                value == best_values[accumulator]
                                                and index < best_indices[accumulator]
                                            )
                                        )
                                    )
                                )
                            if better:
                                best_values[accumulator] = value
                                best_indices[accumulator] = T.cast(index, "int32")

                best_value = T.alloc_var(T.float32)
                best_index = T.alloc_var(T.int32)
                best_value = best_values[0]
                best_index = best_indices[0]
                for accumulator in T.serial(1, num_accumulators):
                    other_value = best_values[accumulator]
                    other_index = best_indices[accumulator]
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                for stage in T.serial(5):
                    mask = T.int32(16) >> stage
                    other_value = T.shfl_xor(best_value, mask)
                    other_index = T.shfl_xor(best_index, mask)
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                if lane == 0:
                    warp_values[warp] = best_value
                    warp_indices[warp] = best_index
                T.sync_threads()

                if lane < num_warps:
                    best_value = warp_values[lane]
                    best_index = warp_indices[lane]
                else:
                    if op_kind == "argmax":
                        best_value = -T.infinity("float32")
                    else:
                        best_value = T.infinity("float32")
                    best_index = T.int32(N)

                if warp == 0:
                    for stage in T.serial(5):
                        mask = T.int32(16) >> stage
                        other_value = T.shfl_xor(best_value, mask)
                        other_index = T.shfl_xor(best_index, mask)
                        other_nan = T.isnan(other_value)
                        best_nan = T.isnan(best_value)
                        if op_kind == "argmax":
                            better = (
                                (other_nan and not best_nan)
                                or (
                                    other_nan == best_nan
                                    and (
                                        other_value > best_value
                                        or (
                                            other_value == best_value
                                            and other_index < best_index
                                        )
                                    )
                                )
                            )
                        else:
                            better = (
                                (other_nan and not best_nan)
                                or (
                                    other_nan == best_nan
                                    and (
                                        other_value < best_value
                                        or (
                                            other_value == best_value
                                            and other_index < best_index
                                        )
                                    )
                                )
                            )
                        if better:
                            best_value = other_value
                            best_index = other_index
                    if lane == 0:
                        partial_values[partial] = best_value
                        partial_indices[partial] = best_index

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _argreduce_multicta_final_kernel(
    M: int,
    N: int,
    op_kind: str,
    ctas_per_row: int,
):
    """Build stage two, reducing multi-CTA value/index partials per row."""
    num_partials = M * ctas_per_row
    rows_per_block = 8
    threads = rows_per_block * _WARP_SIZE

    @tilelang.jit(out_idx=[2])
    def _func():
        @T.prim_func
        def main(
            partial_values: T.Tensor[(num_partials,), "float32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, rows_per_block), threads=threads) as pid:
                tx = T.get_thread_binding()
                row_in_block = tx // _WARP_SIZE
                lane = tx % _WARP_SIZE
                row = pid * rows_per_block + row_in_block

                best_value = T.alloc_var(T.float32)
                best_index = T.alloc_var(T.int32)
                if op_kind == "argmax":
                    best_value = -T.infinity("float32")
                else:
                    best_value = T.infinity("float32")
                best_index = T.int32(N)

                if row < M and lane < ctas_per_row:
                    best_value = partial_values[row * ctas_per_row + lane]
                    best_index = partial_indices[row * ctas_per_row + lane]

                for stage in T.serial(5):
                    mask = T.int32(16) >> stage
                    other_value = T.shfl_xor(best_value, mask)
                    other_index = T.shfl_xor(best_index, mask)
                    other_nan = T.isnan(other_value)
                    best_nan = T.isnan(best_value)
                    if op_kind == "argmax":
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value > best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    else:
                        better = (
                            (other_nan and not best_nan)
                            or (
                                other_nan == best_nan
                                and (
                                    other_value < best_value
                                    or (
                                        other_value == best_value
                                        and other_index < best_index
                                    )
                                )
                            )
                        )
                    if better:
                        best_value = other_value
                        best_index = other_index

                if row < M and lane == 0:
                    out[row] = T.cast(best_index, "int64")

        return main

    return _func


@torch.library.custom_op("top::argreduce_fwd", mutates_args=())
def _argreduce_fwd_wrapped(
    M: int,
    N: int,
    inner_stride: int,
    op_kind: str,
    dtype_str: str,
    strategy: str,
    ctas_per_row: int,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    if strategy == "output":
        return _argreduce_output_kernel(
            M, N, inner_stride, op_kind, dtype_str
        )(block_m, threads)(x)
    if strategy == "cta":
        return _argreduce_cta_kernel(
            M, N, op_kind, dtype_str
        )(block_m, threads)(x)
    if strategy == "multi_cta":
        partial_values, partial_indices = _argreduce_multicta_partial_kernel(
            M, N, op_kind, dtype_str, ctas_per_row
        )(threads)(x)
        return _argreduce_multicta_final_kernel(
            M, N, op_kind, ctas_per_row
        )()(partial_values, partial_indices)
    return _argreduce_warp_kernel(
        M, N, op_kind, dtype_str
    )(block_m, threads)(x)


@_argreduce_fwd_wrapped.register_fake
def _(
    M,
    N,
    inner_stride,
    op_kind,
    dtype_str,
    strategy,
    ctas_per_row,
    block_m,
    threads,
    x,
):
    return torch.empty((M,), dtype=torch.int64, device=x.device)


class ArgreduceKernel(Kernel):
    """Adaptive streaming argmax/argmin kernel.

    ``inner_stride == 1`` means the reduction axis is contiguous.  A warp
    processes each ordinary row; very long rows use a full CTA and shared
    memory only for the small set of warp partials.  ``inner_stride > 1``
    selects output-parallel traversal, which keeps global accesses coalesced
    without transposing the input.
    """

    supported_archs: list[int] = [80, 86, 89, 90, 100]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        inner_stride: int = 1,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. "
                f"Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        if N <= 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        if inner_stride <= 0 or M % inner_stride != 0:
            raise ValueError(
                f"Invalid inner_stride={inner_stride} for M={M}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.inner_stride = inner_stride
        self.ctas_per_row = 1
        if inner_stride > 1:
            self.strategy = "output"
            self.kernel = _argreduce_output_kernel(
                M, N, inner_stride, op_kind, self.dtype_str
            )
        elif N >= 32768 and M < 64:
            self.strategy = "multi_cta"
            # About one 4K-element chunk per CTA, capped at one warp of
            # partials so the final reduction remains a single shuffle tree.
            self.ctas_per_row = min(32, (N + 4095) // 4096)
            self.kernel = _argreduce_multicta_partial_kernel(
                M, N, op_kind, self.dtype_str, self.ctas_per_row
            )
        elif N >= 4096:
            self.strategy = "cta"
            self.kernel = _argreduce_cta_kernel(M, N, op_kind, self.dtype_str)
        else:
            self.strategy = "warp"
            self.kernel = _argreduce_warp_kernel(M, N, op_kind, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.strategy in {"cta", "multi_cta"}:
            return {"block_m": 1, "threads": 256}
        if self.strategy == "output":
            threads = 256
            return {"block_m": threads, "threads": threads}
        lanes = _lanes_per_row(self.N)
        target_threads = 256 if self.M >= 8 else max(32, self.M * lanes)
        block_m = max(1, target_threads // lanes)
        return {"block_m": block_m, "threads": block_m * lanes}

    @property
    def autotune_configs(self) -> list[dict]:
        if self.strategy in {"cta", "multi_cta"}:
            return [
                {"block_m": 1, "threads": threads}
                for threads in (128, 256, 512)
            ]
        if self.strategy == "output":
            return [
                {"block_m": threads, "threads": threads}
                for threads in (128, 256, 512)
            ]
        lanes = _lanes_per_row(self.N)
        configs = []
        for target_threads in (64, 128, 256, 512):
            block_m = max(1, target_threads // lanes)
            configs.append(
                {"block_m": block_m, "threads": block_m * lanes}
            )
        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selected arg-reduction strategy on a flattened input."""
        return _argreduce_fwd_wrapped(
            self.M,
            self.N,
            self.inner_stride,
            self.op_kind,
            self.dtype_str,
            self.strategy,
            self.ctas_per_row,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
