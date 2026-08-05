"""Streaming argmax and argmin kernels.

Values and indices are reduced together so first-index and NaN semantics are
preserved without materializing an input tile. Launch geometry adapts to the
row length; input layout remains the responsibility of the reduction Op layer.
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
    lanes = 1
    while lanes < min(n, _WARP_SIZE):
        lanes *= 2
    return lanes


def _make_pair_ops(op_kind: str, n: int):
    """Create argreduce-local pair operations shared by all launch paths."""

    @T.macro
    def set_identity(values, indices, slot):
        if op_kind == "argmax":
            values[slot] = -T.infinity("float32")
        else:
            values[slot] = T.infinity("float32")
        indices[slot] = T.int32(n)

    @T.macro
    def init_accumulators(values, indices):
        for accumulator in T.serial(_NUM_ACCUMULATORS):
            set_identity(values, indices, accumulator)

    @T.macro
    def update(values, indices, slot, candidate_value, candidate_index):
        """Merge a candidate into a slot under PyTorch's argreduce ordering.

        NaN outranks every number, equal rank breaks to the lower index, and
        two NaNs are equal in rank — the last of which no float comparison can
        express, since every comparison between them is false.
        """
        candidate_nan = T.isnan(candidate_value)
        current_nan = T.isnan(values[slot])
        both_numeric = not candidate_nan and not current_nan
        if op_kind == "argmax":
            outranks = both_numeric and candidate_value > values[slot]
        else:
            outranks = both_numeric and candidate_value < values[slot]
        same_rank = (candidate_nan and current_nan) or (
            both_numeric and candidate_value == values[slot]
        )
        better = (
            (candidate_nan and not current_nan)
            or outranks
            or (same_rank and candidate_index < indices[slot])
        )
        if better:
            values[slot] = candidate_value
            indices[slot] = T.cast(candidate_index, "int32")

    @T.macro
    def merge_accumulators(values, indices, best_value, best_index):
        best_value[0] = values[0]
        best_index[0] = indices[0]
        for accumulator in T.serial(1, _NUM_ACCUMULATORS):
            update(
                best_value,
                best_index,
                0,
                values[accumulator],
                indices[accumulator],
            )

    @T.macro
    def warp_reduce(best_value, best_index, stages, width):
        for stage in T.serial(stages):
            mask = T.int32(width // 2) >> stage
            candidate_value = T.shfl_xor(best_value[0], mask, width=width)
            candidate_index = T.shfl_xor(best_index[0], mask, width=width)
            update(
                best_value,
                best_index,
                0,
                candidate_value,
                candidate_index,
            )

    return (
        set_identity,
        init_accumulators,
        update,
        merge_accumulators,
        warp_reduce,
    )


def _make_block_reduce(
    set_identity,
    merge_accumulators,
    warp_reduce,
    num_warps: int,
):
    """Create the register-to-block pair reduction used by CTA kernels."""

    @T.macro
    def block_reduce(
        values,
        indices,
        best_value,
        best_index,
        warp_values,
        warp_indices,
        tx,
    ):
        lane = tx % _WARP_SIZE
        warp = tx // _WARP_SIZE

        merge_accumulators(values, indices, best_value, best_index)
        warp_reduce(best_value, best_index, 5, _WARP_SIZE)

        if lane == 0:
            warp_values[warp] = best_value[0]
            warp_indices[warp] = best_index[0]
        T.sync_threads()

        set_identity(best_value, best_index, 0)
        if lane < num_warps:
            best_value[0] = warp_values[lane]
            best_index[0] = warp_indices[lane]

        if warp == 0:
            warp_reduce(best_value, best_index, 5, _WARP_SIZE)

    return block_reduce


@functools.lru_cache(maxsize=64)
def _argreduce_warp_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build the subgroup-per-row kernel used for ordinary row lengths."""
    lanes = _lanes_per_row(N)
    items_per_iteration = lanes * _NUM_ACCUMULATORS
    iterations = (N + items_per_iteration - 1) // items_per_iteration
    log_lanes = lanes.bit_length() - 1

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        (
            _,
            init_accumulators,
            update,
            merge_accumulators,
            warp_reduce,
        ) = _make_pair_ops(op_kind, N)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * block_m + tx // lanes
                lane = tx % lanes

                values = T.alloc_local((_NUM_ACCUMULATORS,), "float32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                best_value = T.alloc_local((1,), "float32")
                best_index = T.alloc_local((1,), "int32")
                init_accumulators(values, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = iteration * items_per_iteration + accumulator * lanes + lane
                        if row < M and index < N:
                            update(
                                values,
                                indices,
                                accumulator,
                                T.cast(x[row, index], "float32"),
                                index,
                            )

                merge_accumulators(values, indices, best_value, best_index)
                warp_reduce(best_value, best_index, log_lanes, lanes)

                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _argreduce_cta_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build the block-per-row kernel used for long rows."""

    @tilelang.jit(out_idx=[1])
    def _func(threads: int):
        num_warps = threads // _WARP_SIZE
        iterations = (N + threads * _NUM_ACCUMULATORS - 1) // (threads * _NUM_ACCUMULATORS)
        (
            set_identity,
            init_accumulators,
            update,
            merge_accumulators,
            warp_reduce,
        ) = _make_pair_ops(op_kind, N)
        block_reduce = _make_block_reduce(
            set_identity,
            merge_accumulators,
            warp_reduce,
            num_warps,
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                warp_values = T.alloc_shared((num_warps,), "float32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                values = T.alloc_local((_NUM_ACCUMULATORS,), "float32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                best_value = T.alloc_local((1,), "float32")
                best_index = T.alloc_local((1,), "int32")
                init_accumulators(values, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = iteration * threads * _NUM_ACCUMULATORS + accumulator * threads + tx
                        if index < N:
                            update(
                                values,
                                indices,
                                accumulator,
                                T.cast(x[row, index], "float32"),
                                index,
                            )

                block_reduce(
                    values,
                    indices,
                    best_value,
                    best_index,
                    warp_values,
                    warp_indices,
                    tx,
                )
                if tx == 0:
                    out[row] = T.cast(best_index[0], "int64")

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
    """Build the partial stage for rows split across multiple blocks."""
    chunk_size = (N + ctas_per_row - 1) // ctas_per_row
    num_partials = M * ctas_per_row

    @tilelang.jit(out_idx=[1, 2])
    def _func(threads: int):
        num_warps = threads // _WARP_SIZE
        iterations = (chunk_size + threads * _NUM_ACCUMULATORS - 1) // (threads * _NUM_ACCUMULATORS)
        (
            set_identity,
            init_accumulators,
            update,
            merge_accumulators,
            warp_reduce,
        ) = _make_pair_ops(op_kind, N)
        block_reduce = _make_block_reduce(
            set_identity,
            merge_accumulators,
            warp_reduce,
            num_warps,
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            partial_values: T.Tensor[(num_partials,), "float32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
        ):
            with T.Kernel(num_partials, threads=threads) as partial:
                tx = T.get_thread_binding()
                row = partial // ctas_per_row
                split = partial % ctas_per_row
                chunk_start = split * chunk_size
                chunk_end = T.min(chunk_start + chunk_size, N)

                warp_values = T.alloc_shared((num_warps,), "float32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                values = T.alloc_local((_NUM_ACCUMULATORS,), "float32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                best_value = T.alloc_local((1,), "float32")
                best_index = T.alloc_local((1,), "int32")
                init_accumulators(values, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = (
                            chunk_start
                            + iteration * threads * _NUM_ACCUMULATORS
                            + accumulator * threads
                            + tx
                        )
                        if index < chunk_end:
                            update(
                                values,
                                indices,
                                accumulator,
                                T.cast(x[row, index], "float32"),
                                index,
                            )

                block_reduce(
                    values,
                    indices,
                    best_value,
                    best_index,
                    warp_values,
                    warp_indices,
                    tx,
                )
                if tx == 0:
                    partial_values[partial] = best_value[0]
                    partial_indices[partial] = best_index[0]

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _argreduce_multicta_final_kernel(
    M: int,
    N: int,
    op_kind: str,
    ctas_per_row: int,
):
    """Build the final reduction over per-row block partials."""
    num_partials = M * ctas_per_row
    rows_per_block = 8
    threads = rows_per_block * _WARP_SIZE

    @tilelang.jit(out_idx=[2])
    def _func():
        set_identity, _, _, _, warp_reduce = _make_pair_ops(op_kind, N)

        @T.prim_func
        def main(
            partial_values: T.Tensor[(num_partials,), "float32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, rows_per_block), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * rows_per_block + tx // _WARP_SIZE
                lane = tx % _WARP_SIZE
                best_value = T.alloc_local((1,), "float32")
                best_index = T.alloc_local((1,), "int32")
                set_identity(best_value, best_index, 0)

                if row < M and lane < ctas_per_row:
                    best_value[0] = partial_values[row * ctas_per_row + lane]
                    best_index[0] = partial_indices[row * ctas_per_row + lane]

                warp_reduce(best_value, best_index, 5, _WARP_SIZE)
                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main

    return _func


@torch.library.custom_op("top::argreduce_fwd", mutates_args=())
def _argreduce_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    strategy: str,
    ctas_per_row: int,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    if strategy == "cta":
        return _argreduce_cta_kernel(M, N, op_kind, dtype_str)(threads)(x)
    if strategy == "multi_cta":
        partial_values, partial_indices = _argreduce_multicta_partial_kernel(
            M, N, op_kind, dtype_str, ctas_per_row
        )(threads)(x)
        return _argreduce_multicta_final_kernel(M, N, op_kind, ctas_per_row)()(
            partial_values, partial_indices
        )
    return _argreduce_warp_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_argreduce_fwd_wrapped.register_fake
def _(
    M,
    N,
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
    """Adaptive streaming argmax/argmin over contiguous rows."""

    supported_archs: list[int] = [80, 86, 89, 90, 100]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. " f"Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        if N <= 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )

        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.ctas_per_row = 1

        if N >= 32768 and M < 64:
            self.strategy = "multi_cta"
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

    def _knobs(self, config: dict) -> dict:
        """Keep only the knobs this strategy's kernel actually takes.

        ``block_m`` packs several rows into one block, which only the warp
        layout does — the other two give a row its own block or its own group of
        them. Deriving the keys from the built kernel keeps a config space from
        naming a knob the kernel would reject.
        """
        parameters = self.kernel.signature.parameters
        return {name: value for name, value in config.items() if name in parameters}

    @property
    def default_config(self) -> dict:
        lanes = _lanes_per_row(self.N)
        target_threads = 256 if self.M >= 8 else max(32, self.M * lanes)
        block_m = max(1, target_threads // lanes)
        if self.strategy in {"cta", "multi_cta"}:
            block_m, threads = 1, 256
        else:
            threads = block_m * lanes
        return self._knobs({"block_m": block_m, "threads": threads})

    @property
    def autotune_configs(self) -> list[dict]:
        if self.strategy in {"cta", "multi_cta"}:
            candidates = [{"block_m": 1, "threads": t} for t in (128, 256, 512)]
        else:
            lanes = _lanes_per_row(self.N)
            candidates = []
            for target_threads in (64, 128, 256, 512):
                block_m = max(1, target_threads // lanes)
                candidates.append({"block_m": block_m, "threads": block_m * lanes})
        return [self._knobs(candidate) for candidate in candidates]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _argreduce_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.strategy,
            self.ctas_per_row,
            self.config.get("block_m", 1),
            self.config["threads"],
            x,
        )
