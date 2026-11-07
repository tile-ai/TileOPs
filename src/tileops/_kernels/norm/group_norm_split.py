"""GroupNorm with each row cut into pieces across blocks.

Each block holds a piece in registers and writes its mean and variance contribution; one
block a row merges them into the row's mean and, for each channel, rstd times the weight;
a map writes ``(x - mean) * scale + bias``. The grid spans pieces, so few rows still fill
the device.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.norm._config import row_padding, select_row_config_by_width
from tileops._kernels.norm.call_spec import GroupNormCall, GroupNormFwdInterface
from tileops._kernels.norm.group_norm import GroupNormKernel
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["GroupNormSplitKernel"]


def _block_sum(threads):
    """A macro ``(value, sums, slots, phase, tx, total)`` summing ``value[:sums]`` across a block
    of *threads* into ``total``, through the shared rows of *slots* from *phase* on."""
    warps = threads // WARP_LANES

    @T.macro
    def block_sum(value, sums, slots, phase, tx, total):
        for st in T.unroll(WARP_SHUFFLE_STAGES):
            for s in T.unroll(sums):
                value[s] += T.shfl_xor(value[s], T.shift_left(1, st))
        if tx % WARP_LANES == 0:
            for s in T.unroll(sums):
                slots[phase + s, tx // WARP_LANES] = value[s]
        T.sync_threads()
        for s in T.unroll(sums):
            total[s] = T.cast(0, "float32")
            for w in T.unroll(warps):
                total[s] += slots[phase + s, w]

    return block_sum


@target_cache(maxsize=32)
def _group_norm_split_kernel(M, D, eps, dtype, num_groups, channels_per_group, affine):
    """Return ``(vec, stats, merge, apply)`` for ``M`` rows: the columns one access reads
    and the three stages' JIT factories."""
    accum_dtype = "float32"
    # The widest access that divides the row, so every row and piece starts on one.
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    while D % vec:
        vec //= 2
    spatial = D // channels_per_group
    C = num_groups * channels_per_group

    @tilelang.jit
    def stats(threads, accesses):
        piece = threads * accesses * vec
        pieces = -(-D // piece)
        block_sum = _block_sum(threads)

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            piece_ref: T.Tensor[(M, pieces), accum_dtype],
            piece_centre: T.Tensor[(M, pieces), accum_dtype],
            piece_m2: T.Tensor[(M, pieces), accum_dtype],
        ):
            with T.Kernel(M, pieces, threads=threads) as (row, p):
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
                value = T.alloc_local([2], accum_dtype)
                total = T.alloc_local([2], accum_dtype)
                lead = T.alloc_local([1], accum_dtype)
                warp_sums = T.alloc_shared([3, threads // WARP_LANES], accum_dtype)
                start = p * piece
                count = T.cast(T.min(piece, D - start), accum_dtype)
                # The piece's first element, in a two-column loop (tile-ai/tilelang#3453).
                for j in T.unroll(2):
                    if j == 0:
                        lead[0] = T.cast(x[row, start + j], accum_dtype)
                for a in T.unroll(accesses):
                    col = start + (a * threads + tx) * vec
                    if col < D:
                        for i in T.vectorized(vec):
                            held[a * vec + i] = x[row, col + i]
                value[0] = T.cast(0, accum_dtype)
                for a in T.unroll(accesses):
                    if start + (a * threads + tx) * vec < D:
                        for i in T.unroll(vec):
                            value[0] += T.cast(held[a * vec + i], accum_dtype) - lead[0]
                block_sum(value, 1, warp_sums, 0, tx, total)
                ref = lead[0] + total[0] / count
                value[0] = T.cast(0, accum_dtype)
                value[1] = T.cast(0, accum_dtype)
                for a in T.unroll(accesses):
                    if start + (a * threads + tx) * vec < D:
                        for i in T.unroll(vec):
                            d = T.cast(held[a * vec + i], accum_dtype) - ref
                            value[0] += d
                            value[1] += d * (d * (1.0 / D))
                block_sum(value, 2, warp_sums, 1, tx, total)
                if tx == 0:
                    piece_ref[row, p] = ref
                    piece_centre[row, p] = total[0] / count
                    # M2 / D keeps partial and merged sums within the variance range.
                    piece_m2[row, p] = total[1] - (total[0] * (1.0 / D)) * (total[0] / count)

        return main

    @tilelang.jit
    def merge(threads, piece):
        pieces = -(-D // piece)
        block_sum = _block_sum(threads)

        @T.prim_func
        def main(
            piece_ref: T.Tensor[(M, pieces), accum_dtype],
            piece_centre: T.Tensor[(M, pieces), accum_dtype],
            piece_m2: T.Tensor[(M, pieces), accum_dtype],
            weight: T.Tensor[(C if affine else 1,), dtype],
            row_ref: T.Tensor[(M,), accum_dtype],
            row_centre: T.Tensor[(M,), accum_dtype],
            scale: T.Tensor[(M * channels_per_group,), accum_dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                value = T.alloc_local([1], accum_dtype)
                total = T.alloc_local([1], accum_dtype)
                rstd = T.alloc_local([1], accum_dtype)
                warp_sums = T.alloc_shared([3, threads // WARP_LANES], accum_dtype)
                # The pieces' mean reference, so no sum across pieces rounds at a far piece's
                # distance.
                value[0] = T.cast(0, accum_dtype)
                for k in T.serial(T.ceildiv(pieces, threads)):
                    q = k * threads + tx
                    if q < pieces:
                        value[0] += piece_ref[row, q] - piece_ref[row, 0]
                block_sum(value, 1, warp_sums, 0, tx, total)
                ref = piece_ref[row, 0] + total[0] * (1.0 / pieces)
                # Chan's merge: the row mean weights each piece by its length, and the
                # spread of the piece means about it joins the pieces' own.
                value[0] = T.cast(0, accum_dtype)
                for k in T.serial(T.ceildiv(pieces, threads)):
                    q = k * threads + tx
                    if q < pieces:
                        length = T.cast(T.min(piece, D - q * piece), accum_dtype)
                        value[0] += length * ((piece_ref[row, q] - ref) + piece_centre[row, q])
                block_sum(value, 1, warp_sums, 1, tx, total)
                centre = total[0] * (1.0 / D)
                value[0] = T.cast(0, accum_dtype)
                for k in T.serial(T.ceildiv(pieces, threads)):
                    q = k * threads + tx
                    if q < pieces:
                        length = T.cast(T.min(piece, D - q * piece), accum_dtype)
                        d = (piece_ref[row, q] - ref) + piece_centre[row, q] - centre
                        value[0] += piece_m2[row, q] + (length * (1.0 / D)) * d * d
                block_sum(value, 1, warp_sums, 2, tx, total)
                rstd[0] = T.rsqrt(total[0] + eps)
                if tx == 0:
                    row_ref[row] = ref
                    row_centre[row] = centre
                for k in T.serial(T.ceildiv(channels_per_group, threads)):
                    c = k * threads + tx
                    if c < channels_per_group:
                        if affine:
                            channel = (row % num_groups) * channels_per_group + c
                            scale[row * channels_per_group + c] = rstd[0] * T.cast(
                                weight[channel], accum_dtype
                            )
                        else:
                            scale[row * channels_per_group + c] = rstd[0]

        return main

    @tilelang.jit(out_idx=[-1])
    def apply(threads, accesses):
        span = threads * accesses * vec
        # One channel owns every column of an access exactly when channels start on one.
        access_in_channel = spatial % vec == 0

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            row_ref: T.Tensor[(M,), accum_dtype],
            row_centre: T.Tensor[(M,), accum_dtype],
            scale: T.Tensor[(M * channels_per_group,), accum_dtype],
            bias: T.Tensor[(C if affine else 1,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(M, T.ceildiv(D, span), threads=threads) as (row, b):
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
                ref = row_ref[row]
                centre = row_centre[row]
                for a in T.unroll(accesses):
                    col = b * span + (a * threads + tx) * vec
                    if col < D:
                        for i in T.vectorized(vec):
                            held[a * vec + i] = x[row, col + i]
                for a in T.unroll(accesses):
                    col = b * span + (a * threads + tx) * vec
                    if col < D:
                        for i in T.unroll(vec):
                            c = col // spatial if access_in_channel else (col + i) // spatial
                            normed = (
                                (T.cast(held[a * vec + i], accum_dtype) - ref) - centre
                            ) * scale[row * channels_per_group + c]
                            if affine:
                                normed = normed + T.cast(
                                    bias[(row % num_groups) * channels_per_group + c], accum_dtype
                                )
                            held[a * vec + i] = T.cast(normed, dtype)
                        for i in T.vectorized(vec):
                            y[row, col + i] = held[a * vec + i]

        return main

    return vec, stats, merge, apply


class GroupNormSplitKernel(Kernel, GroupNormFwdInterface):
    """GroupNorm, affine or not, for the rows the one-block kernels cannot hold or run slower.

    Args:
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        num_groups: Number of groups G.
        channels_per_group: C / G.
        affine: Whether the call passes weight and bias.
        config: Optional ``{"threads", "accesses"}``.
    """

    supported_archs = [80, 86, 89, 90]
    preferred_over = frozenset({"group_norm", "group_norm_no_affine"})

    # Past 64 columns for each of 1024 threads, a row held in registers runs faster cut up.
    _REGISTER_ROW = 1024 * 64
    # One block a row takes time by the row's length, the pieces by its bytes: past this many
    # columns an fp16 or bf16 row staged through shared memory runs faster cut up.
    _STAGED_TWO_BYTE_ROW = 16384

    @classmethod
    def refusal(cls, call: GroupNormCall) -> "str | None":
        d = call.c // call.num_groups * call.spatial
        d_padded = row_padding(d, call.dtype.itemsize)
        if GroupNormKernel._holds_row_in_registers(d, d_padded):
            split = d_padded > cls._REGISTER_ROW
        elif call.dtype.itemsize == 2 and d_padded > cls._STAGED_TWO_BYTE_ROW:
            split = True
        else:
            # A staged row sits in shared memory beside the reduction's scratch, a word a thread.
            config = select_row_config_by_width(
                d_padded, GroupNormKernel._row_widths_for(d, d_padded)
            )
            split = d_padded * call.dtype.itemsize + config["threads"] * 4 > call.smem_budget
        if not split:
            return f"a group row of {d} elements fits one block"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GroupNormCall) -> Entry:
        cpg = call.c // call.num_groups
        identity = (
            cpg * call.spatial,
            call.eps,
            call.dtype,
            call.num_groups,
            cpg,
            call.passes_affine,
        )
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        D: int,
        eps: float,
        dtype: torch.dtype,
        num_groups: int,
        channels_per_group: int,
        affine: bool,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.D, self.eps, self.dtype = D, eps, dtype
        self.num_groups, self.channels_per_group, self.affine = (
            num_groups,
            channels_per_group,
            affine,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": 256, "accesses": 4}

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize *x*'s groups; *weight* and *bias* come exactly when built with the affine.

        Raises:
            ValueError: An input is off CUDA, or the affine pair does not match the build.
        """
        self._require_devices(x=x, weight=weight, bias=bias)
        if (weight is not None, bias is not None) != (self.affine, self.affine):
            raise ValueError(
                f"{type(self).__name__} was built {'with' if self.affine else 'without'} an "
                "affine; pass weight and bias exactly when it has one."
            )
        rows = x.reshape(-1, self.D)
        if not self.affine:
            weight = bias = rows.new_empty(1)
        return self.program(self._launcher, rows.shape[0])(rows, weight, bias).reshape(x.shape)

    def _launcher(self, cfg: dict, m: int) -> Callable:
        """What one launch of *m* rows under *cfg* runs on ``(rows, weight, bias)``: the
        statistics, merge and apply programs with the scratch they pass between them."""
        vec, stats, merge, apply = _group_norm_split_kernel(
            m,
            self.D,
            self.eps,
            self.dtype_str,
            self.num_groups,
            self.channels_per_group,
            self.affine,
        )
        threads, accesses = cfg["threads"], cfg["accesses"]
        piece = threads * accesses * vec
        pieces = -(-self.D // piece)
        stats = CompiledProgram(stats(threads, accesses))
        merge = CompiledProgram(merge(128, piece))
        apply = CompiledProgram(apply(threads, accesses))
        channels_per_group = self.channels_per_group

        def launch(rows, weight, bias):
            f32 = functools.partial(torch.empty, device=rows.device, dtype=torch.float32)
            piece_ref, piece_centre, piece_m2 = f32(m, pieces), f32(m, pieces), f32(m, pieces)
            row_ref, row_centre, scale = f32(m), f32(m), f32(m * channels_per_group)
            stats(rows, piece_ref, piece_centre, piece_m2)
            merge(piece_ref, piece_centre, piece_m2, weight, row_ref, row_centre, scale)
            return apply(rows, row_ref, row_centre, scale, bias)

        return launch
