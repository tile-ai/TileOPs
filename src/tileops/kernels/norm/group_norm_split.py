"""GroupNorm with each row cut into pieces across blocks.

Each block holds a piece in registers and writes its mean and centered square sum; one
block a row merges them into the row's mean and, for each channel, rstd times the weight;
a map writes ``(x - mean) * scale + bias``. The grid spans pieces, so few rows still fill
the device.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.norm._config import row_padding, select_row_config_by_width
from tileops.kernels.norm.call_spec import GroupNormCall, GroupNormFwdInterface
from tileops.kernels.norm.group_norm import GroupNormKernel
from tileops.utils import WARP_LANES

__all__ = ["GroupNormSplitKernel"]


def _block_sum(threads):
    """A macro summing one fp32 value a thread across the block into ``total``."""
    warps = threads // WARP_LANES

    @T.macro
    def block_sum(tx, acc, warp_sums, slot, total):
        for k in T.unroll(WARP_LANES.bit_length() - 1):
            acc[0] += T.shfl_xor(acc[0], T.shift_left(1, k))
        if tx % WARP_LANES == 0:
            warp_sums[slot, tx // WARP_LANES] = acc[0]
        T.sync_threads()
        total[0] = T.cast(0, "float32")
        for w in T.unroll(warps):
            total[0] += warp_sums[slot, w]

    return block_sum


@functools.lru_cache(maxsize=32)
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
            piece_mean: T.Tensor[(M, pieces), accum_dtype],
            piece_m2: T.Tensor[(M, pieces), accum_dtype],
        ):
            with T.Kernel(M, pieces, threads=threads) as (row, p):
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
                acc = T.alloc_local([1], accum_dtype)
                total = T.alloc_local([1], accum_dtype)
                mean = T.alloc_local([1], accum_dtype)
                warp_sums = T.alloc_shared([2, threads // WARP_LANES], accum_dtype)
                start = p * piece
                acc[0] = T.cast(0, accum_dtype)
                for a in T.unroll(accesses):
                    col = start + (a * threads + tx) * vec
                    if col < D:
                        for i in T.vectorized(vec):
                            held[a * vec + i] = x[row, col + i]
                    else:
                        for i in T.vectorized(vec):
                            held[a * vec + i] = T.cast(0, dtype)
                for j in T.unroll(accesses * vec):
                    acc[0] += T.cast(held[j], accum_dtype)
                block_sum(tx, acc, warp_sums, 0, total)
                mean[0] = total[0] / T.cast(T.min(piece, D - start), accum_dtype)
                acc[0] = T.cast(0, accum_dtype)
                for a in T.unroll(accesses):
                    col = start + (a * threads + tx) * vec
                    if col < D:
                        for i in T.unroll(vec):
                            d = T.cast(held[a * vec + i], accum_dtype) - mean[0]
                            acc[0] += d * d
                block_sum(tx, acc, warp_sums, 1, total)
                if tx == 0:
                    piece_mean[row, p] = mean[0]
                    piece_m2[row, p] = total[0]

        return main

    @tilelang.jit
    def merge(threads, piece):
        pieces = -(-D // piece)
        block_sum = _block_sum(threads)

        @T.prim_func
        def main(
            piece_mean: T.Tensor[(M, pieces), accum_dtype],
            piece_m2: T.Tensor[(M, pieces), accum_dtype],
            weight: T.Tensor[(C if affine else 1,), dtype],
            row_mean: T.Tensor[(M,), accum_dtype],
            scale: T.Tensor[(M * channels_per_group,), accum_dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                acc = T.alloc_local([1], accum_dtype)
                total = T.alloc_local([1], accum_dtype)
                mean = T.alloc_local([1], accum_dtype)
                rstd = T.alloc_local([1], accum_dtype)
                warp_sums = T.alloc_shared([2, threads // WARP_LANES], accum_dtype)
                # Chan's merge: the row mean weights each piece by its length, and the
                # spread of the piece means about it joins the pieces' own.
                acc[0] = T.cast(0, accum_dtype)
                for k in T.serial(T.ceildiv(pieces, threads)):
                    q = k * threads + tx
                    if q < pieces:
                        length = T.cast(T.min(piece, D - q * piece), accum_dtype)
                        acc[0] += length * piece_mean[row, q]
                block_sum(tx, acc, warp_sums, 0, total)
                mean[0] = total[0] / float(D)
                acc[0] = T.cast(0, accum_dtype)
                for k in T.serial(T.ceildiv(pieces, threads)):
                    q = k * threads + tx
                    if q < pieces:
                        length = T.cast(T.min(piece, D - q * piece), accum_dtype)
                        d = piece_mean[row, q] - mean[0]
                        acc[0] += piece_m2[row, q] + length * d * d
                block_sum(tx, acc, warp_sums, 1, total)
                rstd[0] = T.rsqrt(total[0] / float(D) + eps)
                if tx == 0:
                    row_mean[row] = mean[0]
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
            row_mean: T.Tensor[(M,), accum_dtype],
            scale: T.Tensor[(M * channels_per_group,), accum_dtype],
            bias: T.Tensor[(C if affine else 1,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(M, T.ceildiv(D, span), threads=threads) as (row, b):
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
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
                                T.cast(held[a * vec + i], accum_dtype) - row_mean[row]
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
        self._require_cuda(x=x, weight=weight, bias=bias)
        x, weight, bias = vector_aligned(x), vector_aligned(weight), vector_aligned(bias)
        if (weight is not None, bias is not None) != (self.affine, self.affine):
            raise ValueError(
                f"{type(self).__name__} was built {'with' if self.affine else 'without'} an "
                "affine; pass weight and bias exactly when it has one."
            )
        rows = x.reshape(-1, self.D)
        m = rows.shape[0]
        vec, stats, merge, apply = _group_norm_split_kernel(
            m,
            self.D,
            self.eps,
            self.dtype_str,
            self.num_groups,
            self.channels_per_group,
            self.affine,
        )
        threads, accesses = self.config["threads"], self.config["accesses"]
        piece = threads * accesses * vec
        pieces = -(-self.D // piece)
        f32 = functools.partial(torch.empty, device=x.device, dtype=torch.float32)
        piece_mean, piece_m2 = f32(m, pieces), f32(m, pieces)
        row_mean, scale = f32(m), f32(m * self.channels_per_group)
        if not self.affine:
            weight = bias = rows.new_empty(1)
        stats(threads, accesses)(rows, piece_mean, piece_m2)
        merge(128, piece)(piece_mean, piece_m2, weight, row_mean, scale)
        return apply(threads, accesses)(rows, row_mean, scale, bias).reshape(x.shape)
