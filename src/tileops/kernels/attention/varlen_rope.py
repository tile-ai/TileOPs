"""Rotary Position Embedding (RoPE) for packed variable-length attention.

The two sides are placed differently because they are read differently. A rotated key row
is read by every query tile of its request and by every query head of its KV group, so it
is rotated once, in its own launch, before the attention kernel runs. A rotated query tile
is read by the one CTA that owns it, so rotating it inside that CTA costs the tile's
registers and nothing else, where materializing it would cost a full round trip through
global memory.

Query token ``i`` of request ``b`` sits at position ``kv_len_b - q_len_b + i`` and key
token ``j`` at position ``j``.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import MAX_BLOCK_THREADS, VECTOR_ACCESS_BYTES
from tileops.kernels.grouped_tiling import GroupTiling

__all__ = ["VarlenKeyRoPE", "make_varlen_query_rope", "rope_channel_pair"]


def rope_channel_pair(rope_layout: str, half: int, freq):
    """The two channels frequency *freq* rotates together under *rope_layout*.

    ``neox`` pairs channel ``f`` with ``f + rotary_dim / 2``; ``interleaved`` pairs the
    adjacent channels ``2f`` and ``2f + 1``.
    """
    if rope_layout == "neox":
        return freq, freq + half
    return freq * 2, freq * 2 + 1


def make_varlen_query_rope(
    rows: int,
    rotary_dim: int,
    rope_layout: str,
    max_position: int,
    dtype: str,
    rope_dtype: str = "",
    tile_axes: int = 0,
    positions: int = 0,
) -> Callable:
    """A macro rotating a ``[rows, dim]`` shared query tile in place.

    The tile's first row sits at *base*; rows past the request's end hold padding whose
    rotation is discarded, so their position is clamped to the table rather than guarded.

    Args:
        rows: Query rows the tile holds.
        rotary_dim: Rotated width; the channels past it are left alone.
        rope_layout: ``"neox"`` or ``"interleaved"``.
        max_position: Rows of the table, which bounds the position an index may take.
        dtype: Element type of the tile.
        rope_dtype: Element type of the table, which an FP8 call carries at 16 bits while
            the tile it rotates is 8. Empty means the tile's.
        positions: Query positions the tile spans, which is fewer than its rows when a
            KV group's query heads are packed into it: row ``i`` then carries position
            ``i % positions``. Zero means one position per row.
        tile_axes: Axes in front of the tile's own two. A warp-specialized kernel holds
            its query tiles in a double-buffered, per-warpgroup array and passes 2, with
            the slot and the warpgroup as the two leading arguments of the macro; a
            kernel whose tile is the whole buffer passes 0.
    """
    half = rotary_dim // 2
    span = positions or rows
    accum = "float"
    # Frequencies one thread owns: enough that the two channel runs the pair indexing
    # selects form one widest access together. Re-fit by changing the divisor.
    itemsize = torch.empty(0, dtype=getattr(torch, dtype)).element_size()
    run = max(1, VECTOR_ACCESS_BYTES // (2 * itemsize))
    while half % run:
        run -= 1

    @T.macro
    def rotate_query_tile(q_shared, rope_cos, rope_sin, base, slot=0, warpgroup=0):
        for i, chunk in T.Parallel(rows, half // run):
            pos = T.min(base + i % span, max_position - 1)
            for step in T.serial(run):
                freq = chunk * run + step
                d0, d1 = rope_channel_pair(rope_layout, half, freq)
                c = T.Cast(accum, rope_cos[pos, freq])
                s = T.Cast(accum, rope_sin[pos, freq])
                if tile_axes == 2:
                    x0 = T.Cast(accum, q_shared[slot, warpgroup, i, d0])
                    x1 = T.Cast(accum, q_shared[slot, warpgroup, i, d1])
                    q_shared[slot, warpgroup, i, d0] = T.Cast(dtype, x0 * c - x1 * s)
                    q_shared[slot, warpgroup, i, d1] = T.Cast(dtype, x1 * c + x0 * s)
                else:
                    x0 = T.Cast(accum, q_shared[i, d0])
                    x1 = T.Cast(accum, q_shared[i, d1])
                    q_shared[i, d0] = T.Cast(dtype, x0 * c - x1 * s)
                    q_shared[i, d1] = T.Cast(dtype, x1 * c + x0 * s)

    return rotate_query_tile


@functools.lru_cache(maxsize=32)
def _varlen_rope_keys_kernel(
    batch: int,
    heads_kv: int,
    dim: int,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    dtype: str,
    rope_dtype: str,
    block_t: int,
    threads: int,
) -> Callable:
    """Build the key-rotation program; *block_t* key rows per CTA, *threads* rotating them."""
    half = rotary_dim // 2
    accum = "float"
    tail = dim - rotary_dim

    @tilelang.jit(
        out_idx=[4],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _varlen_rope_keys_func() -> Callable:
        total_kv = T.dynamic("total_kv")
        kv_shape = (total_kv, heads_kv, dim)
        tiling = GroupTiling(batch, block_t)
        num_tiles = tiling.tile_upper_bound(total_kv)

        @T.prim_func
        def _varlen_rope_keys_main(
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            rope_cos: T.Tensor([max_position, half], rope_dtype),  # type: ignore
            rope_sin: T.Tensor([max_position, half], rope_dtype),  # type: ignore
            k_rot: T.Tensor(kv_shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_tiles, heads_kv, threads=threads) as (row_tile, head):
                tile_cum = T.alloc_shared([batch + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                request = T.alloc_local([1], "int32")
                row = T.alloc_local([1], "int32")

                tiling.cumsum_offsets(cu_seqlens_kv, tile_cum)
                if row_tile < tile_cum[batch]:
                    tiling.decode(row_tile, tile_cum, lo, hi, request, row)
                    kv_start = cu_seqlens_kv[request[0]]
                    kv_len = cu_seqlens_kv[request[0] + 1] - kv_start
                    base = row[0]
                    # A tile ending inside the request runs unguarded, so each half of the
                    # rotated channels crosses global memory as a vector; only the request's
                    # last tile carries a predicate.
                    if base + block_t <= kv_len:
                        for i, freq in T.Parallel(block_t, half):
                            d0, d1 = rope_channel_pair(rope_layout, half, freq)
                            # Key token j of a request sits at position j.
                            pos = T.min(base + i, max_position - 1)
                            c = T.Cast(accum, rope_cos[pos, freq])
                            s = T.Cast(accum, rope_sin[pos, freq])
                            x0 = T.Cast(accum, k[kv_start + base + i, head, d0])
                            x1 = T.Cast(accum, k[kv_start + base + i, head, d1])
                            k_rot[kv_start + base + i, head, d0] = T.Cast(dtype, x0 * c - x1 * s)
                            k_rot[kv_start + base + i, head, d1] = T.Cast(dtype, x1 * c + x0 * s)
                        for i, channel in T.Parallel(block_t, tail):
                            k_rot[kv_start + base + i, head, rotary_dim + channel] = k[
                                kv_start + base + i, head, rotary_dim + channel
                            ]
                    else:
                        for i, freq in T.Parallel(block_t, half):
                            if base + i < kv_len:
                                d0, d1 = rope_channel_pair(rope_layout, half, freq)
                                pos = T.min(base + i, max_position - 1)
                                c = T.Cast(accum, rope_cos[pos, freq])
                                s = T.Cast(accum, rope_sin[pos, freq])
                                x0 = T.Cast(accum, k[kv_start + base + i, head, d0])
                                x1 = T.Cast(accum, k[kv_start + base + i, head, d1])
                                k_rot[kv_start + base + i, head, d0] = T.Cast(
                                    dtype, x0 * c - x1 * s
                                )
                                k_rot[kv_start + base + i, head, d1] = T.Cast(
                                    dtype, x1 * c + x0 * s
                                )
                        for i, channel in T.Parallel(block_t, tail):
                            if base + i < kv_len:
                                k_rot[kv_start + base + i, head, rotary_dim + channel] = k[
                                    kv_start + base + i, head, rotary_dim + channel
                                ]

        return _varlen_rope_keys_main

    return _varlen_rope_keys_func


class VarlenKeyRoPE:
    """Rotate every packed key once, before an attention kernel reads it."""

    # Key rows a CTA rotates. Fitted by sweeping it against the launch's device_busy on
    # the manifest's packed shapes; re-run that sweep to change it.
    _KEY_ROWS_PER_CTA: int = 16

    def __init__(
        self,
        batch: int,
        heads_kv: int,
        dim: int,
        max_position: int,
        rotary_dim: int,
        rope_layout: str,
        dtype: str,
        rope_dtype: str = "",
    ) -> None:
        self.kernel = _varlen_rope_keys_kernel(
            batch,
            heads_kv,
            dim,
            max_position,
            rotary_dim,
            rope_layout,
            dtype,
            rope_dtype or dtype,
            self._KEY_ROWS_PER_CTA,
            self._thread_count(rotary_dim, dtype),
        )

    @classmethod
    def _thread_count(cls, rotary_dim: int, dtype: str) -> int:
        """Threads that give each one a whole vector of each rotated half.

        A thread rotates one run of ``freq`` and reads both channels that run pairs with,
        so the widest vectorized access fixes how many frequencies it takes.
        """
        itemsize = torch.empty(0, dtype=getattr(torch, dtype)).element_size()
        per_thread = max(1, VECTOR_ACCESS_BYTES // itemsize)
        pairs = cls._KEY_ROWS_PER_CTA * (rotary_dim // 2)
        return max(32, min(MAX_BLOCK_THREADS, pairs // per_thread))

    def __call__(
        self,
        k: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if rope_cos is None or rope_sin is None:
            raise ValueError("fused RoPE requires rope_cos and rope_sin")
        return self.kernel()(k, cu_seqlens_kv, rope_cos, rope_sin)
