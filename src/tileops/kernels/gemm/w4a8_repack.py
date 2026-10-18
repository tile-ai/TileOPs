"""Repack a signed INT4 weight into the order the W4A8 GEMM kernels read."""

import functools
from dataclasses import dataclass
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import WARP_MMA_ROWS, WGMMA_ROWS
from tileops.kernels.gemm.call_spec import W4A8RepackCall, W4A8RepackFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GROUP_SIZE", "W4A8_LAYOUT", "W4A8Layout", "W4A8RepackKernel"]


@dataclass(frozen=True)
class W4A8Layout:
    """Extents fixed by the prepacked-weight ABI the repack and both W4A8 GEMMs share.

    A unit is 512 bytes, ``unit_rows`` rows by ``unit_k`` K: one warp's slice of the
    ``m64nNk32`` FP8 WGMMA A operand for two K/32 steps, each lane's 16 bytes the fragment
    of rows ``r`` and ``r + 8``. A tile is the ``tile_rows`` by ``tile_k`` block of units,
    one WGMMA M extent, stored contiguously; the last row block holds
    ``(N % tile_rows) / unit_rows`` units per K half when ``tile_rows`` does not divide N.
    """

    unit_rows: int = WARP_MMA_ROWS
    unit_k: int = 64
    tile_rows: int = WGMMA_ROWS
    tile_k: int = 128

    @property
    def tile_bytes(self) -> int:
        """Bytes of one full tile, two INT4 per byte."""
        return self.tile_rows * self.tile_k // 2

    @property
    def k_half_bytes(self) -> int:
        """Bytes of one full tile's ``unit_k``-wide K half: ``tile_rows / unit_rows`` units."""
        return self.tile_bytes * self.unit_k // self.tile_k

    @property
    def unit_bytes(self) -> int:
        """Bytes of one unit."""
        return self.unit_rows * self.unit_k // 2


# Read by the repack, the op's input check and both GEMM kernels, which decode this order.
W4A8_LAYOUT = W4A8Layout()
# The weight group along K both W4A8 GEMMs are compiled for, and the only one they serve.
GROUP_SIZE = 128


@functools.lru_cache(maxsize=32)
def _w4a8_repack_kernel(n: int, k: int) -> Callable:
    unit_rows, unit_k = W4A8_LAYOUT.unit_rows, W4A8_LAYOUT.unit_k
    tile_rows, tile_k = W4A8_LAYOUT.tile_rows, W4A8_LAYOUT.tile_k
    # A unit's 512 bytes are 128 32-bit words, one per thread.
    words_per_unit = unit_rows * unit_k // 8
    units_per_tile_half = tile_rows // unit_rows
    full_blocks = n // tile_rows
    ragged_units = (n % tile_rows) // unit_rows
    words_per_block = (k // tile_k) * 2 * units_per_tile_half * words_per_unit

    @tilelang.jit(out_idx=[-1])
    def build() -> Callable:
        def encode(nibble):
            # Two's complement q in [-8, 7] to the code the GEMMs decode: q - 1 for q in [1, 7],
            # 7 for q = 0, 8 | (-q - 1) for q < 0.
            return T.if_then_else(nibble >= 8, nibble ^ 7, (nibble + 7) & 7)

        def encode_byte(byte):
            wide = T.cast(byte, "uint32")
            return encode(wide & T.uint32(0xF)) | (encode(wide >> 4) << 4)

        @T.prim_func
        def main(
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            prepacked: T.Tensor((n * k // 8,), "uint32"),  # type: ignore
        ) -> None:
            with T.Kernel(n // unit_rows, k // unit_k, threads=words_per_unit) as (
                unit,
                half,
            ):
                block = unit // units_per_tile_half
                units = T.if_then_else(block < full_blocks, units_per_tile_half, ragged_units)
                for word in T.Parallel(words_per_unit):
                    # word = 4 * (4r + t) + 2p + hr holds row r + 8hr of the unit, K 4t..4t+3
                    # and 4t+16..4t+19 of its K/32 step p.
                    lane = word // 4
                    row = unit_rows * unit + 8 * (word % 2) + lane // 4
                    col = (unit_k // 2) * half + 16 * ((word // 2) % 2) + 2 * (lane % 4)
                    prepacked[
                        block * words_per_block
                        + (half // 2) * units * 2 * words_per_unit
                        + (half % 2) * units * words_per_unit
                        + (unit % units_per_tile_half) * words_per_unit
                        + word
                    ] = (
                        encode_byte(packed_weight[row, col])
                        | (encode_byte(packed_weight[row, col + 1]) << 8)
                        | (encode_byte(packed_weight[row, col + 8]) << 16)
                        | (encode_byte(packed_weight[row, col + 9]) << 24)
                    )

        return main

    return build


class W4A8RepackKernel(Kernel, W4A8RepackFwdInterface):
    """Move each weight into its tensor-core fragment slot and re-encode its nibble.

    Args:
        n: Output columns of the weight this repacks, a multiple of 16.
        k: Contraction length of the weight, a multiple of 128.
        config: Unused; the program has no tunable parameter.
        device_index: CUDA device the kernel is built for.
    """

    # `packed_weight` is a uint8 payload, not an extent.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [90]

    @classmethod
    def entry_for(cls, call: W4A8RepackCall) -> Entry:
        """The program depends on ``(n, k)`` only; the architecture is validated, not read."""
        return (call.n, call.k), lambda: cls(call.n, call.k, device_index=call.device.index)

    def __init__(
        self,
        n: int,
        k: int,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.k = k
        self.kernel = _w4a8_repack_kernel(n, k)
        self.init_config(config)

    def forward(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Reorder and re-encode a ``[N, K/2]`` row-major packed weight.

        Args:
            packed_weight: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``.

        Returns:
            A ``[N, K/2]`` ``torch.uint8`` tensor in the order the W4A8 GEMMs read.
        """
        words = self.kernel()(packed_weight)
        return words.view(torch.uint8).reshape(packed_weight.shape)
