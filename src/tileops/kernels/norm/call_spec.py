"""Call records for normalization kernels."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.constants import VECTOR_ACCESS_BYTES

__all__ = ["BatchNormCall"]


@dataclasses.dataclass(frozen=True)
class BatchNormCall(CallSpec):
    """The facts that select a batch normalization implementation and build it.

    The input is ``(n, c, *spatial)``; ``spatial`` is the product of the trailing axes.
    ``eps`` and ``momentum`` are the op's construction parameters, which the programs
    compile in. The properties are the boundaries two candidate regions share.
    """

    n: int = 0
    c: int = 0
    spatial: int = 0
    dtype: torch.dtype = torch.float16
    eps: float = 1e-5
    momentum: float = 0.1

    # The longest channel of one element per batch item that one thread holds.
    _THREAD_MAX_L: ClassVar[int] = 32

    # Elements one thread of a register-holding block keeps, over every tensor it holds.
    _BLOCK_MAX_HELD: ClassVar[int] = 256
    _BLOCK_THREADS: ClassVar[int] = 256
    _BLOCK_MAX_THREADS: ClassVar[int] = 1024
    # The widest block once the grid alone covers every SM.
    _BLOCK_MAX_THREADS_FULL_GRID: ClassVar[int] = 512

    # At or above this many channels one block per channel already fills the device.
    _SPLIT_MAX_C: ClassVar[int] = 1024
    _SPLIT_MIN_L: ClassVar[int] = 1 << 16
    # Blocks a split grid aims for before tuning.
    _SPLIT_TARGET_BLOCKS: ClassVar[int] = 512

    @property
    def fits_one_thread(self) -> bool:
        """Whether a channel is one element per batch item and short enough for one thread."""
        return self.spatial <= 1 and self.n * self.spatial <= self._THREAD_MAX_L

    def block_launch(self, held_tensors: int) -> Optional[tuple[int, int]]:
        """The ``(threads, num_per_thread)`` of a block holding one channel in registers.

        ``None`` where the channel does not fit. A thread holds *held_tensors* elements per
        channel element, and its vector never straddles two batch items.
        """
        L = self.n * self.spatial
        full_grid = self.sm_count <= self.c
        widest = self._BLOCK_MAX_THREADS_FULL_GRID if full_grid else self._BLOCK_MAX_THREADS
        vector = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        for num_per_thread in (vector >> k for k in range(vector.bit_length())):
            if self.spatial % num_per_thread:
                continue
            # Halve the block while the channel would leave half of it empty.
            threads = self._BLOCK_THREADS
            while threads > 32 and threads * num_per_thread >= L * 2:
                threads //= 2
            steps = -(-L // (threads * num_per_thread))
            # Widen while a step is left partly empty and a wider block takes fewer steps;
            # a one-element thread stays, since a wider block only scatters more requests.
            while (
                num_per_thread > 1
                and threads < widest
                and steps > 1
                and steps * threads * num_per_thread != L
            ):
                threads *= 2
                steps = -(-L // (threads * num_per_thread))
            if steps * num_per_thread * held_tensors <= self._BLOCK_MAX_HELD:
                return threads, num_per_thread
        return None

    def fits_one_block(self, held_tensors: int) -> bool:
        """Whether one block's registers hold a channel of *held_tensors* tensors."""
        return self.block_launch(held_tensors) is not None

    @property
    def splittable(self) -> bool:
        """Whether a channel is long enough, among few enough channels, to cut across blocks."""
        return self.c < self._SPLIT_MAX_C and self.n * self.spatial >= self._SPLIT_MIN_L

    @property
    def split_seed(self) -> int:
        """Pieces a split channel is cut into before anything is measured."""
        return max(1, min(self.n * self.spatial, -(-self._SPLIT_TARGET_BLOCKS // self.c)))
