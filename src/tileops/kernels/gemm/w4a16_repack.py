"""Repack a W4A16 weight into the layout `GemmW4A16Kernel` reads.

The GEMM decodes a 32-bit word with four LOP3s, which needs nibble ``j`` and
nibble ``j+4`` of that word to be the two halves of one source byte. No
permutation of whole bytes produces that pairing, so the nibbles themselves
move, once, when a checkpoint loads.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .w4a16 import _LANES, MMA_STEP_K

__all__ = ["W4A16RepackKernel"]


@functools.lru_cache(maxsize=32)
def _w4a16_repack_kernel(n: int, packed_k: int, step_k: int = MMA_STEP_K) -> Callable:
    step = step_k // 2
    if packed_k % step or step % (4 * _LANES):
        raise ValueError(
            f"K/2={packed_k} must be a multiple of step_k/2={step}, which must itself"
            f" be a multiple of {4 * _LANES}"
        )
    words_per_step = step // 4
    words_per_lane = words_per_step // _LANES
    n_steps = packed_k // step

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def build(block_n: int = 64, threads: int = 128) -> Callable:
        @T.prim_func
        def main(
            packed_weight: T.Tensor((n, packed_k), "uint8"),  # type: ignore
            prepacked: T.Tensor((n, packed_k // 4), "uint32"),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), n_steps, threads=threads) as (bn, bs):
                for row, word in T.Parallel(block_n, words_per_step):
                    lane = word // words_per_lane
                    index = word % words_per_lane
                    acc = T.alloc_local((1,), "uint32")
                    acc[0] = T.uint32(0)
                    for pair in T.serial(4):
                        byte = packed_weight[
                            bn * block_n + row, bs * step + 16 * index + 4 * pair + lane
                        ]
                        acc[0] = (
                            acc[0]
                            | (T.cast(byte, "uint32") & T.uint32(0xF)) << (4 * pair)
                            | ((T.cast(byte, "uint32") >> 4) & T.uint32(0xF)) << (4 * pair + 16)
                        )
                    prepacked[bn * block_n + row, bs * words_per_step + word] = acc[0]

        return main

    return build


class W4A16RepackKernel(Kernel):
    """Move each K step's nibbles into the order the prepacked GEMM decodes.

    Args:
        n: Output columns of the weight this repacks.
        packed_k: Bytes per weight row, ``K / 2``.
        step_k: Weights per MMA step; must match the GEMM kernel's.
        config: Optional explicit config; defaults to :attr:`default_config`.
        tune: Whether to autotune over :attr:`autotune_configs`.
        device_index: CUDA device the kernel is built for.
    """

    # `packed_weight` is a uint8 payload, not an extent.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        packed_k: int,
        step_k: int = MMA_STEP_K,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.packed_k = packed_k
        self.step_k = step_k
        self.kernel = _w4a16_repack_kernel(n, packed_k, step_k)
        self.init_config(config, tune)
        while n % self.config["block_n"]:
            self.config["block_n"] //= 2

    @property
    def default_config(self) -> dict:
        return {"block_n": 64, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_n": block_n, "threads": threads}
            for block_n in (32, 64, 128)
            for threads in (128, 256)
            if self.n % block_n == 0
        ]

    def forward(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Permute a ``[N, K/2]`` packed weight in place of its own layout.

        Args:
            packed_weight: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``.

        Returns:
            A ``[N, K/2]`` ``torch.uint8`` tensor carrying the same nibbles in the
            order :class:`~tileops.kernels.gemm.GemmW4A16Kernel` reads.
        """
        compiled = self.kernel(**self.config)
        words = compiled(packed_weight)
        return words.view(torch.uint8).reshape(packed_weight.shape)
