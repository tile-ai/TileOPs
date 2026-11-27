"""Inverted dropout over Philox4x32-10 draws; the seed and offset are launch arguments."""

import dataclasses
import fractions
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.elementwise._dtype import FLOAT_DTYPES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._kernels.sampling.philox import mix

__all__ = ["DropoutCall", "DropoutFwdInterface", "DropoutKernel"]


@dataclasses.dataclass(frozen=True)
class DropoutCall(CallSpec):
    """One dropout call; the probability is baked into the program, the seed is not."""

    count: int = 0
    p: float = 0.5
    dtype: torch.dtype | None = None


class DropoutFwdInterface(KernelInterface):
    """Inverted dropout over a flat run of elements."""

    request = DropoutCall
    # A call uses at most max(count, MAX_DRAWS_PER_STREAM) draws of one stream: a thread draws
    # one value per element it holds, so this caps num_per_thread, and the op advances torch's
    # generator past at least this many.
    MAX_DRAWS_PER_STREAM = 32

    @abstractmethod
    def forward(self, x: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
        """Scale the elements a Bernoulli draw keeps; nothing is written in place.

        Element ``i`` becomes ``x[i] / (1 - call.p)`` where its draw keeps it and zero where
        it does not. Each element has a draw of its own, independent of every other
        element's, that drops it with probability ``call.p`` rounded up to a multiple of
        ``2**-32``. The draws come from the Philox stream ``(seed, offset)`` through the
        thread layout the built program fixes, so two calls on one built kernel with the
        same ``seed`` and ``offset`` produce the same mask. An implementation uses at most
        ``max(call.count, MAX_DRAWS_PER_STREAM)`` consecutive draws of each stream, starting
        at ``offset``, so a call at an offset that many or more higher uses values this one
        did not.
        Nothing is written in place, and the output aliases no input.

        Args:
            x: The input, ``(call.count,)`` in ``call.dtype`` on ``call.device``,
                contiguous. The op passes a flat contiguous view of the caller's tensor and
                restores its shape outside the kernel, so an implementation sees one run of
                elements and no stride.
            seed: The Philox seed, a signed 64-bit integer.
            offset: The Philox offset, a non-negative multiple of 4 below ``2**63``.

        Returns:
            A new contiguous ``(call.count,)`` tensor in ``call.dtype`` on ``call.device``.
        """


@T.macro
def _philox_block(words, counter, key, bump, seed, offset, sequence, block):
    """Write draws ``4 * block`` to ``4 * block + 3`` of subsequence *sequence* into ``words``.

    The draws are those cuRAND's ``curandStatePhilox4_32_10_t`` yields after
    ``curand_init(seed, sequence, offset)`` for an *offset* that is a multiple of 4: the
    counter holds the block of four in its low 64 bits and the subsequence in its high 64.
    """
    quad = (offset >> T.int64(2)) + T.int64(block)
    counter[0] = T.cast(quad & T.int64(0xFFFFFFFF), "uint32")
    counter[1] = T.cast(quad >> T.int64(32), "uint32")
    counter[2] = T.cast(sequence, "uint32")
    counter[3] = T.uint32(0)
    key[0] = T.cast(seed & T.int64(0xFFFFFFFF), "uint32")
    key[1] = T.cast((seed >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
    mix(counter, key, bump)
    for k in T.unroll(4):
        words[4 * block + k] = counter[k]


def _half(word, j):
    """The 16-bit half of *word* element ``j`` takes: the low half for even ``j``."""
    return (word >> T.uint32(16 * (j % 2))) & T.uint32(0xFFFF)


def _scaled(value, scale, dtype):
    """*value* times *scale*, in *dtype* where *scale* is finite in it and in float32 where
    it is not: ``1 / (1 - p)`` passes float16's largest value for ``p`` near 1, where the
    scaled element can still fit."""
    if scale <= torch.finfo(getattr(torch, dtype)).max:
        return value * T.cast(scale, dtype)
    return T.cast(T.cast(value, "float32") * T.float32(scale), dtype)


@target_cache(maxsize=32)
def _make_dropout_kernel(N, dtype, p, threads, num_per_thread):
    """Build a dropout kernel over Philox4x32-10 draws.

    Thread ``t`` holds elements ``t * num_per_thread`` onwards and draws ``num_per_thread``
    values of subsequence ``t`` from ``offset``. Element ``j``'s 32-bit draw has half
    ``j % 2`` of value ``j // 2`` as its high 16 bits and half ``j % 2`` of value
    ``num_per_thread // 2 + j // 2`` as its low 16 bits. The first half of the values thus
    decides every element whose high half differs from the threshold's; the rest are drawn
    only in a thread holding an element whose high half equals it, which happens with
    probability ``2**-16`` an element, and never for a threshold whose low half is 0. An
    element whose draw is below ``ceil(p * 2**32)`` is dropped and the rest are scaled by
    ``1 / (1 - p)``.

    *num_per_thread* is a multiple of 4 and at most
    ``DropoutFwdInterface.MAX_DRAWS_PER_STREAM``.
    """
    block_size = threads * num_per_thread
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    # The least 32-bit draw that keeps an element, so a draw drops with probability p rounded
    # up to the next multiple of 2**-32; 2**32 keeps none.
    threshold = math.ceil(fractions.Fraction(p) * 2**32)
    high, low = threshold >> 16, threshold & 0xFFFF

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        # Philox blocks a thread draws, and how many of them hold the high halves.
        blocks = npt_arg // 4
        high_blocks = T.ceildiv(npt_arg // 2, 4)

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
            seed: T.int64,
            offset: T.int64,
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                thread = bx * threads_arg + T.get_thread_binding()
                base = thread * npt_arg
                words = T.alloc_local((npt_arg,), "uint32")
                counter = T.alloc_local((4,), "uint32")
                key = T.alloc_local((2,), "uint32")
                bump = T.alloc_local((2,), "uint32")
                for b in T.unroll(high_blocks):
                    _philox_block(words, counter, key, bump, seed, offset, thread, b)
                kept = T.alloc_local((npt_arg,), "bool")
                for j in T.unroll(npt_arg):
                    kept[j] = _half(words[j // 2], j) >= T.uint32(high + (low > 0))
                if low > 0:
                    tied = T.alloc_local((1,), "bool")
                    tied[0] = False
                    for j in T.unroll(npt_arg):
                        tied[0] = tied[0] or _half(words[j // 2], j) == T.uint32(high)
                    if tied[0]:
                        for b in T.unroll(high_blocks, blocks):
                            _philox_block(words, counter, key, bump, seed, offset, thread, b)
                        for j in T.unroll(npt_arg):
                            kept[j] = kept[j] or (
                                _half(words[j // 2], j) == T.uint32(high)
                                and _half(words[npt_arg // 2 + j // 2], j) >= T.uint32(low)
                            )
                x_run = T.alloc_local((npt_arg,), dtype)
                y_run = T.alloc_local((npt_arg,), dtype)
                if base + npt_arg <= N:
                    for j in T.vectorized(npt_arg):
                        x_run[j] = x[base + j]
                    for j in T.vectorized(npt_arg):
                        x_run[j] = _scaled(x_run[j], scale, dtype)
                    for j in T.unroll(npt_arg):
                        y_run[j] = T.if_then_else(kept[j], x_run[j], T.cast(0.0, dtype))
                    for j in T.vectorized(npt_arg):
                        y[base + j] = y_run[j]
                else:
                    for j in T.unroll(npt_arg):
                        if base + j < N:
                            y[base + j] = T.if_then_else(
                                kept[j], _scaled(x[base + j], scale, dtype), T.cast(0.0, dtype)
                            )

        return main

    return kernel


class DropoutKernel(Kernel, DropoutFwdInterface):
    """Inverted dropout with a Philox4x32-10 mask, one independent draw per element.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for input (float16, bfloat16, float32).
        p: Drop probability in [0, 1].
        config: Optional dict with "threads" and "num_per_thread"; ``num_per_thread`` is a
            multiple of 4 and at most ``MAX_DRAWS_PER_STREAM``.

    Raises:
        ValueError: If ``p`` is outside [0, 1] or ``num_per_thread`` is not such a count.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    # The kernel scales only floating-point elements.
    SUPPORTED_DTYPES = FLOAT_DTYPES

    @classmethod
    def entry_for(cls, call: DropoutCall) -> Entry:
        return call, lambda: cls(call.count, call.dtype, p=call.p)

    def __init__(self, N_total, dtype, p=0.5, config=None):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.N_total = N_total
        self.dtype = dtype
        self.p = p
        self.init_config(config)
        npt = self.config["num_per_thread"]
        if npt <= 0 or npt % 4 or npt > self.MAX_DRAWS_PER_STREAM:
            raise ValueError(
                "num_per_thread must be a positive multiple of 4 and at most "
                f"{self.MAX_DRAWS_PER_STREAM}, got {npt}"
            )
        self.kernel = self._build_kernel(self.config)

    def _build_kernel(self, cfg: dict):
        # The grid bakes in ``threads * num_per_thread``, so it is built from the compiled config.
        return _make_dropout_kernel(
            self.N_total,
            self.dtype_str,
            self.p,
            threads=cfg["threads"],
            num_per_thread=cfg["num_per_thread"],
        )

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self._build_kernel(cfg)(cfg["threads"], cfg["num_per_thread"])

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x, seed, offset):
        return self.program(self._compile)(x, seed, offset)
