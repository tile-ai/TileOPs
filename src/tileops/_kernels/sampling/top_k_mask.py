"""Per-row top-k logit mask: a cluster of CTAs holds each row in registers and selects its
k-th largest value by a most-significant-digit radix select over the key bits."""

from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import SamplingCall, TopKMaskFwdInterface
from tileops._kernels.sampling.radix_select import cluster_plan, key_select, row_refusal

__all__ = ["TopKMaskFwdKernel"]


@target_cache(maxsize=32)
def _top_k_mask_kernel(batch: int, vocab: int, dtype: str):
    """Build the top-k mask of ``batch`` rows of ``vocab`` logits."""
    n = batch * vocab

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h", "top_k_mask_helper.h"))
    def _top_k_mask_func(threads: int, cluster: int, slots: int):
        rows = key_select(vocab, dtype, threads, cluster, slots)
        words, held, halves = rows.words, rows.held, rows.halves
        word_neg_inf = rows.word_neg_inf

        @T.prim_func
        def _top_k_mask_main(
            x: T.Tensor((n,), dtype),
            k: T.Tensor((batch,), "int32"),
            out: T.Tensor((n,), dtype),
        ):
            with (
                T.Kernel(batch, threads=threads)
                if cluster == 1
                else T.ClusterKernel(batch * cluster, cluster_dims=cluster, threads=threads)
            ) as bx:
                vals = T.alloc_local((held,), "uint32")
                outw = T.alloc_local((words,), "uint32")
                acc = T.alloc_local((4,), "int32")
                sum32 = T.alloc_local((1,), "uint32")
                above = T.alloc_local((1,), "int32")
                digit_shift = T.alloc_local((1,), "uint32")
                settled = T.alloc_local((1,), "uint32")
                counted = T.alloc_local((2,), "int32")
                hist = T.alloc_shared((2 * rows.slabs,), "uint32")
                total = T.alloc_shared((rows.slabs,), "uint32")
                red = T.alloc_shared((rows.warps,), "int32")
                # The key bits the select has settled.
                pre = T.alloc_shared((1,), "uint32")
                st = T.alloc_shared((9,), "int32")

                row = bx if cluster == 1 else bx // cluster
                rank = 0 if cluster == 1 else T.block_rank_in_cluster()
                base = row * vocab + rank * rows.chunk
                out_of_row = vocab - rank * rows.chunk

                rows.load_row(vals, x, base, out_of_row)
                kk = k[row]
                if kk >= vocab:
                    for j in T.unroll(slots):
                        for c in T.unroll(words):
                            outw[c] = vals[j * words + c]
                        rows.store_slot(outw, j, out_of_row, out, base)
                else:
                    rows.to_keys(vals)
                    rows.select_kth(
                        vals,
                        kk,
                        0,
                        out_of_row,
                        hist,
                        total,
                        red,
                        pre,
                        st,
                        acc,
                        sum32,
                        above,
                        digit_shift,
                        settled,
                        counted,
                    )
                    thr = rows.keeps_both_zeros(pre[0]) * T.uint32(halves)
                    # Each value at least ``thr`` keeps its bits; the rest, and every NaN,
                    # become -inf.
                    for j in T.unroll(slots):
                        for c in T.unroll(words):
                            key = vals[j * words + c]
                            if rows.packed:
                                back = T.uint32(0xFFFFFFFF) - (
                                    (key >> T.uint32(15)) & T.uint32(halves)
                                ) * T.uint32(0x7FFF)
                                keep = T.call_extern(
                                    "uint32", "__vcmpgeu2", key, thr
                                ) & ~T.call_extern("uint32", "__vcmpeq2", key, T.uint32(0xFFFFFFFF))
                                outw[c] = ((key ^ back) & keep) | (T.uint32(word_neg_inf) & ~keep)
                            else:
                                outw[c] = T.if_then_else(
                                    (key >= thr) & (key != T.uint32(0xFFFFFFFF)),
                                    key
                                    ^ T.if_then_else(
                                        (key >> T.uint32(31)) != T.uint32(0),
                                        T.uint32(0x80000000),
                                        T.uint32(0xFFFFFFFF),
                                    ),
                                    T.uint32(word_neg_inf),
                                )
                        rows.store_slot(outw, j, out_of_row, out, base)
                if cluster > 1:
                    # Peers read this CTA's counts, so it stays alive to the last read.
                    T.cluster_arrive()
                    T.cluster_wait()

        return _top_k_mask_main

    return _top_k_mask_func


class TopKMaskFwdKernel(Kernel, TopKMaskFwdInterface):
    """Mask each row of logits to its ``k`` largest, reading and writing each row once.

    A cluster of CTAs holds a row in registers and selects its k-th largest key one 8-bit
    digit at a time. One sample per thread brackets the digit values the k-th key can take,
    so a pass counts only the keys near it; a bracket that misses costs one more pass and
    changes no result. The select is exact for every input, ties and NaN included.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``cluster`` and ``slots``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    # Block width for the supported cluster sizes.
    _THREADS: ClassVar[int] = 1024
    # Most 16-byte vectors a thread holds before its registers cost the SM a resident block.
    _MAX_SLOTS: ClassVar[int] = 16

    @classmethod
    def refusal(cls, call: SamplingCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        return row_refusal(call, cls._THREADS, cls._MAX_SLOTS)

    def __init__(self, call: SamplingCall, config: dict | None = None):
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        self.kernel = _top_k_mask_kernel(call.batch, call.vocab, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return cluster_plan(self.call, self._THREADS, self._MAX_SLOTS)

    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._require_devices(logits=logits, k=k)
        out = torch.empty_like(logits)
        if logits.numel() == 0:
            return out
        self.program(self._compile)(logits.view(-1), k, out.view(-1))
        return out

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(**cfg)
