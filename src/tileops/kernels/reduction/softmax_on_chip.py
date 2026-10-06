"""Softmax / log-softmax of long rows read once and kept on chip.

A CTA holds its part of a row partly in registers and the rest in shared memory, reads it
once, and writes the result from there. A row larger than one SM's on-chip storage is split
across a thread-block cluster; the CTAs of a cluster exchange their partial ``(max, sum)``
through each other's shared memory.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._csrc import csrc_path
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.reduction._primitives import (
    LOG2E,
    exp_shifted,
    restore_same_shape,
)
from tileops.kernels.reduction.call_spec import SoftmaxCall
from tileops.kernels.reduction.softmax import _SoftmaxKernelBase
from tileops.utils import WARP_LANES

__all__ = ["SoftmaxOnChipKernel"]


@functools.lru_cache(maxsize=32)
def _softmax_on_chip_kernel(
    M, N, op_kind, dtype, out_dtype, cluster, threads, held_vectors, ctas_per_sm
):
    """Build the program for ``(M, N)`` rows, each split across ``cluster`` CTAs.

    A CTA takes ``N // cluster`` columns: ``held_vectors`` 16-byte vectors a thread in
    registers, the rest staged in shared memory.
    The buffers are viewed as ``(M * cluster, N // cluster)``, one CTA a row of the view.
    """
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    chunk = N // cluster
    held = threads * held_vectors * vec
    staged = chunk - held
    staged_vectors = staged // (threads * vec)
    warps = threads // WARP_LANES
    # Floats a CTA's partial (max, sum) takes in a peer's shared memory.
    slot = 2
    # Warp maxima and sums, then one slot a CTA of the cluster; a CTA's own stays unused.
    own = 2 * warps
    neg_inf = float("-inf")
    # The output's exponent is raised by this many before the exp2 and the result scaled
    # back by a multiply, which keeps a subnormal probability.
    exp_bias = 64
    # One MUFU.EX2, which flushes a subnormal result to zero. ``exp2f`` guards the same
    # instruction with a test and two multiplies to keep it.
    prelude = r"""
static __device__ __forceinline__ float tl_approx_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
"""

    @tilelang.jit(out_idx=[1], compile_flags=["-include", csrc_path("cluster_partials.h")])
    def build():
        def fold(peak, total, other_peak, other_total):
            """The ``(max, sum)`` of two partial pairs; two empty ones stay empty."""
            top = T.max(peak, other_peak)
            both = total * exp_shifted(peak, top) + other_total * exp_shifted(other_peak, top)
            return top, T.if_then_else(top == neg_inf, T.cast(0, "float32"), both)

        def term(value, peak):
            """``exp(value - peak)`` of one element for the sum, which is at least one: a
            subnormal term reads as zero."""
            return T.call_extern(
                "float32", "tl_approx_exp2", (T.cast(value, "float32") - peak) * LOG2E
            )

        def row_scale(total):
            """What a row's result takes from its sum: ``log2(sum)`` less the exponent
            bias, or ``log(sum)``."""
            if op_kind == "softmax":
                return T.log2(total) - float(exp_bias)
            return T.log(total)

        def finish(value, peak, scale):
            """The result for one element: exp(x - max) / sum, or x - max - log(sum)."""
            if op_kind == "softmax":
                # The division is one more term of the exponent, which the scaling
                # takes in the same FFMA.
                biased = T.call_extern(
                    "float32", "tl_approx_exp2", (T.cast(value, "float32") - peak) * LOG2E - scale
                )
                return biased * (2.0**-exp_bias)
            return T.cast(value, "float32") - peak - scale

        @T.prim_func
        def main(
            x: T.Tensor[(M * cluster, chunk), dtype],
            y: T.Tensor[(M * cluster, chunk), out_dtype],
        ):
            with T.ClusterKernel(
                M * cluster, threads=threads, cluster_dims=cluster, prelude=prelude
            ) as cta:
                T.annotate_min_blocks_per_sm(ctas_per_sm)
                tx = T.get_thread_binding()
                rank = cta % cluster
                values = T.alloc_local([held_vectors * vec], dtype)
                piece = T.alloc_local([vec], dtype)
                peaks = T.alloc_local([vec], dtype)
                out = T.alloc_local([vec], out_dtype)
                stat = T.alloc_local([2], "float32")  # running (max, sum)
                peer = T.alloc_local([2], "float32")
                tile = T.alloc_shared((1, staged), dtype)
                # One buffer for the warp pairs and the partials: a peer writes its pair while
                # this CTA still folds its warps, so no slot may share their storage.
                sums = T.alloc_shared([own + slot * cluster], "float32")
                # One arrival, here; the peers' stores complete its bytes. A store reads
                # registers, so a CTA may leave once it has received every partial.
                received = T.alloc_barrier([1])
                if tx == 0:
                    T.call_extern(
                        "handle",
                        "tl::tileops_expect_partials",
                        T.address_of(received[0]),
                        8 * (cluster - 1),
                    )
                # Peers may write here once this CTA's barrier is initialized; their wait
                # for that overlaps the loads below.
                T.cluster_arrive_relaxed()

                for v in T.unroll(held_vectors):
                    for i in T.vectorized(vec):
                        values[v * vec + i] = x[cta, (v * threads + tx) * vec + i]
                T.copy(x[cta : cta + 1, held:chunk], tile)

                # The maximum is one of the values: taken in their own type, two 16-bit
                # elements a lane-wide instruction.
                for i in T.vectorized(vec):
                    peaks[i] = values[i]
                for v in T.unroll(1, held_vectors):
                    for i in T.vectorized(vec):
                        peaks[i] = T.max(peaks[i], values[v * vec + i])
                for v in T.serial(staged_vectors):
                    for i in T.vectorized(vec):
                        piece[i] = tile[0, (v * threads + tx) * vec + i]
                    for i in T.vectorized(vec):
                        peaks[i] = T.max(peaks[i], piece[i])
                stat[0] = T.cast(peaks[0], "float32")
                for i in T.unroll(1, vec):
                    stat[0] = T.max(stat[0], T.cast(peaks[i], "float32"))
                stat[1] = T.cast(0, "float32")
                if stat[0] != neg_inf:
                    for j in T.unroll(held_vectors * vec):
                        stat[1] += term(values[j], stat[0])
                    for v in T.serial(staged_vectors):
                        for i in T.vectorized(vec):
                            piece[i] = tile[0, (v * threads + tx) * vec + i]
                        for i in T.unroll(vec):
                            stat[1] += term(piece[i], stat[0])
                for step in T.unroll(WARP_LANES.bit_length() - 1):
                    peer[0] = T.shfl_xor(stat[0], T.shift_left(1, step))
                    peer[1] = T.shfl_xor(stat[1], T.shift_left(1, step))
                    stat[0], stat[1] = fold(stat[0], stat[1], peer[0], peer[1])
                if tx % WARP_LANES == 0:
                    sums[2 * (tx // WARP_LANES)] = stat[0]
                    sums[2 * (tx // WARP_LANES) + 1] = stat[1]
                T.sync_threads()
                stat[0] = T.cast(neg_inf, "float32")
                stat[1] = T.cast(0, "float32")
                for w in T.unroll(warps):
                    stat[0], stat[1] = fold(stat[0], stat[1], sums[2 * w], sums[2 * w + 1])

                # Every peer's barrier is initialized past this wait.
                T.cluster_wait()
                # st.async takes a peer's memory only, so a CTA keeps its own pair in
                # registers.
                if tx < cluster and tx != rank:
                    T.call_extern(
                        "handle",
                        "tl::tileops_send_partial2",
                        T.address_of(sums[own + slot * rank]),
                        T.address_of(received[0]),
                        tx,
                        stat[0],
                        stat[1],
                    )
                T.mbarrier_wait_parity(received[0], 0)
                for p in T.unroll(cluster):
                    if p != rank:
                        stat[0], stat[1] = fold(
                            stat[0],
                            stat[1],
                            sums[own + slot * p],
                            sums[own + slot * p + 1],
                        )
                scale = row_scale(stat[1])

                for v in T.unroll(held_vectors):
                    for i in T.unroll(vec):
                        out[i] = T.cast(
                            finish(values[v * vec + i], stat[0], scale),
                            out_dtype,
                        )
                    for i in T.vectorized(vec):
                        y[cta, (v * threads + tx) * vec + i] = out[i]
                for v in T.serial(staged_vectors):
                    for i in T.vectorized(vec):
                        piece[i] = tile[0, (v * threads + tx) * vec + i]
                    for i in T.unroll(vec):
                        out[i] = T.cast(finish(piece[i], stat[0], scale), out_dtype)
                    for i in T.vectorized(vec):
                        y[cta, held + (v * threads + tx) * vec + i] = out[i]

        return main

    return build


class SoftmaxOnChipKernel(_SoftmaxKernelBase):
    """Softmax / log-softmax reading each row once from registers and shared memory.

    A row is split across the smallest power-of-two cluster of two to eight CTAs whose
    CTAs each take at most 64 KB of a 16-bit row or 32 KB of an fp32 one. Several CTAs
    share an SM, so one computes its exponentials while another loads or stores. Rows
    narrower than 32768 elements, and rows :class:`SoftmaxSplitKernel` splits, stay with
    the other kernels.
    """

    supported_archs = [90]
    preferred_over = frozenset({"softmax_streaming"})

    @classmethod
    def _plan(cls, call: SoftmaxCall) -> Optional[tuple]:
        """``(cluster, threads, held_vectors, ctas_per_sm)`` for *call*'s rows, or ``None``."""
        elem = call.dtype.itemsize
        # A 16384-element fp32 row runs faster on SoftmaxKernel.
        if call.n < 32768 or cls.split_seg_n(call) != 0:
            return None
        # A 16-bit element costs the same exponential as a 32-bit one in half the bytes,
        # so a 16-bit CTA takes twice the bytes for the same work.
        cta_bytes = 64 * 1024 if elem == 2 else 32 * 1024
        cluster = 2
        while call.n * elem > cluster * cta_bytes and cluster < 8:
            cluster *= 2
        if call.n % cluster:
            return None
        share = call.n // cluster * elem
        # (threads, bytes a CTA holds in registers, CTAs an SM) by the share a CTA takes;
        # the rest of a share is staged in shared memory.
        if share <= 32 * 1024:
            threads, held_bytes, ctas_per_sm = 128, 16 * 1024, 4
        elif share <= 64 * 1024:
            threads, held_bytes, ctas_per_sm = 256, 32 * 1024, 3
        else:
            threads, held_bytes, ctas_per_sm = 128, 64 * 1024, 1
        staged = share - held_bytes
        if staged <= 0 or share % (threads * VECTOR_ACCESS_BYTES):
            return None
        # Each CTA keeps 1 KB for its sums and barrier rather than the row.
        if staged * ctas_per_sm > call.smem_budget - ctas_per_sm * 1024:
            return None
        return cluster, threads, held_bytes // (threads * VECTOR_ACCESS_BYTES), ctas_per_sm

    @classmethod
    def applies(cls, call: SoftmaxCall) -> bool:
        return cls._plan(call) is not None

    def __init__(self, call: SoftmaxCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        self.cluster, threads, held_vectors, ctas_per_sm = self._plan(call)
        self.kernel = _softmax_on_chip_kernel(
            call.m,
            call.n,
            call.op_kind,
            self.dtype_str,
            self.dtype_to_str(call.out_dtype),
            self.cluster,
            threads,
            held_vectors,
            ctas_per_sm,
        )
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of the contiguous input *x*."""
        rows = self._rows(x)
        m = rows.shape[0]
        chunk = self.call.n // self.cluster
        y = self.kernel()(rows.reshape(m * self.cluster, chunk))
        return restore_same_shape(y.reshape(m, self.call.n), self.call.shape, (self.call.axis,))
