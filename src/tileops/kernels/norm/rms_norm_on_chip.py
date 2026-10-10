"""RMS normalization of rows too wide for a register-held block, read once and kept on chip.

A CTA holds its part of a row partly in registers and the rest in shared memory, reads it
once, and writes the result from there. A row larger than one SM's on-chip storage is split
across a thread-block cluster; the CTAs of a cluster exchange their partial sums of squares
through each other's shared memory.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import MAX_PORTABLE_CLUSTER_BLOCKS, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops.utils import WARP_LANES

__all__ = ["RMSNormOnChipKernel"]


@functools.lru_cache(maxsize=32)
def _rms_norm_on_chip_kernel(
    M, N, eps, dtype, has_weight, cluster, threads, held_vectors, async_stage
):
    """Build the program for ``(M, N)`` rows, each split across ``cluster`` CTAs.

    A CTA takes ``N // cluster`` columns: ``held_vectors`` 16-byte vectors a thread in
    registers, the rest staged in shared memory, by cp.async where ``async_stage``.
    The buffers are viewed as ``(M * cluster, N // cluster)``, one CTA a row of the view.
    """
    vec = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    chunk = N // cluster
    held = threads * held_vectors * vec
    staged = chunk - held
    staged_vectors = staged // (threads * vec)
    warps = threads // WARP_LANES
    # Warp sums, then one slot a CTA of the cluster; a CTA's own stays unused.
    own = warps
    clustered = cluster > 1

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("cluster_partials.h"))
    def build():
        def normalize(value, rrms, scale):
            """*value* scaled by the row's reciprocal RMS and, where there is one, the weight."""
            normed = T.cast(value, "float32") * rrms
            return normed * T.cast(scale, "float32") if has_weight else normed

        @T.prim_func
        def main(
            x: T.Tensor[(M * cluster, chunk), dtype],
            weight: T.Tensor[(cluster if has_weight else 1, chunk if has_weight else 1), dtype],
            y: T.Tensor[(M * cluster, chunk), dtype],
        ):
            with T.ClusterKernel(M * cluster, threads=threads, cluster_dims=cluster) as cta:
                tx = T.get_thread_binding()
                rank = cta % cluster
                values = T.alloc_local([held_vectors * vec], dtype)
                piece = T.alloc_local([vec], dtype)
                scale = T.alloc_local([vec], dtype)
                total = T.alloc_local([1], "float32")
                tile = T.alloc_shared((1, staged), dtype)
                # One buffer for the warp sums and the partials: a peer writes its partial
                # while this CTA still folds its warps, so no slot may share their storage.
                sums = T.alloc_shared([own + cluster], "float32")
                if clustered:
                    # One arrival, here; the peers' stores complete its bytes. A store reads
                    # registers, so a CTA may leave once it has received every partial.
                    received = T.alloc_barrier([1])
                    if tx == 0:
                        T.call_extern(
                            "handle",
                            "tileops::expect_partials",
                            T.address_of(received[0]),
                            4 * (cluster - 1),
                        )
                    # Peers may write here once this CTA's barrier is initialized; their wait
                    # for that overlaps the loads below.
                    T.cluster_arrive_relaxed()

                if async_stage:
                    T.async_copy(x[cta : cta + 1, held:chunk], tile)
                for v in T.unroll(held_vectors):
                    for i in T.vectorized(vec):
                        values[v * vec + i] = x[cta, (v * threads + tx) * vec + i]
                if async_stage:
                    T.ptx_wait_group(0)
                    T.sync_threads()
                else:
                    T.copy(x[cta : cta + 1, held:chunk], tile)

                total[0] = T.cast(0, "float32")
                for j in T.unroll(held_vectors * vec):
                    value = T.cast(values[j], "float32")
                    total[0] += value * value
                for v in T.unroll(staged_vectors):
                    for i in T.vectorized(vec):
                        piece[i] = tile[0, (v * threads + tx) * vec + i]
                    for i in T.unroll(vec):
                        value = T.cast(piece[i], "float32")
                        total[0] += value * value
                for step in T.unroll(WARP_LANES.bit_length() - 1):
                    total[0] += T.shfl_xor(total[0], T.shift_left(1, step))
                if tx % WARP_LANES == 0:
                    sums[tx // WARP_LANES] = total[0]
                T.sync_threads()
                total[0] = T.cast(0, "float32")
                for w in T.unroll(warps):
                    total[0] += sums[w]

                if clustered:
                    # Every peer's barrier is initialized past this wait.
                    T.cluster_wait()
                    # st.async takes a peer's memory only, so a CTA keeps its own sum in
                    # registers.
                    if tx < cluster and tx != rank:
                        T.call_extern(
                            "handle",
                            "tileops::send_partial",
                            T.address_of(sums[own + rank]),
                            T.address_of(received[0]),
                            tx,
                            total[0],
                        )
                    T.mbarrier_wait_parity(received[0], 0)
                    for peer in T.unroll(cluster):
                        if peer != rank:
                            total[0] += sums[own + peer]
                rrms = T.rsqrt(total[0] / float(N) + eps)

                for v in T.unroll(held_vectors):
                    if has_weight:
                        for i in T.vectorized(vec):
                            scale[i] = weight[rank, (v * threads + tx) * vec + i]
                    for i in T.unroll(vec):
                        piece[i] = T.cast(normalize(values[v * vec + i], rrms, scale[i]), dtype)
                    for i in T.vectorized(vec):
                        y[cta, (v * threads + tx) * vec + i] = piece[i]
                for v in T.unroll(staged_vectors):
                    if has_weight:
                        for i in T.vectorized(vec):
                            scale[i] = weight[rank, held + (v * threads + tx) * vec + i]
                    for i in T.vectorized(vec):
                        piece[i] = tile[0, (v * threads + tx) * vec + i]
                    for i in T.unroll(vec):
                        piece[i] = T.cast(normalize(piece[i], rrms, scale[i]), dtype)
                    for i in T.vectorized(vec):
                        y[cta, held + (v * threads + tx) * vec + i] = piece[i]

        return main

    return build


class RMSNormOnChipKernel(Kernel, RMSNormFwdInterface):
    """RMS normalization reading each row once, for rows wider than 40960 16-bit elements.

    1024 threads a CTA hold a share of the row in registers and the rest in shared memory.
    A row that one CTA cannot hold is split across the smallest power-of-two cluster whose
    CTAs can, at most a portable cluster wide.
    """

    supported_archs = [90]
    preferred_over = frozenset({"rms_norm_streaming"})

    _THREADS = 1024

    @classmethod
    def _plan(cls, n: int, dtype: torch.dtype, smem_budget: int) -> Optional[tuple]:
        """``(cluster, held_vectors, async_stage)`` for rows of *n*, or ``None``."""
        elem = dtype.itemsize
        vec = VECTOR_ACCESS_BYTES // elem
        # Rows of at most 80 KB stay with the register-held kernel.
        if n * elem <= 80 * 1024:
            return None
        # A CTA's share past 128 KB stages its rest by cp.async.
        async_stage_bytes = 128 * 1024
        # Shared memory a CTA keeps for its sums and barrier rather than the row.
        reserved_bytes = 1024
        cluster = 1
        while cluster <= MAX_PORTABLE_CLUSTER_BLOCKS:
            chunk = n // cluster
            share = chunk * elem
            async_stage = share > async_stage_bytes
            # 16-byte vectors a thread holds in registers.
            held_vectors = 4 if async_stage else 3
            staged = chunk - cls._THREADS * held_vectors * vec
            if (
                n % cluster == 0
                and chunk % (cls._THREADS * vec) == 0
                and staged > 0
                and staged * elem <= smem_budget - reserved_bytes
            ):
                return cluster, held_vectors, async_stage
            cluster *= 2
        return None

    @classmethod
    def refusal(cls, call: LayerNormCall) -> "str | None":
        if cls._plan(call.n, call.dtype, call.smem_budget) is None:
            return f"a row of {call.n} {call.dtype} elements does not fit on chip"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype, call.smem_budget)
        return identity, lambda: cls(*identity)

    def __init__(self, n: int, eps: float, dtype: torch.dtype, smem_budget: int) -> None:
        super().__init__()
        self.n, self.eps, self.dtype = n, eps, dtype
        self.cluster, self.held_vectors, self.async_stage = self._plan(n, dtype, smem_budget)
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        x = vector_aligned(x)
        weight = vector_aligned(weight)
        rows = x.reshape(-1, self.n)
        m = rows.shape[0]
        has_weight = weight is not None
        chunk = self.n // self.cluster
        weight = weight.reshape(self.cluster, chunk) if has_weight else rows.new_empty(1, 1)
        self.kernel = _rms_norm_on_chip_kernel(
            m,
            self.n,
            self.eps,
            self.dtype_str,
            has_weight,
            self.cluster,
            self._THREADS,
            self.held_vectors,
            self.async_stage,
        )
        y = self.kernel()(rows.reshape(m * self.cluster, chunk), weight)
        return y.reshape_as(x)
