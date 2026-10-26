"""Fused Add + Norm forward kernels using TileLang.

FusedAddLayerNorm: y = LayerNorm(x + residual), also outputs (x + residual)
FusedAddRMSNorm:   y = RMSNorm(x + residual),   also outputs (x + residual)

Fusing the residual add into the normalization kernel eliminates one global
memory round-trip compared to separate add + norm.  Both kernels return dual
outputs ``(y, x + residual)`` so downstream residual connections can reuse the
pre-norm sum without recomputation.

FusedAddLayerNorm has two kernels. One holds a row, or on SM90 a cluster CTA's share of one,
in registers from the load through the store, through accesses of up to 16 bytes. The other
holds a row block in a register fragment, rows padded to 256 elements, and serves the rows
that queue on every SM while a thread of the first would hold more than its registers.
FusedAddRMSNorm reads it through 16-byte accesses and writes nothing until the reduction
has settled, so the two reads and the two writes reach memory as two runs rather than
interleaved; the chunk a CTA writes waits in shared memory in between. Its rows are padded
to 256 elements, so a 16-byte access always divides the row across a warp.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops._csrc import csrc_include
from tileops._kernels.constants import MAX_PORTABLE_CLUSTER_BLOCKS, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm._config import select_row_config, select_row_configs
from tileops._kernels.norm.call_spec import (
    FusedAddLayerNormFwdInterface,
    FusedAddRMSNormFwdInterface,
    LayerNormCall,
)
from tileops._kernels.tiling import ALIGNMENT, align_up
from tileops._utils import WARP_LANES, get_sm_count, get_sm_version

__all__ = ["FusedAddLayerNormHeldKernel", "FusedAddLayerNormKernel", "FusedAddRMSNormKernel"]


@functools.lru_cache(maxsize=32)
def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[4, 5])
    def _func(block_m, threads):
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            bias: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                add_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Both operands are read once and reduced along the row the thread
                # already owns, so neither has to pass through shared memory.
                if row_guard:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.if_then_else(
                            pid_m * block_m + i < M,
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = T.if_then_else(
                            pid_m * block_m + i < M,
                            residual[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                else:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = x[pid_m * block_m + i, j]
                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = residual[pid_m * block_m + i, j]

                # Fused add: compute (x + residual) in fp32, rounded to the storage dtype as
                # residual_out holds it, so the statistics and the output see one row.
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = T.cast(
                        T.cast(
                            T.cast(x_local[i, j], "float32") + T.cast(r_local[i, j], "float32"),
                            dtype,
                        ),
                        "float32",
                    )

                # Store pre-norm sum back in x_local (native dtype) for output
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = add_f32[i, j]

                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N)

                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = T.if_then_else(
                        j < N,
                        (add_f32[i, j] - mean_val[i]) * (add_f32[i, j] - mean_val[i]),
                        0.0,
                    )

                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(acc[i] / float(N) + eps)

                # Re-cast from x_local (which holds the pre-norm sum in native dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    if (not row_guard) or pid_m * block_m + i < M:
                        y[pid_m * block_m + i, j] = T.cast(
                            (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            * T.cast(weight[j], "float32")
                            + T.cast(bias[j], "float32"),
                            dtype,
                        )

                for i, j in T.Parallel(block_m, N_padded):
                    if (not row_guard) or pid_m * block_m + i < M:
                        residual_out[pid_m * block_m + i, j] = x_local[i, j]

        return main

    return _func


class FusedAddLayerNormKernel(Kernel, FusedAddLayerNormFwdInterface):
    """Fused Add + LayerNorm forward kernel.

    Computes ``y = LayerNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for the shared
    buffer the register-relief path fills; every other path holds the row in a
    register fragment from the load through the store.

    Serves the calls :class:`FusedAddLayerNormHeldKernel` refuses: rows that queue on
    every SM while a thread of that kernel would hold more than its registers.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: LayerNormCall) -> "str | None":
        if not FusedAddLayerNormHeldKernel._queued_past_registers(
            call.n, call.dtype, call.rows, call.arch, call.sm_count
        ):
            return f"{call.rows} rows of {call.n} fit a thread's registers or leave SMs idle"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is compiled on the first launch with that row count.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return select_row_config()

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype, num_buffers=2)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + LayerNorm over the trailing ``N`` elements.

        Flattening to 2-D rows happens here, as does the alignment padding the prim_func
        requires.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.
            bias: Affine shift holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, residual=residual, weight=weight, bias=bias)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        residual = residual.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        bias = bias.reshape(self.N)

        program = self.program(self._compile, rows.shape[0])

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
            bias = F.pad(bias, (0, pad))
        outputs = program(rows, residual, weight, bias)
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]

    def _compile(self, cfg: dict, m: int):
        """The program for *m* rows."""
        self.kernel = kernel = _fused_add_layer_norm_kernel(m, self.N, self.eps, self.dtype_str)
        return kernel(cfg["block_m"], cfg["threads"])


@functools.lru_cache(maxsize=32)
def _fused_add_layer_norm_held_kernel(M, N, eps, dtype, vec, cluster):
    """Return the factory for ``M`` rows held in registers, each split across ``cluster`` CTAs.

    The tensors are viewed as ``(M * cluster, N // cluster)``, one CTA a row of the view.
    ``vec`` divides a CTA's share; an access past the share's end is skipped whole.
    """
    chunk = N // cluster

    @tilelang.jit(out_idx=[4, 5], compile_flags=csrc_include("cluster_partials.h"))
    def _func(threads):
        steps = -(-chunk // (threads * vec))
        held = steps * vec
        warps = threads // WARP_LANES
        whole = chunk % (threads * vec) == 0

        @T.macro
        def row_sum(acc, part, peers, received, tx, rank, phase):
            """``acc[0]`` summed over the row into ``acc[0]`` of every thread of every CTA.

            Every CTA adds the same partials in the same order, so all hold one bit-identical
            sum, and a cluster's CTAs center and scale their shares alike.
            """
            for k in T.unroll(WARP_LANES.bit_length() - 1):
                acc[0] += T.shfl_xor(acc[0], T.shift_left(1, k))
            if warps > 1:
                if tx % WARP_LANES == 0:
                    part[phase, tx // WARP_LANES] = acc[0]
                T.sync_threads()
                acc[0] = T.float32(0)
                for w in T.unroll(warps):
                    acc[0] += part[phase, w]
            if cluster > 1:
                # st.async writes a peer's shared memory only, so a CTA keeps its own sum
                # in registers.
                if tx < cluster and tx != rank:
                    T.call_extern(  # lint: extern-target allow: st.async to a peer CTA has no TileLang primitive
                        "handle",
                        "tileops::send_partial",
                        T.address_of(peers[phase, rank]),
                        T.address_of(received[phase]),
                        tx,
                        acc[0],
                    )
                T.mbarrier_wait_parity(received[phase], 0)
                acc[1] = acc[0]
                acc[0] = T.float32(0)
                for peer in T.unroll(cluster):
                    acc[0] += T.if_then_else(peer == rank, acc[1], peers[phase, peer])

        @T.prim_func
        def main(
            x: T.Tensor[(M * cluster, chunk), dtype],
            residual: T.Tensor[(M * cluster, chunk), dtype],
            weight: T.Tensor[(cluster, chunk), dtype],
            bias: T.Tensor[(cluster, chunk), dtype],
            y: T.Tensor[(M * cluster, chunk), dtype],
            residual_out: T.Tensor[(M * cluster, chunk), dtype],
        ):
            with (
                T.Kernel(M, threads=threads)
                if cluster == 1
                else T.ClusterKernel(M * cluster, cluster_dims=cluster, threads=threads)
            ) as cta:
                tx = T.get_thread_binding()
                rank = cta % cluster
                a = T.alloc_local([held], dtype)
                b = T.alloc_local([held], dtype)
                w = T.alloc_local([held], dtype)
                c = T.alloc_local([held], dtype)
                d = T.alloc_local([held], "float32")
                acc = T.alloc_local([2], "float32")
                # One slot a warp for each of the two sums, so the second needs no barrier
                # to keep it off the first one's slots.
                part = T.alloc_shared([2, warps], "float32")
                peers = None
                received = None
                if cluster > 1:
                    peers = T.alloc_shared([2, cluster], "float32")
                    received = T.alloc_barrier([1, 1])
                    if tx == 0:
                        T.mbarrier_arrive_expect_tx(received[0], 4 * (cluster - 1))
                        T.mbarrier_arrive_expect_tx(received[1], 4 * (cluster - 1))
                    # Peers may send once this CTA's barriers are initialized; their wait
                    # for that overlaps the loads below.
                    T.cluster_arrive_relaxed()

                # All four operands are read before the first reduction, so their reads
                # share one trip to memory instead of the affine pair waiting on the sums.
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        for i in T.vectorized(vec):
                            a[k * vec + i] = x[cta, base + i]
                        for i in T.vectorized(vec):
                            b[k * vec + i] = residual[cta, base + i]
                        for i in T.vectorized(vec):
                            w[k * vec + i] = weight[rank, base + i]
                        for i in T.vectorized(vec):
                            c[k * vec + i] = bias[rank, base + i]

                # The row is the sum rounded to the storage dtype: residual_out holds it,
                # and the statistics and the output are taken over the same values.
                acc[0] = T.float32(0)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        for i in T.unroll(vec):
                            a[k * vec + i] = T.cast(
                                T.cast(a[k * vec + i], "float32")
                                + T.cast(b[k * vec + i], "float32"),
                                dtype,
                            )
                            d[k * vec + i] = T.cast(a[k * vec + i], "float32")
                            acc[0] += d[k * vec + i]
                # Written while the sum waits on the other warps.
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        for i in T.vectorized(vec):
                            residual_out[cta, base + i] = a[k * vec + i]
                if cluster > 1:
                    # Every peer's barriers are initialized past this wait.
                    T.cluster_wait()
                row_sum(acc, part, peers, received, tx, rank, 0)
                mean = acc[0] * (1.0 / N)

                # Centered before squaring: a row whose mean outgrows its spread keeps its
                # variance.
                acc[0] = T.float32(0)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        for i in T.unroll(vec):
                            d[k * vec + i] = d[k * vec + i] - mean
                            acc[0] += d[k * vec + i] * d[k * vec + i]
                row_sum(acc, part, peers, received, tx, rank, 1)
                rstd = T.rsqrt(acc[0] * (1.0 / N) + eps)

                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        for i in T.unroll(vec):
                            b[k * vec + i] = T.cast(
                                d[k * vec + i] * rstd * T.cast(w[k * vec + i], "float32")
                                + T.cast(c[k * vec + i], "float32"),
                                dtype,
                            )
                        for i in T.vectorized(vec):
                            y[cta, base + i] = b[k * vec + i]

        return main

    return _func


class FusedAddLayerNormHeldKernel(Kernel, FusedAddLayerNormFwdInterface):
    """Fused Add + LayerNorm forward kernel holding a row, or its share of one, in registers.

    On SM90 a row may be split across a cluster; see :meth:`_cluster`. ``config["threads"]``
    left ``None`` sizes the block per call by :meth:`_threads`, the untuned default, which
    :meth:`request_tune` tunes from.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    # 16-byte (or narrower) accesses in a CTA's share of the build tuning measures.
    _tune_accesses: Optional[int] = None

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is compiled on the first launch with that row count.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": None}

    @property
    def autotune_configs(self) -> list[dict]:
        """Every power-of-two block from 128 to 1024 threads that the tuned build's share fills.

        Tuning measures the first row count built, seeded with its :meth:`_threads`, and
        publishes one width for every later launch.
        """
        accesses = self._tune_accesses or -(-self.N // self._vec(self.N, self.dtype, 1))
        return [
            {"threads": t}
            for t in (128, 256, 512, 1024)
            if t == 128 or t <= -(-accesses // WARP_LANES) * WARP_LANES
        ]

    @staticmethod
    def _cluster(n: int, dtype: torch.dtype, rows: int, arch: int, sm_count: int) -> int:
        """CTAs each row is split across: the width whose busiest SM finishes first.

        An SM's cost is its CTAs times each one's share of ``x`` at one SM's bandwidth, plus
        an exchange per CTA when split. A split needs SM90, 16-byte accesses and a share that
        covers its own latency; ties take the narrower width.
        """
        # Bytes of a row one SM reads per microsecond when it holds the whole row.
        sm_bytes_per_us = 10.4 * 1024
        # Microseconds the two cross-CTA exchanges of a split row add to its CTA.
        exchange_us = 0.6
        # Smallest share that covers a split CTA's latency: alone on its SM, and queued.
        min_share_alone, min_share_queued = 2048, 6144
        elem = dtype.itemsize
        vec = VECTOR_ACCESS_BYTES // elem
        best_cost, best = None, 1
        for cluster in (1, 2, 4, 8):
            if cluster > 1 and (arch < 90 or cluster > MAX_PORTABLE_CLUSTER_BLOCKS):
                break
            if n % (cluster * vec):
                continue
            share = n * elem / cluster
            ctas_per_sm = -(-rows * cluster // sm_count)
            if cluster > 1 and share < (min_share_alone if ctas_per_sm == 1 else min_share_queued):
                continue
            cost = ctas_per_sm * (share / sm_bytes_per_us + (exchange_us if cluster > 1 else 0))
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, cluster
        return best

    @staticmethod
    def _threads(n: int, cluster: int, vec: int, rows: int, sm_count: int) -> int:
        """Threads a CTA takes for its share of ``n / cluster`` elements, untuned.

        The whole-warp count dividing the share's accesses, so no round leaves threads idle,
        nearest a target: one access a thread on a split share, two on an SM's only CTA,
        else 128 so more CTAs share an SM.
        """
        accesses = -(-n // (cluster * vec))
        ctas_per_sm = -(-rows * cluster // sm_count)
        if cluster > 1:
            target = accesses if accesses <= 256 else (256 if accesses <= 512 else 128)
        elif ctas_per_sm == 1:
            target = min(512, accesses // 2)
        else:
            target = 128
        max_threads = 1024
        max_accesses = 32
        dividing = [
            w
            for w in range(WARP_LANES, max_threads + 1, WARP_LANES)
            if accesses % w == 0 and accesses // w <= max_accesses
        ]
        if dividing:
            return min(dividing, key=lambda w: (abs(w - target), w))
        threads = min(max(target, WARP_LANES), -(-accesses // WARP_LANES) * WARP_LANES)
        return -(-threads // WARP_LANES) * WARP_LANES

    @staticmethod
    def _vec(n: int, dtype: torch.dtype, cluster: int) -> int:
        """Elements of the widest access of at most 16 bytes that divides a CTA's share."""
        widest = VECTOR_ACCESS_BYTES // dtype.itemsize
        share = n // cluster
        return next(v for v in (widest, widest // 2, widest // 4, 1) if v and share % v == 0)

    @classmethod
    def _queued_past_registers(
        cls, n: int, dtype: torch.dtype, rows: int, arch: int, sm_count: int
    ) -> bool:
        """Whether CTAs queue on an SM while each thread holds more than its registers.

        A thread holds the row's ``x``, ``residual``, weight and bias in the storage dtype
        and the centered row in fp32 for every element of its share. When that exceeds its
        registers and CTAs queue on an SM, the fragment kernel, which holds two fewer
        operands a thread, serves the call.
        """
        # A thread's registers: 255 of 4 bytes each.
        register_bytes = 255 * 4
        cluster = cls._cluster(n, dtype, rows, arch, sm_count)
        threads = cls._threads(n, cluster, cls._vec(n, dtype, cluster), rows, sm_count)
        held_bytes = -(-n // (cluster * threads)) * (4 * dtype.itemsize + 4)
        return rows * cluster > sm_count and held_bytes > register_bytes

    @classmethod
    def refusal(cls, call: LayerNormCall) -> "str | None":
        if cls._queued_past_registers(call.n, call.dtype, call.rows, call.arch, call.sm_count):
            return f"{call.rows} rows of {call.n} queue on every SM past a thread's registers"
        return super().refusal(call)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + LayerNorm over the trailing ``N`` elements.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.
            bias: Affine shift holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, residual=residual, weight=weight, bias=bias)

        original_shape = x.shape
        m = x.numel() // self.N
        index = x.device.index
        sm_count = get_sm_count(index)
        cluster = self._cluster(self.N, self.dtype, m, get_sm_version(index), sm_count)
        chunk = self.N // cluster
        program = self.program(self._compile, m, cluster, sm_count)
        outputs = program(
            x.reshape(m * cluster, chunk),
            residual.reshape(m * cluster, chunk),
            weight.reshape(cluster, chunk),
            bias.reshape(cluster, chunk),
        )
        return [out.reshape(original_shape) for out in outputs]

    def _compile(self, cfg: dict, m: int, cluster: int, sm_count: int):
        """The program for *m* rows, each split across *cluster* CTAs, on *sm_count* SMs."""
        share = self.N // cluster
        vec = self._vec(self.N, self.dtype, cluster)
        if cfg["threads"] is None:
            cfg = {**cfg, "threads": self._threads(self.N, cluster, vec, m, sm_count)}
        kernel = _fused_add_layer_norm_held_kernel(
            m, self.N, self.eps, self.dtype_str, vec, cluster
        )
        program = kernel(cfg["threads"])
        # Set after the build succeeds, so tuning never reads one that failed.
        self.kernel = kernel
        if self._tune_target is None:
            self._tune_target = (kernel, cfg)
            self._tune_accesses = -(-share // vec)
        return program


# Fused Add + RMSNorm kernel


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_kernel(M, N, eps, dtype, splits, vec):
    """Return the factory for one CTA per ``(row, chunk)`` pair.

    A CTA reduces a whole row and writes one chunk of it. Nothing is written until the
    reduction has settled, so the two reads and the two writes reach memory as two runs
    rather than interleaved.

    With ``splits > 1`` the CTAs sharing a row each reduce the whole row, paying reads
    that L2 serves to put a single row on more than one SM in one launch.
    """
    N_padded = align_up(N, ALIGNMENT)
    chunk = N_padded // splits

    @tilelang.jit(out_idx=[3, 4])
    def _func(threads):
        own = chunk // (threads * vec)  # accesses this CTA reduces and writes
        rest = (N_padded - chunk) // (threads * vec)  # accesses it only reduces
        warps = threads // WARP_LANES
        inv_n = 1.0 / N  # N, not N_padded: a padded column holds zero

        @T.macro
        def block_rsqrt(tx, acc, warp_sums, rrms):
            """Fold one fp32 partial per thread into ``rrms[0]``."""
            for step in T.serial(WARP_LANES.bit_length() - 1):
                acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
            if tx % WARP_LANES == 0:
                warp_sums[tx // WARP_LANES] = acc[0]
            T.sync_threads()
            # One thread adds the per-warp totals: a block holds at most 32 warps.
            if tx == 0:
                acc[0] = T.cast(0, "float32")
                for w in T.serial(warps):
                    acc[0] += warp_sums[w]
                rrms[0] = T.rsqrt(acc[0] * inv_n + eps)
            T.sync_threads()

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(M * splits, threads=threads) as pid:
                row = pid // splits
                mine = (pid % splits) * chunk
                tx = T.get_thread_binding()
                a = T.alloc_local([vec], dtype)
                b = T.alloc_local([vec], dtype)
                summed = T.alloc_shared([chunk], dtype)
                acc = T.alloc_local([1], "float32")
                warp_sums = T.alloc_shared([warps], "float32")
                rrms = T.alloc_shared([1], "float32")

                acc[0] = T.cast(0, "float32")
                for step in T.serial(own):
                    base = (step * threads + tx) * vec
                    for i in T.vectorized(vec):
                        a[i] = x[row, mine + base + i]
                    for i in T.vectorized(vec):
                        b[i] = residual[row, mine + base + i]
                    for i in T.serial(vec):
                        # Adding in the storage dtype is bit-identical to rounding the
                        # f32 sum: f32 holds the exact sum of two 16-bit floats.
                        b[i] = a[i] + b[i]
                        v = T.cast(b[i], "float32")
                        acc[0] += v * v
                    for i in T.vectorized(vec):
                        summed[base + i] = b[i]

                # The rest of the row, reduced but not written. Starting past this CTA's
                # own chunk keeps the CTAs sharing a row off the same lines at once.
                for step in T.serial(rest):
                    base = (mine + chunk + (step * threads + tx) * vec) % N_padded
                    for i in T.vectorized(vec):
                        a[i] = x[row, base + i]
                    for i in T.vectorized(vec):
                        b[i] = residual[row, base + i]
                    for i in T.serial(vec):
                        v = T.cast(a[i] + b[i], "float32")
                        acc[0] += v * v

                block_rsqrt(tx, acc, warp_sums, rrms)

                scale = rrms[0]
                for step in T.serial(own):
                    base = (step * threads + tx) * vec
                    for i in T.vectorized(vec):
                        a[i] = summed[base + i]
                    for i in T.vectorized(vec):
                        b[i] = weight[mine + base + i]
                    for i in T.serial(vec):
                        b[i] = T.cast(
                            T.cast(a[i], "float32") * scale * T.cast(b[i], "float32"), dtype
                        )
                    for i in T.vectorized(vec):
                        residual_out[row, mine + base + i] = a[i]
                    for i in T.vectorized(vec):
                        y[row, mine + base + i] = b[i]

        return main

    return _func


class FusedAddRMSNormKernel(Kernel, FusedAddRMSNormFwdInterface):
    """Fused Add + RMSNorm forward kernel.

    Computes ``y = RMSNorm(x + residual)`` and returns both ``y`` and ``x + residual``,
    in the four passes over the row that two inputs and two outputs require.

    Supports SM80+ architectures. One CTA reduces a row and writes one chunk of it,
    through 16-byte accesses, parking that chunk in shared memory while the reduction
    settles.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    # Rows at or below which a row-per-CTA call takes the narrow block: while the shared
    # park is small a narrow block keeps more CTAs resident on an SM, and past this width
    # the park caps residency whatever the block is.
    _NARROW_ROW_COLUMNS = 4096
    _NARROW_THREADS = 128
    _WIDE_THREADS = 512

    # Accesses each CTA makes over the row it reduces, when several CTAs share one row.
    _ROW_ACCESSES = 4

    # Most CTAs a single row is cut across. A power of two, so halving reaches every
    # candidate the row may admit.
    _ROW_SPLITS = 4

    # Threads CUDA allows in one block.
    _MAX_THREADS = 1024

    def _widest_dividing(self, target: int) -> int:
        """The widest block at or below *target* cutting the row into whole accesses.

        Floored at one warp, which a row padded to :data:`ALIGNMENT` always admits.
        """
        threads = self._MAX_THREADS
        while threads > WARP_LANES and (threads > target or (self.N_padded // self._vec) % threads):
            threads //= 2
        return threads

    def _row_threads(self) -> int:
        """Block width for a call holding one row per CTA."""
        narrow = self.N_padded <= self._NARROW_ROW_COLUMNS
        return self._widest_dividing(self._NARROW_THREADS if narrow else self._WIDE_THREADS)

    def _split_row_threads(self) -> int:
        """Block width for a single-row call, where the CTAs share one row.

        Sized by the whole row every CTA reduces rather than the chunk it writes,
        which is why it differs from :meth:`_row_threads`.
        """
        return self._widest_dividing(self.N_padded // (self._vec * self._ROW_ACCESSES))

    def _row_splits(self, threads: int, rows: int) -> int:
        """CTAs to put on one row -- more than one only when the call is a single row.

        A call with rows to spare fills the grid with them. A single row instead leaves
        one CTA holding it and the rest of the device idle.
        """
        if rows != 1:
            return 1
        accesses = self.N_padded // (self._vec * threads)
        splits = self._ROW_SPLITS
        while splits > 1 and accesses % splits:
            splits //= 2
        return splits

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is compiled on the first launch with that row count.

        Args:
            N: Hidden size the rows are normalized over.
            eps: Epsilon for numerical stability.
            dtype: Element type the rows are stored in.
            config: Optional ``{"threads": ...}`` override, for the row-per-CTA case.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self._vec = VECTOR_ACCESS_BYTES // 2  # Eight 16-bit elements per access.
        self.N_padded = align_up(N, ALIGNMENT)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": self._row_threads()}

    @property
    def autotune_configs(self) -> list[dict]:
        """The one width :meth:`_row_threads` gives, so a tuned build equals an untuned one.

        The autotuner times repeats of one candidate back to back, which reads the row
        out of L2 and so ranks a kernel this one never runs as.
        """
        return [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + RMSNorm over the trailing ``N`` elements.

        Flattening to 2-D rows happens here, as does the alignment padding the prim_func
        requires.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, residual=residual, weight=weight)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        residual = residual.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        program = self.program(self._compile, rows.shape[0])

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
        outputs = program(rows, residual, weight)
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]

    def _compile(self, cfg: dict, m: int):
        """The program for *m* rows."""
        # A single row puts several CTAs on it, which wants a different width from the
        # row-per-CTA case and so does not read the configured one.
        threads = self._split_row_threads() if m == 1 else cfg["threads"]
        splits = self._row_splits(threads, m)

        self.kernel = kernel = _fused_add_rms_norm_kernel(
            m, self.N, self.eps, self.dtype_str, splits, self._vec
        )
        return kernel(threads)
