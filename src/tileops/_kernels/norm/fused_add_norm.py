"""Fused Add + Norm forward kernels using TileLang, and the register-held LayerNorm kernels.

FusedAddLayerNorm: y = LayerNorm(x + residual), also outputs (x + residual)
FusedAddRMSNorm:   y = RMSNorm(x + residual),   also outputs (x + residual)

The register-held kernel family also serves LayerNorm and AdaLayerNorm (with and without the
Zero gate) unfused; ``layer_norm.py`` and ``ada_layer_norm.py`` dispatch to it.

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

from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    MAX_PORTABLE_CLUSTER_BLOCKS,
    SM90_REGISTERS,
    VECTOR_ACCESS_BYTES,
)
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm._config import (
    fold_chains,
    make_corrected_row_reduce,
    make_raw_sums_hold,
    pairwise_sum,
    read_first,
    select_row_config,
    select_row_configs,
    warp_sum,
)
from tileops._kernels.norm.call_spec import (
    AdaLayerNormFwdInterface,
    AdaLayerNormZeroFwdInterface,
    FusedAddLayerNormFwdInterface,
    FusedAddRMSNormFwdInterface,
    LayerNormCall,
    LayerNormFwdInterface,
)
from tileops._kernels.tiling import ALIGNMENT, align_up
from tileops._utils import WARP_LANES

__all__ = [
    "AdaLayerNormHeldKernel",
    "AdaLayerNormZeroHeldKernel",
    "FusedAddLayerNormHeldKernel",
    "FusedAddLayerNormKernel",
    "FusedAddRMSNormKernel",
    "LayerNormHeldKernel",
]


@target_cache(maxsize=32)
def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[4, 5])
    def _func(block_m, threads):
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0
        row_reduce = make_corrected_row_reduce(block_m, N, N_padded, eps)

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
                centered = T.alloc_fragment((block_m, N_padded), "float32")
                squares = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                acc_squares = T.alloc_fragment((block_m,), "float32")
                shift = T.alloc_fragment((block_m,), "float32")
                centre = T.alloc_fragment((block_m,), "float32")
                offset = T.alloc_fragment((block_m,), "float32")
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

                # Center on a row element to keep the offset out of the reductions.
                # Clamp padded rows to the last row.
                for i in T.Parallel(block_m):
                    row = T.min(pid_m * block_m + i, M - 1)
                    read_first(
                        shift,
                        i,
                        lambda j, row=row: T.cast(
                            T.cast(
                                T.cast(x[row, T.min(j, N - 1)], "float32")
                                + T.cast(residual[row, T.min(j, N - 1)], "float32"),
                                dtype,
                            ),
                            "float32",
                        ),
                    )

                # Fused add in the storage dtype, as residual_out holds it, so the statistics
                # and the output see one row. It equals rounding the fp32 sum: fp32's 24 bits
                # are at least 2p + 2 for bf16 and fp16 (p = 8, 11).
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = x_local[i, j] + r_local[i, j]
                for i, j in T.Parallel(block_m, N_padded):
                    if (not row_guard) or pid_m * block_m + i < M:
                        residual_out[pid_m * block_m + i, j] = x_local[i, j]

                for i, j in T.Parallel(block_m, N_padded):
                    centered[i, j] = T.cast(x_local[i, j], "float32")
                row_reduce(centered, squares, shift, acc, acc_squares, centre, offset, rstd)

                for i, j in T.Parallel(block_m, N_padded):
                    if (not row_guard) or pid_m * block_m + i < M:
                        y[pid_m * block_m + i, j] = T.cast(
                            (centered[i, j] - offset[i]) * rstd[i] * T.cast(weight[j], "float32")
                            + T.cast(bias[j], "float32"),
                            dtype,
                        )

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


@target_cache(maxsize=32)
def _held_layer_norm_kernel(
    M, N, eps, dtype, vec, cluster, alone, fused, has_weight, has_bias, modulated, gated
):
    """Return the factory for ``M`` rows held in registers, each split across ``cluster`` CTAs.

    The tensors are viewed as ``(M * cluster, N // cluster)``, one CTA a row of the view.
    ``vec`` divides a CTA's share; an access past the share's end is skipped whole.
    ``alone`` builds for CTAs that each have an SM to themselves. ``fused`` normalizes
    ``x + residual`` and writes it as ``residual_out`` (FusedAddLayerNorm); otherwise ``x``
    (LayerNorm), whose affine pair may be absent. ``modulated`` takes the affine pair per
    element, shaped like ``x`` (AdaLayerNorm's scale and shift), and ``gated`` multiplies the
    output by a third such tensor (AdaLayerNormZero's gate).
    """
    chunk = N // cluster

    out_idx = [4, 5] if fused else [4 if gated else 3]

    @tilelang.jit(out_idx=out_idx, compile_flags=csrc_include("cluster_partials.h"))
    def _func(threads, ctas_per_sm):
        steps = -(-chunk // (threads * vec))
        held = steps * vec
        warps = threads // WARP_LANES
        whole = chunk % (threads * vec) == 0
        # Accesses every thread makes, and the threads that make one more.
        full_steps, last_step = divmod(chunk // vec, threads)
        # One exchange buffer, so no exchange shares bytes with another: a peer's store may land
        # while this CTA still reads an earlier exchange.
        peer_words = 4 * cluster if cluster > 1 else 0
        warp_words = 3 if alone else 4
        raw_peer = -(-(peer_words + warp_words * warps) // 4) * 4
        raw_words = raw_peer + (2 * cluster if cluster > 1 else 0)
        raw_adds = (
            steps * -(-vec // 4)
            # The pairwise fold of the four chains.
            + 2
            + (WARP_LANES.bit_length() - 1)
            + (warps - 1).bit_length()
            + (cluster - 1).bit_length()
        )

        def held_count(warp):
            """`held_by` of a warp known at build time."""
            return vec * (
                full_steps * WARP_LANES + max(min(last_step - warp * WARP_LANES, WARP_LANES), 0)
            )

        live_warps = [v for v in range(warps) if held_count(v)]
        cluster_ranks = list(range(cluster))
        warp_ranks = list(range(warps))

        def held_by(warp):
            """Elements ``warp`` holds: zero for a warp past the share."""
            return vec * (
                full_steps * WARP_LANES + T.max(T.min(last_step - warp * WARP_LANES, WARP_LANES), 0)
            )

        @T.macro
        def centred_sums(d, partial, acc, centre, tx, across_warp=True):
            """Rewrite ``d`` less *centre*; sum ``d`` and ``d^2 / N`` into ``acc[0:2]``."""
            for j in T.unroll(8):
                partial[j] = T.float32(0)
            for k in T.unroll(steps):
                base = (k * threads + tx) * vec
                if whole or base < chunk:
                    for i in T.unroll(vec):
                        d[k * vec + i] = d[k * vec + i] - centre
                        partial[i % 4] += d[k * vec + i]
                        partial[4 + i % 4] += d[k * vec + i] * (d[k * vec + i] * (1.0 / N))
            acc[0] = fold_chains(partial)
            acc[1] = fold_chains(partial, 4)
            if across_warp:
                warp_sum(acc, 2)

        @T.macro
        def send(slots, received, tx, rank, word, phase, first, second):
            """Store ``first`` and ``second`` at ``word`` of this CTA's peer slots in every peer
            CTA."""
            # st.async writes a peer's shared memory only, so a CTA keeps its own values
            # in registers.
            if tx < cluster and tx != rank:
                T.call_extern(  # lint: extern-target allow: st.async to a peer CTA has no TileLang primitive
                    "handle",
                    "tileops::send_partial2",
                    T.address_of(slots[rank * 4 + word]),
                    T.address_of(received[phase]),
                    tx,
                    first,
                    second,
                )
            T.mbarrier_wait_parity(received[phase], 0)

        @T.macro
        def body(x, residual, weight, bias, gate, y, residual_out):
            with (
                T.Kernel(M, threads=threads)
                if cluster == 1
                else T.ClusterKernel(M * cluster, cluster_dims=cluster, threads=threads)
            ) as cta:
                if ctas_per_sm and not alone:
                    T.annotate_min_blocks_per_sm(ctas_per_sm)
                tx = T.get_thread_binding()
                rank = cta % cluster
                warp = tx // WARP_LANES
                # Queued unfused CTAs read the affine pair late: held, it caps the CTAs an SM keeps.
                late_affine = not (fused or alone)
                affine_n = vec if late_affine else held
                a = T.alloc_local([held], dtype)
                # The residual, then the output; an unfused row's output reuses ``a``.
                b = T.alloc_local([held], dtype) if fused else a
                w = T.alloc_local([affine_n if has_weight else 1], dtype)
                c = T.alloc_local([affine_n if has_bias else 1], dtype)
                g = T.alloc_local([affine_n if gated else 1], dtype)
                affine_row = cta if modulated else rank
                d = T.alloc_local([held], "float32")
                acc = T.alloc_local([2], "float32")
                # Four chains a thread for each sum.
                partial = T.alloc_local([8], "float32")
                stats = T.alloc_local([4], "float32")
                # (rstd, the mean's share of the output).
                scaled = T.alloc_local([2], "float32")
                slots = T.alloc_shared([raw_words + 2 * warps], "float32")
                received = None
                if cluster > 1:
                    # The raw sums' exchange, then one more when alone, two when queued.
                    received = T.alloc_barrier([1, 1] if alone else [1, 1, 1])
                    if tx == 0:
                        T.mbarrier_arrive_expect_tx(received[0], 8 * (cluster - 1))
                        if alone:
                            T.mbarrier_arrive_expect_tx(received[1], 16 * (cluster - 1))
                        else:
                            T.mbarrier_arrive_expect_tx(received[1], 8 * (cluster - 1))
                            T.mbarrier_arrive_expect_tx(received[2], 8 * (cluster - 1))
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
                        if fused:
                            for i in T.vectorized(vec):
                                b[k * vec + i] = residual[cta, base + i]
                        if has_weight and not late_affine:
                            for i in T.vectorized(vec):
                                w[k * vec + i] = weight[affine_row, base + i]
                        if has_bias and not late_affine:
                            for i in T.vectorized(vec):
                                c[k * vec + i] = bias[affine_row, base + i]
                        if gated and not late_affine:
                            for i in T.vectorized(vec):
                                g[k * vec + i] = gate[affine_row, base + i]
                    else:
                        # An access past the share holds zeros: a warp's lead may come from it.
                        for i in T.vectorized(vec):
                            a[k * vec + i] = T.cast(0, dtype)
                        if fused:
                            for i in T.vectorized(vec):
                                b[k * vec + i] = T.cast(0, dtype)

                if fused:
                    # Normalize the stored residual sum. Native fp16/bf16 addition equals the
                    # rounded fp32 sum: fp32 has at least 2p + 2 bits (p = 11/8).
                    for k in T.unroll(steps):
                        base = (k * threads + tx) * vec
                        if whole or base < chunk:
                            for i in T.vectorized(vec):
                                a[k * vec + i] = a[k * vec + i] + b[k * vec + i]

                # Raw sums, kept where `make_raw_sums_hold` holds; a row's CTAs share one verdict.
                for j in T.unroll(8):
                    partial[j] = T.float32(0)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        for i in T.unroll(vec):
                            d[k * vec + i] = T.cast(a[k * vec + i], "float32")
                            partial[i % 4] += d[k * vec + i]
                            partial[4 + i % 4] += d[k * vec + i] * d[k * vec + i]
                    else:
                        for i in T.unroll(vec):
                            d[k * vec + i] = T.float32(0)
                acc[0] = fold_chains(partial)
                acc[1] = fold_chains(partial, 4)
                if fused:
                    # Written while the sum crosses the warp.
                    for k in T.unroll(steps):
                        base = (k * threads + tx) * vec
                        if whole or base < chunk:
                            for i in T.vectorized(vec):
                                residual_out[cta, base + i] = a[k * vec + i]

                warp_sum(acc, 2)
                if warps > 1:
                    if tx % WARP_LANES == 0:
                        slots[raw_words + warp] = acc[0]
                        slots[raw_words + warps + warp] = acc[1]
                    T.sync_threads()
                    acc[0] = pairwise_sum([slots[raw_words + v] for v in warp_ranks])
                    acc[1] = pairwise_sum([slots[raw_words + warps + v] for v in warp_ranks])
                if cluster > 1:
                    # Every peer's barriers are initialized past this wait.
                    T.cluster_wait()
                    if tx < cluster and tx != rank:
                        T.call_extern(  # lint: extern-target allow: st.async to a peer CTA has no TileLang primitive
                            "handle",
                            "tileops::send_partial2",
                            T.address_of(slots[raw_peer + rank * 2]),
                            T.address_of(received[0]),
                            tx,
                            acc[0],
                            acc[1],
                        )
                    T.mbarrier_wait_parity(received[0], 0)
                    cta_sum = acc[0]
                    cta_square = acc[1]
                    acc[0] = pairwise_sum(
                        [
                            T.if_then_else(p == rank, cta_sum, slots[raw_peer + p * 2])
                            for p in cluster_ranks
                        ]
                    )
                    acc[1] = pairwise_sum(
                        [
                            T.if_then_else(p == rank, cta_square, slots[raw_peer + p * 2 + 1])
                            for p in cluster_ranks
                        ]
                    )
                raw_mean = acc[0] * (1.0 / N)
                raw_square = acc[1] * (1.0 / N)
                raw_var = raw_square - raw_mean * raw_mean
                # Built at the use: the program's closure holds plain values, which tuning
                # hashes.
                if make_raw_sums_hold(dtype, raw_adds)(raw_square, raw_var):
                    scaled[0] = T.rsqrt(raw_var + eps)
                    scaled[1] = raw_mean * scaled[0]
                else:
                    # Sums less a lead, a thread's (alone) or its warp's (queued) first element.
                    lead = d[0] if alone else T.shfl_sync(d[0], 0)
                    for j in T.unroll(4):
                        partial[j] = T.float32(0)
                    for k in T.unroll(steps):
                        base = (k * threads + tx) * vec
                        if whole or base < chunk:
                            for i in T.unroll(vec):
                                partial[i % 4] += d[k * vec + i] - lead
                    acc[0] = fold_chains(partial)
                    if alone:
                        # Alone: each thread centres on its own mean, and lanes then warps merge by
                        # Chan's update, one exchange. A share every thread fills alike merges at
                        # counts known at build time.
                        uniform = last_step == 0
                        count = T.cast(
                            vec * (full_steps + T.if_then_else(tx < last_step, 1, 0)), "float32"
                        )
                        # The centre only needs to be near the mean: the hardware reciprocal serves.
                        inverse = (
                            1.0 / (vec * full_steps)
                            if uniform
                            else T.if_then_else(count > 0, T.fast_rcp(T.max(count, 1.0)), 0.0)
                        )
                        centre = lead + acc[0] * inverse
                        centred_sums(d, partial, acc, centre, tx, False)
                        # (centre, mean offset, centred square sum / N, count).
                        stats[0] = centre
                        stats[1] = acc[0] * inverse
                        stats[2] = acc[1] - stats[1] * (acc[0] * (1.0 / N))
                        stats[3] = count
                        for st in T.unroll(WARP_LANES.bit_length() - 1):
                            other_centre = T.shfl_xor(stats[0], T.shift_left(1, st))
                            other_mean = T.shfl_xor(stats[1], T.shift_left(1, st))
                            other_square = T.shfl_xor(stats[2], T.shift_left(1, st))
                            delta = (other_centre - stats[0]) + (other_mean - stats[1])
                            if uniform:
                                # Two groups of one count.
                                stats[1] = stats[1] + delta * 0.5
                                stats[2] = (
                                    stats[2]
                                    + other_square
                                    + (
                                        T.cast(T.shift_left(1, st), "float32")
                                        * (0.5 * vec * full_steps / N)
                                    )
                                    * delta
                                    * delta
                                )
                            else:
                                other_count = T.shfl_xor(stats[3], T.shift_left(1, st))
                                share = other_count * T.fast_rcp(T.max(stats[3] + other_count, 1.0))
                                stats[1] = stats[1] + delta * share
                                stats[2] = (
                                    stats[2] + other_square + (stats[3] * share / N) * delta * delta
                                )
                                stats[3] = stats[3] + other_count
                        if warps > 1:
                            if tx % WARP_LANES == 0:
                                slots[peer_words + warp] = stats[0]
                                slots[peer_words + warps + warp] = stats[1]
                                slots[peer_words + 2 * warps + warp] = stats[2]
                            T.sync_threads()
                            # Merged less a reference near the merged mean, so no term rounds at a
                            # far warp's distance.
                            stats[0] = slots[peer_words]
                            stats[0] = stats[0] + pairwise_sum(
                                [
                                    float(held_count(v))
                                    * (
                                        (slots[peer_words + v] - stats[0])
                                        + slots[peer_words + warps + v]
                                    )
                                    for v in live_warps
                                ]
                            ) * (1.0 / chunk)
                            means = [
                                (slots[peer_words + v] - stats[0]) + slots[peer_words + warps + v]
                                for v in live_warps
                            ]
                            stats[1] = pairwise_sum(
                                [
                                    float(held_count(v)) * m
                                    for v, m in zip(live_warps, means, strict=True)
                                ]
                            ) * (1.0 / chunk)
                            stats[2] = (
                                pairwise_sum(
                                    [
                                        slots[peer_words + 2 * warps + v]
                                        + (float(held_count(v)) / N) * m * m
                                        for v, m in zip(live_warps, means, strict=True)
                                    ]
                                )
                                - (float(chunk) / N) * stats[1] * stats[1]
                            )
                        if cluster > 1:
                            if tx < cluster and tx != rank:
                                T.call_extern(  # lint: extern-target allow: st.async to a peer CTA has no TileLang primitive
                                    "handle",
                                    "tileops::send_partial4",
                                    T.address_of(slots[rank * 4]),
                                    T.address_of(received[1]),
                                    tx,
                                    stats[0],
                                    stats[1],
                                    stats[2],
                                    T.float32(0),
                                )
                            T.mbarrier_wait_parity(received[1], 0)
                            # A CTA reads its own statistics from registers, a peer's from slots.
                            cta_centre = stats[0]
                            cta_mean = stats[1]
                            cta_square = stats[2]
                            stats[0] = T.if_then_else(rank == 0, cta_centre, slots[0])
                            stats[0] = stats[0] + pairwise_sum(
                                [
                                    (T.if_then_else(p == rank, cta_centre, slots[p * 4]) - stats[0])
                                    + T.if_then_else(p == rank, cta_mean, slots[p * 4 + 1])
                                    for p in cluster_ranks
                                ]
                            ) * (1.0 / cluster)
                            means = [
                                (T.if_then_else(p == rank, cta_centre, slots[p * 4]) - stats[0])
                                + T.if_then_else(p == rank, cta_mean, slots[p * 4 + 1])
                                for p in cluster_ranks
                            ]
                            stats[1] = pairwise_sum(means) * (1.0 / cluster)
                            stats[2] = (
                                pairwise_sum(
                                    [
                                        T.if_then_else(p == rank, cta_square, slots[p * 4 + 2])
                                        + (float(chunk) / N) * m * m
                                        for p, m in zip(cluster_ranks, means, strict=True)
                                    ]
                                )
                                - stats[1] * stats[1]
                            )
                        # The row's mean less this warp's centre: the output centers on the row.
                        offset = stats[1] - (centre - stats[0])
                    else:
                        # Queued: two passes, fewer registers.
                        warp_sum(acc, 1)
                        stats[0] = lead
                        if warps > 1:
                            if tx % WARP_LANES == 0:
                                slots[peer_words + warp] = acc[0]
                                slots[peer_words + warps + warp] = lead
                            T.sync_threads()
                            stats[0] = slots[peer_words + warps]
                            acc[0] = T.float32(0)
                            for v in T.unroll(warps):
                                if held_by(v) > 0:
                                    acc[0] += slots[peer_words + v] + T.cast(
                                        held_by(v), "float32"
                                    ) * (slots[peer_words + warps + v] - stats[0])
                        if cluster > 1:
                            send(slots, received, tx, rank, 0, 1, acc[0], stats[0])
                            cta_sum = acc[0]
                            cta_lead = stats[0]
                            stats[0] = T.if_then_else(rank == 0, cta_lead, slots[1])
                            acc[0] = T.float32(0)
                            for p in T.unroll(cluster):
                                acc[0] += T.if_then_else(p == rank, cta_sum, slots[p * 4]) + float(
                                    chunk
                                ) * (
                                    T.if_then_else(p == rank, cta_lead, slots[p * 4 + 1]) - stats[0]
                                )
                        centre = stats[0] + acc[0] * (1.0 / N)
                        centred_sums(d, partial, acc, centre, tx)
                        if warps > 1:
                            if tx % WARP_LANES == 0:
                                slots[peer_words + 2 * warps + warp] = acc[0]
                                slots[peer_words + 3 * warps + warp] = acc[1]
                            T.sync_threads()
                            acc[0] = T.float32(0)
                            acc[1] = T.float32(0)
                            for v in T.unroll(warps):
                                acc[0] += slots[peer_words + 2 * warps + v]
                                acc[1] += slots[peer_words + 3 * warps + v]
                        if cluster > 1:
                            send(slots, received, tx, rank, 2, 2, acc[0], acc[1])
                            cta_sum = acc[0]
                            cta_square = acc[1]
                            acc[0] = T.float32(0)
                            acc[1] = T.float32(0)
                            for p in T.unroll(cluster):
                                acc[0] += T.if_then_else(p == rank, cta_sum, slots[p * 4 + 2])
                                acc[1] += T.if_then_else(p == rank, cta_square, slots[p * 4 + 3])
                        offset = acc[0] * (1.0 / N)
                        stats[2] = acc[1] - offset * (acc[0] * (1.0 / N))
                    rstd = T.rsqrt(stats[2] + eps)
                    scaled[0] = rstd
                    scaled[1] = offset * rstd

                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        at = 0 if late_affine else k * vec
                        if has_weight and late_affine:
                            for i in T.vectorized(vec):
                                w[i] = weight[affine_row, base + i]
                        if has_bias and late_affine:
                            for i in T.vectorized(vec):
                                c[i] = bias[affine_row, base + i]
                        if gated and late_affine:
                            for i in T.vectorized(vec):
                                g[i] = gate[affine_row, base + i]
                        for i in T.vectorized(vec):
                            normed = d[k * vec + i] * scaled[0] - scaled[1]
                            if has_weight:
                                normed = normed * T.cast(w[at + i], "float32")
                            if has_bias:
                                normed = normed + T.cast(c[at + i], "float32")
                            if gated:
                                normed = normed * T.cast(g[at + i], "float32")
                            b[k * vec + i] = T.cast(normed, dtype)
                        for i in T.vectorized(vec):
                            y[cta, base + i] = b[k * vec + i]

        if fused:

            @T.prim_func
            def main(
                x: T.Tensor[(M * cluster, chunk), dtype],
                residual: T.Tensor[(M * cluster, chunk), dtype],
                weight: T.Tensor[(cluster, chunk), dtype],
                bias: T.Tensor[(cluster, chunk), dtype],
                y: T.Tensor[(M * cluster, chunk), dtype],
                residual_out: T.Tensor[(M * cluster, chunk), dtype],
            ):
                body(x, residual, weight, bias, None, y, residual_out)

            return main

        if gated:

            @T.prim_func
            def main_gated(
                x: T.Tensor[(M * cluster, chunk), dtype],
                scale: T.Tensor[(M * cluster, chunk), dtype],
                shift: T.Tensor[(M * cluster, chunk), dtype],
                gate: T.Tensor[(M * cluster, chunk), dtype],
                y: T.Tensor[(M * cluster, chunk), dtype],
            ):
                body(x, None, scale, shift, gate, y, None)

            return main_gated

        pair_rows = M * cluster if modulated else cluster

        @T.prim_func
        def main_layer_norm(
            x: T.Tensor[(M * cluster, chunk), dtype],
            weight: T.Tensor[(pair_rows, chunk) if has_weight else (1,), dtype],
            bias: T.Tensor[(pair_rows, chunk) if has_bias else (1,), dtype],
            y: T.Tensor[(M * cluster, chunk), dtype],
        ):
            body(x, None, weight, bias, None, y, None)

        return main_layer_norm

    return _func


class _HeldLayerNormKernel(Kernel):
    """LayerNorm, fused with a residual add or not, holding a row, or its share of one, in
    registers.

    On SM90 a row may be split across a cluster; see :meth:`_cluster`. ``config["threads"]``
    left ``None`` sizes the block per call by :meth:`_threads`, the untuned default, which
    :meth:`request_tune` tunes from.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    # 16-byte (or narrower) accesses in a CTA's share of the build tuning measures.
    _tune_accesses: Optional[int] = None
    # Whether the row is ``x + residual``, written back as ``residual_out``.
    _fused: bool = True
    # Whether the affine pair is per element (AdaLayerNorm), and a gate multiplies the output.
    _modulated: bool = False
    _gated: bool = False
    # Tensors a thread holds in the storage dtype, an element each, through the reductions.
    _operands: int = 4

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
        return {"threads": None, "ctas_per_sm": None}

    @property
    def autotune_configs(self) -> list[dict]:
        """Every power-of-two block from 128 to 1024 threads that the tuned build's share fills,
        each with the compiler's registers and with every CTAs-an-SM count that leaves a thread
        64 to 96 of them. Tuning measures the first row count built, seeded by :meth:`_threads`."""
        accesses = self._tune_accesses or -(-self.N // self._vec(self.N, self.dtype, 1))
        return [
            {"threads": t, "ctas_per_sm": ctas}
            for t in (128, 256, 512, 1024)
            if t == 128 or t <= -(-accesses // WARP_LANES) * WARP_LANES
            for ctas in (0, *range(1, 9))
            if not ctas or 64 <= SM90_REGISTERS // (t * ctas) < 96
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

        A thread holds ``_operands`` tensors in the storage dtype (the row's ``x``, the
        ``residual`` when fused, the affine pair, and the gate when gated) and the centered row
        in fp32 for every element of its share. When that exceeds its registers and CTAs queue
        on an SM, the class's fragment kernel serves the call.
        """
        # A thread's registers: 255 of 4 bytes each.
        register_bytes = 255 * 4
        cluster = cls._cluster(n, dtype, rows, arch, sm_count)
        threads = cls._threads(n, cluster, cls._vec(n, dtype, cluster), rows, sm_count)
        held_bytes = -(-n // (cluster * threads)) * (cls._operands * dtype.itemsize + 4)
        return rows * cluster > sm_count and held_bytes > register_bytes

    @classmethod
    def refusal(cls, call: LayerNormCall) -> "str | None":
        if cls._queued_past_registers(call.n, call.dtype, call.rows, call.arch, call.sm_count):
            return f"{call.rows} rows of {call.n} queue on every SM past a thread's registers"
        return super().refusal(call)

    def _launch_shape(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        """``(rows, SM count, cluster, a CTA's share of a row)`` of a launch on *x*."""
        m = x.numel() // self.N
        target = self.build_target
        cluster = self._cluster(self.N, self.dtype, m, target.arch, target.sm_count)
        return m, target.sm_count, cluster, self.N // cluster

    def _compile(self, cfg: dict, m: int, cluster: int, sm_count: int, affine=(True, True)):
        """The program for *m* rows, each split across *cluster* CTAs, on *sm_count* SMs,
        with or without a weight and a bias."""
        share = self.N // cluster
        vec = self._vec(self.N, self.dtype, cluster)
        if cfg["threads"] is None:
            cfg = {**cfg, "threads": self._threads(self.N, cluster, vec, m, sm_count)}
        if cfg["ctas_per_sm"] is None:
            # Untuned, queued CTAs ask for as many an SM as the held row's registers allow, at least
            # 64 a thread: the compiler, choosing for one CTA alone, takes more.
            held = -(-share // (cfg["threads"] * vec)) * vec
            words = held * self.dtype.itemsize * (2 if self._fused else 1) // 4
            registers = max(64, -(-(held + words + 16) // 8) * 8)
            cfg = {**cfg, "ctas_per_sm": SM90_REGISTERS // (cfg["threads"] * registers)}
        kernel = _held_layer_norm_kernel(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            vec,
            cluster,
            m * cluster <= sm_count,
            self._fused,
            *affine,
            self._modulated,
            self._gated,
        )
        program = kernel(cfg["threads"], cfg["ctas_per_sm"])
        # Set after the build succeeds, so tuning never reads one that failed.
        self.kernel = kernel
        if self._tune_target is None:
            self._tune_target = (kernel, cfg)
            self._tune_accesses = -(-share // vec)
        return program


class FusedAddLayerNormHeldKernel(_HeldLayerNormKernel, FusedAddLayerNormFwdInterface):
    """Fused Add + LayerNorm forward kernel holding a row, or its share of one, in registers.

    See `_HeldLayerNormKernel`.
    """

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
        m, sm_count, cluster, chunk = self._launch_shape(x)
        program = self.program(self._compile, m, cluster, sm_count)
        outputs = program(
            x.reshape(m * cluster, chunk),
            residual.reshape(m * cluster, chunk),
            weight.reshape(cluster, chunk),
            bias.reshape(cluster, chunk),
        )
        return [out.reshape(original_shape) for out in outputs]


class LayerNormHeldKernel(_HeldLayerNormKernel, LayerNormFwdInterface):
    """LayerNorm forward kernel holding a row, or its share of one, in registers.

    See `_HeldLayerNormKernel`. Serves the rows that fit: the fragment kernel serves rows
    that queue on every SM past a thread's registers.
    """

    _fused = False
    _operands = 3

    def forward(
        self, x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, or ``None`` to scale by one.
            bias: Affine shift holding ``N`` elements, or ``None`` to shift by zero.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, weight=weight, bias=bias)
        original_shape = x.shape
        m, sm_count, cluster, chunk = self._launch_shape(x)
        affine = (weight is not None, bias is not None)
        # An absent tensor is a one-element placeholder the program never reads.
        weight = weight.reshape(cluster, chunk) if affine[0] else x.new_empty(1)
        bias = bias.reshape(cluster, chunk) if affine[1] else x.new_empty(1)
        program = self.program(self._compile, m, cluster, sm_count, affine)
        return program(x.reshape(m * cluster, chunk), weight, bias).reshape(original_shape)


class AdaLayerNormHeldKernel(_HeldLayerNormKernel, AdaLayerNormFwdInterface):
    """Adaptive LayerNorm, ``scale * LayerNorm(x) + shift`` with a scale and a shift per
    element, holding a row, or its share of one, in registers. See `_HeldLayerNormKernel`.
    """

    _fused = False
    _modulated = True
    _operands = 3

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize and modulate ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axis is ``N``, contiguous, on a CUDA device.
            scale: Per-element scale shaped like *x*.
            shift: Per-element shift shaped like *x*.
            gate: Per-element gate shaped like *x*, for the gated variant only.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device, or the gate does not match the
                variant.
        """
        self._require_devices(x=x, scale=scale, shift=shift, gate=gate)
        if (gate is not None) != self._gated:
            raise ValueError(
                f"{type(self).__name__} takes a gate exactly when it is the gated variant"
            )
        original_shape = x.shape
        m, sm_count, cluster, chunk = self._launch_shape(x)
        view = (m * cluster, chunk)
        program = self.program(self._compile, m, cluster, sm_count)
        operands = [t.reshape(view) for t in (x, scale, shift, gate) if t is not None]
        return program(*operands).reshape(original_shape)


class AdaLayerNormZeroHeldKernel(AdaLayerNormHeldKernel, AdaLayerNormZeroFwdInterface):
    """The AdaLN-Zero variant, ``gate * (scale * LayerNorm(x) + shift)``: see
    `AdaLayerNormHeldKernel`."""

    _gated = True
    _operands = 4


# Fused Add + RMSNorm kernel


@target_cache(maxsize=32)
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
