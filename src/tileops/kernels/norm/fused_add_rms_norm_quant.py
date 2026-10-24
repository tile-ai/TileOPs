"""Fused residual add, RMSNorm and dynamic per-row quantization.

The arithmetic is vLLM's ``rms_norm_dynamic_per_token_quant``: the sum ``x + residual`` is
reduced unrounded, the normalized value is rounded to the storage dtype before and after the
weight multiply, and the codes are the normalized values times the reciprocal of the scale.
Two row reductions run per row: the sum of squares gives the inverse RMS, and the normalized
row's absolute maximum gives the scale.

:class:`FusedAddRMSNormPerTokenQuantKernel` holds a row in registers, one CTA a row or, on
SM90, a thread-block cluster a row whose CTAs exchange their partials through each other's
shared memory. :class:`FusedAddRMSNormPerTokenQuantStreamKernel` serves the rows no CTA can
hold: it reads a row's inputs once per reduction and once to write the codes.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.norm.call_spec import (
    FusedAddRMSNormPerTokenQuantFwdInterface,
    RMSNormQuantCall,
)
from tileops.kernels.tiling import align_up
from tileops.utils import WARP_LANES, get_sm_count, get_sm_version

__all__ = ["FusedAddRMSNormPerTokenQuantKernel", "FusedAddRMSNormPerTokenQuantStreamKernel"]


def _vec(n: int) -> int:
    """The widest access of at most 16 bytes of 16-bit elements that divides *n*."""
    return next(v for v in (VECTOR_ACCESS_BYTES // 2, 4, 2, 1) if n % v == 0)


def _held_cluster(n: int, rows: int, arch: int, sm_count: int) -> Optional[int]:
    """CTAs a row is split across in registers, or ``None`` when no cluster holds it.

    A row of more than 8192 elements splits on SM90 into the widest power-of-two cluster,
    up to the 8 CTAs every SM90 device launches, that keeps the launch to one CTA an SM and
    a CTA's share at 1024 elements or more. The split stands when a share is then at most
    4096 elements, or when the row needs a cluster anyway: a row wider than one CTA holds
    takes at least the narrowest cluster that holds it.
    """
    # Elements one CTA holds in registers at most: 64 a thread at 512 threads fit the 128
    # registers a thread has there; more spill.
    max_held = 32768
    vec = _vec(n)
    max_cluster = 8 if arch >= 90 else 1

    def divides(cluster):
        return n % (cluster * vec) == 0

    cluster = 1
    while n // cluster > max_held:
        cluster *= 2
        if cluster > max_cluster or not divides(cluster):
            return None
    if n > 8192:
        wide = cluster
        while (
            wide < max_cluster
            and rows * wide * 2 <= sm_count
            and n // (wide * 2) >= 1024
            and divides(wide * 2)
        ):
            wide *= 2
        if cluster > 1 or n // wide <= 4096:
            cluster = wide
    return cluster


def _pairwise_sum(part, row, count):
    """``part[row, :count]`` summed as a pairwise tree."""
    terms = [part[row, i] for i in range(count)]
    while len(terms) > 1:
        pairs = [terms[i] + terms[i + 1] for i in range(0, len(terms) - 1, 2)]
        terms = pairs + terms[len(pairs) * 2 :]
    return terms[0]


def _pair(buf, at):
    """The two elements of *buf* from *at*, as one packed value."""
    return buf[T.Ramp(at, 1, 2)]


def _constants(dtype, quant_dtype, scale_ub):
    """``(fp8, qmax, min_scale, bound)`` of a code dtype and amax bound."""
    fp8 = quant_dtype == "float8_e4m3fn"
    # vLLM's floor on a row's scale, so an all-zero row divides by a positive number.
    min_scale = 1.0 / (448.0 * 512.0) if fp8 else float(torch.finfo(torch.float32).eps)
    bound = float("inf") if scale_ub is None else scale_ub
    return fp8, 448.0 if fp8 else 127.0, min_scale, bound


@T.macro
def _load_inputs(a, b, at, x, residual, row, col, vec):
    """``vec`` elements of ``x`` and ``residual`` at ``(row, col)`` into ``a[at:]``, ``b[at:]``."""
    if vec * 2 == VECTOR_ACCESS_BYTES:
        # The L2 evict-last cache policy has no TileLang primitive.
        T.call_extern(  # lint: extern-target allow: streaming_load.h's evict-last load
            "handle", "tileops::load16_evict_last", T.address_of(a[at]), T.address_of(x[row, col])
        )
        T.call_extern(  # lint: extern-target allow: streaming_load.h's evict-last load
            "handle",
            "tileops::load16_evict_last",
            T.address_of(b[at]),
            T.address_of(residual[row, col]),
        )
    else:
        for i in T.vectorized(vec):
            a[at + i] = x[row, col + i]
        for i in T.vectorized(vec):
            b[at + i] = residual[row, col + i]


@T.macro
def _add(a, b, s, acc, at, vec):
    """``s = a + b`` unrounded; ``acc[0]``, ``acc[1]`` gather the squares of the even and
    odd elements."""
    for i in T.vectorized(vec):
        s[at + i] = T.cast(a[at + i], "float32") + T.cast(b[at + i], "float32")
    for i in T.unroll(vec):
        acc[i % 2] = T.ieee_fmaf(s[at + i], s[at + i], acc[i % 2])


@T.macro
def _store_sum(s, a, at, residual_out, row, col, vec, dtype):
    """``s[at:at + vec]`` rounded into ``a`` and written to ``residual_out[row, col:]``."""
    for i in T.vectorized(vec):
        a[at + i] = T.cast(s[at + i], dtype)
    for i in T.vectorized(vec):
        residual_out[row, col + i] = a[at + i]


@T.macro
def _normalize(s, w, y, peak, at, wat, rrms, vec, dtype, fp8):
    """``y = T(T(s * rrms) * w)``; ``peak`` keeps each lane's largest ``|y|``, skipping a NaN.

    With FP8 codes a NaN ``y`` is held as +inf, which the codes turn into +448 as vLLM's
    ``fminf`` before ``fmaxf`` does.
    """
    inf = T.cast(T.float32(float("inf")), dtype)
    if vec % 2 == 0:
        for j in T.unroll(vec // 2):
            # The packed multiply rounds as the float32 product rounded once does.
            p = T.mul2(
                T.cast(_pair(s, at + 2 * j) * T.Broadcast(rrms, 2), f"{dtype}x2"),
                _pair(w, wat + 2 * j),
            )
            peak[T.Ramp(0, 1, 2)] = T.max2(_pair(peak, 0), T.abs2(p))
            y[T.Ramp(at + 2 * j, 1, 2)] = T.min2(p, T.Broadcast(inf, 2)) if fp8 else p
    else:
        p1 = T.cast(s[at] * rrms, dtype) * w[wat]
        peak[T.Ramp(0, 1, 2)] = T.max2(_pair(peak, 0), T.abs2(T.Broadcast(p1, 2)))
        y[at] = T.if_then_else(T.isnan(T.cast(p1, "float32")), inf, p1) if fp8 else p1


@T.macro
def _codes(y, at, wide, codes, inv_scale, vec, quant_dtype, fp8, nan_clamp):
    """Codes of ``y[at:at + vec] * inv_scale`` into ``codes``; ``nan_clamp`` makes a NaN
    product +448."""
    for i in T.vectorized(vec):
        wide[i] = T.cast(y[at + i], "float32") * inv_scale
    if fp8:
        if nan_clamp:
            for i in T.vectorized(vec):
                wide[i] = T.min(wide[i], T.float32(448.0))
        # Saturates finite values and infinities to +-448.
        for i in T.vectorized(vec):
            codes[i] = T.cast(wide[i], quant_dtype)
    else:
        # Round half to even and saturate; a NaN becomes 0, as vLLM's cvt.rni.sat makes it.
        # Adding 1.5 * 2^23 rounds a value within +-2^22 half to even into the low bits of
        # the sum, whose low byte is then the int8 code.
        for i in T.unroll(vec):
            codes[i] = T.cast(
                T.reinterpret(
                    T.if_then_else(T.isnan(wide[i]), T.float32(0), T.clamp(wide[i], -128.0, 127.0))
                    + T.float32(12582912.0),
                    "int32",
                ),
                quant_dtype,
            )


@T.macro
def _warp_sum(acc):
    """``acc[0] + acc[1]`` summed over the warp into ``acc[0]`` of every lane."""
    acc[0] += acc[1]
    for k in T.unroll(WARP_LANES.bit_length() - 1):
        acc[0] += T.shfl_xor(acc[0], T.shift_left(1, k))


@T.macro
def _block_sum(acc, part, tx, warps):
    """The warps' ``acc[0]`` summed over the CTA into ``acc[0]`` of every thread."""
    if warps > 1:
        if tx % WARP_LANES == 0:
            part[0, tx // WARP_LANES] = acc[0]
        T.sync_threads()
        acc[0] = _pairwise_sum(part, 0, warps)


@T.macro
def _block_amax_bits(peak, bits, part, tx, warps):
    """The largest lane of ``peak`` over the CTA, as int32 bits, into ``bits[0]``.

    A non-negative float without NaN orders as its int32 bits, which a warp folds with one
    ``redux.sync``.
    """
    bits[0] = T.warp_reduce_max(
        T.reinterpret(T.max(T.cast(peak[0], "float32"), T.cast(peak[1], "float32")), "int32")
    )
    if warps > 1:
        if tx % WARP_LANES == 0:
            part[1, tx // WARP_LANES] = T.reinterpret(bits[0], "float32")
        T.sync_threads()
        bits[0] = T.warp_reduce_max(T.reinterpret(part[1, tx % WARP_LANES % warps], "int32"))


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_quant_kernel(M, N, eps, dtype, quant_dtype, scale_ub, vec, cluster):
    """Return the factory for ``M`` rows held in registers, each split across ``cluster``
    CTAs.

    The tensors are viewed as ``(M * cluster, N // cluster)``, one CTA a row of the view.
    ``vec`` divides a CTA's share; an access past the share's end is skipped whole.
    """
    fp8, qmax, min_scale, bound = _constants(dtype, quant_dtype, scale_ub)
    chunk = N // cluster

    @tilelang.jit(
        out_idx=[3, 4, 5], compile_flags=csrc_include("cluster_partials.h", "streaming_load.h")
    )
    def _func(threads):
        steps = -(-chunk // (threads * vec))
        held = steps * vec
        warps = threads // WARP_LANES
        whole = chunk % (threads * vec) == 0

        @T.macro
        def exchange(peers, at, received, tx, rank, value):
            """Send *value* to slot ``peers[at, rank]`` of every peer CTA, then wait for
            theirs."""
            if tx < cluster and tx != rank:
                T.call_extern(  # lint: extern-target allow: st.async to a peer CTA has no TileLang primitive
                    "handle",
                    "tileops::send_partial",
                    T.address_of(peers[at, rank]),
                    T.address_of(received[at]),
                    tx,
                    value,
                )
            T.mbarrier_wait_parity(received[at], 0)

        @T.macro
        def codes_pass(cta, tx, a, wide, codes, q, inv_scale, nan_clamp):
            for k in T.unroll(steps):
                base = (k * threads + tx) * vec
                if whole or base < chunk:
                    _codes(a, k * vec, wide, codes, inv_scale, vec, quant_dtype, fp8, nan_clamp)
                    for i in T.vectorized(vec):
                        q[cta, base + i] = codes[i]

        @T.prim_func
        def main(
            x: T.Tensor[(M * cluster, chunk), dtype],
            residual: T.Tensor[(M * cluster, chunk), dtype],
            weight: T.Tensor[(cluster, chunk), dtype],
            q: T.Tensor[(M * cluster, chunk), quant_dtype],
            scale: T.Tensor[(M, 1), T.float32],
            residual_out: T.Tensor[(M * cluster, chunk), dtype],
        ):
            with (
                T.Kernel(M, threads=threads)
                if cluster == 1
                else T.ClusterKernel(M * cluster, cluster_dims=cluster, threads=threads)
            ) as cta:
                tx = T.get_thread_binding()
                lane = tx % WARP_LANES
                rank = cta % cluster
                a = T.alloc_local([held], dtype)
                b = T.alloc_local([held], dtype)
                w = T.alloc_local([held], dtype)
                s = T.alloc_local([held], "float32")
                wide = T.alloc_local([vec], "float32")
                codes = T.alloc_local([vec], quant_dtype)
                acc = T.alloc_local([2], "float32")
                peak = T.alloc_local([2], dtype)
                bits = T.alloc_local([1], "int32")
                # One slot a warp for each fold, so the folds need no barrier between them.
                part = T.alloc_shared([2, warps], "float32")
                if cluster > 1:
                    # One slot a peer CTA for each fold; a CTA keeps its own in registers.
                    peers = T.alloc_shared([2, cluster], "float32")
                    received = T.alloc_barrier([1, 1])
                    if tx == 0:
                        T.mbarrier_arrive_expect_tx(received[0], 4 * (cluster - 1))
                        T.mbarrier_arrive_expect_tx(received[1], 4 * (cluster - 1))
                    # Peers may send once this CTA's barriers are initialized.
                    T.cluster_arrive_relaxed()

                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _load_inputs(a, b, k * vec, x, residual, cta, base, vec)
                        for i in T.vectorized(vec):
                            w[k * vec + i] = weight[rank, base + i]

                acc[0] = T.float32(0)
                acc[1] = T.float32(0)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _add(a, b, s, acc, k * vec, vec)
                _warp_sum(acc)
                # Written while the warp's sum waits for the other warps'.
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _store_sum(s, a, k * vec, residual_out, cta, base, vec, dtype)
                _block_sum(acc, part, tx, warps)
                if cluster > 1:
                    T.cluster_wait()
                    exchange(peers, 0, received, tx, rank, acc[0])
                    # Every CTA folds the same partials in the same lanes, so all take one
                    # bit-identical sum.
                    acc[0] = T.warp_reduce_sum(
                        T.if_then_else(
                            lane == rank,
                            acc[0],
                            T.if_then_else(
                                lane < cluster, peers[0, T.min(lane, cluster - 1)], T.float32(0)
                            ),
                        )
                    )
                rrms = T.rsqrt(acc[0] / T.float32(N) + eps)

                peak[0] = T.cast(T.float32(0), dtype)
                peak[1] = T.cast(T.float32(0), dtype)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _normalize(s, w, a, peak, k * vec, k * vec, rrms, vec, dtype, fp8)
                _block_amax_bits(peak, bits, part, tx, warps)
                if cluster > 1:
                    exchange(peers, 1, received, tx, rank, T.reinterpret(bits[0], "float32"))
                    bits[0] = T.warp_reduce_max(
                        T.if_then_else(
                            T.And(lane < cluster, lane != rank),
                            T.reinterpret(peers[1, T.min(lane, cluster - 1)], "int32"),
                            bits[0],
                        )
                    )

                row_max = T.min(T.reinterpret(bits[0], "float32"), T.float32(bound))
                row_scale = T.max(row_max / qmax, T.float32(min_scale))
                inv_scale = T.float32(1.0) / row_scale
                if tx == 0 and rank == 0:
                    scale[cta // cluster, 0] = row_scale
                # Uniform across the row: only an infinite scale makes NaN products.
                if fp8 and inv_scale == T.float32(0):
                    codes_pass(cta, tx, a, wide, codes, q, inv_scale, True)
                else:
                    codes_pass(cta, tx, a, wide, codes, q, inv_scale, False)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_quant_stream_kernel(M, N, eps, dtype, quant_dtype, scale_ub, vec):
    """Return the factory for ``M`` rows, one CTA a row, read in tiles three times.

    The first read sums the squares and writes ``residual_out``, the second finds the
    absolute maximum, the third writes the codes, walking the tiles back from the row's end
    so the tiles read last are read first, while they are likeliest in L2.
    """
    fp8, qmax, min_scale, bound = _constants(dtype, quant_dtype, scale_ub)

    @tilelang.jit(out_idx=[3, 4, 5], compile_flags=csrc_include("streaming_load.h"))
    def _func(threads, accesses):
        tile = threads * accesses * vec
        tiles = -(-N // tile)
        warps = threads // WARP_LANES
        held = accesses * vec

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            residual: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N,), dtype],
            q: T.Tensor[(M, N), quant_dtype],
            scale: T.Tensor[(M, 1), T.float32],
            residual_out: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                a = T.alloc_local([held], dtype)
                b = T.alloc_local([held], dtype)
                w = T.alloc_local([vec], dtype)
                s = T.alloc_local([held], "float32")
                wide = T.alloc_local([vec], "float32")
                codes = T.alloc_local([vec], quant_dtype)
                acc = T.alloc_local([2], "float32")
                peak = T.alloc_local([2], dtype)
                bits = T.alloc_local([1], "int32")
                part = T.alloc_shared([2, warps], "float32")

                acc[0] = T.float32(0)
                acc[1] = T.float32(0)
                for t in T.serial(tiles):
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _load_inputs(a, b, k * vec, x, residual, row, col, vec)
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _add(a, b, s, acc, k * vec, vec)
                            _store_sum(s, a, k * vec, residual_out, row, col, vec, dtype)
                _warp_sum(acc)
                _block_sum(acc, part, tx, warps)
                rrms = T.rsqrt(acc[0] / T.float32(N) + eps)

                peak[0] = T.cast(T.float32(0), dtype)
                peak[1] = T.cast(T.float32(0), dtype)
                for t in T.serial(tiles):
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _load_inputs(a, b, k * vec, x, residual, row, col, vec)
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _add(a, b, s, acc, k * vec, vec)
                            for i in T.vectorized(vec):
                                w[i] = weight[col + i]
                            _normalize(s, w, a, peak, k * vec, 0, rrms, vec, dtype, fp8)
                _block_amax_bits(peak, bits, part, tx, warps)

                row_max = T.min(T.reinterpret(bits[0], "float32"), T.float32(bound))
                row_scale = T.max(row_max / qmax, T.float32(min_scale))
                inv_scale = T.float32(1.0) / row_scale
                if tx == 0:
                    scale[row, 0] = row_scale
                for t_back in T.serial(tiles):
                    t = tiles - 1 - t_back
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _load_inputs(a, b, k * vec, x, residual, row, col, vec)
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _add(a, b, s, acc, k * vec, vec)
                            for i in T.vectorized(vec):
                                w[i] = weight[col + i]
                            _normalize(s, w, a, peak, k * vec, 0, rrms, vec, dtype, fp8)
                            # Uniform across the row: only an infinite scale makes NaN
                            # products.
                            if fp8 and inv_scale == T.float32(0):
                                _codes(
                                    a, k * vec, wide, codes, inv_scale, vec, quant_dtype, fp8, True
                                )
                            else:
                                _codes(
                                    a, k * vec, wide, codes, inv_scale, vec, quant_dtype, fp8, False
                                )
                            for i in T.vectorized(vec):
                                q[row, col + i] = codes[i]

        return main

    return _func


class _FusedAddRMSNormPerTokenQuantBase(Kernel, FusedAddRMSNormPerTokenQuantFwdInterface):
    """Construction shared by both implementations."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: RMSNormQuantCall) -> Entry:
        identity = (call.n, call.eps, call.dtype, call.quant_dtype, call.scale_ub)
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        scale_ub: Optional[float],
        config: Optional[dict] = None,
    ):
        """Build for a hidden size, the storage and code dtypes and the amax bound.

        The program for a given row count is resolved in ``forward``.

        Args:
            N: Hidden size the rows are normalized over.
            eps: Epsilon for numerical stability.
            dtype: Element type ``x``, ``residual`` and ``weight`` are stored in.
            quant_dtype: ``torch.float8_e4m3fn`` or ``torch.int8``.
            scale_ub: Upper bound on a row's absolute maximum, or ``None``.
            config: Optional launch override; see ``default_config``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.quant_dtype = quant_dtype
        self.scale_ub = scale_ub
        self.vec = _vec(N)
        self.init_config(config)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]


class FusedAddRMSNormPerTokenQuantKernel(_FusedAddRMSNormPerTokenQuantBase):
    """Fused residual add, RMSNorm and dynamic per-row quantization, a row held in registers.

    Supports SM80+ architectures. A row is read through accesses of up to 16 bytes. On SM90 a
    row wider than 8192 elements is split across a cluster of CTAs when the launch has SMs to
    spare, and a row wider than one CTA holds always is. ``config["threads"]`` is the fewest
    threads a CTA takes; see ``_threads``.
    """

    @classmethod
    def refusal(cls, call: RMSNormQuantCall) -> "str | None":
        if _held_cluster(call.n, call.rows, call.arch, call.sm_count) is None:
            return f"no cluster on SM{call.arch} holds a row of {call.n} in registers"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    def _threads(self, rows: int, cluster: int, sm_count: int) -> int:
        """Threads a CTA takes for its share of ``N / cluster`` elements.

        At least ``config["threads"]``, doubled up to 1024 until a thread makes at most 8
        accesses. A launch of few CTAs an SM spreads a share further, to two accesses a
        thread: up to 512 threads when a CTA has an SM to itself, 512 / k when an SM holds k
        CTAs (k rounded up to a power of two). Whole warps, no more than cover the share.
        """
        accesses = self.N // (cluster * self.vec)
        fewest = self.config["threads"]
        while fewest < 1024 and fewest * 8 < accesses:
            fewest *= 2
        ctas_per_sm = -(-rows * cluster // sm_count)
        spread = min(512 >> (ctas_per_sm - 1).bit_length(), align_up(-(-accesses // 2), WARP_LANES))
        return min(max(fewest, spread), align_up(accesses, WARP_LANES))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the fused add, RMSNorm and quantization over the trailing ``N`` elements.

        Args:
            x: ``(rows, N)`` on a CUDA device.
            residual: Shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.

        Returns:
            ``[q, scale, residual_out]``.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, residual=residual, weight=weight)
        x, residual, weight = (vector_aligned(t) for t in (x, residual, weight))
        m = x.shape[0]
        sm_count = get_sm_count(x.device.index)
        cluster = _held_cluster(self.N, m, get_sm_version(x.device.index), sm_count)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_rms_norm_quant_kernel(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            self.dtype_to_str(self.quant_dtype),
            self.scale_ub,
            self.vec,
            cluster,
        )
        chunk = self.N // cluster
        q, scale, residual_out = self.kernel(self._threads(m, cluster, sm_count))(
            x.reshape(m * cluster, chunk),
            residual.reshape(m * cluster, chunk),
            weight.reshape(cluster, chunk),
        )
        return [q.reshape(m, self.N), scale, residual_out.reshape(m, self.N)]


class FusedAddRMSNormPerTokenQuantStreamKernel(_FusedAddRMSNormPerTokenQuantBase):
    """Fused residual add, RMSNorm and dynamic per-row quantization of rows no CTA holds.

    Supports SM80+ architectures. One CTA a row reads the row's inputs in tiles three times:
    to sum the squares, to find the absolute maximum, and to write the codes. Serves the rows
    :class:`FusedAddRMSNormPerTokenQuantKernel` cannot hold in registers.
    ``config["threads"]`` and ``config["accesses"]`` set a tile: accesses of up to 16 bytes
    a thread.
    """

    @classmethod
    def refusal(cls, call: RMSNormQuantCall) -> "str | None":
        if _held_cluster(call.n, call.rows, call.arch, call.sm_count) is not None:
            return f"a row of {call.n} fits the registers of a CTA or a cluster on SM{call.arch}"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        return {"threads": 512, "accesses": 4}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the fused add, RMSNorm and quantization over the trailing ``N`` elements.

        Args:
            x: ``(rows, N)`` on a CUDA device.
            residual: Shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.

        Returns:
            ``[q, scale, residual_out]``.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, residual=residual, weight=weight)
        x, residual, weight = (vector_aligned(t) for t in (x, residual, weight))
        self.kernel = _fused_add_rms_norm_quant_stream_kernel(
            x.shape[0],
            self.N,
            self.eps,
            self.dtype_str,
            self.dtype_to_str(self.quant_dtype),
            self.scale_ub,
            self.vec,
        )
        return self.kernel(self.config["threads"], self.config["accesses"])(x, residual, weight)
