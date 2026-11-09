"""RMSNorm kernel using TileLang.

y = x * rsqrt(mean(x^2) + eps) * weight

``RMSNormKernel`` runs one of three bodies, chosen by ``_row_body`` from the row's width and
dtype alone: a group of a warp's lanes a row for rows of 256 to 2048 elements, a row held in
a block's registers past 4096, and otherwise the fragment body. Tuning sizes the chosen body
and never switches bodies. ``RMSNormTwoPassKernel`` reads a row of two or more whole thread
chunks twice; it leaves to the held body a launch of rows of two vectors a thread shorter
than its ``_MIN_WAVES`` waves, and a launch of at most four rows an SM of rows up to 16384.
``RMSNormPairedKernel`` holds a row at two 16-byte vectors a thread for launches of many
waves.

In the fragment body the normalized row goes from the register fragment straight to
global memory. Only the partial reduction reads shared memory, because a thread walking a
strided run of the row can reach it there and cannot reach another thread's registers.
The fragment may be wider than the row: columns past the row load zero and are never
stored, and the mean divides by N.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm._config import select_row_config, select_row_configs
from tileops._kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops._kernels.streaming_load import load16, load16_evict_last
from tileops._kernels.tiling import ALIGNMENT, align_up
from tileops._utils import (
    RESIDENT_CTAS,
    WARP_LANES,
    WARP_SHUFFLE_STAGES,
)

__all__ = ["RMSNormKernel", "RMSNormPairedKernel", "RMSNormTwoPassKernel"]


def _widen(buf, pair: int, half: int, dtype: str):
    """Element ``2 * pair + half`` of the local *buf* as f32; a bf16 widens from its 4-byte pair."""
    if dtype != "bfloat16":
        return T.cast(buf[pair * 2 + half], "float32")
    name = "tileops::widen_bf16_hi" if half else "tileops::widen_bf16_lo"
    return T.call_extern("float32", name, T.address_of(buf[pair * 2]))


def _square(buf, pair: int, half: int, dtype: str):
    """The f32 square of element ``2 * pair + half`` of the local *buf*."""
    v = _widen(buf, pair, half, dtype)
    return v * v


def _scaled(held, pair: int, half: int, rrms, scale, scale_pair: int, dtype: str, has_weight: bool):
    """Element ``2 * pair + half`` of *held* times *rrms*, times the matching element of *scale*
    where *has_weight*, rounded once to *dtype*."""
    v = _widen(held, pair, half, dtype) * rrms
    if has_weight:
        v = v * _widen(scale, scale_pair, half, dtype)
    return T.cast(v, dtype)


def _two_pass_shape(n: int, elem_bytes: int, threads_per_sm: int) -> tuple:
    """``(threads, vectors a thread reads at a time, blocks an SM, paired)`` of a two-pass row
    of *n*.

    ``paired`` marks a row of two 16-byte vectors a thread of a 512-thread block; it takes
    every block the SM's threads admit, whose rows stay in L1 until the second read.
    """
    threads = 512
    if n * elem_bytes <= 2 * threads * VECTOR_ACCESS_BYTES:
        return threads, 1, threads_per_sm // threads, True
    return (threads, 1, 3, False) if n <= 16384 else (1024, 2, 2, False)


def _warp_row_shape(n: int, elem_bytes: int) -> tuple:
    """``(lanes, vectors a lane holds of a row, rows a lane group holds)`` of a warp row of *n*.

    A lane takes a 16-byte vector of the row up to a warp a row, then more vectors. A lane
    group takes as many rows as fit four vectors a lane, its loads in flight, and at least
    one.
    """
    vectors = n * elem_bytes // VECTOR_ACCESS_BYTES
    per_lane = max(1, vectors // WARP_LANES)
    return vectors // per_lane, per_lane, max(1, 4 // per_lane)


def _paired_width(n: int, elem_bytes: int) -> Optional[int]:
    """The block width giving each thread two 16-byte vectors of a row of *n*, where that
    width is 64, 128 or 256 threads, else ``None``.

    Two vectors keep a thread's registers low; below 64 threads the SM's resident-block cap
    stops short of its threads.
    """
    threads = n * elem_bytes // (2 * VECTOR_ACCESS_BYTES)
    whole = threads * 2 * VECTOR_ACCESS_BYTES == n * elem_bytes
    return threads if whole and threads in (64, 128, 256) else None


def _held_widths(n: int, elem_bytes: int) -> tuple:
    """Block widths that hold a row of *n* in whole 16-byte vectors, within the element cap."""
    vec = VECTOR_ACCESS_BYTES // elem_bytes
    return tuple(
        threads
        for threads in (128, 256)
        # Past 64 elements a thread the registers cap the resident warps.
        if n % (threads * vec) == 0 and n // threads <= 64
    )


def _row_body(n: int, n_padded: int, elem_bytes: int) -> str:
    """The ``RMSNormKernel`` body serving rows of *n* elements: ``"warp_rows"``,
    ``"held_row"`` or ``"fragment"``.

    A function of the row alone, never of the launch config, so tuning cannot switch bodies.
    """
    vec = VECTOR_ACCESS_BYTES // elem_bytes
    if n_padded != n or n % vec:
        return "fragment"
    lanes, per_lane, _ = _warp_row_shape(n, elem_bytes)
    if 256 <= n <= 2048 and lanes * per_lane * vec == n and lanes & (lanes - 1) == 0:
        return "warp_rows"
    if n > 4096 and _held_widths(n, elem_bytes):
        return "held_row"
    return "fragment"


def _held_row(M, N, eps, dtype, has_weight, threads, vec, blocks=1):
    """One row a block, each thread holding ``N / threads`` elements in registers.

    Thread ``t`` holds vectors ``k * threads + t``. The sum of
    squares folds across a warp with shuffles, then across warps through shared memory.
    ``blocks`` is the blocks an SM the launch bound asks the compiler to fit.
    """
    steps = N // (threads * vec)
    held_n = steps * vec
    num_warps = threads // WARP_LANES

    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N if has_weight else 1,), dtype],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(M, threads=threads) as row:
            if blocks > 1:
                T.annotate_min_blocks_per_sm(blocks)
            tx = T.get_thread_binding()
            held = T.alloc_local((held_n,), dtype)
            scale = T.alloc_local((held_n,), dtype)
            out = T.alloc_local((held_n,), dtype)
            part = T.alloc_local((1,), "float32")
            warp_sum = T.alloc_shared((num_warps,), "float32")

            for k in T.unroll(steps):
                load16(held, k * vec, x, (row, (k * threads + tx) * vec))
            part[0] = T.cast(0.0, "float32")
            for p in T.unroll(held_n // 2):
                part[0] += _square(held, p, 0, dtype)
                part[0] += _square(held, p, 1, dtype)
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                part[0] += T.shfl_xor(part[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES)
            if tx % WARP_LANES == 0:
                warp_sum[tx // WARP_LANES] = part[0]
            if has_weight:
                for k in T.unroll(steps):
                    for c in T.vectorized(vec):
                        scale[k * vec + c] = weight[(k * threads + tx) * vec + c]
            T.sync_threads()
            part[0] = warp_sum[0]
            for w in T.unroll(1, num_warps):
                part[0] += warp_sum[w]
            # N, not a padded width: every column of the row is held.
            part[0] = T.rsqrt(part[0] / float(N) + eps)
            for p in T.unroll(held_n // 2):
                out[p * 2] = _scaled(held, p, 0, part[0], scale, p, dtype, has_weight)
                out[p * 2 + 1] = _scaled(held, p, 1, part[0], scale, p, dtype, has_weight)
            for k in T.unroll(steps):
                for c in T.vectorized(vec):
                    y[row, (k * threads + tx) * vec + c] = out[k * vec + c]

    return main


def _two_pass_row(M, N, eps, dtype, has_weight, threads, vec, chunk, blocks, tree, weight_first):
    """One row a block, read twice: evict-last to sum its squares, then evict-first to scale it.

    A thread walks its vectors ``chunk`` at a time and holds those, not the row: the second
    read finds the row in L2, where the first pass's policy kept it, and lowers each line's
    priority as it goes, so the rows still to be read a second time stay resident. ``blocks`` is the
    blocks an SM the launch bound asks the compiler to fit.

    Where ``tree``, each warp folds the warp sums itself by shuffles rather than walking them
    in turn. The shuffles take fewer registers than the walk; the caller asks for them only
    where ``blocks`` already fill the SM's threads, since elsewhere those registers admit
    another block, whose row evicts the others' from L1 before their second read.

    Where ``weight_first``, a thread loads its weight vectors beside its first read and holds
    them through the barrier, so the second pass waits on no weight load; the caller asks for
    it where those vectors fit the registers ``blocks`` leave a thread.
    """
    steps = N // (threads * vec)
    num_warps = threads // WARP_LANES
    width = chunk * vec
    # The held weight is indexed by the step, so its loops unroll to keep it in registers.
    walk = T.unroll if weight_first else T.serial

    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N if has_weight else 1,), dtype],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(M, threads=threads) as row:
            # ``blocks`` blocks an SM where the SM's threads hold them (ptxas ignores a bound
            # past them): without it the compiler spends the registers they would leave free.
            T.annotate_min_blocks_per_sm(blocks)
            tx = T.get_thread_binding()
            held = T.alloc_local((width,), dtype)
            scale = T.alloc_local((steps * vec if weight_first else width,), dtype)
            out = T.alloc_local((width,), dtype)
            part = T.alloc_local((1,), "float32")
            warp_sum = T.alloc_shared((num_warps,), "float32")

            part[0] = T.cast(0.0, "float32")
            for s in walk(steps // chunk):
                for k in T.unroll(chunk):
                    load16_evict_last(
                        held, k * vec, x, (row, ((s * chunk + k) * threads + tx) * vec)
                    )
                if weight_first and has_weight:
                    for k in T.unroll(chunk):
                        at = ((s * chunk + k) * threads + tx) * vec
                        for c in T.vectorized(vec):
                            scale[(s * chunk + k) * vec + c] = weight[at + c]
                for p in T.unroll(width // 2):
                    part[0] += _square(held, p, 0, dtype)
                    part[0] += _square(held, p, 1, dtype)
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                part[0] += T.shfl_xor(part[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES)
            if tx % WARP_LANES == 0:
                warp_sum[tx // WARP_LANES] = part[0]
            T.sync_threads()
            if tree:
                # Lane l holds warp l's sum; the warps are a power of two of at most 32.
                part[0] = warp_sum[tx % num_warps]
                for stage in T.unroll(num_warps.bit_length() - 1):
                    part[0] += T.shfl_xor(
                        part[0], T.int32(num_warps // 2) >> stage, width=WARP_LANES
                    )
            else:
                part[0] = warp_sum[0]
                for w in T.unroll(1, num_warps):
                    part[0] += warp_sum[w]
            part[0] = T.rsqrt(part[0] / float(N) + eps)
            for s in walk(steps // chunk):
                for k in T.unroll(chunk):
                    at = ((s * chunk + k) * threads + tx) * vec
                    T.call_extern(
                        "handle",
                        "tileops::load16_evict_first",
                        T.address_of(held[k * vec]),
                        T.address_of(x[row, at]),
                    )
                    if has_weight and not weight_first:
                        for c in T.vectorized(vec):
                            scale[k * vec + c] = weight[at + c]
                for p in T.unroll(width // 2):
                    w = s * width // 2 + p if weight_first else p
                    out[p * 2] = _scaled(held, p, 0, part[0], scale, w, dtype, has_weight)
                    out[p * 2 + 1] = _scaled(held, p, 1, part[0], scale, w, dtype, has_weight)
                for k in T.unroll(chunk):
                    for c in T.vectorized(vec):
                        y[row, ((s * chunk + k) * threads + tx) * vec + c] = out[k * vec + c]

    return main


def _warp_rows(M, N, eps, dtype, has_weight, lanes, per_lane, warps, rows):
    """Rows of ``lanes * per_lane`` vectors, ``lanes`` adjacent lanes of a warp a row, ``rows``
    rows a lane group.

    ``block_m`` does not apply: a block of ``warps`` warps takes as many rows as its lanes
    hold.

    A lane holds vectors ``k * lanes + lane`` of each of its rows, which lie a block's lane
    groups apart, and the group's lanes fold each row's sum of squares by shuffles: no shared
    memory, no barrier. A lane past the last row reads the last row and stores nothing.
    """
    vec = N // (lanes * per_lane)
    held_n = per_lane * vec
    groups = WARP_LANES // lanes * warps
    stages = lanes.bit_length() - 1

    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N if has_weight else 1,), dtype],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(T.ceildiv(M, groups * rows), threads=warps * WARP_LANES) as bx:
            tx = T.get_thread_binding()
            lane = tx % lanes
            first = bx * groups * rows + tx // lanes
            held = T.alloc_local((rows * held_n,), dtype)
            scale = T.alloc_local((held_n,), dtype)
            out = T.alloc_local((rows * held_n,), dtype)
            part = T.alloc_local((rows,), "float32")
            for r in T.unroll(rows):
                for k in T.unroll(per_lane):
                    load16(
                        held,
                        r * held_n + k * vec,
                        x,
                        (T.min(first + r * groups, M - 1), (k * lanes + lane) * vec),
                    )
            if has_weight:
                for k in T.unroll(per_lane):
                    for c in T.vectorized(vec):
                        scale[k * vec + c] = weight[(k * lanes + lane) * vec + c]
            for r in T.unroll(rows):
                part[r] = T.cast(0.0, "float32")
                for p in T.unroll(held_n // 2):
                    part[r] += _square(held, r * held_n // 2 + p, 0, dtype)
                    part[r] += _square(held, r * held_n // 2 + p, 1, dtype)
            # The rows' shuffles interleave, so one row's latency hides behind the next's.
            for s in T.unroll(stages):
                for r in T.unroll(rows):
                    part[r] += T.shfl_xor(part[r], T.shift_left(1, s))
            for r in T.unroll(rows):
                part[r] = T.rsqrt(part[r] / float(N) + eps)
                for p in T.unroll(held_n // 2):
                    q = r * held_n // 2 + p
                    out[q * 2] = _scaled(held, q, 0, part[r], scale, p, dtype, has_weight)
                    out[q * 2 + 1] = _scaled(held, q, 1, part[r], scale, p, dtype, has_weight)
            for r in T.unroll(rows):
                if first + r * groups < M:
                    for k in T.unroll(per_lane):
                        for c in T.vectorized(vec):
                            y[first + r * groups, (k * lanes + lane) * vec + c] = out[
                                r * held_n + k * vec + c
                            ]

    return main


@target_cache(maxsize=32)
def _rms_norm_kernel(M, N, N_padded, eps, dtype, has_weight, sm_count, smem_budget):
    col_guard = N_padded != N
    elem_bytes = DataType(dtype).bits // 8

    vec = VECTOR_ACCESS_BYTES // elem_bytes
    body = _row_body(N, N_padded, elem_bytes)

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h", "bf16_widen.h"))
    def _func(block_m, threads):
        if body == "held_row":
            widths = _held_widths(N, elem_bytes)
            if block_m != 1 or threads not in widths:
                raise ValueError(f"a held row of {N} takes one row a block at {widths} threads")
            return _held_row(M, N, eps, dtype, has_weight, threads, vec)
        if body == "warp_rows":
            # block_m does not apply: a block takes as many rows as its lanes hold.
            lanes, per_lane, rows = _warp_row_shape(N, elem_bytes)
            return _warp_rows(
                M, N, eps, dtype, has_weight, lanes, per_lane, threads // WARP_LANES, rows
            )

        # A partial per thread trades the fp32 fragment's N/threads registers,
        # which cap the resident warps, for a serial walk of shared memory. Only
        # a grid that oversubscribes the device is paid back for the walk, and
        # only once a thread owns 32 elements.
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0
        per_thread_partial = (
            not col_guard
            and -(-M // block_m) > sm_count
            # A thread count that does not divide the row truncates the walk.
            and N_padded % threads == 0
            and N_padded // threads >= 32
            # The staged rows share the block's shared memory with the reduction's
            # scratch, one fp32 a thread.
            and block_m * N_padded * elem_bytes + threads * 4 <= smem_budget
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                reduce_width = threads if per_thread_partial else N_padded
                xsq_f32 = T.alloc_fragment((block_m, reduce_width), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                if per_thread_partial:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.clear(xsq_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            v = T.cast(shared_buf[i, k * threads + j], "float32")
                            xsq_f32[i, j] += v * v

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    T.copy(shared_buf, x_local)
                else:
                    for i, j in T.Parallel(block_m, N_padded):
                        if row_guard and col_guard:
                            x_local[i, j] = T.if_then_else(
                                (pid_m * block_m + i < M) & (j < N),
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                        elif row_guard:
                            x_local[i, j] = T.if_then_else(
                                pid_m * block_m + i < M,
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                        elif col_guard:
                            x_local[i, j] = T.if_then_else(
                                j < N, x[pid_m * block_m + i, j], T.cast(0.0, dtype)
                            )
                        else:
                            x_local[i, j] = x[pid_m * block_m + i, j]

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = x * rrms * weight, written from the fragment holding the row.
                for i, j in T.Parallel(block_m, N_padded):
                    # Nested: `and` between a Python bool and a TIR comparison is not TIR.
                    if (not row_guard) or pid_m * block_m + i < M:  # noqa: SIM102
                        if (not col_guard) or j < N:
                            if has_weight:
                                y[pid_m * block_m + i, j] = T.cast(
                                    T.cast(x_local[i, j], "float32")
                                    * rrms[i]
                                    * T.cast(weight[j], "float32"),
                                    dtype,
                                )
                            else:
                                y[pid_m * block_m + i, j] = T.cast(
                                    T.cast(x_local[i, j], "float32") * rrms[i], dtype
                                )

        return main

    return _func


class RMSNormKernel(Kernel, RMSNormFwdInterface):
    """RMSNorm kernel.

    Supports SM80+ architectures. The row's width and dtype select one of the module's
    bodies (``_row_body``); rows of at most ``_SHARED_ROW_MAX`` elements share a block.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    # Rows at most this wide share a block of the default 128 threads: one of them gives
    # each thread less than one 16-byte access of a 16-bit dtype.
    _SHARED_ROW_MAX = 512

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
        # A row sharing a block is held at a power of two, so whole rows tile the fragment.
        self.N_padded = (
            1 << (N - 1).bit_length() if N <= self._SHARED_ROW_MAX else align_up(N, ALIGNMENT)
        )
        self.init_config(config)

    @property
    def _body(self) -> str:
        return _row_body(self.N, self.N_padded, self.dtype.itemsize)

    @property
    def default_config(self) -> dict:
        config = select_row_config()
        if self._body == "held_row":
            config["threads"] = _held_widths(self.N, self.dtype.itemsize)[0]
        elif self._body == "warp_rows":
            # The fewest threads a block whose resident-block cap still fills the SM's threads;
            # block_m does not apply.
            target = self.build_target
            config["threads"] = target.threads_per_sm // RESIDENT_CTAS[target.arch]
        elif self.N_padded <= self._SHARED_ROW_MAX:
            # 2048 elements a block: two 16-byte accesses a thread.
            config["block_m"] = 2048 // self.N_padded
        elif self.N_padded * self.dtype.itemsize > STATIC_SHARED_BYTES:
            config["threads"] = 256
        return config

    @property
    def autotune_configs(self) -> list[dict]:
        """Configs of the body ``_row_body`` chose; the default alone for rows of at most
        ``_SHARED_ROW_MAX`` elements, which L2-warm tuning misranks."""
        if self._body == "held_row":
            return [
                {"block_m": 1, "threads": threads}
                for threads in _held_widths(self.N, self.dtype.itemsize)
            ]
        if self.N_padded <= self._SHARED_ROW_MAX:
            return [self.default_config]
        if self._body == "warp_rows":
            # The block widths the fragment configs offer a row this wide, one row a block.
            configs = select_row_configs(self.N_padded, self.dtype, every_width_past_budget=True)
            return [{"block_m": 1, "threads": t} for t in sorted({c["threads"] for c in configs})]
        if self.default_config["block_m"] > 1:
            return [self.default_config]
        # Past the shared budget the best width still varies with the row.
        return select_row_configs(self.N_padded, self.dtype, every_width_past_budget=True)

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device,
                or ``None`` to scale by one.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: Either input is not on a device this kernel's ``devices`` names.
        """
        self._require_devices(x=x, weight=weight)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        has_weight = weight is not None
        # The prim_func keeps its weight parameter without one; it reads nothing from it.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        # The device the input is on, not whichever is current.
        program = self.program(self._compile, rows.shape[0], has_weight, x.device.index)
        return program(rows, weight).reshape(original_shape)

    def _compile(self, cfg: dict, m: int, has_weight: bool, device_index: int):
        """The program for *m* rows, with or without a weight, on device *device_index*."""
        self.kernel = kernel = _rms_norm_kernel(
            m,
            self.N,
            self.N_padded,
            self.eps,
            self.dtype_str,
            has_weight,
            self.build_target.sm_count,
            self.build_target.smem_budget,
        )
        block_m = cfg["block_m"]
        if self.N_padded <= self._SHARED_ROW_MAX:
            block_m = min(block_m, 1 << (m - 1).bit_length())
        return kernel(block_m, cfg["threads"])


class _RMSNormRowsKernel(Kernel, RMSNormFwdInterface):
    """An RMS normalization kernel built per hidden size and dtype, compiling a program per
    row count, with or without a weight.

    A subclass names its builder ``_builder(M, N, eps, dtype, has_weight)``, whose program
    takes the launch config as keyword arguments; a builder that also takes a device fact
    overrides ``_compile`` to pass it.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"rms_norm"})

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def __init__(self, N: int, eps: float, dtype: torch.dtype, config: Optional[dict] = None):
        """Build for a hidden size and dtype; ``forward`` resolves the program per row count."""
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device,
                or ``None`` to scale by one.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: Either input is not on a device this kernel's ``devices`` names.
        """
        self._require_devices(x=x, weight=weight)
        rows = x.reshape(-1, self.N)
        has_weight = weight is not None
        # The prim_func keeps its weight parameter without one; it reads nothing from it.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        program = self.program(self._compile, rows.shape[0], has_weight)
        return program(rows, weight).reshape(x.shape)

    def _compile(self, cfg: dict, m: int, has_weight: bool):
        """The program for *m* rows, with or without a weight."""
        self.kernel = kernel = self._builder(m, self.N, self.eps, self.dtype_str, has_weight)
        return kernel(**cfg)


@target_cache(maxsize=32)
def _rms_norm_paired_kernel(M, N, eps, dtype, has_weight, threads_per_sm, resident_ctas):
    elem_bytes = DataType(dtype).bits // 8
    threads = _paired_width(N, elem_bytes)
    # Where the resident-block cap, not the SM's threads, bounds the blocks, the block asks
    # for the cap: unbounded, ptxas takes more registers than the cap's blocks leave a thread
    # and the SM holds fewer of the short rows.
    blocks = resident_ctas if threads_per_sm // threads >= resident_ctas else 1

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h", "bf16_widen.h"))
    def build():
        return _held_row(
            M, N, eps, dtype, has_weight, threads, VECTOR_ACCESS_BYTES // elem_bytes, blocks
        )

    return build


class RMSNormPairedKernel(_RMSNormRowsKernel):
    """RMS normalization holding a row in registers at two 16-byte vectors a thread.

    Serves rows a 64-, 128- or 256-thread block holds that way, in sustained multi-wave
    launches: at least ``_MIN_WAVES`` waves of those blocks, as many as the SM's threads and
    resident-block cap hold, a conservative performance cutoff.
    """

    _builder = staticmethod(_rms_norm_paired_kernel)
    _MIN_WAVES = 6

    @classmethod
    def refusal(cls, call: LayerNormCall) -> Optional[str]:
        threads = _paired_width(call.n, call.dtype.itemsize)
        if threads is None:
            return f"a row of {call.n} is not two vectors a thread of 64 to 256 threads"
        per_sm = min(call.threads_per_sm // threads, RESIDENT_CTAS[call.arch])
        wave = per_sm * call.sm_count
        if call.rows < cls._MIN_WAVES * wave:
            return f"{call.rows} rows are fewer than {cls._MIN_WAVES} waves of {wave}"
        return super().refusal(call)

    def _compile(self, cfg: dict, m: int, has_weight: bool):
        """The program for *m* rows, with or without a weight, sized to the build target's SM."""
        target = self.build_target
        self.kernel = kernel = self._builder(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            has_weight,
            target.threads_per_sm,
            RESIDENT_CTAS[target.arch],
        )
        return kernel(**cfg)


@target_cache(maxsize=32)
def _rms_norm_two_pass_kernel(M, N, eps, dtype, has_weight, threads_per_sm):
    elem_bytes = DataType(dtype).bits // 8
    vec = VECTOR_ACCESS_BYTES // elem_bytes
    shape_threads, chunk, blocks, paired = _two_pass_shape(N, elem_bytes, threads_per_sm)

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h", "bf16_widen.h"))
    def _func(threads):
        if threads != shape_threads:
            raise ValueError(f"a row of {N} read twice takes {shape_threads} threads")
        tree = threads * blocks == threads_per_sm
        # Two weight vectors fit the registers a thread keeps with every block the SM's
        # threads admit resident.
        return _two_pass_row(
            M, N, eps, dtype, has_weight, threads, vec, chunk, blocks, tree, weight_first=paired
        )

    return _func


class RMSNormTwoPassKernel(_RMSNormRowsKernel):
    """Root Mean Square (RMS) normalization reading each row twice: once to sum its squares,
    once to scale it.

    One row a block; the second read finds the row in L1 or L2. Rows of two or more whole
    thread chunks, except two launches ``RMSNormKernel`` holds in registers: rows of two
    vectors a thread in fewer than ``_MIN_WAVES`` waves, and at most four rows an SM of rows
    up to 16384.
    """

    _builder = staticmethod(_rms_norm_two_pass_kernel)
    # Below this many waves, rows of two vectors a thread are faster held in registers.
    _MIN_WAVES = 16

    @classmethod
    def refusal(cls, call: LayerNormCall) -> Optional[str]:
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        shape = _two_pass_shape(call.n, call.dtype.itemsize, call.threads_per_sm)
        threads, chunk, blocks, paired = shape
        width = threads * chunk * vec
        if call.n % width or call.n < 2 * width:
            return f"a row of {call.n} elements is not two or more {threads}-thread chunks"
        wave = blocks * call.sm_count
        if paired and call.rows < cls._MIN_WAVES * wave:
            return f"{call.rows} rows are fewer than {cls._MIN_WAVES} waves of {wave}"
        # The second read adds latency that four or fewer rows an SM cannot hide.
        if call.n <= 16384 and call.rows <= 4 * call.sm_count:
            return f"{call.rows} rows of {call.n} elements are held in registers"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        shape = _two_pass_shape(self.N, self.dtype.itemsize, self.build_target.threads_per_sm)
        return {"threads": shape[0]}

    def _compile(self, cfg: dict, m: int, has_weight: bool):
        """The program for *m* rows, with or without a weight, sized to the build target's SM."""
        threads_per_sm = self.build_target.threads_per_sm
        self.kernel = kernel = self._builder(
            m, self.N, self.eps, self.dtype_str, has_weight, threads_per_sm
        )
        return kernel(**cfg)
