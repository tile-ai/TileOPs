"""RMSNorm kernel using TileLang.

y = x * rsqrt(mean(x^2) + eps) * weight

``RMSNormKernel`` runs one of three bodies, chosen by ``_row_body`` from the row's width and
dtype alone: a group of a warp's lanes a row for rows of 256 to 2048 elements, a row held in
a block's registers past 4096, and otherwise the fragment body. Tuning sizes the chosen body
and never switches bodies. ``RMSNormTwoPassKernel`` reads a row past 8192 elements twice; it
leaves a launch of at most four rows an SM, rows up to 16384, to the held body.

In the fragment body the normalized row goes from the register fragment straight to
global memory. Only the partial reduction reads shared memory, because a thread walking a
strided run of the row can reach it there and cannot reach another thread's registers.
The fragment may be wider than the row: columns past the row load zero and are never
stored, and the mean divides by N.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._csrc import csrc_include
from tileops.kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.norm._config import select_row_config, select_row_configs
from tileops.kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES, get_shared_memory_optin, get_sm_count

__all__ = ["RMSNormKernel", "RMSNormTwoPassKernel"]


def _two_pass_shape(n: int) -> tuple:
    """``(threads, vectors a thread reads at a time, blocks an SM)`` of a two-pass row of *n*."""
    return (512, 1, 3) if n <= 16384 else (1024, 2, 2)


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
    vectors = n // vec
    per_lane = max(2, vectors // WARP_LANES)
    lanes = vectors // per_lane
    if 256 <= n <= 2048 and vectors % per_lane == 0 and lanes & (lanes - 1) == 0:
        return "warp_rows"
    if n > 4096 and _held_widths(n, elem_bytes):
        return "held_row"
    return "fragment"


@T.macro
def _load16_evict_last(dst, slot, src, at):
    """Read the 16-byte vector of *src* at *at* into local *dst* from *slot*, evict-last in L2."""
    T.call_extern(
        "handle", "tileops::load16_evict_last", T.address_of(dst[slot]), T.address_of(src[at])
    )


def _held_row(M, N, eps, dtype, has_weight, threads, vec):
    """One row a block, each thread holding ``N / threads`` elements in registers.

    Thread ``t`` holds vectors ``k * threads + t``, read evict-last in L2. The sum of
    squares folds across a warp with shuffles, then across warps through shared memory.
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
            tx = T.get_thread_binding()
            held = T.alloc_local((held_n,), dtype)
            scale = T.alloc_local((held_n,), dtype)
            out = T.alloc_local((held_n,), dtype)
            part = T.alloc_local((1,), "float32")
            warp_sum = T.alloc_shared((num_warps,), "float32")

            for k in T.unroll(steps):
                _load16_evict_last(held, k * vec, x, (row, (k * threads + tx) * vec))
            part[0] = T.cast(0.0, "float32")
            for i in T.unroll(held_n):
                part[0] += T.cast(held[i], "float32") * T.cast(held[i], "float32")
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
            for i in T.unroll(held_n):
                if has_weight:
                    out[i] = T.cast(
                        T.cast(held[i], "float32") * part[0] * T.cast(scale[i], "float32"), dtype
                    )
                else:
                    out[i] = T.cast(T.cast(held[i], "float32") * part[0], dtype)
            for k in T.unroll(steps):
                for c in T.vectorized(vec):
                    y[row, (k * threads + tx) * vec + c] = out[k * vec + c]

    return main


def _two_pass_row(M, N, eps, dtype, has_weight, threads, vec, chunk, blocks):
    """One row a block, read twice: evict-last to sum its squares, then evict-first to scale it.

    A thread walks its vectors ``chunk`` at a time and holds those, not the row: the second
    read finds the row in L2, where the first pass's policy kept it. ``blocks`` is the
    blocks an SM the launch bound asks the compiler to fit.
    """
    steps = N // (threads * vec)
    num_warps = threads // WARP_LANES
    width = chunk * vec

    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N if has_weight else 1,), dtype],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(M, threads=threads) as row:
            # ``blocks`` blocks an SM where they fit (SM80, SM90; ptxas ignores it on
            # SM86/SM89): without it the compiler spends the registers they would leave free.
            T.annotate_min_blocks_per_sm(blocks)
            tx = T.get_thread_binding()
            held = T.alloc_local((width,), dtype)
            scale = T.alloc_local((width,), dtype)
            out = T.alloc_local((width,), dtype)
            part = T.alloc_local((1,), "float32")
            warp_sum = T.alloc_shared((num_warps,), "float32")

            part[0] = T.cast(0.0, "float32")
            for s in T.serial(steps // chunk):
                for k in T.unroll(chunk):
                    _load16_evict_last(
                        held, k * vec, x, (row, ((s * chunk + k) * threads + tx) * vec)
                    )
                for i in T.unroll(width):
                    part[0] += T.cast(held[i], "float32") * T.cast(held[i], "float32")
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                part[0] += T.shfl_xor(part[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES)
            if tx % WARP_LANES == 0:
                warp_sum[tx // WARP_LANES] = part[0]
            T.sync_threads()
            part[0] = warp_sum[0]
            for w in T.unroll(1, num_warps):
                part[0] += warp_sum[w]
            part[0] = T.rsqrt(part[0] / float(N) + eps)
            for s in T.serial(steps // chunk):
                for k in T.unroll(chunk):
                    at = ((s * chunk + k) * threads + tx) * vec
                    T.call_extern(
                        "handle",
                        "tileops::load16_evict_first",
                        T.address_of(held[k * vec]),
                        T.address_of(x[row, at]),
                    )
                    if has_weight:
                        for c in T.vectorized(vec):
                            scale[k * vec + c] = weight[at + c]
                for i in T.unroll(width):
                    if has_weight:
                        out[i] = T.cast(
                            T.cast(held[i], "float32") * part[0] * T.cast(scale[i], "float32"),
                            dtype,
                        )
                    else:
                        out[i] = T.cast(T.cast(held[i], "float32") * part[0], dtype)
                for k in T.unroll(chunk):
                    for c in T.vectorized(vec):
                        y[row, ((s * chunk + k) * threads + tx) * vec + c] = out[k * vec + c]

    return main


def _warp_rows(M, N, eps, dtype, has_weight, lanes, per_lane, warps):
    """Rows of ``lanes * per_lane`` vectors, ``lanes`` adjacent lanes of a warp a row.

    ``block_m`` does not apply: a block of ``warps`` warps takes as many rows as its lanes
    hold.

    A lane holds vectors ``k * lanes + lane`` of its row, read evict-last in L2, and the
    group's lanes fold the sum of squares by shuffles: no shared memory, no barrier. A lane
    past the last row reads the last row and stores nothing.
    """
    vec = N // (lanes * per_lane)
    held_n = per_lane * vec
    rows_per_cta = WARP_LANES // lanes * warps
    stages = lanes.bit_length() - 1

    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N if has_weight else 1,), dtype],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(T.ceildiv(M, rows_per_cta), threads=warps * WARP_LANES) as bx:
            tx = T.get_thread_binding()
            lane = tx % lanes
            row = T.min(bx * rows_per_cta + tx // lanes, M - 1)
            held = T.alloc_local((held_n,), dtype)
            scale = T.alloc_local((held_n,), dtype)
            out = T.alloc_local((held_n,), dtype)
            part = T.alloc_local((1,), "float32")
            for k in T.unroll(per_lane):
                _load16_evict_last(held, k * vec, x, (row, (k * lanes + lane) * vec))
            if has_weight:
                for k in T.unroll(per_lane):
                    for c in T.vectorized(vec):
                        scale[k * vec + c] = weight[(k * lanes + lane) * vec + c]
            part[0] = T.cast(0.0, "float32")
            for i in T.unroll(held_n):
                part[0] += T.cast(held[i], "float32") * T.cast(held[i], "float32")
            for s in T.unroll(stages):
                part[0] += T.shfl_xor(part[0], T.shift_left(1, s))
            part[0] = T.rsqrt(part[0] / float(N) + eps)
            for i in T.unroll(held_n):
                if has_weight:
                    out[i] = T.cast(
                        T.cast(held[i], "float32") * part[0] * T.cast(scale[i], "float32"), dtype
                    )
                else:
                    out[i] = T.cast(T.cast(held[i], "float32") * part[0], dtype)
            if bx * rows_per_cta + tx // lanes < M:
                for k in T.unroll(per_lane):
                    for c in T.vectorized(vec):
                        y[row, (k * lanes + lane) * vec + c] = out[k * vec + c]

    return main


@functools.lru_cache(maxsize=32)
def _rms_norm_kernel(M, N, N_padded, eps, dtype, has_weight, sm_count, smem_budget):
    col_guard = N_padded != N
    elem_bytes = DataType(dtype).bits // 8

    vec = VECTOR_ACCESS_BYTES // elem_bytes
    body = _row_body(N, N_padded, elem_bytes)

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h"))
    def _func(block_m, threads):
        if body == "held_row":
            widths = _held_widths(N, elem_bytes)
            if block_m != 1 or threads not in widths:
                raise ValueError(f"a held row of {N} takes one row a block at {widths} threads")
            return _held_row(M, N, eps, dtype, has_weight, threads, vec)
        if body == "warp_rows":
            # block_m does not apply: a block takes as many rows as its lanes hold.
            vectors = N // vec
            per_lane = max(2, vectors // WARP_LANES)
            lanes = vectors // per_lane
            return _warp_rows(M, N, eps, dtype, has_weight, lanes, per_lane, threads // WARP_LANES)

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

        The program for a given row count is resolved in ``forward``, memoized by
        ``_rms_norm_kernel``.
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
            # Four warps a block; block_m does not apply.
            pass
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
        x = vector_aligned(x)
        weight = vector_aligned(weight)
        self._require_devices(x=x, weight=weight)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        has_weight = weight is not None
        # The prim_func keeps its weight parameter without one; it reads nothing from it.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        m = rows.shape[0]

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _rms_norm_kernel(
            m,
            self.N,
            self.N_padded,
            self.eps,
            self.dtype_str,
            has_weight,
            # The device the input is on, not whichever is current.
            get_sm_count(x.device.index),
            get_shared_memory_optin(x.device.index),
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        block_m = self.config["block_m"]
        if self.N_padded <= self._SHARED_ROW_MAX:
            block_m = min(block_m, 1 << (m - 1).bit_length())
        y = self.kernel(block_m, self.config["threads"])(rows, weight)
        return y.reshape(original_shape)


@functools.lru_cache(maxsize=32)
def _rms_norm_two_pass_kernel(M, N, eps, dtype, has_weight):
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    shape_threads, chunk, blocks = _two_pass_shape(N)

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h"))
    def _func(threads):
        if threads != shape_threads:
            raise ValueError(f"a row of {N} read twice takes {shape_threads} threads")
        return _two_pass_row(M, N, eps, dtype, has_weight, threads, vec, chunk, blocks)

    return _func


class RMSNormTwoPassKernel(Kernel, RMSNormFwdInterface):
    """Root Mean Square (RMS) normalization reading each row twice: once to sum its squares,
    once to scale it.

    One row a block; the second read finds the row in L2. Rows past 8192 elements that
    divide into whole thread chunks, except a launch of at most four rows an SM of rows up
    to 16384, which ``RMSNormKernel`` holds in registers.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"rms_norm"})

    @classmethod
    def refusal(cls, call: LayerNormCall) -> Optional[str]:
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        threads, chunk, _ = _two_pass_shape(call.n)
        if call.n <= 8192:
            return f"a row of {call.n} elements is held in registers or a block's fragment"
        if call.n % (threads * chunk * vec):
            return f"a row of {call.n} elements does not divide into {threads}-thread chunks"
        # The second read adds latency that four or fewer rows an SM cannot hide.
        if call.n <= 16384 and call.rows <= 4 * call.sm_count:
            return f"{call.rows} rows of {call.n} elements are held in registers"
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
        """Build for a hidden size and dtype; ``forward`` resolves the program per row count."""
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": _two_pass_shape(self.N)[0]}

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
            ValueError: Either input is not on a CUDA device.
        """
        x = vector_aligned(x)
        weight = vector_aligned(weight)
        if not (x.is_cuda and (weight is None or weight.is_cuda)):
            weight_device = None if weight is None else weight.device
            raise ValueError(
                f"{type(self).__name__} is a CUDA kernel; got x on {x.device} and weight on "
                f"{weight_device}. Another target's backend serves other devices."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        has_weight = weight is not None
        # The prim_func keeps its weight parameter without one; it reads nothing from it.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _rms_norm_two_pass_kernel(
            rows.shape[0], self.N, self.eps, self.dtype_str, has_weight
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()
        y = self.kernel(self.config["threads"])(rows, weight)
        return y.reshape(original_shape)
