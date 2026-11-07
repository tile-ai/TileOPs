"""LayerNorm with register fragments or shared-memory row reductions.

Rows are padded to 256 elements; padded values are excluded from statistics."""

from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm._config import (
    make_centred_row_reduce,
    make_raw_sums_hold,
    make_read_leads,
    pairwise_sum,
    select_row_config,
    select_row_configs,
)
from tileops._kernels.norm.call_spec import LayerNormCall, LayerNormFwdInterface
from tileops._kernels.norm.fused_add_norm import LayerNormHeldKernel
from tileops._kernels.tiling import ALIGNMENT, align_up
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES, get_sm_count

__all__ = ["LayerNormKernel"]


def _walk_terms(buf, i, start, count, centre, width=None):
    """``buf[i, start : start + count]`` less *centre* in fp32, zero at or past *width*."""
    terms = []
    for v in range(count):
        d = T.cast(buf[i, start + v], "float32") - centre
        if width is not None:
            d = T.if_then_else(start + v < width, d, 0.0)
        terms.append(d)
    return terms


@target_cache(maxsize=32)
def _layer_norm_kernel(M, N, eps, dtype, has_weight, has_bias, sm_count):
    N_padded = align_up(N, ALIGNMENT)
    needs_pad = N_padded != N

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):
        # A partial per thread trades the fp32 fragment's N/threads registers,
        # which cap the resident warps, for a serial walk of shared memory. Only
        # a grid that oversubscribes the device is paid back for the walk, and
        # only once a thread owns 64 elements: twice RMSNorm's, for two walks.
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0
        per_thread_partial = (
            # The walk holds one row a block, which is all autotuning offers.
            block_m == 1
            and -(-M // block_m) > sm_count
            # A thread count that does not divide the row truncates the walk.
            and N_padded % threads == 0
            and N_padded // threads >= 64
            # FIXME(staged-rollout): an odd walk is routed to the fragment path
            #
            # Broken invariant: the shared-memory walk serves every count it divides.
            # Why: TileLang generates an out-of-bounds shared read for some odd counts,
            #      95 elements a thread at 256 threads among them.
            # Cleanup: drop this clause once tile-ai/tilelang#3438 is fixed.
            and N_padded // threads % 2 == 0
        )

        walk_vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
        while (N_padded // threads) % walk_vec:
            walk_vec //= 2
        walk_steps = N_padded // (threads * walk_vec)
        warps = threads // WARP_LANES
        warp_ranks = list(range(warps))
        raw_adds = (
            (walk_vec - 1).bit_length()
            + walk_steps
            + WARP_SHUFFLE_STAGES
            + (warps - 1).bit_length()
        )
        read_leads = make_read_leads(block_m, M, N)
        row_reduce = make_centred_row_reduce(block_m, N, N_padded, eps)

        def affine(normed, weight, bias, j):
            """*normed* scaled by the weight and shifted by the bias the call passed."""
            if has_weight:
                normed = normed * T.cast(weight[j], "float32")
            if has_bias:
                normed = normed + T.cast(bias[j], "float32")
            return normed

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            bias: T.Tensor[(N if has_bias else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if per_thread_partial:
                    tx = T.get_thread_binding()
                    warp = tx // WARP_LANES
                    shared_buf = T.alloc_shared((1, N_padded), dtype)
                    sums = T.alloc_local([3], "float32")
                    # (centre, rstd, the mean's share of the output).
                    scaled = T.alloc_local([3], "float32")
                    slots = T.alloc_shared([5 * warps], "float32")
                    width = N if needs_pad else None
                    if needs_pad:
                        for i, j in T.Parallel(1, N_padded):
                            shared_buf[i, j] = T.if_then_else(
                                j < N, x[pid_m, j], T.cast(0.0, dtype)
                            )
                    else:
                        T.copy(x[pid_m, 0], shared_buf)
                    # Raw sums, kept where `make_raw_sums_hold` holds; the block shares one verdict.
                    sums[1] = T.float32(0)
                    sums[2] = T.float32(0)
                    for k in T.serial(walk_steps):
                        terms = _walk_terms(
                            shared_buf, 0, (k * threads + tx) * walk_vec, walk_vec, 0.0, width
                        )
                        sums[1] += pairwise_sum(terms)
                        sums[2] += pairwise_sum([t * t for t in terms])
                    for st in T.unroll(WARP_SHUFFLE_STAGES):
                        sums[1] += T.shfl_xor(sums[1], T.shift_left(1, st))
                        sums[2] += T.shfl_xor(sums[2], T.shift_left(1, st))
                    if tx % WARP_LANES == 0:
                        slots[3 * warps + warp] = sums[1]
                        slots[4 * warps + warp] = sums[2]
                    T.sync_threads()
                    raw_sum = pairwise_sum([slots[3 * warps + v] for v in warp_ranks])
                    raw_mean = raw_sum * (1.0 / N)
                    raw_square = pairwise_sum([slots[4 * warps + v] for v in warp_ranks]) * (
                        1.0 / N
                    )
                    raw_var = raw_square - raw_mean * raw_mean
                    if make_raw_sums_hold(dtype, raw_adds)(raw_square, raw_var):
                        scaled[0] = raw_mean
                        scaled[1] = T.rsqrt(raw_var + eps)
                        scaled[2] = T.float32(0)
                    else:
                        lead = T.cast(shared_buf[0, 0], "float32")

                        sums[0] = T.float32(0)
                        for k in T.serial(walk_steps):
                            sums[0] += pairwise_sum(
                                _walk_terms(
                                    shared_buf,
                                    0,
                                    (k * threads + tx) * walk_vec,
                                    walk_vec,
                                    lead,
                                    width,
                                )
                            )
                        for st in T.unroll(WARP_SHUFFLE_STAGES):
                            sums[0] += T.shfl_xor(sums[0], T.shift_left(1, st))
                        if tx % WARP_LANES == 0:
                            slots[warp] = sums[0]
                        T.sync_threads()
                        sums[0] = T.float32(0)
                        for v in T.unroll(warps):
                            sums[0] += slots[v]
                        centre = lead + sums[0] * (1.0 / N)

                        sums[1] = T.float32(0)
                        sums[2] = T.float32(0)
                        for k in T.serial(walk_steps):
                            terms = _walk_terms(
                                shared_buf,
                                0,
                                (k * threads + tx) * walk_vec,
                                walk_vec,
                                centre,
                                width,
                            )
                            sums[1] += pairwise_sum(terms)
                            sums[2] += pairwise_sum([t * (t * (1.0 / N)) for t in terms])
                        for st in T.unroll(WARP_SHUFFLE_STAGES):
                            sums[1] += T.shfl_xor(sums[1], T.shift_left(1, st))
                            sums[2] += T.shfl_xor(sums[2], T.shift_left(1, st))
                        if tx % WARP_LANES == 0:
                            slots[warps + warp] = sums[1]
                            slots[2 * warps + warp] = sums[2]
                        T.sync_threads()
                        sums[1] = T.float32(0)
                        sums[2] = T.float32(0)
                        for v in T.unroll(warps):
                            sums[1] += slots[warps + v]
                            sums[2] += slots[2 * warps + v]
                        mean = sums[1] * (1.0 / N)
                        scaled[0] = centre
                        scaled[1] = T.rsqrt(sums[2] - mean * mean + eps)
                        scaled[2] = -mean * scaled[1]

                    for i, j in T.Parallel(1, N_padded):
                        if (not needs_pad) or j < N:
                            y[pid_m, j] = T.cast(
                                affine(
                                    (T.cast(shared_buf[i, j], "float32") - scaled[0]) * scaled[1]
                                    + scaled[2],
                                    weight,
                                    bias,
                                    j,
                                ),
                                dtype,
                            )
                else:
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    shift = T.alloc_fragment((block_m,), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    centre = T.alloc_fragment((block_m,), "float32")
                    offset = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")
                    read_leads(x, pid_m, shift)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    # The fragment holds the row for the output pass while the
                    # fp32 copy below is reduced and overwritten.
                    if needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                    elif row_guard:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(
                                pid_m * block_m + i < M,
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = x[pid_m * block_m + i, j]
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.if_then_else(
                            (not needs_pad) or j < N,
                            T.cast(x_local[i, j], "float32") - shift[i],
                            0.0,
                        )
                    row_reduce(x_local, x_f32, shift, acc, centre, offset, rstd)

                    for i, j in T.Parallel(block_m, N_padded):
                        if (not needs_pad and not row_guard) or T.And(
                            pid_m * block_m + i < M, j < N
                        ):
                            y[pid_m * block_m + i, j] = T.cast(
                                affine(
                                    (T.cast(x_local[i, j], "float32") - centre[i]) * rstd[i]
                                    + offset[i],
                                    weight,
                                    bias,
                                    j,
                                ),
                                dtype,
                            )

        return main

    return _func


class LayerNormKernel(Kernel, LayerNormFwdInterface):
    """LayerNorm kernel.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for the shared buffer the partial reduction walks; every other
    path holds the row in a register fragment from the load through the store.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: LayerNormCall) -> "str | None":
        if not LayerNormHeldKernel._queued_past_registers(
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
        return select_row_configs(self.N_padded, self.dtype, block_ms=(1,))

    def forward(
        self, x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Flattening to 2-D rows and a flat weight and bias happens here; the prim_func
        handles the non-aligned tail itself.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device, or
                ``None`` to scale by one.
            bias: Affine shift holding ``N`` elements, contiguous, on the same device, or
                ``None`` to shift by zero.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, weight=weight, bias=bias)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        has_weight, has_bias = weight is not None, bias is not None
        # An absent tensor is a one-element placeholder the program never reads.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        bias = bias.reshape(self.N) if has_bias else rows.new_empty(1)

        # The device the input is on, not whichever is current.
        program = self.program(
            self._compile, rows.shape[0], has_weight, has_bias, rows.device.index
        )
        return program(rows, weight, bias).reshape(original_shape)

    def _compile(self, cfg: dict, m: int, has_weight: bool, has_bias: bool, device_index: int):
        """The program for *m* rows, with or without a weight and a bias, on *device_index*."""
        sm_count = get_sm_count(device_index)
        kernel = _layer_norm_kernel(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            has_weight,
            has_bias,
            sm_count,
        )
        program = kernel(cfg["block_m"], cfg["threads"])
        # Set after the build succeeds, so tuning never reads one that failed.
        self.kernel = kernel
        if self._tune_target is None:
            self._tune_target = (kernel, cfg)
        return program
