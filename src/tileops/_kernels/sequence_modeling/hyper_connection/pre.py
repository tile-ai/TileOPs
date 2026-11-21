"""The manifold hyper-connection pre step.

The projection launch multiplies the flattened streams by ``fn`` on the tensor cores and
sums their squares, split over the reduction axis when the tokens are too few to fill the
device. ``fn`` takes part as three bfloat16 terms whose sum is its float32 value, so every
product is exact, and each 64-channel step's products are added to float32 accumulators. A
warpgroup of the projection splits each step's slice of ``fn`` as it arrives. The mix launch
reduces the splits and, one block per token, computes the mixes, runs Sinkhorn in one warp
and writes the weighted stream sum.
"""

import math

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import WARPGROUP_THREADS, WGMMA_ROWS
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.sequence_modeling.hyper_connection.call_spec import (
    HyperConnectionPreCall,
    HyperConnectionPreFwdInterface,
)
from tileops._utils import WARP_LANES

__all__ = ["HyperConnectionPreKernel"]

_STEP = 64


@target_cache(maxsize=32)
def _prenorm_gemm_kernel(
    n: int,
    c: int,
    streams: int,
    splits: int,
    block_m: int,
    num_stages: int,
):
    """Per split of the ``n * c`` reduction: ``partial = x @ fn^T`` and ``sqr = sum(x^2)``.

    ``x`` holds ``streams`` stored streams a token, 1 for a broadcast residual, whose one
    stream stands for all ``n``. Split ``s`` takes the 64-channel steps from ``s * steps //
    splits`` up to the next split's first, so a split count that does not divide the steps
    still fills the SMs. The last warp loads each step's ``block_m`` tokens and float32 slice
    of ``fn`` into a ring of ``num_stages`` slots. The warpgroup before it splits the slice
    into its three bfloat16 terms in the slot. Each warpgroup before that takes 64 of the
    tokens into registers, sums their squares and multiplies them by the terms, one 64 x 72
    WGMMA per 16 channels, the step's products added into float32.
    """
    k_total = n * c
    q = n * n + 2 * n
    rows = 3 * q
    steps = k_total // _STEP
    workers = block_m // WGMMA_ROWS * WARPGROUP_THREADS
    threads = workers + WARPGROUP_THREADS + WARP_LANES
    regs = rows // 2
    split_tile = f"tileops::hc_split_tile<{q}, {WARPGROUP_THREADS}>"
    wgmma = f"tileops::dense_wgmma_64xN_k64_rs<tl::DataType::kBFloat16, {rows}>"
    tokens = T.dynamic("tokens", "int64")

    @tilelang.jit(
        out_idx=[2, 3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            *csrc_include("dense_wgmma_helper.h", "hyper_connection_helper.h"),
        ],
    )
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((tokens, streams * c), "bfloat16"),
            fn: T.Tensor((q, k_total), "float32"),
            partial: T.Tensor((splits, tokens, q), "float32"),
            sqr: T.Tensor((splits, tokens), "float32"),
        ):
            with T.Kernel(T.ceildiv(tokens, block_m), splits, threads=threads) as (bm, bs):
                x_s = T.alloc_shared((num_stages, block_m, _STEP), "bfloat16")
                fn_s = T.alloc_shared((num_stages, q, _STEP), "float32")
                w_s = T.alloc_shared((num_stages, rows, _STEP), "bfloat16")
                T.annotate_layout(
                    {
                        x_s: tilelang.layout.make_swizzled_layout(x_s),
                        w_s: tilelang.layout.make_swizzled_layout(w_s),
                    }
                )
                # A slot is full once the loader's tokens land and the splitter's terms are
                # stored; the slice the splitter reads has a barrier of its own.
                full = T.alloc_barrier([WARP_LANES + WARPGROUP_THREADS] * num_stages)
                sliced = T.alloc_barrier([WARP_LANES] * num_stages)
                empty = T.alloc_barrier([workers] * num_stages)
                a = T.alloc_local((16,), "uint32")
                tile = T.alloc_local((regs,), "float32")
                total = T.alloc_local((regs,), "float32")
                squares = T.alloc_local((2,), "float32")
                tx = T.get_thread_binding()
                m0 = bm * block_m
                first = bs * steps // splits
                count = (bs + 1) * steps // splits - first
                if tx >= workers + WARPGROUP_THREADS:
                    for g in T.serial(count):
                        slot = g % num_stages
                        T.barrier_wait(empty[slot], ((g // num_stages) & 1) ^ 1)
                        k0 = (first + g) * _STEP
                        xk = k0 % c if streams == 1 else k0
                        T.tma_copy(
                            x[m0 : m0 + block_m, xk : xk + _STEP],
                            x_s[slot, :, :],
                            barrier=full[slot],
                        )
                        T.tma_copy(fn[0:q, k0 : k0 + _STEP], fn_s[slot, :, :], barrier=sliced[slot])
                        T.barrier_arrive(sliced[slot])
                        T.barrier_arrive(full[slot])
                elif tx >= workers:
                    for g in T.serial(count):
                        slot = g % num_stages
                        T.barrier_wait(sliced[slot], (g // num_stages) & 1)
                        T.call_extern(
                            "handle",
                            split_tile,
                            T.address_of(fn_s[slot, 0, 0]),
                            T.address_of(w_s[slot, 0, 0]),
                            tx - workers,
                        )
                        T.fence_proxy_async()
                        T.barrier_arrive(full[slot])
                else:
                    wg = tx // WARPGROUP_THREADS
                    wt = tx % WARPGROUP_THREADS
                    for r in T.unroll(regs):
                        total[r] = 0.0
                    squares[0] = 0.0
                    squares[1] = 0.0
                    for g in T.serial(count):
                        slot = g % num_stages
                        T.barrier_wait(full[slot], (g // num_stages) & 1)
                        T.call_extern(
                            "handle",
                            "tileops::dense_load_a_64x64",
                            T.address_of(x_s[slot, wg * WGMMA_ROWS, 0]),
                            a.data,
                        )
                        T.call_extern("handle", "tileops::hc_square_regs", a.data, squares.data)
                        T.call_extern(
                            "handle", wgmma, tile.data, a.data, T.address_of(w_s[slot, 0, 0]), 0
                        )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(tile, num_regs=regs)
                        T.barrier_arrive(empty[slot])
                        for r in T.unroll(regs):
                            total[r] += tile[r]
                    # A thread holds rows ``row`` and ``row + 8`` of its warpgroup's 64.
                    row = m0 + wg * WGMMA_ROWS + wt // WARP_LANES * 16 + wt % WARP_LANES // 4
                    upper = T.call_extern("float32", "tileops::hc_quad_sum", squares[0])
                    lower = T.call_extern("float32", "tileops::hc_quad_sum", squares[1])
                    if wt % 4 == 0:
                        if row < tokens:
                            sqr[bs, row] = upper
                        if row + 8 < tokens:
                            sqr[bs, row + 8] = lower
                    for cg in T.unroll(q // 8):
                        for h in T.unroll(2):
                            for e in T.unroll(2):
                                if row + 8 * h < tokens:
                                    partial[bs, row + 8 * h, 8 * cg + 2 * (wt % 4) + e] = total[
                                        4 * cg + 2 * h + e
                                    ] + (
                                        total[4 * (cg + q // 8) + 2 * h + e]
                                        + total[4 * (cg + q // 4) + 2 * h + e]
                                    )

        return main

    return _func()


@target_cache(maxsize=32)
def _pre_mix_kernel(
    n: int,
    c: int,
    streams: int,
    splits: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """One block a token. Every thread sums a share of the splits; then the first warp
    computes the post and comb mixes and runs Sinkhorn while the other two write the stream
    sum, staged through shared memory up to 512 channels at a time. The launch may start
    while the projection drains: only the shares wait for it, and the two stream warps issue
    their first loads before summing theirs."""
    q = n * n + 2 * n
    parts = min(4, splits)
    block = math.gcd(512, c)
    tokens = T.dynamic("tokens", "int64")

    @tilelang.jit(
        out_idx=[5, 6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
        compile_flags=["-O3", *csrc_include("hyper_connection_helper.h")],
    )
    def _func():
        @T.macro
        def share(partial, sqr, part_s, t, tx):
            # Thread tx sums part tx // q of the splits of mix tx % q, part p being splits
            # [p * splits // parts, (p + 1) * splits // parts), and each of the first `parts`
            # threads part tx of the squares. A part one split short reads its last split
            # again and adds it as zero, so no load waits on a branch. The splits are the
            # projection's output, so the wait for it comes first.
            T.pdl_sync()
            if tx < parts * q:
                lo = tx // q * splits // parts
                hi = (tx // q + 1) * splits // parts
                acc = T.alloc_var("float32")
                acc = 0.0
                for sp in T.unroll(-(-splits // parts)):
                    acc += (
                        T.if_then_else(lo + sp < hi, 1.0, 0.0)
                        * partial[T.min(lo + sp, hi - 1), t, tx % q]
                    )
                part_s[tx // q, tx % q] = acc
            if tx < parts:
                lo = tx * splits // parts
                hi = (tx + 1) * splits // parts
                acc = T.alloc_var("float32")
                acc = 0.0
                for sp in T.unroll(-(-splits // parts)):
                    acc += T.if_then_else(lo + sp < hi, 1.0, 0.0) * sqr[T.min(lo + sp, hi - 1), t]
                part_s[tx, q] = acc

        @T.macro
        def mix(part_s, j):
            # Mix j of the token, from the parts the block summed.
            whole = T.alloc_var("float32")
            squares = T.alloc_var("float32")
            whole = 0.0
            squares = 0.0
            for p in T.unroll(parts):
                whole += part_s[p, j]
                squares += part_s[p, q]
            return whole * T.rsqrt(squares / (n * c) + rms_eps)

        @T.prim_func
        def main(
            x: T.Tensor((tokens, streams, c), "bfloat16"),
            partial: T.Tensor((splits, tokens, q), "float32"),
            sqr: T.Tensor((splits, tokens), "float32"),
            hc_scale: T.Tensor((3,), "float32"),
            hc_base: T.Tensor((q,), "float32"),
            post_mix: T.Tensor((tokens, n), "float32"),
            comb_mix: T.Tensor((tokens, n * n), "float32"),
            layer_input: T.Tensor((tokens, c), "bfloat16"),
        ):
            with T.Kernel(tokens, threads=3 * WARP_LANES) as t:
                tx = T.get_thread_binding()
                part_s = T.alloc_shared((parts, q + 1), "float32")
                # Every share is in ``part_s`` once all three warps arrive; the two stream warps
                # then wait for the pre weights.
                summed = 1
                weighed = 2
                if tx < WARP_LANES:
                    share(partial, sqr, part_s, t, tx)
                    T.sync_threads(barrier_id=summed, arrive_count=3 * WARP_LANES)
                    if tx < n:
                        post_mix[t, tx] = (
                            T.sigmoid(mix(part_s, n + tx) * hc_scale[1] + hc_base[n + tx])
                            * hc_post_mult_value
                        )
                    lane = tx % (n * n)
                    comb = T.call_extern(
                        "float32",
                        "tileops::hc_sinkhorn4",
                        mix(part_s, 2 * n + lane) * hc_scale[2] + hc_base[2 * n + lane],
                        sinkhorn_repeat,
                        T.float32(hc_sinkhorn_eps),
                    )
                    if tx < n * n:
                        comb_mix[t, lane] = comb
                else:
                    pre = T.alloc_shared((n,), "float32")
                    for b in T.Pipelined(c // block, num_stages=3):
                        xs = T.alloc_shared((streams, block), "bfloat16")
                        xl = T.alloc_fragment((streams, block), "float32")
                        T.copy(x[t, 0, b * block], xs)
                        if b == 0:
                            share(partial, sqr, part_s, t, tx)
                            T.sync_threads(barrier_id=summed, arrive_count=3 * WARP_LANES)
                            if tx < WARP_LANES + n:
                                i = tx - WARP_LANES
                                pre[i] = (
                                    T.sigmoid(mix(part_s, i) * hc_scale[0] + hc_base[i])
                                    + hc_pre_eps
                                )
                            T.sync_threads(barrier_id=weighed, arrive_count=2 * WARP_LANES)
                        T.copy(xs, xl)
                        part = T.alloc_fragment((block,), "float32")
                        T.clear(part)
                        for i in T.serial(n):
                            for k in T.Parallel(block):
                                part[k] += pre[i] * xl[0 if streams == 1 else i, k]
                        T.copy(part, layer_input[t, b * block])

        return main

    return _func()


class HyperConnectionPreKernel(Kernel, HyperConnectionPreFwdInterface):
    """The manifold hyper-connection pre step on SM90 tensor cores, for four streams.

    The projection's products are exact and each 64-channel step's add into float32, so the
    mixes are as accurate as a float32 matrix product. The token count is a runtime extent:
    a launch picks the tile and the reduction splits from it, so a new token count builds
    programs only where it picks a new pair.

    Args:
        n: Streams a token.
        c: Channels a stream.
        broadcast: Whether the residual's stream axis has stride 0, its first stream
            standing for all ``n``.
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat: The op's
            constants.
        sm_count: Device SM count, which sizes the reduction splits.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: HyperConnectionPreCall) -> str | None:
        if call.n != 4:
            return f"serves four streams, got {call.n}"
        if not call.c or call.c % 128:
            return f"requires the stream width to be a positive multiple of 128, got {call.c}"
        if call.sinkhorn_repeat < 1:
            return f"requires at least one Sinkhorn iteration, got {call.sinkhorn_repeat}"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HyperConnectionPreCall) -> Entry:
        return call, lambda: cls(
            call.n,
            call.c,
            call.broadcast,
            call.rms_eps,
            call.hc_pre_eps,
            call.hc_sinkhorn_eps,
            call.hc_post_mult_value,
            call.sinkhorn_repeat,
            sm_count=call.sm_count,
        )

    def __init__(
        self,
        n: int,
        c: int,
        broadcast: bool,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        *,
        sm_count: int,
    ):
        super().__init__()
        self.n = n
        self.c = c
        self.broadcast = broadcast
        self.rms_eps = rms_eps
        self.hc_pre_eps = hc_pre_eps
        self.hc_sinkhorn_eps = hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat
        self.sm_count = sm_count

    def _launch(self, tokens: int) -> tuple[int, int, int]:
        """The ``block_m``, ``num_stages`` and ``splits`` a call of *tokens* tokens launches
        with; *tokens* is positive."""
        # A call within one WGMMA's rows would leave a second warpgroup idle; its splits run a
        # few steps, which four slots cover, and six cover a 128-token tile's load latency.
        small = tokens <= WGMMA_ROWS
        block_m, num_stages = (WGMMA_ROWS, 4) if small else (2 * WGMMA_ROWS, 6)
        # The splits fill the SMs the token tiles leave. The projection lasts as long as its
        # longest split, and of the counts with that longest split the fewest leave the mix
        # launch the fewest partial sums.
        steps = self.n * self.c // _STEP
        fill = max(1, min(steps, self.sm_count // -(-tokens // block_m)))
        return block_m, num_stages, -(-steps // -(-steps // fill))

    def forward(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_devices(residual=residual, fn=fn, hc_scale=hc_scale, hc_base=hc_base)
        lead = residual.shape[:-2]
        tokens = math.prod(lead)
        stored = residual[..., :1, :] if self.broadcast else residual
        x = stored.reshape(tokens, stored.shape[-2], self.c).contiguous()
        gemm, mix = self.program(self._compile, *self._launch(tokens))
        partial, sqr = gemm(x.view(tokens, -1), fn)
        post_mix, comb_mix, layer_input = mix(x, partial, sqr, hc_scale, hc_base)
        return (
            post_mix.view(*lead, self.n, 1),
            comb_mix.view(*lead, self.n, self.n),
            layer_input.view(*lead, self.c),
        )

    def _compile(
        self, cfg: dict, block_m: int, num_stages: int, splits: int
    ) -> tuple[CompiledProgram, CompiledProgram]:
        """The projection and mix programs a launch of these facts runs."""
        streams = 1 if self.broadcast else self.n
        gemm = _prenorm_gemm_kernel(self.n, self.c, streams, splits, block_m, num_stages)
        mix = _pre_mix_kernel(
            self.n,
            self.c,
            streams,
            splits,
            self.rms_eps,
            self.hc_pre_eps,
            self.hc_sinkhorn_eps,
            self.hc_post_mult_value,
            self.sinkhorn_repeat,
        )
        return CompiledProgram(gemm), CompiledProgram(mix)
