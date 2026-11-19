"""The manifold hyper-connection pre step.

The projection launch multiplies the flattened streams by ``fn`` on the tensor cores and
sums their squares, split over the reduction axis when the tokens are too few to fill the
device. ``fn`` takes part as three bfloat16 terms whose sum is its float32 value, so every
product is exact, and each 64-channel slot's products are added to float32 accumulators.
A small call splits ``fn`` in the projection, a large one in a launch before it. The mix
launch reduces the splits and, one block per token, computes the mixes, runs Sinkhorn in
one warp and writes the weighted stream sum.
"""

import math

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    SHARED_BUFFER_ALIGN_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.sequence_modeling.hyper_connection.call_spec import (
    HyperConnectionPreCall,
    HyperConnectionPreFwdInterface,
)
from tileops._utils import WARP_LANES

__all__ = ["HyperConnectionPreKernel"]

_STEP = 64


@target_cache(maxsize=8)
def _split_fn_kernel(rows: int, cols: int):
    """``fn`` ``(rows, cols)`` float32 to its three bfloat16 terms stacked ``(3 * rows, cols)``,
    eight values a thread."""
    chunks = rows * cols // 8
    threads = WARPGROUP_THREADS

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", *csrc_include("hyper_connection_helper.h")])
    def _func():
        @T.prim_func
        def main(
            fn: T.Tensor((rows, cols), "float32"),
            stacked: T.Tensor((3 * rows, cols), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(chunks, threads), threads=threads) as b:
                ch = b * threads + T.get_thread_binding()
                if ch < chunks:
                    T.call_extern(
                        "handle",
                        "tileops::hc_split_global",
                        T.address_of(fn[ch // (cols // 8), ch % (cols // 8) * 8]),
                        T.address_of(stacked[ch // (cols // 8), ch % (cols // 8) * 8]),
                        rows * cols,
                    )

        return main

    return _func()


@target_cache(maxsize=32)
def _prenorm_gemm_kernel(
    n: int,
    c: int,
    streams: int,
    splits: int,
    block_m: int,
    num_stages: int,
    resident: bool,
):
    """Per split of the ``n * c`` reduction: ``partial = x @ fn^T`` and ``sqr = sum(x^2)``.

    ``x`` holds ``streams`` stored streams a token, 1 for a broadcast residual, whose one
    stream stands for all ``n``. The last warp loads ``block_m`` tokens 64 channels at a
    time into a ring of ``num_stages`` slots. Each warpgroup before it takes 64 of the
    tokens: one 64 x 72 WGMMA per 16 channels against the stacked terms, the slot's
    products added into float32, the slot's squares summed. With *resident* the program
    takes ``fn`` in float32 and the warpgroups split the split's whole slice into shared
    memory before the first slot; otherwise it takes the stacked terms and loads each
    slot's with its tokens.
    """
    k_total = n * c
    q = n * n + 2 * n
    rows = 3 * q
    k_split = k_total // splits
    steps = k_split // _STEP
    workers = block_m // 64 * WARPGROUP_THREADS
    threads = workers + WARP_LANES
    regs = rows // 2
    terms_barrier = 8
    wgmma = f"tileops::dense_wgmma_64xN_k64<tl::DataType::kBFloat16, {rows}>"
    split_slice = f"tileops::hc_split_slice<{q}, {steps}, {workers}, 4>"
    weight_shape = (q, k_total) if resident else (rows, k_total)
    weight_dtype = "float32" if resident else "bfloat16"
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
            weight: T.Tensor(weight_shape, weight_dtype),
            partial: T.Tensor((splits, tokens, q), "float32"),
            sqr: T.Tensor((splits, tokens), "float32"),
        ):
            with T.Kernel(T.ceildiv(tokens, block_m), splits, threads=threads) as (bm, bs):
                x_s = T.alloc_shared((num_stages, block_m, _STEP), "bfloat16")
                w_s = T.alloc_shared((steps if resident else num_stages, rows, _STEP), "bfloat16")
                T.annotate_layout(
                    {
                        x_s: tilelang.layout.make_swizzled_layout(x_s),
                        w_s: tilelang.layout.make_swizzled_layout(w_s),
                    }
                )
                full = T.alloc_barrier([WARP_LANES] * num_stages)
                empty = T.alloc_barrier([workers] * num_stages)
                tile = T.alloc_local((regs,), "float32")
                total = T.alloc_local((regs,), "float32")
                sq = T.alloc_local((1,), "float32")
                tx = T.get_thread_binding()
                m0 = bm * block_m
                k_base = bs * k_split
                if tx >= workers:
                    for g in T.Pipelined(steps, num_stages=0):
                        slot = g % num_stages
                        T.barrier_wait(empty[slot], ((g // num_stages) & 1) ^ 1)
                        k0 = k_base + g * _STEP
                        xk = k0 % c if streams == 1 else k0
                        T.tma_copy(
                            x[m0 : m0 + block_m, xk : xk + _STEP],
                            x_s[slot, :, :],
                            barrier=full[slot],
                        )
                        if not resident:
                            T.tma_copy(
                                weight[0:rows, k0 : k0 + _STEP],
                                w_s[slot, :, :],
                                barrier=full[slot],
                            )
                        T.barrier_arrive(full[slot])
                else:
                    wg = tx // WARPGROUP_THREADS
                    wt = tx % WARPGROUP_THREADS
                    for r in T.unroll(regs):
                        total[r] = 0.0
                    sq[0] = 0.0
                    if resident:
                        T.call_extern(
                            "handle",
                            split_slice,
                            T.address_of(weight[0, k_base]),
                            k_total,
                            T.address_of(w_s[0, 0, 0]),
                            tx,
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=terms_barrier, arrive_count=workers)
                    for g in T.Pipelined(steps, num_stages=0):
                        slot = g % num_stages
                        T.barrier_wait(full[slot], (g // num_stages) & 1)
                        T.call_extern(
                            "handle",
                            wgmma,
                            tile.data,
                            T.address_of(x_s[slot, wg * 64, 0]),
                            T.address_of(w_s[g if resident else slot, 0, 0]),
                            0,
                        )
                        sq[0] += T.call_extern(
                            "float32",
                            "tileops::hc_square_sum",
                            T.address_of(x_s[slot, wg * 64, 0]),
                            wt,
                        )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(tile, num_regs=regs)
                        T.barrier_arrive(empty[slot])
                        for r in T.unroll(regs):
                            total[r] += tile[r]
                    whole = T.call_extern("float32", "tileops::hc_pair_sum", sq[0])
                    if wt % 2 == 0 and m0 + wg * 64 + wt // 2 < tokens:
                        sqr[bs, m0 + wg * 64 + wt // 2] = whole
                    row = m0 + wg * 64 + wt // WARP_LANES * 16 + wt % WARP_LANES // 4
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
    staged: bool,
):
    """One block a token: the splits' sums and the mixes, then one warp writes the post and
    comb mixes while two write the stream sum, staged through shared memory up to 512
    channels at a time when *staged*, else loaded whole into registers."""
    q = n * n + 2 * n
    parts = math.gcd(4, splits)
    block = math.gcd(512, c)
    tokens = T.dynamic("tokens", "int64")

    @tilelang.jit(
        out_idx=[5, 6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
    )
    def _func():
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
                part_s = T.alloc_shared((parts, q + 1), "float32")
                for p, j in T.Parallel(parts, q):
                    acc = T.alloc_var("float32")
                    acc = 0.0
                    for sp in T.unroll(splits // parts):
                        acc += partial[p * (splits // parts) + sp, t, j]
                    part_s[p, j] = acc
                for p in T.Parallel(parts):
                    acc = T.alloc_var("float32")
                    acc = 0.0
                    for sp in T.unroll(splits // parts):
                        acc += sqr[p * (splits // parts) + sp, t]
                    part_s[p, q] = acc
                total = T.alloc_shared((q + 1,), "float32")
                for j in T.Parallel(q + 1):
                    whole = T.alloc_var("float32")
                    whole = 0.0
                    for p in T.unroll(parts):
                        whole += part_s[p, j]
                    total[j] = whole
                mixes = T.alloc_shared((q,), "float32")
                for j in T.Parallel(q):
                    mixes[j] = total[j] * T.rsqrt(total[q] / (n * c) + rms_eps)
                if T.get_thread_binding() < WARP_LANES:
                    for i in T.Parallel(n):
                        post_mix[t, i] = (
                            T.sigmoid(mixes[n + i] * hc_scale[1] + hc_base[n + i])
                            * hc_post_mult_value
                        )
                    cm = T.alloc_fragment((n, n), "float32")
                    row = T.alloc_fragment((n,), "float32")
                    col = T.alloc_fragment((n,), "float32")
                    for i, j in T.Parallel(n, n):
                        cm[i, j] = (
                            mixes[2 * n + i * n + j] * hc_scale[2] + hc_base[2 * n + i * n + j]
                        )
                    T.reduce_max(cm, row, dim=1)
                    for i, j in T.Parallel(n, n):
                        cm[i, j] = T.exp(cm[i, j] - row[i])
                    T.reduce_sum(cm, row, dim=1)
                    for i, j in T.Parallel(n, n):
                        cm[i, j] = cm[i, j] / row[i] + hc_sinkhorn_eps
                    T.reduce_sum(cm, col, dim=0)
                    for i, j in T.Parallel(n, n):
                        cm[i, j] = cm[i, j] / (col[j] + hc_sinkhorn_eps)
                    for _ in T.serial(sinkhorn_repeat - 1):
                        T.reduce_sum(cm, row, dim=1)
                        for i, j in T.Parallel(n, n):
                            cm[i, j] = cm[i, j] / (row[i] + hc_sinkhorn_eps)
                        T.reduce_sum(cm, col, dim=0)
                        for i, j in T.Parallel(n, n):
                            cm[i, j] = cm[i, j] / (col[j] + hc_sinkhorn_eps)
                    for i, j in T.Parallel(n, n):
                        comb_mix[t, i * n + j] = cm[i, j]
                else:
                    pre = T.alloc_shared((n,), "float32")
                    for i in T.Parallel(n):
                        pre[i] = T.sigmoid(mixes[i] * hc_scale[0] + hc_base[i]) + hc_pre_eps
                    if staged:
                        for b in T.Pipelined(c // block, num_stages=2):
                            xs = T.alloc_shared((streams, block), "bfloat16")
                            xl = T.alloc_fragment((streams, block), "float32")
                            T.copy(x[t, 0, b * block], xs)
                            T.copy(xs, xl)
                            part = T.alloc_fragment((block,), "float32")
                            T.clear(part)
                            for i in T.serial(n):
                                for k in T.Parallel(block):
                                    part[k] += pre[i] * xl[0 if streams == 1 else i, k]
                            T.copy(part, layer_input[t, b * block])
                    else:
                        out = T.alloc_fragment((c,), "float32")
                        T.clear(out)
                        for i in T.unroll(n):
                            for k in T.Parallel(c):
                                out[k] += pre[i] * T.cast(
                                    x[t, 0 if streams == 1 else i, k], "float32"
                                )
                        for k in T.Parallel(c):
                            layer_input[t, k] = T.cast(out[k], "bfloat16")

        return main

    return _func()


class HyperConnectionPreKernel(Kernel, HyperConnectionPreFwdInterface):
    """The manifold hyper-connection pre step on SM90 tensor cores, for four streams.

    The projection's products are exact and each 64-channel slot's add into float32, so the
    mixes are as accurate as a float32 matrix product. The token count is a runtime extent:
    a launch picks the tile, the reduction splits and the paths from it, so a new token count
    builds programs only where it picks a new set. A call of few tokens whose slice of the terms
    fits in shared memory beside the token ring splits ``fn`` in the projection; any other
    splits it in a launch of its own.

    Args:
        n: Streams a token.
        c: Channels a stream.
        broadcast: Whether the residual's stream axis has stride 0, its first stream
            standing for all ``n``.
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat: The op's
            constants.
        sm_count: Device SM count, which sizes the reduction splits.
        smem_budget: Shared memory a block may take, which bounds the slice of the terms
            the projection holds.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: HyperConnectionPreCall) -> str | None:
        if call.n != 4:
            return f"serves four streams, got {call.n}"
        if call.c % 128:
            return f"requires the stream width to be a multiple of 128, got {call.c}"
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
            smem_budget=call.smem_budget,
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
        smem_budget: int,
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
        self.smem_budget = smem_budget
        self.init_config()

    def _launch(self, tokens: int) -> tuple[int, int, int, bool, bool]:
        """The ``block_m``, ``num_stages``, ``splits``, ``resident`` and ``staged_mix`` a call
        of *tokens* tokens launches with."""
        # Two warpgroups a tile share each slot's terms; a call one WGMMA's rows hold would
        # leave the second idle.
        small = tokens <= WGMMA_ROWS
        block_m, num_stages = (WGMMA_ROWS, 4) if small else (2 * WGMMA_ROWS, 6)
        # More splits fill more SMs but add partial sums for the mix launch to read, more of
        # them a split the larger the tile.
        max_splits = 128 if small else 64
        steps = self.n * self.c // _STEP
        splits = max(1, min(max_splits, self.sm_count // -(-tokens // block_m)))
        while steps % splits:
            splits -= 1
        q = self.n * self.n + 2 * self.n
        terms = steps // splits * 3 * q * _STEP * 2
        ring = num_stages * block_m * _STEP * 2
        # The barriers take one aligned slot ahead of the slice and the ring.
        fits = SHARED_BUFFER_ALIGN_BYTES + terms + ring <= self.smem_budget
        # Every token tile splits the fn slice itself, work that outgrows the launch it saves.
        resident_max = 512
        # Staging overlaps a block's loads with its sums while few blocks share an SM; more
        # tokens fill the SMs, and whole rows in registers skip the copy.
        staged_mix_max = 4096
        return (
            block_m,
            num_stages,
            splits,
            fits and tokens <= resident_max,
            tokens <= staged_mix_max,
        )

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
        split, gemm, mix = self.program(self._compile, *self._launch(tokens))
        weight = fn if split is None else split(fn)
        partial, sqr = gemm(x.view(tokens, -1), weight)
        post_mix, comb_mix, layer_input = mix(x, partial, sqr, hc_scale, hc_base)
        return (
            post_mix.view(*lead, self.n, 1),
            comb_mix.view(*lead, self.n, self.n),
            layer_input.view(*lead, self.c),
        )

    def _compile(
        self,
        cfg: dict,
        block_m: int,
        num_stages: int,
        splits: int,
        resident: bool,
        staged_mix: bool,
    ) -> tuple[CompiledProgram | None, CompiledProgram, CompiledProgram]:
        """The split, projection and mix programs a launch of these facts runs."""
        streams = 1 if self.broadcast else self.n
        q = self.n * self.n + 2 * self.n
        split = None if resident else CompiledProgram(_split_fn_kernel(q, self.n * self.c))
        gemm = _prenorm_gemm_kernel(self.n, self.c, streams, splits, block_m, num_stages, resident)
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
            staged_mix,
        )
        return split, CompiledProgram(gemm), CompiledProgram(mix)
