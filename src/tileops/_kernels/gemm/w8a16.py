"""Batched W8A16 NT GEMM: bfloat16 activations against e4m3 weights under 128x128 block scales,
the weights widened to bfloat16 in registers and multiplied on bfloat16 WGMMA.

Each 64 weight rows of a unit take one consumer warpgroup, which computes the transposed
product ``w @ a^T``: the weight rows run along WGMMA's 64-row axis, as its register operand, and
the unit's activation rows along its column axis, so a decode step's single token pads to
eight columns rather than 64 rows. A producer warpgroup fills a TMA ring with each K step's
128 weight columns in e4m3 and the activation's in bfloat16. Every e4m3 code is a bfloat16
value, so the widened weights and their products with the activations are exact; a K step's
float32 sum is multiplied by its block's scale, exactly for a power-of-two scale, and added to
the unit's float32 sum. For power-of-two scales the result is the bfloat16 batched matmul of
the dequantized weight, up to the order of the float32 additions.
"""

from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import (
    BLOCK_SHARED_BYTES_OPT_IN,
    SETMAXNREG_GRANULARITY,
    SHARED_BUFFER_ALIGN_BYTES,
    SM90_REGISTERS,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.gemm.bmm import _declared_strides
from tileops._kernels.gemm.call_spec import BMMW8A16Call, BMMW8A16FwdInterface
from tileops._kernels.gemm.heuristics import _STATIC_SMEM_BYTES
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["BMMW8A16Kernel"]

_SCALE_BLOCK = 128
_HALF_K = 64
_PRODUCER_REGS = 24


def _in_place(t: torch.Tensor) -> bool:
    """Whether the program reads or writes *t* where it lies: TMA and the 16-byte stores
    address a unit last stride, other strides of whole 16-byte steps, and a 16-byte aligned
    start."""
    step = 16 // t.element_size()
    return (
        t.stride(-1) == 1 and all(s % step == 0 for s in t.stride()[:-1]) and t.data_ptr() % 16 == 0
    )


def _stage_bytes(tile_m: int, wgs: int) -> int:
    """Shared bytes one ring stage holds: the weight rows' e4m3 step and the activation's."""
    return wgs * WGMMA_ROWS * _SCALE_BLOCK + tile_m * _SCALE_BLOCK * 2


def _fixed_bytes(k: int, tile: dict) -> int:
    """Shared bytes beside the ring, each buffer in whole KiB as TileLang aligns them: the
    output staging tile, the unit's scales, the sums of every K part but the first, which only
    a tile that splits K holds, and the static barriers."""
    tile_m, wgs, ksplit = tile["tile_m"], tile["wgs"], tile["ksplit"]
    kib = [
        wgs * tile_m * WGMMA_ROWS * 2,
        wgs * (k // _SCALE_BLOCK) * 4,
        (ksplit - 1) * wgs * tile_m // 2 * WARPGROUP_THREADS * 4,
    ]
    align = SHARED_BUFFER_ALIGN_BYTES
    return _STATIC_SMEM_BYTES + sum(-(-b // align) * align for b in kib)


def _tile_config(m: int, n: int, batch: int, sm_count: int) -> dict:
    """The unit's shape and its consumer warpgroups for ``m`` activation rows.

    Up to 64 rows run as one tile, padded to WGMMA's eight-column step, under one weight
    warpgroup whose K steps several warpgroups share, so they widen weights at once: seven
    for an eight-column tile, which with the producer fill the block's 1024-thread limit, four
    for wider ones, whose accumulators take more registers.
    More rows run 128 at a time over two weight warpgroups sharing the activation tile,
    unless the 128-row units would leave SMs idle that 64-row ones fill.
    """
    if m <= 64:
        return {"tile_m": -(-m // 8) * 8, "wgs": 1, "ksplit": 7 if m <= 8 else 4}
    wide = {"tile_m": 128, "wgs": 2, "ksplit": 1}
    units = batch * -(-m // wide["tile_m"]) * -(-n // (wide["wgs"] * WGMMA_ROWS))
    if units < sm_count:
        return {"tile_m": 64, "wgs": 1, "ksplit": 1}
    return wide


def _ring_limits(m: int, n: int, k: int, batch: int, sm_count: int, tile: dict) -> tuple:
    """The deepest ring shared memory holds beside ``tile``'s fixed buffers, the K steps one
    CTA streams over all of its units, and the fewest CTAs that run the units in as many
    rounds as ``sm_count`` CTAs."""
    tile_m, wgs = tile["tile_m"], tile["wgs"]
    units = batch * -(-m // tile_m) * -(-n // (wgs * WGMMA_ROWS))
    rounds = -(-units // sm_count)
    deepest = (BLOCK_SHARED_BYTES_OPT_IN[90] - _fixed_bytes(k, tile)) // _stage_bytes(tile_m, wgs)
    return deepest, rounds * (k // _SCALE_BLOCK), -(-units // rounds)


def _ring_config(m: int, n: int, k: int, batch: int, sm_count: int, tile: dict) -> dict:
    """``tile`` with its ring depth, as many stages as shared memory holds and at most the K
    steps one CTA streams, and its CTA count, from :func:`_ring_limits`."""
    deepest, steps, ctas = _ring_limits(m, n, k, batch, sm_count, tile)
    return {**tile, "num_stages": min(deepest, steps), "ctas": ctas}


@target_cache(maxsize=32)
def _bmm_w8a16_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    wgs: int,
    ksplit: int,
) -> Callable:
    """Build the batched W8A16 program; the returned factory takes the ring, the grid and the
    strides of ``a`` and ``c``: each tuple of element strides, ending in 1, or empty for
    contiguous storage.

    Unit ``u`` is weight-row block ``u % nblk`` (``64 * wgs`` rows), activation-row block
    ``u // nblk % mblk`` (``tile_m`` rows) of batch item ``u // (nblk * mblk)``; CTA ``p`` runs
    units ``p``, ``p + ctas``, ... through one ring. ``wgs * ksplit`` consumer warpgroups run
    a unit: warpgroup ``w`` takes weight rows ``64 * (w % wgs)`` on and the unit's K steps
    ``w // wgs``, ``w // wgs + ksplit``, ...; the K parts' sums meet in shared memory. The
    producer warpgroup, which only issues TMA, drops to 24 registers, the fewest
    ``setmaxnreg`` sets, and the consumers share the rest of the block's register file, at
    most 240 each, the count the FP8 stream kernel's consumer takes.

    Args:
        batch: Batch items.
        m: Activation rows per item.
        n: Weight rows per item, the output's columns; a multiple of 8.
        k: Contraction dim, a multiple of 128.
        tile_m: Activation rows of a unit, a multiple of 8 up to 128.
        wgs: Consumer warpgroups along a unit's weight rows, 64 rows each.
        ksplit: Consumer warpgroups along a unit's K steps.

    Returns:
        A ``@tilelang.jit`` factory taking ``(num_stages, ctas, a_strides, c_strides)``.
    """
    rows = wgs * WGMMA_ROWS
    k_steps = k // _SCALE_BLOCK
    nblk = -(-n // rows)
    mblk = -(-m // tile_m)
    units = batch * nblk * mblk
    scale_rows = -(-n // _SCALE_BLOCK)
    regs = tile_m // 2
    groups = wgs * ksplit
    threads = WARPGROUP_THREADS
    consumers = threads * groups
    out_chunks = tile_m * WGMMA_ROWS // 8
    part_steps = -(-k_steps // ksplit)
    step = SETMAXNREG_GRANULARITY
    launch_regs = SM90_REGISTERS // (consumers + threads) // step * step
    pool = (consumers + threads) * launch_regs - threads * _PRODUCER_REGS
    consumer_regs = min(240, pool // consumers // step * step)

    @tilelang.jit(
        out_idx=[],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            *CUDA_COMPILE_FLAGS,
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            *csrc_include("w8a16_dequant.h"),
        ],
    )
    def _bmm_w8a16_func(
        num_stages: int, ctas: int, a_strides: tuple = (), c_strides: tuple = ()
    ) -> Callable:
        per_cta = -(-units // ctas)
        a_param = (
            T.StridedTensor((batch, m, k), a_strides, "bfloat16")
            if a_strides
            else T.Tensor((batch, m, k), "bfloat16")
        )
        c_param = (
            T.StridedTensor((batch, m, n), c_strides, "bfloat16")
            if c_strides
            else T.Tensor((batch, m, n), "bfloat16")
        )

        @T.prim_func
        def _bmm_w8a16_main(
            a: a_param,  # type: ignore
            b: T.Tensor((batch, n, k), "float8_e4m3fn"),  # type: ignore
            scale: T.Tensor((batch, scale_rows, k_steps), "float32"),  # type: ignore
            c: c_param,  # type: ignore
        ) -> None:
            with T.Kernel(ctas, threads=consumers + threads) as bx:
                codes = T.alloc_shared((num_stages, rows, _SCALE_BLOCK), "float8_e4m3fn")
                acts = T.alloc_shared((num_stages, 2, tile_m, _HALF_K), "bfloat16")
                staged = T.alloc_shared((wgs, tile_m, WGMMA_ROWS), "bfloat16")
                scales = T.alloc_shared((wgs, k_steps), "float32")
                if ksplit > 1:
                    parts = T.alloc_shared((ksplit - 1, wgs, regs, threads), "float32")
                T.annotate_layout(
                    {
                        codes: tilelang.layout.make_swizzled_layout(codes),
                        acts: tilelang.layout.make_swizzled_layout(acts),
                    }
                )
                acc = T.alloc_local((regs,), "float")
                part = T.alloc_local((regs,), "float")
                full = T.alloc_barrier([threads] * num_stages)
                empty = T.alloc_barrier([threads * wgs] * num_stages)
                unit_step = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()

                if tx < consumers:
                    T.inc_max_nreg(consumer_regs)
                    w = tx // threads
                    wg = w % wgs
                    kp = w // wgs
                    t = tx % threads
                    lane = t % 32
                    row0 = t // 32 * 16 + lane // 4
                    col0 = lane % 4 * 2
                    for s in T.serial(per_cta):
                        unit = bx + s * ctas
                        live = T.min(unit, units - 1)
                        n0 = live % nblk * rows + wg * WGMMA_ROWS
                        m0 = live // nblk % mblk * tile_m
                        g = live // (nblk * mblk)
                        unit_step[0] = s * k_steps
                        for r in T.unroll(regs):
                            acc[r] = 0.0
                        if kp == 0:
                            for i in T.serial(T.ceildiv(k_steps, threads)):
                                if i * threads + t < k_steps:
                                    scales[wg, i * threads + t] = scale[
                                        g,
                                        T.min(n0 // _SCALE_BLOCK, scale_rows - 1),
                                        i * threads + t,
                                    ]
                        T.sync_threads(barrier_id=15, arrive_count=consumers)
                        for p in T.serial(part_steps):
                            ki = p * ksplit + kp
                            if ki < k_steps:
                                step = unit_step[0] + ki
                                slot = step % num_stages
                                T.barrier_wait(full[slot], (step % (2 * num_stages)) // num_stages)
                                T.call_extern(
                                    "handle",
                                    f"tileops::w8a16_wgmma_step<{tile_m}>",
                                    T.address_of(part[0]),
                                    T.access_ptr(codes[slot, wg * WGMMA_ROWS, 0], "r"),
                                    T.access_ptr(acts[slot, 0, 0, 0], "r"),
                                )
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(part, num_regs=regs)
                                T.barrier_arrive(empty[slot])
                                for r in T.unroll(regs):
                                    acc[r] += part[r] * scales[wg, ki]
                        if ksplit > 1:
                            if kp > 0:
                                for r in T.unroll(regs):
                                    parts[kp - 1, wg, r, t] = acc[r]
                            T.sync_threads(barrier_id=15, arrive_count=consumers)
                            if kp == 0:
                                for q in T.serial(ksplit - 1):
                                    for r in T.unroll(regs):
                                        acc[r] += parts[q, wg, r, t]
                        if kp == 0:
                            for r in T.unroll(regs):
                                staged[wg, r // 4 * 8 + col0 + r % 2, row0 + r % 4 // 2 * 8] = (
                                    T.cast(acc[r], "bfloat16")
                                )
                            T.sync_threads(barrier_id=4 + w, arrive_count=threads)
                            for i in T.serial(T.ceildiv(out_chunks, threads)):
                                idx = i * threads + t
                                mi = idx // 8
                                j = idx % 8 * 8
                                if unit < units and idx < out_chunks and m0 + mi < m and n0 + j < n:
                                    for v in T.vectorized(8):
                                        c[g, m0 + mi, n0 + j + v] = staged[wg, mi, j + v]
                else:
                    T.dec_max_nreg(_PRODUCER_REGS)
                    for s in T.serial(per_cta):
                        unit = bx + s * ctas
                        live = T.min(unit, units - 1)
                        n0 = live % nblk * rows
                        m0 = live // nblk % mblk * tile_m
                        g = live // (nblk * mblk)
                        unit_step[0] = s * k_steps
                        for ki in T.serial(k_steps):
                            step = unit_step[0] + ki
                            slot = step % num_stages
                            k0 = ki * _SCALE_BLOCK
                            T.barrier_wait(
                                empty[slot], ((step % (2 * num_stages)) // num_stages) ^ 1
                            )
                            if unit < units:
                                T.tma_copy(
                                    b[g, n0 : n0 + rows, k0 : k0 + _SCALE_BLOCK],
                                    codes[slot, :, :],
                                    barrier=full[slot],
                                    eviction_policy="evict_first",
                                )
                                for h in T.unroll(2):
                                    T.tma_copy(
                                        a[
                                            g,
                                            m0 : m0 + tile_m,
                                            k0 + h * _HALF_K : k0 + (h + 1) * _HALF_K,
                                        ],
                                        acts[slot, h, :, :],
                                        barrier=full[slot],
                                    )
                            T.barrier_arrive(full[slot])

        return _bmm_w8a16_main

    return _bmm_w8a16_func


class BMMW8A16Kernel(Kernel, BMMW8A16FwdInterface):
    """Batched W8A16 NT GEMM on SM90: ``c[i] = a[i] @ dequant(b[i])^T``.

    Args:
        batch: Batch items.
        m: Activation rows per item.
        n: Weight rows per item.
        k: Contraction dim.
        config: Kernel config override; unset keys take their default.
        device_index: The device the kernel is built for.
        sm_count: SMs of that device.
    """

    supported_archs = [90]
    general = True

    @classmethod
    def refusal(cls, call: BMMW8A16Call) -> "str | None":
        if call.k == 0 or call.k % _SCALE_BLOCK:
            return f"k={call.k} is not a positive multiple of {_SCALE_BLOCK}, the scale block"
        if call.n % 8:
            return f"n={call.n} is not a multiple of 8, the output's 16-byte store"
        if not call.batch * call.m * call.n:
            return super().refusal(call)
        tile = _tile_config(call.m, call.n, call.batch, call.sm_count)
        deepest, steps, _ = _ring_limits(call.m, call.n, call.k, call.batch, call.sm_count, tile)
        if deepest < min(tile["ksplit"], steps):
            return (
                f"k={call.k} leaves shared memory {max(deepest, 0)} ring stages for "
                f"{steps} K steps, fewer than the {tile['ksplit']} K parts that each wait on a "
                "stage of their own"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: BMMW8A16Call) -> Entry:
        index = call.device_index
        identity = (call.batch, call.m, call.n, call.k, index)
        return identity, lambda: cls(
            call.batch, call.m, call.n, call.k, device_index=index, sm_count=call.sm_count
        )

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        config: dict | None = None,
        device_index: int | None = None,
        *,
        sm_count: int,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.sm_count = sm_count
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        tile = _tile_config(self.m, self.n, self.batch, self.sm_count)
        return _ring_config(self.m, self.n, self.k, self.batch, self.sm_count, tile)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = a if _in_place(a) else a.contiguous()
        shape = (self.batch, self.m, self.n)
        target = out
        if out is None or not _in_place(out):
            target = torch.empty(shape, dtype=torch.bfloat16, device=a.device)
        self.program(self._compile, _declared_strides(a), _declared_strides(target))(
            a, b, scale, target
        )
        if out is None or target is out:
            return target
        return out.copy_(target)

    def _compile(self, cfg: dict, a_strides: tuple, c_strides: tuple) -> Callable:
        """The program ``forward`` launches, compiled from *cfg* for ``a`` and ``out``
        storage of *a_strides* and *c_strides*."""
        kernel = _bmm_w8a16_kernel(
            self.batch,
            self.m,
            self.n,
            self.k,
            cfg["tile_m"],
            cfg["wgs"],
            cfg["ksplit"],
        )
        self.kernel = kernel
        return kernel(
            num_stages=cfg["num_stages"], ctas=cfg["ctas"], a_strides=a_strides, c_strides=c_strides
        )
