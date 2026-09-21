"""W4A16 GEMM over a prepacked weight, with the weights as the MMA A operand.

Dequantizing into a shared tile and feeding a shared-source WGMMA spends the
tile budget on the round trip. Here the dequantized halves land in the A-operand
registers instead (``wgmma_rs``), which removes the shared stores and pays for a
larger tile per warpgroup.

The price is the weight layout: a lane's bytes have to be contiguous, which
`W4A16RepackKernel` arranges. The permutation stays *within* a weight row, so
the tensor keeps its ``[N, K/2]`` shape and the copy still lowers to TMA.
Permuting N and K together produces a shape no ``T.copy`` will lower, and
TileLang only emits TMA under warp specialization, so losing TMA would cost the
producer/consumer schedule too.
"""

import functools
import os
import warnings
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count, is_h200

from .call_spec import GemmCall

# Weights are quantized per group of this many K elements, which fixes the
# ``[N, K / GROUP_SIZE]`` shape of weight_scale and weight_zero.
GROUP_SIZE = 128

_DECODE_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_w4a16_decode_helper.h")
)

# Lanes sharing a weight row in the A fragment: thread ``t = 32w + 4rp + c``
# holds bytes ``jj = c, c + 4, ...``, and the repack gathers each lane's into a
# contiguous run.
_LANES = 4

# Every M tile re-streams its whole weight column, so past this many tiles a
# tile cannot win -- the cost model never picks one -- and the cap only keeps
# the legal space, which autotune compiles in full, from growing with m.
_MAX_M_TILES = 64

# Weights per MMA K step: what one unrolled sub-step consumes and what the
# repack permutes within. Not a knob -- another value is another layout.
MMA_STEP_K = 128

# FP16 0x6400 is 1024.0; ORed with an unsigned nibble it reads back as 1024 + q,
# so the decode takes ``1024 + zero`` as its bias and the subtraction is exact.
_FP16_NIBBLE_BIAS = 1024

# Named barriers for the math warpgroups alone, above the ids TileLang's own
# lowering emits (a bare ``T.sync_threads`` is barrier 0).
_MATH_BARRIER_PROLOGUE = 9
_MATH_BARRIER_EPILOGUE = 8

# ``setmaxnreg`` split beside a producer warpgroup: what the producer keeps,
# and what each consumer grows to.
_PRODUCER_REG = 24
_CONSUMER_REG = 240

# Dynamic shared memory an SM90 CTA may request; every tile is checked against
# it rather than failing at launch.
_SM90_SMEM_BYTES = 232448

__all__ = ["GROUP_SIZE", "MMA_STEP_K", "GemmW4A16Kernel"]


def _smem_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    k: int,
    group_size: int,
    tma_threads: int = 0,
) -> int:
    """Shared memory one CTA of the kernel below needs, in bytes.

    The pipelined activation and packed tiles plus the staged scale and zero.
    It reproduces the sizes ptxas reports to the byte, which is what lets the
    selector and the autotune list reject a config without building it.
    The budget it is compared against is the per-block opt-in maximum (227 KB),
    not the 228 KB a Hopper SM carries: asking for the SM figure fails at launch
    with "Failed to set the allowed dynamic shared memory size".
    """
    per_tile_meta = threads == 128 and block_k <= 256
    meta_groups = (
        block_k // group_size if per_tile_meta else -(-k // block_k) * (block_k // group_size)
    )
    total = (
        num_stages * (block_m * block_k * 2 + block_n * block_k // 2) + block_n * meta_groups * 3
    )
    if tma_threads:
        # The hand-written ring lives across the whole K loop, so the epilogue
        # tile cannot reuse its bytes the way it does under `T.Pipelined`.
        total += block_m * block_n * 2
    return total


# Times measured on an H200 (SM90, 132 SMs), not a formula. The board is in the
# name because `default_config` ranks by these on whatever sm90 device it finds,
# and `_warn_off_calibration_board` is what tells a caller that happened. The
# `_ws` terms apply only to the automatic path.
_H200_COST_AUTOMATIC = {
    "dequant": 1.8958e-08,
    "dequant_ws": 1.3231e-08,
    "mma": 2.9199e-10,
    "mma_big": 4.6797e-11,
    "iter_ws": 1.3708e-04,
    "iter_ns": 1.0580e-04,
    "wave": 9.6752e-04,
}
# The fit under-ranks the path it scores: no shape tried selects a tile with a
# TMA producer, but the best one measures 1.09x the selected tile at m=128
# 8192x8192 (interleaved, both exact). Refitting is a change of its own; the
# path is not dead code.
_H200_COST_HAND_WRITTEN = {
    "dequant": 2.9226e-08,
    "dequant_ws": 0.0,
    "mma": 3.6020e-10,
    "mma_big": 9.4408e-11,
    "iter_ws": 0.0,
    "iter_ns": 4.7413e-04,
    "wave": 1.5662e-03,
}
# Costs this close are a tie the fit cannot call; the wider K tile wins it.
_H200_COST_TIE_WINDOW = 0.02


@functools.lru_cache(maxsize=8)
def _warn_off_calibration_board(device_index: Optional[int]) -> None:
    """Warn once per device that the tile ranking is running off its fit."""
    if not is_h200(device_index):
        warnings.warn(
            f"{torch.cuda.get_device_name(device_index)} is not the H200 the W4A16 tile "
            "cost model was fitted on, so `default_config` ranks tiles by coefficients "
            "that do not describe this board; pass `config=` to choose one yourself",
            RuntimeWarning,
            stacklevel=3,
        )


def _config_cost(m: int, n: int, k: int, cfg: dict, sms: int) -> float:
    """Modelled milliseconds for one launch of ``cfg``."""
    block_m, block_n = cfg["block_m"], cfg["block_n"]
    block_k, num_stages, threads = cfg["block_k"], cfg["num_stages"], cfg["threads"]
    tma_threads = cfg.get("tma_threads", 0)
    c = _H200_COST_HAND_WRITTEN if tma_threads else _H200_COST_AUTOMATIC
    waves = -(-((-(-m // block_m)) * (-(-n // block_n))) // sms)
    # block_n is the WGMMA M.
    math_warpgroups = min((threads - tma_threads) // 128, block_n // 64)
    rows_per_warpgroup = block_n / math_warpgroups
    specialized = threads > 128 and not tma_threads
    return waves * (
        rows_per_warpgroup * k * (c["dequant"] + c["dequant_ws"] * specialized)
        + rows_per_warpgroup * block_m * k * (c["mma"] + c["mma_big"] * (block_m >= 256))
        + (k / block_k) * (c["iter_ws"] * specialized + c["iter_ns"] / num_stages)
        + c["wave"]
    )


def _legal_configs(m: int, n: int, k: int, group_size: int):
    """Every tile the builder accepts for this shape."""
    for block_m in (8, 16, 32, 64, 128, 256):
        if block_m > max(8, 2 * m) or -(-m // block_m) > _MAX_M_TILES:
            continue
        for block_n in (64, 128):
            # (total threads, threads reserved for the TMA producer). A
            # nonzero reserve selects the hand-written producer/consumer split.
            for threads, tma_threads in ((128, 0), (256, 0), (384, 128)):
                math_threads = threads - tma_threads
                # One warpgroup per 64 weight rows, and at most 256 fp32
                # accumulator elements per thread before they spill.
                if math_threads < 128 * (block_n // 64) or block_m * block_n // math_threads > 256:
                    continue
                # Splitting a 64-row weight tile across two warpgroups is
                # silently wrong, so it is never offered.
                if block_n == 64 and math_threads > 128:
                    continue
                for block_k in (128, 256, 512):
                    for num_stages in (2, 3, 4):
                        if (
                            _smem_bytes(
                                block_m,
                                block_n,
                                block_k,
                                num_stages,
                                threads,
                                k,
                                group_size,
                                tma_threads,
                            )
                            > _SM90_SMEM_BYTES
                        ):
                            continue
                        yield {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "step_k": MMA_STEP_K,
                            "num_stages": num_stages,
                            "threads": threads,
                            "producer_reg": _PRODUCER_REG if threads > 128 else 0,
                            "consumer_reg": _CONSUMER_REG if threads > 128 else 0,
                            "tma_threads": tma_threads,
                        }


def _select_config(m: int, n: int, k: int, group_size: int, sms: int) -> dict:
    """The cheapest legal tile under :func:`_config_cost`, widest K tile on a tie."""
    scored = [(_config_cost(m, n, k, cfg, sms), cfg) for cfg in _legal_configs(m, n, k, group_size)]
    if not scored:
        raise ValueError(f"no legal W4A16 tile for m={m}, n={n}, k={k}")
    floor = min(cost for cost, _ in scored) * (1 + _H200_COST_TIE_WINDOW)
    tied = [(cost, cfg) for cost, cfg in scored if cost <= floor]
    return min(tied, key=lambda t: (-t[1]["block_k"], t[0]))[1]


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "-include", _DECODE_HELPER_PATH],
    )
    def build(
        block_m: int = 256,
        block_n: int = 128,
        block_k: int = 128,
        step_k: int = MMA_STEP_K,
        num_stages: int = 2,
        threads: int = 256,
        producer_reg: int = _PRODUCER_REG,
        consumer_reg: int = _CONSUMER_REG,
        tma_threads: int = 0,
        swizzle_packed: bool = True,
    ) -> Callable:
        packed_k = block_k // 2
        run = (step_k // 2) // _LANES
        steps = block_k // step_k
        if steps not in (1, 2, 3, 4):
            raise ValueError(
                f"block_k // step_k must be 1..4, got {steps}: the sub-steps are unrolled so"
                " each can own a weight fragment, and there are four of those"
            )
        all_groups = k // group_size
        tile_groups = block_k // group_size
        # A partial K tile is zero-filled, so only the scale and zero reads
        # need the clamp below.
        padded_groups = -(-k // block_k) * tile_groups
        # Per-K-tile staging frees shared budget at the cost of re-reading. The
        # thread guard is not a trade-off: two math warpgroups writing the tile
        # inside the pipelined loop hang the GPU.
        per_tile_meta = threads == 128 and block_k <= 256
        meta_groups = tile_groups if per_tile_meta else padded_groups
        tiles_m = -(-m // block_m)
        tiles_n = -(-n // block_n)
        math_threads = threads - tma_threads
        math_warps = math_threads // 32
        k_iters = -(-k // block_k)
        if tma_threads and (math_threads <= 0 or math_threads % 128):
            raise ValueError(
                f"threads={threads} leaves {math_threads} for math beside a "
                f"{tma_threads}-thread producer; it must be a positive multiple of 128"
            )

        @T.macro
        def decode_step(
            ks,
            weight_frag,
            packed_word,
            scale_shared,
            zero_shared,
            run_local,
            scale_local,
            group,
        ):
            """Dequantize one K sub-step into ``weight_frag``.

            ``packed_word(i, w)`` reads word ``w`` of weight row ``i`` from whatever
            shared buffer the caller pipelines with.
            """
            # ``c`` must stay parallel: TileLang refuses the fragment store
            # outright if it is serial.
            for i, c in T.Parallel(block_n, _LANES):
                for v in T.vectorized(run // 4):
                    run_local[v] = packed_word(i, ks * (step_k // 8) + c * (run // 4) + v)
                scale_local[0] = scale_shared[i, group]
                for wd in T.serial(run // 4):
                    for j in T.serial(4):
                        for v in T.vectorized(2):
                            weight_frag[i, 32 * wd + 8 * j + 2 * c + v] = T.call_extern(
                                dtype,
                                "tileops_w4a16_dequant_word",
                                run_local[wd],
                                T.cast(
                                    _FP16_NIBBLE_BIAS + T.cast(zero_shared[i, group], "int32"),
                                    dtype,
                                ),
                                scale_local[0],
                                j,
                                v,
                            )

        @T.macro
        def sub_step(
            ks,
            weight_frag,
            packed_shared,
            scale_shared,
            zero_shared,
            activation_shared,
            output_local,
            run_local,
            scale_local,
            k_start,
            overlapped,
        ):
            """Decode one K sub-step, then issue its WGMMA.

            With ``overlapped`` the WGMMA is the async form, which emits no
            implicit wait, so the caller can keep one in flight while the next
            sub-step dequantizes. A lone sub-step has nothing to overlap with and
            keeps ``T.gemm``, whose wait the lowering places from the pipeline's
            own buffer rotation -- placing it by hand there races.
            """
            group = (
                (ks * step_k) // group_size
                if per_tile_meta
                else (k_start // group_size + (ks * step_k) // group_size)
            )
            decode_step(
                ks,
                weight_frag,
                lambda i, w: packed_shared[i, w],
                scale_shared,
                zero_shared,
                run_local,
                scale_local,
                group,
            )
            if overlapped:
                T.wgmma_gemm(
                    weight_frag,
                    activation_shared[:, ks * step_k : (ks + 1) * step_k],
                    output_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            else:
                T.gemm(
                    weight_frag,
                    activation_shared[:, ks * step_k : (ks + 1) * step_k],
                    output_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

        @T.macro
        def ring_step(
            ks,
            weight_frag,
            packed_ring,
            scale_shared,
            zero_shared,
            activation_ring,
            output_local,
            run_local,
            scale_local,
            slot,
            k_start,
        ):
            """`sub_step` against a ring slot, for the hand-written pipeline.

            `scale_shared` holds every group of K, so the index carries the K
            tile as well as the sub-step.
            """
            group = (k_start + ks * step_k) // group_size
            decode_step(
                ks,
                weight_frag,
                lambda i, w: packed_ring[slot, i, w],
                scale_shared,
                zero_shared,
                run_local,
                scale_local,
                group,
            )
            T.wgmma_gemm(
                weight_frag,
                activation_ring[slot, :, ks * step_k : (ks + 1) * step_k],
                output_local,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )

        @T.prim_func
        def specialized(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 8), "uint32"),  # type: ignore
            weight_scale: T.Tensor((n, all_groups), dtype),  # type: ignore
            weight_zero: T.Tensor((n, all_groups), "uint8"),  # type: ignore
            output: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            """One CTA per tile, but with the producer/consumer split written out.

            Identical arithmetic to `main`; the difference is that `tma_threads`
            threads do nothing but issue TMA and the rest do nothing but decode
            and multiply, which is what lets a third warpgroup exist.
            """
            with T.Kernel(tiles_m, tiles_n, threads=threads) as (bx, by):
                activation_ring = T.alloc_shared((num_stages, block_m, block_k), dtype)
                packed_ring = T.alloc_shared((num_stages, block_n, packed_k // 4), "uint32")
                scale_shared = T.alloc_shared((block_n, all_groups), dtype)
                zero_shared = T.alloc_shared((block_n, all_groups), "uint8")
                frag_a = T.alloc_fragment((block_n, step_k), dtype)
                frag_b = T.alloc_fragment((block_n, step_k), dtype)
                frag_c = T.alloc_fragment((block_n, step_k), dtype)
                frag_d = T.alloc_fragment((block_n, step_k), dtype)
                output_local = T.alloc_fragment((block_n, block_m), "float")
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                run_local = T.alloc_local((run // 4,), "uint32")
                scale_local = T.alloc_local((1,), dtype)

                layouts = {activation_ring: tilelang.layout.make_swizzled_layout(activation_ring)}
                if swizzle_packed:
                    layouts[packed_ring] = tilelang.layout.make_half_bank_swizzled_layout(
                        packed_ring
                    )
                T.annotate_layout(layouts)

                # Producer arrival and consumer release, one pair per slot.
                full = T.alloc_barrier([tma_threads] * num_stages)
                empty = T.alloc_barrier([math_warps] * num_stages)
                tx = T.get_thread_binding()
                m_start = bx * block_m
                n_start = by * block_n

                if tx < tma_threads:
                    T.dec_max_nreg(producer_reg)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) & 1
                        k_start = ki * block_k
                        T.barrier_wait(empty[slot], phase ^ 1)
                        T.tma_copy(
                            activation[m_start : m_start + block_m, k_start : k_start + block_k],
                            activation_ring[slot, :, :],
                            barrier=full[slot],
                        )
                        T.tma_copy(
                            packed_weight[
                                n_start : n_start + block_n,
                                k_start // 8 : k_start // 8 + packed_k // 4,
                            ],
                            packed_ring[slot, :, :],
                            barrier=full[slot],
                        )
                        T.barrier_arrive(full[slot])
                else:
                    T.inc_max_nreg(consumer_reg)
                    lane = (tx - tma_threads) % 32
                    # The whole of K, once.
                    for i, g in T.Parallel(block_n, all_groups):
                        scale_shared[i, g] = weight_scale[n_start + i, g]
                        zero_shared[i, g] = weight_zero[n_start + i, g]
                    # Named so it counts only the math threads: a plain
                    # `T.sync_threads` would wait for the producer warpgroup,
                    # which never arrives here.
                    T.sync_threads(barrier_id=_MATH_BARRIER_PROLOGUE, arrive_count=math_threads)
                    T.clear(output_local)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) & 1
                        k_start = ki * block_k
                        T.barrier_wait(full[slot], phase)
                        ring_step(
                            0,
                            frag_a,
                            packed_ring,
                            scale_shared,
                            zero_shared,
                            activation_ring,
                            output_local,
                            run_local,
                            scale_local,
                            slot,
                            k_start,
                        )
                        if steps > 1:
                            ring_step(
                                1,
                                frag_b,
                                packed_ring,
                                scale_shared,
                                zero_shared,
                                activation_ring,
                                output_local,
                                run_local,
                                scale_local,
                                slot,
                                k_start,
                            )
                        if steps > 2:
                            ring_step(
                                2,
                                frag_c,
                                packed_ring,
                                scale_shared,
                                zero_shared,
                                activation_ring,
                                output_local,
                                run_local,
                                scale_local,
                                slot,
                                k_start,
                            )
                        if steps > 3:
                            ring_step(
                                3,
                                frag_d,
                                packed_ring,
                                scale_shared,
                                zero_shared,
                                activation_ring,
                                output_local,
                                run_local,
                                scale_local,
                                slot,
                                k_start,
                            )
                        T.wait_wgmma(0)
                        if lane == 0:
                            T.mbarrier_arrive(empty[slot])
                    # Through shared, so the global write is one coalesced row
                    # per token.
                    T.sync_threads(barrier_id=_MATH_BARRIER_EPILOGUE, arrive_count=math_threads)
                    for i, j in T.Parallel(block_n, block_m):
                        out_shared[j, i] = T.cast(output_local[i, j], dtype)
                    T.sync_threads(barrier_id=_MATH_BARRIER_EPILOGUE, arrive_count=math_threads)
                    T.copy(
                        out_shared,
                        output[m_start : m_start + block_m, n_start : n_start + block_n],
                    )

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 8), "uint32"),  # type: ignore
            weight_scale: T.Tensor((n, all_groups), dtype),  # type: ignore
            weight_zero: T.Tensor((n, all_groups), "uint8"),  # type: ignore
            output: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(tiles_m, tiles_n, threads=threads) as (bx, by):
                activation_shared = T.alloc_shared((block_m, block_k), dtype)
                packed_shared = T.alloc_shared((block_n, packed_k // 4), "uint32")
                # Only the K tile's own groups.
                scale_shared = T.alloc_shared((block_n, meta_groups), dtype)
                zero_shared = T.alloc_shared((block_n, meta_groups), "uint8")
                # One per sub-step, so none is overwritten while its WGMMA is
                # still reading; that keeps `steps - 1` in flight.
                frag_a = T.alloc_fragment((block_n, step_k), dtype)
                frag_b = T.alloc_fragment((block_n, step_k), dtype)
                frag_c = T.alloc_fragment((block_n, step_k), dtype)
                frag_d = T.alloc_fragment((block_n, step_k), dtype)
                output_local = T.alloc_fragment((block_n, block_m), "float")
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                run_local = T.alloc_local((run // 4,), "uint32")
                scale_local = T.alloc_local((1,), dtype)

                layouts = {
                    activation_shared: tilelang.layout.make_swizzled_layout(activation_shared)
                }
                if swizzle_packed:
                    # Every packed row starts in the same bank, and only the
                    # half-bank swizzle removes the resulting conflicts.
                    layouts[packed_shared] = tilelang.layout.make_half_bank_swizzled_layout(
                        packed_shared
                    )
                T.annotate_layout(layouts)

                # Minimum budget to the producer, the rest to the consumers.
                if producer_reg > 0:
                    T.annotate_producer_reg_dealloc(producer_reg)
                if consumer_reg > 0:
                    T.annotate_consumer_reg_alloc(consumer_reg)

                m_start = bx * block_m
                n_start = by * block_n
                if not per_tile_meta:
                    for i, g in T.Parallel(block_n, padded_groups):
                        g_src = T.min(g, all_groups - 1)
                        scale_shared[i, g] = weight_scale[n_start + i, g_src]
                        zero_shared[i, g] = weight_zero[n_start + i, g_src]
                T.clear(output_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    if per_tile_meta:
                        for i, g in T.Parallel(block_n, tile_groups):
                            g_src = T.min(k_start // group_size + g, all_groups - 1)
                            scale_shared[i, g] = weight_scale[n_start + i, g_src]
                            zero_shared[i, g] = weight_zero[n_start + i, g_src]
                    T.copy(
                        activation[m_start : m_start + block_m, k_start : k_start + block_k],
                        activation_shared,
                    )
                    T.copy(
                        packed_weight[
                            n_start : n_start + block_n,
                            k_start // 8 : k_start // 8 + packed_k // 4,
                        ],
                        packed_shared,
                    )

                    sub_step(
                        0,
                        frag_a,
                        packed_shared,
                        scale_shared,
                        zero_shared,
                        activation_shared,
                        output_local,
                        run_local,
                        scale_local,
                        k_start,
                        steps > 1,
                    )
                    if steps > 1:
                        sub_step(
                            1,
                            frag_b,
                            packed_shared,
                            scale_shared,
                            zero_shared,
                            activation_shared,
                            output_local,
                            run_local,
                            scale_local,
                            k_start,
                            steps > 1,
                        )
                    if steps > 2:
                        sub_step(
                            2,
                            frag_c,
                            packed_shared,
                            scale_shared,
                            zero_shared,
                            activation_shared,
                            output_local,
                            run_local,
                            scale_local,
                            k_start,
                            steps > 1,
                        )
                    if steps > 3:
                        sub_step(
                            3,
                            frag_d,
                            packed_shared,
                            scale_shared,
                            zero_shared,
                            activation_shared,
                            output_local,
                            run_local,
                            scale_local,
                            k_start,
                            steps > 1,
                        )
                    if steps > 1:
                        T.wait_wgmma(0)
                for i, j in T.Parallel(block_n, block_m):
                    out_shared[j, i] = T.cast(output_local[i, j], dtype)
                T.copy(
                    out_shared,
                    output[m_start : m_start + block_m, n_start : n_start + block_n],
                )

        return specialized if tma_threads else main

    return build


class GemmW4A16Kernel(Kernel):
    """W4A16 NT GEMM reading a prepacked weight, dequantized into the A fragment.

    The weight must have been through `W4A16RepackKernel`. The row-major nibble
    packing has the same shape, so passing it produces wrong results rather than
    an error.

    Args:
        m: Token rows.
        n: Output columns.
        k: Contraction dim.
        dtype: Activation/output torch dtype; the accumulation is FP32.
        config: Optional explicit config; defaults to :attr:`default_config`.
        tune: Whether to autotune over :attr:`autotune_configs`.
        group_size: Weights per dequantization group.
        device_index: CUDA device the kernel is built for.
    """

    # ``packed_weight`` and ``weight_zero`` are uint8 payloads, not extents.
    autotune_accepts_random_int_inputs: bool = True

    # WGMMA, warp specialization and `setmaxnreg`; there is no pre-Hopper path.
    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        """Every call, at any token count.

        `GemmW4A16FwdOp` declares the repacked weight order, so a call that
        reaches this kernel is already in it, and the tile is chosen from the
        token count inside :attr:`default_config` rather than by dispatch.
        """
        del call
        return True

    @classmethod
    def entry_for(cls, call: GemmCall) -> tuple:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.dtype, call.group_size, call.tune, index)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            tune=call.tune,
            group_size=call.group_size,
            device_index=index,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        group_size: int = GROUP_SIZE,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if group_size != GROUP_SIZE:
            raise ValueError(f"only group_size={GROUP_SIZE} is supported")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self.kernel = _gemm_w4a16_kernel(m, n, k, self.dtype_str, group_size)
        self.init_config(config, tune)
        # A TMA box overhanging the end of the tensor is far slower than a
        # whole one, so the token count is padded to a whole tile. The tile
        # count is unchanged; the padded rows land in output rows the caller
        # never sees.
        self.m_pad = -(-m // self.config["block_m"]) * self.config["block_m"]
        if self.m_pad != m:
            self.kernel = _gemm_w4a16_kernel(self.m_pad, n, k, self.dtype_str, group_size)

    @property
    def default_config(self) -> dict:
        """The cheapest legal tile, scored against this device's SM count.

        Returns:
            One tile config. The ranking comes from an H200 fit; on another
            board it still returns a legal tile and warns that it did.
        """
        _warn_off_calibration_board(self.device_index)
        return _select_config(
            self.m, self.n, self.k, self.group_size, get_sm_count(self.device_index)
        )

    @property
    def autotune_configs(self) -> list[dict]:
        """The selector's own space plus the register-split variants, so tuning
        cannot reach a tile the builder rejects."""
        return [
            {**cfg, "producer_reg": producer_reg, "consumer_reg": consumer_reg}
            for cfg in _legal_configs(self.m, self.n, self.k, self.group_size)
            for producer_reg, consumer_reg in (
                ((0, 0), (_PRODUCER_REG, _CONSUMER_REG), (32, 224))
                if cfg["threads"] > 128
                else ((0, 0),)
            )
        ]

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply by a prepacked INT4 weight.

        Args:
            activation: Activations, ``[M, K]``.
            packed_weight: Weights from `W4A16RepackKernel`, ``[N, K/2]``.
            weight_scale: Group scales, ``[N, K/128]``, same dtype as activation.
            weight_zero: Group zero points, ``[N, K/128]``, ``torch.uint8``.

        Returns:
            The product, ``[M, N]``, in the activation dtype.
        """
        compiled = self.kernel(**self.config)
        if self.m_pad != self.m:
            # Uninitialized on purpose: an output row depends only on the
            # activation row with the same index, and these are sliced off.
            padded = activation.new_empty((self.m_pad, self.k))
            padded[: self.m] = activation
            activation = padded
        # The kernel decodes 32-bit words; the operand stays UINT8 for the caller.
        out = compiled(activation, packed_weight.view(torch.uint32), weight_scale, weight_zero)
        return out[: self.m] if self.m_pad != self.m else out
