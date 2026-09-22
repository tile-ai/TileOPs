"""W4A16 GEMM over the weight layout produced by ``W4A16RepackKernel``."""

import functools
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count, is_h200

from .call_spec import GemmCall
from .dense import _splitk_reduce_kernel

GROUP_SIZE = 128

_DECODE_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_w4a16_decode_helper.h")
)


@dataclass(frozen=True)
class _Layout:
    """Constants fixed by the packed-weight ABI, not tuning parameters."""

    mma_step_k: int = 128
    lanes: int = 4
    fp16_nibble_bias: int = 1024


@dataclass(frozen=True)
class _Calibration:
    """H200 coefficients used only to rank valid kernel configurations."""

    dequant: float = 1.8958e-08
    dequant_ws: float = 1.3231e-08
    mma: float = 2.9199e-10
    mma_big: float = 4.6797e-11
    iter_ws: float = 1.3708e-04
    iter_ns: float = 1.0580e-04
    wave: float = 9.6752e-04
    tie_window: float = 0.02
    reduce_bytes_per_ms: float = 1.5e9
    launch_ms: float = 0.0015


@dataclass(frozen=True)
class _ConfigSpace:
    """Search space and SM90 resource limits."""

    block_ms: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    block_ns: tuple[int, ...] = (64, 128)
    block_ks: tuple[int, ...] = (128, 256, 512)
    stages: tuple[int, ...] = (2, 3, 4)
    threads: tuple[int, ...] = (128, 256)
    split_ks: tuple[int, ...] = (1, 2, 4, 8, 16)
    max_m_tiles: int = 64
    smem_bytes: int = 227 * 1024
    producer_reg: int = 24
    consumer_reg: int = 240


_LAYOUT = _Layout()
_H200_CALIBRATION = _Calibration()
_CONFIG_SPACE = _ConfigSpace()

__all__ = ["GROUP_SIZE", "GemmW4A16Kernel"]


def _smem_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    k: int,
    group_size: int,
) -> int:
    """Return dynamic shared memory required by one CTA."""
    per_tile_meta = threads == 128 and block_k <= 256
    meta_groups = (
        block_k // group_size if per_tile_meta else -(-k // block_k) * (block_k // group_size)
    )
    total = (
        num_stages * (block_m * block_k * 2 + block_n * block_k // 2) + block_n * meta_groups * 3
    )
    return total


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
    split_k = cfg.get("split_k", 1)
    c = _H200_CALIBRATION
    k_eff = k / split_k
    waves = -(-((-(-m // block_m)) * (-(-n // block_n)) * split_k) // sms)
    math_warpgroups = min(threads // 128, block_n // 64)
    rows_per_warpgroup = block_n / math_warpgroups
    warp_specialized = threads > 128
    cost = waves * (
        rows_per_warpgroup * k_eff * (c.dequant + c.dequant_ws * warp_specialized)
        + rows_per_warpgroup * block_m * k_eff * (c.mma + c.mma_big * (block_m >= 256))
        + (k_eff / block_k) * (c.iter_ws * warp_specialized + c.iter_ns / num_stages)
        + c.wave
    )
    if split_k > 1:
        reduce_bytes = split_k * m * n * 4 + m * n * 2
        cost += reduce_bytes / c.reduce_bytes_per_ms + c.launch_ms
    return cost


def _legal_configs(m: int, n: int, k: int, group_size: int):
    """Every tile the builder accepts for this shape."""
    for block_m in _CONFIG_SPACE.block_ms:
        if block_m > max(8, 2 * m) or -(-m // block_m) > _CONFIG_SPACE.max_m_tiles:
            continue
        for block_n in _CONFIG_SPACE.block_ns:
            for threads in _CONFIG_SPACE.threads:
                if threads < 128 * (block_n // 64) or block_m * block_n // threads > 256:
                    continue
                if block_n == 64 and threads > 128:
                    continue
                for block_k in _CONFIG_SPACE.block_ks:
                    k_iters = -(-k // block_k)
                    for num_stages in _CONFIG_SPACE.stages:
                        if (
                            _smem_bytes(
                                block_m,
                                block_n,
                                block_k,
                                num_stages,
                                threads,
                                k,
                                group_size,
                            )
                            > _CONFIG_SPACE.smem_bytes
                        ):
                            continue
                        for split_k in _CONFIG_SPACE.split_ks:
                            if split_k > 1 and (k_iters % split_k or k_iters // split_k < 2):
                                continue
                            yield {
                                "block_m": block_m,
                                "block_n": block_n,
                                "block_k": block_k,
                                "num_stages": num_stages,
                                "threads": threads,
                                "producer_reg": _CONFIG_SPACE.producer_reg if threads > 128 else 0,
                                "consumer_reg": _CONFIG_SPACE.consumer_reg if threads > 128 else 0,
                                "split_k": split_k,
                            }


# What identifies a tile shape, as opposed to how its K loop is sliced.
_TILE_KEYS = ("block_m", "block_n", "block_k", "num_stages", "threads")


def _select_config(m: int, n: int, k: int, group_size: int, sms: int) -> dict:
    """The cheapest legal tile under :func:`_config_cost`, then the cheapest slicing of it.

    The fit ranks whole tiles against each other; it was not fitted to rank a
    sliced tile of one shape against a whole tile of another, and at one token
    it credits a second warpgroup with bandwidth an SM does not have. So the
    tile is chosen unsliced, widest K tile on a tie, and only that tile's own
    ``split_k`` variants compete afterwards.
    """
    legal = list(_legal_configs(m, n, k, group_size))
    scored = [(_config_cost(m, n, k, cfg, sms), cfg) for cfg in legal if cfg["split_k"] == 1]
    if not scored:
        raise ValueError(f"no legal W4A16 tile for m={m}, n={n}, k={k}")
    floor = min(cost for cost, _ in scored) * (1 + _H200_CALIBRATION.tie_window)
    tied = [(cost, cfg) for cost, cfg in scored if cost <= floor]
    tile = min(tied, key=lambda t: (-t[1]["block_k"], t[0]))[1]
    slicings = [cfg for cfg in legal if all(cfg[key] == tile[key] for key in _TILE_KEYS)]
    return min(slicings, key=lambda cfg: _config_cost(m, n, k, cfg, sms))


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
        num_stages: int = 2,
        threads: int = 256,
        producer_reg: int = _CONFIG_SPACE.producer_reg,
        consumer_reg: int = _CONFIG_SPACE.consumer_reg,
        split_k: int = 1,
    ) -> Callable:
        step_k = _LAYOUT.mma_step_k
        lanes = _LAYOUT.lanes
        fp16_nibble_bias = _LAYOUT.fp16_nibble_bias
        packed_k = block_k // 2
        run = (step_k // 2) // lanes
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
        k_iters = -(-k // block_k)
        if k_iters % split_k:
            raise ValueError(f"split_k={split_k} must divide the {k_iters} K tiles")
        k_slice = k_iters // split_k

        @T.macro
        def decode_step(
            ks,
            weight_frag,
            packed_shared,
            scale_shared,
            zero_shared,
            run_local,
            scale_local,
            group,
        ):
            """Dequantize one K sub-step into ``weight_frag``."""
            for i, c in T.Parallel(block_n, lanes):
                for v in T.vectorized(run // 4):
                    run_local[v] = packed_shared[i, ks * (step_k // 8) + c * (run // 4) + v]
                scale_local[0] = scale_shared[i, group]
                for wd in T.serial(run // 4):
                    for j in T.serial(4):
                        for v in T.vectorized(2):
                            weight_frag[i, 32 * wd + 8 * j + 2 * c + v] = T.call_extern(
                                dtype,
                                "tileops_w4a16_dequant_word",
                                run_local[wd],
                                T.cast(
                                    fp16_nibble_bias + T.cast(zero_shared[i, group], "int32"),
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
            """Decode one K sub-step and issue its WGMMA."""
            group = (
                (ks * step_k) // group_size
                if per_tile_meta
                else (k_start // group_size + (ks * step_k) // group_size)
            )
            decode_step(
                ks,
                weight_frag,
                packed_shared,
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
        def tile_body(
            kk_first,
            kk_count,
            activation,
            packed_weight,
            weight_scale,
            weight_zero,
            activation_shared,
            packed_shared,
            scale_shared,
            zero_shared,
            frag_a,
            frag_b,
            frag_c,
            frag_d,
            output_local,
            run_local,
            scale_local,
            m_start,
            n_start,
        ):
            """Accumulate K tiles ``kk_first .. kk_first + kk_count`` into ``output_local``."""
            if not per_tile_meta:
                for i, g in T.Parallel(block_n, padded_groups):
                    g_src = T.min(g, all_groups - 1)
                    scale_shared[i, g] = weight_scale[n_start + i, g_src]
                    zero_shared[i, g] = weight_zero[n_start + i, g_src]
            T.clear(output_local)

            for kk_local in T.Pipelined(kk_count, num_stages=num_stages):
                kk = kk_local if kk_first == 0 else kk_first + kk_local
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

        def _tile_buffers(out_dtype):
            """The shared and register tiles one CTA of the pipelined path holds."""
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
            out_shared = T.alloc_shared((block_m, block_n), out_dtype)
            run_local = T.alloc_local((run // 4,), "uint32")
            scale_local = T.alloc_local((1,), dtype)

            layouts = {activation_shared: tilelang.layout.make_swizzled_layout(activation_shared)}
            layouts[packed_shared] = tilelang.layout.make_half_bank_swizzled_layout(packed_shared)
            T.annotate_layout(layouts)

            return (
                activation_shared,
                packed_shared,
                scale_shared,
                zero_shared,
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                output_local,
                run_local,
                scale_local,
                out_shared,
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
                (
                    activation_shared,
                    packed_shared,
                    scale_shared,
                    zero_shared,
                    frag_a,
                    frag_b,
                    frag_c,
                    frag_d,
                    output_local,
                    run_local,
                    scale_local,
                    out_shared,
                ) = _tile_buffers(dtype)
                # Minimum budget to the producer, the rest to the consumers.
                if producer_reg > 0:
                    T.annotate_producer_reg_dealloc(producer_reg)
                if consumer_reg > 0:
                    T.annotate_consumer_reg_alloc(consumer_reg)
                m_start = bx * block_m
                n_start = by * block_n
                tile_body(
                    0,
                    k_iters,
                    activation,
                    packed_weight,
                    weight_scale,
                    weight_zero,
                    activation_shared,
                    packed_shared,
                    scale_shared,
                    zero_shared,
                    frag_a,
                    frag_b,
                    frag_c,
                    frag_d,
                    output_local,
                    run_local,
                    scale_local,
                    m_start,
                    n_start,
                )
                for i, j in T.Parallel(block_n, block_m):
                    out_shared[j, i] = T.cast(output_local[i, j], dtype)
                T.copy(
                    out_shared,
                    output[m_start : m_start + block_m, n_start : n_start + block_n],
                )

        @T.prim_func
        def sliced(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 8), "uint32"),  # type: ignore
            weight_scale: T.Tensor((n, all_groups), dtype),  # type: ignore
            weight_zero: T.Tensor((n, all_groups), "uint8"),  # type: ignore
            partials: T.Tensor((split_k, m, n), "float"),  # type: ignore
        ) -> None:
            """`main` over one K slice per grid-z index, leaving fp32 partials to reduce."""
            with T.Kernel(tiles_m, tiles_n, split_k, threads=threads) as (bx, by, bz):
                (
                    activation_shared,
                    packed_shared,
                    scale_shared,
                    zero_shared,
                    frag_a,
                    frag_b,
                    frag_c,
                    frag_d,
                    output_local,
                    run_local,
                    scale_local,
                    out_shared,
                ) = _tile_buffers("float")
                # Minimum budget to the producer, the rest to the consumers.
                if producer_reg > 0:
                    T.annotate_producer_reg_dealloc(producer_reg)
                if consumer_reg > 0:
                    T.annotate_consumer_reg_alloc(consumer_reg)
                m_start = bx * block_m
                n_start = by * block_n
                tile_body(
                    bz * k_slice,
                    k_slice,
                    activation,
                    packed_weight,
                    weight_scale,
                    weight_zero,
                    activation_shared,
                    packed_shared,
                    scale_shared,
                    zero_shared,
                    frag_a,
                    frag_b,
                    frag_c,
                    frag_d,
                    output_local,
                    run_local,
                    scale_local,
                    m_start,
                    n_start,
                )
                for i, j in T.Parallel(block_n, block_m):
                    out_shared[j, i] = output_local[i, j]
                T.copy(
                    out_shared,
                    partials[bz, m_start : m_start + block_m, n_start : n_start + block_n],
                )

        return sliced if split_k > 1 else main

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
        tune: Accepted for the common kernel interface; this composite path uses
            :attr:`default_config` instead of generic autotuning.
        group_size: Weights per dequantization group.
        device_index: CUDA device the kernel is built for.
    """

    # ``packed_weight`` and ``weight_zero`` are uint8 payloads, not extents.
    autotune_accepts_random_int_inputs: bool = True

    # WGMMA, warp specialization and `setmaxnreg`; there is no pre-Hopper path.
    supported_archs: list[int] = [90]

    # The generic tuner measures one JIT launch, while this kernel's config can
    # also change M padding and add a split-K reduction launch.
    autotune_configs = None

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
        split_k = self.config.get("split_k", 1)
        self._reduce = (
            _splitk_reduce_kernel(split_k, self.m_pad, n, self.dtype_str)() if split_k > 1 else None
        )

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

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Keep the calibrated selector; generic tuning cannot measure this composite path."""
        del warmup, rep
        warnings.warn(
            "GemmW4A16Kernel does not support generic autotuning because its config can "
            "change M padding and add a split-K reduction; keeping the calibrated config",
            UserWarning,
            stacklevel=2,
        )

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
        words = packed_weight.view(torch.uint32)
        if self._reduce is None:
            out = compiled(activation, words, weight_scale, weight_zero)
        else:
            # Allocated before the mainloop launches, so the allocation does
            # not sit between the two kernels.
            out = activation.new_empty((self.m_pad, self.n))
            self._reduce(compiled(activation, words, weight_scale, weight_zero), out)
        return out[: self.m] if self.m_pad != self.m else out
