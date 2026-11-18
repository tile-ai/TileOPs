"""FP8 NT GEMM for at most sixteen rows, as the transposed product streamed in 64-row blocks.

At ``m <= 16`` a 64-row WGMMA tile of ``a`` is mostly zero padding, and every K step
multiplies the padding by a full ``b`` tile. Computing ``c^T = b @ a^T`` instead puts 64
rows of ``b`` on WGMMA's 64-row axis and the rows of ``a``, padded to eight or sixteen, on
its column axis. The call reads ``b`` once and nothing else of size, so the kernel is a
weight stream: one producer warpgroup fills a deep TMA ring with 64-row blocks of ``b``,
and one consumer warpgroup runs each K step's WGMMA and adds its partial, under its
scales, to a float32 sum.
"""

from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops._kernels.gemm.call_spec import GEMMFP8Call, GEMMFP8FwdInterface
from tileops._kernels.gemm.dense import splitk_reduce_kernel
from tileops._kernels.gemm.heuristics import (
    _STATIC_SMEM_BYTES,
    FP8_MIN_SLICE_K_TILES,
    stream_width,
)
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel

__all__ = ["GEMMFP8SwapABKernel"]

# The stream's unit, which the builder and the config agree on: 64 rows of ``b`` (one WGMMA's
# M) over 128-wide K steps (one block128 scale step, one 128-byte swizzle atom of fp8).
_ROWS = 64
_BLOCK_K = 128


def _stream_config(m: int, n: int, k: int, sm_count: int, block_scaled: bool) -> dict | None:
    """K split, grid and ring depth of the FP8 weight stream.

    ``None`` where the staged scales leave no room for a ring of two stages (one where a CTA
    streams a single K step).

    The stream is bound by how fast the busiest CTA reads its bytes of ``b``, and a CTA
    reads at most about one SM's share of the DRAM bandwidth.

    - ``split_k``: the most K slices, dividing the K tiles, whose units still fit one
      CTA per SM, each slice keeping ``FP8_MIN_SLICE_K_TILES``; one where the 64-row
      blocks alone fill the SMs. Short of the SM count, an unsplit grid leaves SMs idle
      for the whole launch.
    - ``ctas``: the fewest that run the units in as many rounds as ``sm_count`` CTAs, so
      every CTA streams the same number of units and none runs a tail alone.
    - ``num_stages``: the deepest ring the shared memory holds, at most the K steps a CTA
      streams: a CTA's loads are all of its work, so as many as fit go out before the
      first product.
    """
    mpad = stream_width(m)[0]
    k_tiles = -(-k // _BLOCK_K)
    blocks = -(-n // _ROWS)
    split_k = max(
        s
        for s in range(1, k_tiles + 1)
        if k_tiles % s == 0
        and (s == 1 or (blocks * s <= sm_count and k_tiles // s >= FP8_MIN_SLICE_K_TILES))
    )
    units = blocks * split_k
    rounds = -(-units // sm_count)
    ctas = -(-units // rounds)
    k_iters = k_tiles // split_k
    # A ring stage holds one K step of the block's rows and of a's padded rows.
    stage = (_ROWS + mpad) * _BLOCK_K
    # Beside the ring: the static mbarriers' KiB, the output staging tile (float32 when
    # split) and, under block scales, a's scales over all of K and b's for each unit.
    fixed = _STATIC_SMEM_BYTES + _ROWS * mpad * (2 if split_k == 1 else 4)
    if block_scaled:
        fixed += (k_tiles * mpad + rounds * k_iters) * 4
    deepest = (BLOCK_SHARED_BYTES_OPT_IN[90] - fixed) // stage
    if deepest < min(2, rounds * k_iters):
        return None
    return {"num_stages": min(deepest, rounds * k_iters), "ctas": ctas, "split_k": split_k}


@target_cache(maxsize=32)
def _gemm_fp8_stream_kernel(
    m: int, n: int, k: int, out_dtype: str, block_scaled: bool, split_k: int
) -> Callable:
    """Build the transposed FP8 weight stream; the returned factory takes the ring and grid.

    The work is ``ceil(n / 64) * split_k`` units, one 64-row block of ``b`` over one K
    slice each: slice ``s`` covers K steps ``[s * q, (s + 1) * q)``. CTA ``p`` runs units
    ``p``, ``p + ctas``, ... back to back through one ring, unit ``u`` being block
    ``u % ceil(n / 64)`` of slice ``u // ceil(n / 64)``.

    Args:
        m: Rows of ``a``, at most sixteen.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim, a multiple of 16 (TMA's 16-byte unit in fp8).
        out_dtype: ``"bfloat16"`` or ``"float16"``; with ``split_k > 1`` the program writes
            float32 partials ``[split_k, m, n]`` instead.
        block_scaled: ``scale_a`` per 1x128 block, ``[m, ceil(k / 128)]``, and ``scale_b``
            per 128x128 block, ``[ceil(n / 128), ceil(k / 128)]``; else both ``[1, 1]``.
        split_k: K slices, dividing ``ceil(k / 128)``.

    Returns:
        A ``@tilelang.jit`` factory taking ``(num_stages, ctas)``.
    """
    rows, block_k = _ROWS, _BLOCK_K
    mpad = stream_width(m)[0]
    k_steps = -(-k // block_k)
    k_iters = k_steps // split_k
    nblk = -(-n // rows)
    units = nblk * split_k
    c_dtype = out_dtype if split_k == 1 else "float32"
    # The output tile stores through TMA, which needs 16-byte output rows.
    tma_epilogue = split_k == 1 and (n * 2) % 16 == 0
    # A thread's registers of the 64 x mpad accumulator, in dense_wgmma_8bit_64xN's order.
    regs = mpad // 2

    @tilelang.jit(
        out_idx=[-1],
        # The producer and consumer warpgroups are written by hand below.
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            *csrc_include("dense_wgmma_helper.h"),
        ],
    )
    def _gemm_fp8_stream_func(num_stages: int, ctas: int) -> Callable:
        per_cta = -(-units // ctas)
        # Built here, not closed over: a jit builder closes over scalars only.
        scale_a_shape = (m, k_steps) if block_scaled else (1, 1)
        scale_b_shape = (-(-n // 128), k_steps) if block_scaled else (1, 1)
        c_shape = (m, n) if split_k == 1 else (split_k, m, n)

        @T.prim_func
        def _gemm_fp8_stream_main(
            a: T.Tensor((m, k), "float8_e4m3fn"),  # type: ignore
            b: T.Tensor((n, k), "float8_e4m3fn"),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            c: T.Tensor(c_shape, c_dtype),  # type: ignore
        ) -> None:
            # One consumer warpgroup (threads 0..127) and one producer warpgroup.
            with T.Kernel(ctas, threads=256) as bx:
                b_smem = T.alloc_shared((num_stages, rows, block_k), "float8_e4m3fn")
                a_smem = T.alloc_shared((num_stages, mpad, block_k), "float8_e4m3fn")
                c_smem = T.alloc_shared((mpad, rows), c_dtype)
                if block_scaled:
                    # The scales this CTA reads, staged once: a's per K step and column of the
                    # tile, b's per unit and K step.
                    sa_smem = T.alloc_shared((k_steps, mpad), "float")
                    sb_smem = T.alloc_shared((per_cta, k_iters), "float")
                T.annotate_layout(
                    {
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                    }
                )
                acc = T.alloc_local((regs,), "float")
                part = T.alloc_local((regs,), "float")
                step_scale = T.alloc_local((regs,), "float")
                tensor_scale = T.alloc_local((1,), "float")
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                # The ring step a unit starts at, held in a register: TileLang 0.1.12 folds
                # ``(s * k_iters + k_iters - 1) % (2 * num_stages)`` over the unit loop to a
                # wrong constant, which waits on the wrong barrier phase and hangs.
                unit_step = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()

                # The consumer warpgroup holds threads 0..127: the helper's register order
                # names the lane and warp of an absolute thread.
                if tx < 128:
                    # The consumer holds the sums, partials and step scales; the producer
                    # only issues TMA, so the register file goes to the consumer.
                    T.inc_max_nreg(240)
                    lane = tx % 32
                    # The tile row of this thread's register 0, and its column.
                    row0 = tx // 32 * 16 + lane // 4
                    col0 = lane % 4 * 2
                    if not block_scaled:
                        # Read before the stream, so no load latency follows the last step.
                        tensor_scale[0] = scale_a[0, 0] * scale_b[0, 0]
                    if block_scaled:
                        for i in T.serial(T.ceildiv(k_steps * mpad, 128)):
                            idx = i * 128 + tx
                            if idx < k_steps * mpad:
                                col = idx % mpad
                                sa_smem[idx // mpad, col] = T.if_then_else(
                                    col < m, scale_a[T.min(col, m - 1), idx // mpad], 0.0
                                )
                        for i in T.serial(T.ceildiv(per_cta * k_iters, 128)):
                            idx = i * 128 + tx
                            unit = bx + idx // k_iters * ctas
                            if idx < per_cta * k_iters and unit < units:
                                sb_smem[idx // k_iters, idx % k_iters] = scale_b[
                                    unit % nblk * rows // 128,
                                    unit // nblk * k_iters + idx % k_iters,
                                ]
                        # A named barrier over the consumer warpgroup alone, clear of the ids
                        # TileLang reserves for whole-block and its own partial syncs.
                        T.sync_threads(barrier_id=4, arrive_count=128)
                    for s in T.serial(per_cta):
                        unit = bx + s * ctas
                        n_start = unit % nblk * rows
                        # A CTA past the last unit runs an empty one; its slice stays in range.
                        k_base = T.min(unit // nblk, split_k - 1) * k_iters
                        unit_step[0] = s * k_iters
                        for r in T.unroll(regs):
                            acc[r] = 0.0
                        for ki in T.serial(k_iters):
                            step = unit_step[0] + ki
                            slot = step % num_stages
                            T.barrier_wait(ab_full[slot], (step % (2 * num_stages)) // num_stages)
                            # One K step into a fresh accumulator, left in flight.
                            T.call_extern(
                                "handle",
                                "tileops::dense_wgmma_8bit_64xN<tl::DataType::kFloat8_e4m3, "
                                f"tl::DataType::kFloat32, {mpad}, 1, false>",
                                T.address_of(part[0]),
                                T.address_of(b_smem[slot, 0, 0]),
                                T.address_of(a_smem[slot, 0, 0]),
                                0,
                                0,
                            )
                            if block_scaled:
                                # Read the step's scales while its WGMMAs run.
                                for r in T.unroll(regs):
                                    step_scale[r] = (
                                        sa_smem[k_base + ki, r // 4 * 8 + col0 + r % 2]
                                        * sb_smem[s, ki]
                                    )
                            # The step's partial joins the float32 sum once its WGMMAs retire:
                            # an FP8 WGMMA accumulates below float32 precision, so no sum runs
                            # past one K step in it.
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(part, num_regs=regs)
                            T.barrier_arrive(ab_empty[slot])
                            for r in T.unroll(regs):
                                if block_scaled:
                                    acc[r] += part[r] * step_scale[r]
                                else:
                                    acc[r] += part[r]
                        for r in T.unroll(regs):
                            # Per-tensor scales multiply each finished sum, a slice's included.
                            value = acc[r] if block_scaled else acc[r] * tensor_scale[0]
                            c_smem[r // 4 * 8 + col0 + r % 2, row0 + r % 4 // 2 * 8] = T.cast(
                                value, c_dtype
                            )
                        if tma_epilogue:
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            if unit < units:
                                T.copy(c_smem, c[0, n_start])
                        else:
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            for i in T.serial(T.ceildiv(mpad * rows, 128)):
                                idx = i * 128 + tx
                                mi = idx // rows
                                j = idx % rows
                                if unit < units and mi < m and n_start + j < n:
                                    if split_k == 1:
                                        c[mi, n_start + j] = c_smem[mi, j]
                                    else:
                                        c[unit // nblk, mi, n_start + j] = c_smem[mi, j]
                        # The next unit's tile reuses c_smem.
                        T.sync_threads(barrier_id=4, arrive_count=128)
                else:
                    # The fewest registers setmaxnreg allows: the producer holds counters only.
                    T.dec_max_nreg(24)
                    for s in T.serial(per_cta):
                        unit = bx + s * ctas
                        n_start = unit % nblk * rows
                        k_base = unit // nblk * k_iters
                        unit_step[0] = s * k_iters
                        for ki in T.serial(k_iters):
                            step = unit_step[0] + ki
                            slot = step % num_stages
                            k_start = (k_base + ki) * block_k
                            T.barrier_wait(
                                ab_empty[slot], ((step % (2 * num_stages)) // num_stages) ^ 1
                            )
                            if unit < units:
                                # Each unit reads its rows of b once.
                                T.tma_copy(
                                    b[n_start : n_start + rows, k_start : k_start + block_k],
                                    b_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                    eviction_policy="evict_first",
                                )
                                T.tma_copy(
                                    a[0:mpad, k_start : k_start + block_k],
                                    a_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                            T.barrier_arrive(ab_full[slot])

        return _gemm_fp8_stream_main

    return _gemm_fp8_stream_func


class GEMMFP8SwapABKernel(Kernel, GEMMFP8FwdInterface):
    """FP8 NT GEMM for ``m <= 16``, per-tensor or 1D2D scales, computed as ``(b @ a^T)^T``.

    Args:
        m: Rows of ``a``, at most sixteen.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim.
        dtype: Operand dtype; ``torch.float8_e4m3fn``.
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.
        config: Kernel config override; unset keys take their default.
        device_index: The device the kernel is built for.
        block_scaled: Whether the scales form the 1D2D block128 grid rather than two scalars.
        sm_count: SMs of that device.
    """

    supported_archs = [90]
    preferred_over = frozenset({"gemm_fp8_tensor_scale"})

    @classmethod
    def refusal(cls, call: GEMMFP8Call) -> "str | None":
        per_tensor = call.scale_a_shape == (1, 1) and call.scale_b_shape == (1, 1)
        if not per_tensor and cls.block_scale_grid(call) != "1d2d":
            return "requires one scale per tensor, or 1x128 scales of a with 128x128 of b"
        # Two WGMMA N steps of 8: the widest a's rows run as the transposed tile's columns.
        max_m = 16
        if call.m > max_m:
            return f"m={call.m} exceeds the {max_m} rows one transposed tile holds"
        if call.dtype != torch.float8_e4m3fn:
            return f"requires float8_e4m3fn operands, got {call.dtype}"
        if call.out_dtype not in (torch.bfloat16, torch.float16):
            return f"writes bfloat16 or float16, not {call.out_dtype}"
        if call.has_bias:
            return "does not serve a bias"
        if call.k == 0 or call.k % 16:
            return f"k={call.k} is not a positive multiple of 16 (TMA's 16-byte unit in fp8)"
        if _stream_config(call.m, call.n, call.k, call.sm_count, not per_tensor) is None:
            return (
                f"the scales staged for k={call.k} over n={call.n} leave shared memory no room "
                "for the TMA ring"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GEMMFP8Call) -> Entry:
        index = call.device.index if call.device is not None else None
        # Two (1, 1) scales are one scale per tensor under either grid.
        block_scaled = (call.scale_a_shape, call.scale_b_shape) != ((1, 1), (1, 1))
        identity = (call.m, call.n, call.k, call.dtype, call.out_dtype, block_scaled, index)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.out_dtype,
            block_scaled=block_scaled,
            device_index=index,
            sm_count=call.sm_count,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: dict | None = None,
        device_index: int | None = None,
        *,
        block_scaled: bool,
        sm_count: int,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.block_scaled = block_scaled
        self.sm_count = sm_count
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return _stream_config(self.m, self.n, self.k, self.sm_count, self.block_scaled)

    @property
    def autotune_configs(self) -> list[dict]:
        """The deepest ring and, where it refills its slots, the shallowest that keeps 10 MiB
        loading across the grid.

        10 MiB is the DRAM bandwidth times its loaded latency; past it a deeper ring queues
        loads without adding bandwidth. A ring holding every K step a CTA streams issues all
        of its loads before the first product.
        """
        default = self.default_config
        k_iters = -(-self.k // _BLOCK_K) // default["split_k"]
        units = -(-self.n // _ROWS) * default["split_k"]
        if default["num_stages"] == -(-units // default["ctas"]) * k_iters:
            return [{"num_stages": default["num_stages"], "ctas": default["ctas"]}]
        stage = (_ROWS + stream_width(self.m)[0]) * _BLOCK_K
        in_flight = min(-(-(10 * 2**20) // (default["ctas"] * stage)), default["num_stages"])
        return [
            {"num_stages": stages, "ctas": default["ctas"]}
            for stages in sorted({in_flight, default["num_stages"]})
        ]

    def _tuned_config(self, tuned: dict) -> dict:
        # Tuning sweeps the mainloop's ring; the K split stays the default's.
        return {**self.default_config, **tuned}

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError(f"{type(self).__name__} has no bias epilogue")
        return self.program(self._compile)(a, b, scale_a, scale_b)

    def _compile(self, cfg: dict) -> Callable:
        """The stream program, with its split-K reduce where ``cfg`` splits K."""
        split_k = cfg["split_k"]
        out_dtype = self.dtype_to_str(self.out_dtype)
        kernel = _gemm_fp8_stream_kernel(
            self.m, self.n, self.k, out_dtype, self.block_scaled, split_k
        )
        mainloop = kernel(cfg["num_stages"], cfg["ctas"])
        # Set after the build succeeds, so tuning never reads one that failed.
        self.kernel = kernel
        if self._tune_target is None:
            self._tune_target = (kernel, cfg)
        if split_k == 1:
            return mainloop
        mainloop = CompiledProgram(mainloop)
        reduce = CompiledProgram(splitk_reduce_kernel(split_k, self.m, self.n, out_dtype)())
        m, n, dtype = self.m, self.n, self.out_dtype

        def split(a, b, scale_a, scale_b):
            # Allocated before the mainloop, so no allocation sits between the two launches.
            c = torch.empty((m, n), dtype=dtype, device=a.device)
            slices = mainloop(a, b, scale_a, scale_b)
            reduce(slices, c)
            return c

        return split
