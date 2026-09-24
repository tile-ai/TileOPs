import functools
from typing import Any, Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.trace import trace
from tileops.utils import get_sm_count, str2dtype

from .call_spec import GemmCall
from .heuristics import (
    SWAP_AB_MPAD,
    best_config,
    fp8_ws_config,
    gemv_config,
    small_batch_config,
    swap_ab_grid_underfills,
)

__all__ = [
    "GemmCpAsyncKernel",
    "GemmFp8BlockScaleKernel",
    "GemmFp8TensorScaleKernel",
    "GemmTmaKernel",
    "GemvKernel",
]

# Everything below is read inside a ``prim_func`` or ``T.macro`` body, which is a
# closure in a module-level factory and cannot reach ``self``; that is what keeps
# these at module scope rather than on the kernel classes.

# One named-barrier id per consumer warpgroup: each group arrives on its own
# barrier (``arrive_count=128``) instead of a block-wide sync.
_CONSUMER_BAR_WG0 = 8
_CONSUMER_BAR_WG1 = 9

# Columns in one 128-byte swizzle atom at 2 bytes per element; ``heuristics._SWIZZLE_ATOM_N``.
_COOP2_STAGE_N = 64

# Fixed by the two-consumer split and by the block128 scale grid; not tunable.
_FP8_WS_BLOCK_M = 128
_FP8_WS_HALF_M = _FP8_WS_BLOCK_M // 2
_FP8_WS_BLOCK_K = 128


def _tma_misalignment(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool
) -> Optional[str]:
    """Why TMA cannot address these operands, or ``None`` when it can.

    Every structure ``GemmTmaKernel`` builds loads its tiles through TMA, whose
    descriptors address the innermost (contiguous) dimension in 16-byte units —
    so that extent must be a multiple of ``16 / itemsize`` elements, 8 for
    fp16 / bf16. Which logical dim is innermost follows the layout: ``K`` for a
    non-transposed ``A`` and a transposed ``B``, ``M`` for a transposed ``A``,
    ``N`` for a non-transposed ``B``. The bandwidth-mode kernels load through
    ``cp.async`` and carry no such requirement.

    Undeclared, an unaligned shape reaches TileLang's descriptor check and dies
    as "Check failed: (result.supported) is false", naming nothing to change.
    """
    step = 16 // dtype.itemsize
    a_dim, a_extent = ("m", m) if trans_a else ("k", k)
    b_dim, b_extent = ("k", k) if trans_b else ("n", n)
    offenders = dict.fromkeys(
        f"{d}={v}" for d, v in ((a_dim, a_extent), (b_dim, b_extent)) if v % step
    )
    if not offenders:
        return None
    layout = f"{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}"
    return (
        f"TMA addresses each operand's innermost dimension in 16-byte units, so it "
        f"must be a multiple of {step} elements for {dtype}; the {layout} layout "
        f"makes that {a_dim} for a and {b_dim} for b, and {', '.join(offenders)}"
    )


def _b_eviction(m: int, block_m: int) -> Optional[str]:
    """``"evict_first"`` for a ``B`` tile the grid reads at most twice, else ``None``.

    Each of the ``ceil(m / block_m)`` M-tiles reads the whole of ``B``. At one or two of
    them ``B`` is streamed and the L2 it would hold belongs to ``A``, which every N-tile
    re-reads; above two ``B`` is itself the reused operand.
    """
    return "evict_first" if -(-m // block_m) <= 2 else None


def _dense_entry(cls: type, call: GemmCall) -> Entry:
    """The entry for a kernel taking ``(m, n, k, dtype)`` and both flags.

    The device is in the identity: its SM count and name pick the config.
    """
    index = call.device.index if call.device is not None else None
    identity = (call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b, index)
    return identity, lambda: cls(
        call.m,
        call.n,
        call.k,
        call.dtype,
        tune=call.tune,
        trans_a=call.trans_a,
        trans_b=call.trans_b,
        device_index=index,
    )


class _GemmFp8Kernel(Kernel):
    """Shared body of the two FP8 GEMM kernels; ``BLOCK_SCALED`` picks the scale grid.

    Takes :func:`_gemm_fp8_ws_kernel`, or :func:`_gemm_fp8_kernel` on a call
    :meth:`_ws_refusal` rejects.
    """

    # Whether this kernel reads block128 scale grids rather than per-tensor scalars.
    BLOCK_SCALED = False

    @staticmethod
    def _ws_refusal(m: int, n: int, k: int, dtype: torch.dtype) -> Optional[str]:
        """Why the warp-specialized variant cannot serve this call, or ``None``.

        It loads through TMA, and its epilogue releases a ring slot the mainloop
        named, so it needs at least one K-tile. The fallback carries neither.
        """
        if dtype != torch.float8_e4m3fn:
            return f"the warp-specialized FP8 kernel is e4m3-only, got {dtype}"
        if k == 0:
            return "the warp-specialized mainloop has no K-tile to run at k=0"
        return _tma_misalignment(m, n, k, dtype, trans_a=False, trans_b=True)

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.scale_a_shape,
            call.scale_b_shape,
            call.out_dtype,
            index,
        )
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.out_dtype,
            tune=call.tune,
            device_index=index,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = get_sm_count(self.device_index)
        self.ws_refusal = self._ws_refusal(m, n, k, dtype)
        self.kernel = self._builder()
        self.init_config(config, tune)
        self._unused_bias: Optional[torch.Tensor] = None

    def _builder(self) -> Callable:
        if self.ws_refusal is None:
            return _gemm_fp8_ws_kernel(
                self.m,
                self.n,
                self.k,
                self.dtype_str,
                self.out_dtype_str,
                self.BLOCK_SCALED,
                has_bias=False,
                sm_count=self.sm_count,
            )
        return _gemm_fp8_kernel(
            self.m, self.n, self.k, self.dtype_str, self.out_dtype_str, self.BLOCK_SCALED
        )

    def _run_split_k(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor],
        split_k: int,
        tile_config: dict,
    ) -> torch.Tensor:
        """Run the sliced mainloop and reduce its fp32 workspace into the output."""
        mainloop, reduce = _fp8_ws_splitk_pair(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            self.BLOCK_SCALED,
            bias is not None,
            split_k,
            tile_config["block_n"],
            tile_config["num_stages"],
            tile_config["group_size_m"],
        )
        slices = torch.empty((split_k, self.m, self.n), dtype=torch.float32, device=a.device)
        c = torch.empty((self.m, self.n), dtype=self.out_dtype, device=a.device)
        mainloop(a, b, scale_a, scale_b, self._bias_operand(bias, a), slices)
        reduce(slices, c)
        return c

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        if self.ws_refusal is None:
            return fp8_ws_config(self.m, self.n, self.k, self.sm_count, self.BLOCK_SCALED)
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    def _bias_operand(self, bias: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """The bias operand, or a one-element stand-in the no-bias build never reads."""
        if bias is not None:
            return bias
        if self._unused_bias is None or self._unused_bias.device != like.device:
            self._unused_bias = torch.zeros(1, dtype=self.out_dtype, device=like.device)
        return self._unused_bias

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"{type(self).__name__} only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        if self.ws_refusal is None:
            tile_config = {key: value for key, value in self.config.items() if key != "split_k"}
            split_k = self.config.get("split_k", 1)
            if split_k > 1:
                return self._run_split_k(a, b, scale_a, scale_b, bias, split_k, tile_config)
            builder = self.kernel
            if bias is not None:
                builder = _gemm_fp8_ws_kernel(
                    self.m,
                    self.n,
                    self.k,
                    self.dtype_str,
                    self.out_dtype_str,
                    self.BLOCK_SCALED,
                    has_bias=True,
                    sm_count=self.sm_count,
                )
            return builder(**tile_config)(a, b, scale_a, scale_b, self._bias_operand(bias, a))
        compiled = _gemm_fp8_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            self.BLOCK_SCALED,
            has_bias=bias is not None,
        )(**self.config)
        if bias is not None:
            return compiled(a, b, scale_a, scale_b, bias)
        return compiled(a, b, scale_a, scale_b)


class GemmFp8TensorScaleKernel(_GemmFp8Kernel):
    """FP8 NT GEMM for per-tensor scales; the two scalars land in the epilogue."""

    BLOCK_SCALED = False

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return call.scale_a_shape == (1, 1) and call.scale_b_shape == (1, 1)


class GemmFp8BlockScaleKernel(_GemmFp8Kernel):
    """FP8 NT GEMM for block128 scale grids; each K-step's partial is scaled and folded in."""

    BLOCK_SCALED = True

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        scale_k = (call.k + 127) // 128
        return call.scale_a_shape == (call.m, scale_k) and call.scale_b_shape == (call.n, scale_k)


@functools.lru_cache(maxsize=32)
def _fp8_ws_splitk_pair(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool,
    split_k: int,
    block_n: int,
    num_stages: int,
    group_size_m: int,
) -> tuple[Callable, Callable]:
    """The compiled (mainloop, reduce) pair for one split-K configuration.

    Resolved together so the host is not building the second launch while the
    first is already draining.
    """
    mainloop = _gemm_fp8_ws_splitk_kernel(
        m, n, k, dtype, out_dtype, block_scaled, has_bias, split_k=split_k
    )(block_n, num_stages, group_size_m)
    return mainloop, _splitk_reduce_kernel(split_k, m, n, out_dtype)()


@functools.lru_cache(maxsize=32)
def _gemm_fp8_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool = False,
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_fp8_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 128,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        if block_scaled:
            if block_k > 128:
                raise ValueError(f"block_k must be <= 128 for block128 scaling, got {block_k}")
            if 128 % block_k != 0:
                raise ValueError(f"128 must be divisible by block_k, got {block_k}")
        scale_k = (k + 127) // 128 if block_scaled else 1
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (n, scale_k) if block_scaled else (1, 1)

        @T.prim_func
        def _gemm_fp8_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)
                    # Reuse each row/column scale across the complete output tile instead
                    # of reloading it for every scaled partial element.
                    scale_a_local = T.alloc_fragment((block_m,), accum_dtype)
                    scale_b_local = T.alloc_fragment((block_n,), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    for i, j in T.Parallel(block_m, block_k):
                        a_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            a[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.if_then_else(
                            (n_start + i < n) & (k_start + j < k),
                            b[n_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    if block_scaled:
                        scale_idx = kk * block_k // 128
                        for i in T.Parallel(block_m):
                            scale_a_local[i] = T.if_then_else(
                                m_start + i < m,
                                scale_a[m_start + i, scale_idx],
                                0.0,
                            )
                        for j in T.Parallel(block_n):
                            scale_b_local[j] = T.if_then_else(
                                n_start + j < n,
                                scale_b[n_start + j, scale_idx],
                                0.0,
                            )
                        T.clear(partial)
                        T.gemm(
                            a_shared,
                            b_shared,
                            partial,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += partial[i, j] * scale_a_local[i] * scale_b_local[j]
                    else:
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0]
                            )

        @T.prim_func
        def _gemm_fp8_bias_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor((n,), out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)
                    # Reuse each row/column scale across the complete output tile instead
                    # of reloading it for every scaled partial element.
                    scale_a_local = T.alloc_fragment((block_m,), accum_dtype)
                    scale_b_local = T.alloc_fragment((block_n,), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    for i, j in T.Parallel(block_m, block_k):
                        a_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            a[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.if_then_else(
                            (n_start + i < n) & (k_start + j < k),
                            b[n_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    if block_scaled:
                        scale_idx = kk * block_k // 128
                        for i in T.Parallel(block_m):
                            scale_a_local[i] = T.if_then_else(
                                m_start + i < m,
                                scale_a[m_start + i, scale_idx],
                                0.0,
                            )
                        for j in T.Parallel(block_n):
                            scale_b_local[j] = T.if_then_else(
                                n_start + j < n,
                                scale_b[n_start + j, scale_idx],
                                0.0,
                            )
                        T.clear(partial)
                        T.gemm(
                            a_shared,
                            b_shared,
                            partial,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += partial[i, j] * scale_a_local[i] * scale_b_local[j]
                    else:
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j] + bias[n_start + j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0] + bias[n_start + j]
                            )

        return _gemm_fp8_bias_main if has_bias else _gemm_fp8_main

    return _gemm_fp8_func


@T.macro
def _fp8_ws_tile_id(flat_id, mt, nt, *, group_size_m: int, num_pid_m: int, num_pid_n: int):
    """Write the grouped-rasterization (m, n) tile coordinates of a flat tile id."""
    gin = T.int32(group_size_m * num_pid_n)
    gid = flat_id // gin
    first_m = gid * T.int32(group_size_m)
    gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
    mt[0] = first_m + (flat_id % gin) % gsize
    nt[0] = (flat_id % gin) // gsize


@T.macro
def _fp8_ws_stage(
    a,
    b,
    scale_a,
    scale_b,
    a_top,
    a_bot,
    b_smem,
    sa_stage,
    sb_stage,
    ab_full,
    slot,
    kb,
    m_start,
    n_start,
    *,
    m: int,
    n: int,
    block_n: int,
    block_scaled: bool,
):
    """One K-step of the producer: the step's three TMA boxes and its two scale vectors."""
    half_m = _FP8_WS_HALF_M
    block_m = _FP8_WS_BLOCK_M
    ks = kb * _FP8_WS_BLOCK_K
    T.tma_copy(
        a[m_start : m_start + half_m, ks : ks + _FP8_WS_BLOCK_K],
        a_top[slot, :, :],
        barrier=ab_full[slot],
    )
    T.tma_copy(
        a[m_start + half_m : m_start + block_m, ks : ks + _FP8_WS_BLOCK_K],
        a_bot[slot, :, :],
        barrier=ab_full[slot],
    )
    T.tma_copy(
        b[n_start : n_start + block_n, ks : ks + _FP8_WS_BLOCK_K],
        b_smem[slot, :, :],
        barrier=ab_full[slot],
    )
    if block_scaled:
        for i in T.Parallel(block_m):
            sa_stage[slot, i] = scale_a[T.min(m_start + i, m - 1), kb]
        for j in T.Parallel(block_n):
            sb_stage[slot, j] = scale_b[T.min(n_start + j, n - 1), kb]
        T.fence_proxy_async()
    T.barrier_arrive(ab_full[slot])


@T.macro
def _fp8_ws_scaled_step(
    a_smem,
    b_smem,
    sa_stage,
    sb_stage,
    ab_full,
    ab_empty,
    acc,
    part,
    sa_f,
    sb_f,
    slot,
    phase,
    *,
    row_base: int,
    block_n: int,
):
    """One K-step of a block128 consumer: WGMMA into a fresh accumulator, then promote it.

    Both scale vectors are copied out of the step's ring slot into fragments,
    which keeps the promotion register-local. ``row_base`` is this consumer's
    offset into the staged ``A`` row scales.
    """
    T.barrier_wait(ab_full[slot], phase)
    T.wgmma_gemm(
        a_smem[slot, :, :],
        b_smem[slot, :, :],
        part,
        transpose_B=True,
        policy=T.GemmWarpPolicy.FullRow,
        clear_accum=True,
    )
    T.copy(sa_stage[slot, row_base : row_base + _FP8_WS_HALF_M], sa_f)
    T.copy(sb_stage[slot, :], sb_f)
    T.wait_wgmma(0)
    for i, j in T.Parallel(_FP8_WS_HALF_M, block_n):
        acc[i, j] += part[i, j] * sa_f[i] * sb_f[j]
    T.barrier_arrive(ab_empty[slot])


@T.macro
def _fp8_ws_plain_step(a_smem, b_smem, ab_full, ab_empty, acc, prev, slot, phase, ki):
    """One K-step of a per-tensor consumer: WGMMA accumulates, the previous slot is released.

    The release trails by one step because step ``ki``'s WGMMA still reads slot
    ``ki``; ``prev`` carries the slot the next release belongs to.
    """
    T.barrier_wait(ab_full[slot], phase)
    T.wgmma_gemm(
        a_smem[slot, :, :],
        b_smem[slot, :, :],
        acc,
        transpose_B=True,
        policy=T.GemmWarpPolicy.FullRow,
        clear_accum=(ki == 0),
    )
    if ki > 0:
        T.wait_wgmma(1)
        T.barrier_arrive(ab_empty[prev[0]])
    prev[0] = slot


@T.macro
def _fp8_ws_plain_drain(ab_empty, acc, prev, scale_a, scale_b, *, num_regs: int, block_n: int):
    """Close a per-tensor consumer's mainloop and apply the two scalars."""
    T.wait_wgmma(0)
    T.barrier_arrive(ab_empty[prev[0]])
    T.warpgroup_fence_operand(acc, num_regs=num_regs)
    for i, j in T.Parallel(_FP8_WS_HALF_M, block_n):
        acc[i, j] *= scale_a[0, 0] * scale_b[0, 0]


@T.macro
def _fp8_ws_epilogue(
    c,
    bias,
    acc,
    out,
    c_smem,
    m_start,
    n_start,
    rows,
    cols,
    *,
    bar: int,
    block_n: int,
    n: int,
    out_dtype: str,
    has_bias: bool,
    stage_store: bool,
):
    """Cast one consumer's tile and store it: one TMA box when full, elements otherwise.

    ``stage_store`` is False where no row tile can be full — every ``m`` inside
    one consumer's half — and the staging tile is then not allocated.
    """
    half_m = _FP8_WS_HALF_M
    if has_bias:
        for i, j in T.Parallel(half_m, block_n):
            out[i, j] = T.cast(acc[i, j], out_dtype) + bias[T.min(n_start + j, n - 1)]
    else:
        T.copy(acc, out)
    if not stage_store:
        if rows > T.int32(0):
            for i, j in T.Parallel(half_m, block_n):
                if i < rows and j < cols:
                    c[m_start + i, n_start + j] = out[i, j]
    elif rows == T.int32(half_m) and cols == T.int32(block_n):
        T.sync_threads(barrier_id=bar, arrive_count=128)
        T.copy(out, c_smem)
        T.fence_proxy_async()
        T.sync_threads(barrier_id=bar, arrive_count=128)
        T.copy(c_smem, c[m_start, n_start])
    elif rows > T.int32(0):
        for i, j in T.Parallel(half_m, block_n):
            if i < rows and j < cols:
                c[m_start + i, n_start + j] = out[i, j]


@functools.lru_cache(maxsize=32)
def _gemm_fp8_ws_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool,
    *,
    sm_count: int,
) -> Callable:
    """Warp-specialized FP8 NT GEMM for SM90: 1 producer + 2 consumer warpgroups.

    Same split-A / shared-B layout as :func:`_gemm_coop2_kernel`: a producer
    warpgroup issues the TMA loads, two consumer warpgroups each own 64 of the
    tile's 128 rows and run their own WGMMA over a shared ``B`` ring, and the
    persistent grid sweeps a grouped tile order for L2 reuse.

    Under block128 scaling the producer also stages the K-step's ``A`` row
    scales and ``B`` column scales into that step's ring slot, and the consumer
    folds a fresh WGMMA accumulator in as
    ``acc += partial * scale_a[row] * scale_b[col]``. The staging is what makes
    that affordable: a thread holds ``block_n / 4`` distinct output columns, and
    reading their scales from global inside the mainloop is that many
    uncoalesced sectors per K-step.

    Per-tensor scaling has no such step: WGMMA accumulates across the whole K
    axis and the two scalars land in the epilogue.

    ``M``, ``N`` and ``K`` tails need no predicate on the load side: a TMA box
    past the end of the tensor is zero-filled, so a K tail contributes a zero
    term under any scale, and the epilogue predicates the store.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        dtype: FP8 operand dtype string.
        out_dtype: Output dtype string.
        block_scaled: True for block128 scale grids, False for per-tensor scalars.
        has_bias: Whether the compiled function takes a ``[n]`` bias operand.
        sm_count: Persistent grid width — the device SM count. Part of the cache
            key so a kernel built for one GPU is never reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, num_stages,
        group_size_m)`` returns the compiled ``prim_func``.
    """
    accum_dtype = "float"
    block_m = _FP8_WS_BLOCK_M
    block_k = _FP8_WS_BLOCK_K
    scale_k = (k + 127) // 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16", "-DENABLE_FP8"],
    )
    def _gemm_fp8_ws_func(
        block_n: int = 128,
        num_stages: int = 4,
        group_size_m: int = 8,
    ) -> Callable:
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        grid = min(sm_count, total_tiles)
        max_waves = -(-total_tiles // grid) + 1
        k_iters = -(-k // block_k)
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (n, scale_k) if block_scaled else (1, 1)
        bias_shape = (n,) if has_bias else (1,)
        stage_rows = num_stages if block_scaled else 1
        stage_store = m > half_m
        store_rows = half_m if stage_store else 1

        @T.prim_func
        def _gemm_fp8_ws_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(grid, threads=384) as (pid,):
                a_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_smem_0 = T.alloc_shared((store_rows, block_n), out_dtype)
                c_smem_1 = T.alloc_shared((store_rows, block_n), out_dtype)
                sa_stage = T.alloc_shared((stage_rows, block_m), accum_dtype)
                sb_stage = T.alloc_shared((stage_rows, block_n), accum_dtype)
                acc_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                part_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                part_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                out_0 = T.alloc_fragment((half_m, block_n), out_dtype)
                out_1 = T.alloc_fragment((half_m, block_n), out_dtype)
                sa_0 = T.alloc_fragment((half_m,), accum_dtype)
                sa_1 = T.alloc_fragment((half_m,), accum_dtype)
                sb_0 = T.alloc_fragment((block_n,), accum_dtype)
                sb_1 = T.alloc_fragment((block_n,), accum_dtype)

                layouts = {
                    a_top: tilelang.layout.make_swizzled_layout(a_top),
                    a_bot: tilelang.layout.make_swizzled_layout(a_bot),
                    b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                }
                if stage_store:
                    layouts[c_smem_0] = tilelang.layout.make_swizzled_layout(c_smem_0)
                    layouts[c_smem_1] = tilelang.layout.make_swizzled_layout(c_smem_1)
                T.annotate_layout(layouts)

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            _fp8_ws_tile_id(
                                flat_id,
                                mt,
                                nt,
                                group_size_m=group_size_m,
                                num_pid_m=num_pid_m,
                                num_pid_n=num_pid_n,
                            )
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                _fp8_ws_stage(
                                    a,
                                    b,
                                    scale_a,
                                    scale_b,
                                    a_top,
                                    a_bot,
                                    b_smem,
                                    sa_stage,
                                    sb_stage,
                                    ab_full,
                                    slot,
                                    ki,
                                    m_start,
                                    n_start,
                                    m=m,
                                    n=n,
                                    block_n=block_n,
                                    block_scaled=block_scaled,
                                )
                                gi_prod = gi_prod + 1

                elif tx < 256:
                    T.inc_max_nreg(232)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            _fp8_ws_tile_id(
                                flat_id,
                                mt,
                                nt,
                                group_size_m=group_size_m,
                                num_pid_m=num_pid_m,
                                num_pid_n=num_pid_n,
                            )
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.min(T.int32(half_m), T.int32(m) - m_start)
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            if block_scaled:
                                T.clear(acc_0)
                                for _ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_0 % num_stages
                                    _fp8_ws_scaled_step(
                                        a_top,
                                        b_smem,
                                        sa_stage,
                                        sb_stage,
                                        ab_full,
                                        ab_empty,
                                        acc_0,
                                        part_0,
                                        sa_0,
                                        sb_0,
                                        slot,
                                        (gi_cons_0 // num_stages) & 1,
                                        row_base=0,
                                        block_n=block_n,
                                    )
                                    gi_cons_0 = gi_cons_0 + 1
                            else:
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_0 % num_stages
                                    _fp8_ws_plain_step(
                                        a_top,
                                        b_smem,
                                        ab_full,
                                        ab_empty,
                                        acc_0,
                                        ps0,
                                        slot,
                                        (gi_cons_0 // num_stages) & 1,
                                        ki,
                                    )
                                    gi_cons_0 = gi_cons_0 + 1
                                _fp8_ws_plain_drain(
                                    ab_empty,
                                    acc_0,
                                    ps0,
                                    scale_a,
                                    scale_b,
                                    num_regs=nr,
                                    block_n=block_n,
                                )
                            _fp8_ws_epilogue(
                                c,
                                bias,
                                acc_0,
                                out_0,
                                c_smem_0,
                                m_start,
                                n_start,
                                arows,
                                acols,
                                bar=_CONSUMER_BAR_WG0,
                                block_n=block_n,
                                n=n,
                                out_dtype=out_dtype,
                                has_bias=has_bias,
                                stage_store=stage_store,
                            )

                else:
                    T.inc_max_nreg(232)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            _fp8_ws_tile_id(
                                flat_id,
                                mt,
                                nt,
                                group_size_m=group_size_m,
                                num_pid_m=num_pid_m,
                                num_pid_n=num_pid_n,
                            )
                            m_start = mt[0] * block_m + half_m
                            n_start = nt[0] * block_n
                            brows = T.max(T.int32(0), T.min(T.int32(half_m), T.int32(m) - m_start))
                            bcols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            if block_scaled:
                                T.clear(acc_1)
                                for _ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_1 % num_stages
                                    _fp8_ws_scaled_step(
                                        a_bot,
                                        b_smem,
                                        sa_stage,
                                        sb_stage,
                                        ab_full,
                                        ab_empty,
                                        acc_1,
                                        part_1,
                                        sa_1,
                                        sb_1,
                                        slot,
                                        (gi_cons_1 // num_stages) & 1,
                                        row_base=half_m,
                                        block_n=block_n,
                                    )
                                    gi_cons_1 = gi_cons_1 + 1
                            else:
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_1 % num_stages
                                    _fp8_ws_plain_step(
                                        a_bot,
                                        b_smem,
                                        ab_full,
                                        ab_empty,
                                        acc_1,
                                        ps1,
                                        slot,
                                        (gi_cons_1 // num_stages) & 1,
                                        ki,
                                    )
                                    gi_cons_1 = gi_cons_1 + 1
                                _fp8_ws_plain_drain(
                                    ab_empty,
                                    acc_1,
                                    ps1,
                                    scale_a,
                                    scale_b,
                                    num_regs=nr,
                                    block_n=block_n,
                                )
                            _fp8_ws_epilogue(
                                c,
                                bias,
                                acc_1,
                                out_1,
                                c_smem_1,
                                m_start,
                                n_start,
                                brows,
                                bcols,
                                bar=_CONSUMER_BAR_WG1,
                                block_n=block_n,
                                n=n,
                                out_dtype=out_dtype,
                                has_bias=has_bias,
                                stage_store=stage_store,
                            )

        return _gemm_fp8_ws_main

    return _gemm_fp8_ws_func


@functools.lru_cache(maxsize=32)
def _gemm_fp8_ws_splitk_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool,
    *,
    split_k: int,
) -> Callable:
    """Split-K variant of the warp-specialized FP8 mainloop (NT).

    Slicing K across ``grid.y`` multiplies the grid by ``split_k`` without
    changing the bytes any CTA reads, which is what a decode shape needs: one
    ``block_m`` row tile and few column tiles leave most of the device idle and
    each resident CTA at its own SM's read bandwidth. Each slice writes an fp32
    partial tile into ``slices[split_k, m, n]``; :func:`_splitk_reduce_kernel`
    sums them and casts. Slice 0 carries the bias, so the sum adds it once.

    The mainloop bodies are the macros :func:`_gemm_fp8_ws_kernel` uses; the
    shell differs — a two-dimensional grid instead of a persistent sweep, and
    an fp32 workspace instead of the cast-and-store epilogue.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        dtype: FP8 operand dtype string.
        out_dtype: Output dtype string, which the bias operand also carries.
        block_scaled: True for block128 scale grids, False for per-tensor scalars.
        has_bias: Whether the compiled function takes a ``[n]`` bias operand.
        split_k: Number of K slices; must divide the block128 K-tile count evenly.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, num_stages,
        group_size_m)`` returns the compiled ``prim_func`` producing the fp32
        workspace.
    """
    accum_dtype = "float"
    block_m = _FP8_WS_BLOCK_M
    block_k = _FP8_WS_BLOCK_K
    scale_k = (k + 127) // 128
    k_iters_total = -(-k // block_k)
    if k_iters_total % split_k:
        raise ValueError(
            f"split_k={split_k} must divide the K-tile count evenly "
            f"(k={k}, block_k={block_k} -> {k_iters_total} tiles)"
        )

    @tilelang.jit(
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16", "-DENABLE_FP8"],
    )
    def _gemm_fp8_ws_splitk_func(
        block_n: int = 64,
        num_stages: int = 5,
        group_size_m: int = 8,
    ) -> Callable:
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        k_iters = k_iters_total // split_k
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (n, scale_k) if block_scaled else (1, 1)
        stage_rows = num_stages if block_scaled else 1
        bias_shape = (n,) if has_bias else (1,)

        @T.prim_func
        def _gemm_fp8_ws_splitk_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            slices: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(total_tiles, split_k, threads=384) as (pid, bz):
                a_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                sa_stage = T.alloc_shared((stage_rows, block_m), accum_dtype)
                sb_stage = T.alloc_shared((stage_rows, block_n), accum_dtype)
                acc_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                part_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                part_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                sa_0 = T.alloc_fragment((half_m,), accum_dtype)
                sa_1 = T.alloc_fragment((half_m,), accum_dtype)
                sb_0 = T.alloc_fragment((block_n,), accum_dtype)
                sb_1 = T.alloc_fragment((block_n,), accum_dtype)

                T.annotate_layout(
                    {
                        a_top: tilelang.layout.make_swizzled_layout(a_top),
                        a_bot: tilelang.layout.make_swizzled_layout(a_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                _fp8_ws_tile_id(
                    pid,
                    mt,
                    nt,
                    group_size_m=group_size_m,
                    num_pid_m=num_pid_m,
                    num_pid_n=num_pid_n,
                )
                m_start = mt[0] * block_m
                n_start = nt[0] * block_n
                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.Pipelined(k_iters, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_empty[slot], ((ki // num_stages) & 1) ^ 1)
                        _fp8_ws_stage(
                            a,
                            b,
                            scale_a,
                            scale_b,
                            a_top,
                            a_bot,
                            b_smem,
                            sa_stage,
                            sb_stage,
                            ab_full,
                            slot,
                            bz * k_iters + ki,
                            m_start,
                            n_start,
                            m=m,
                            n=n,
                            block_n=block_n,
                            block_scaled=block_scaled,
                        )

                elif tx < 256:
                    T.inc_max_nreg(232)
                    arows = T.min(T.int32(half_m), T.int32(m) - m_start)
                    acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                    if block_scaled:
                        T.clear(acc_0)
                        for ki in T.Pipelined(k_iters, num_stages=0):
                            _fp8_ws_scaled_step(
                                a_top,
                                b_smem,
                                sa_stage,
                                sb_stage,
                                ab_full,
                                ab_empty,
                                acc_0,
                                part_0,
                                sa_0,
                                sb_0,
                                ki % num_stages,
                                (ki // num_stages) & 1,
                                row_base=0,
                                block_n=block_n,
                            )
                    else:
                        for ki in T.Pipelined(k_iters, num_stages=0):
                            _fp8_ws_plain_step(
                                a_top,
                                b_smem,
                                ab_full,
                                ab_empty,
                                acc_0,
                                ps0,
                                ki % num_stages,
                                (ki // num_stages) & 1,
                                ki,
                            )
                        _fp8_ws_plain_drain(
                            ab_empty,
                            acc_0,
                            ps0,
                            scale_a,
                            scale_b,
                            num_regs=nr,
                            block_n=block_n,
                        )
                    if has_bias and bz == 0:
                        for i, j in T.Parallel(half_m, block_n):
                            acc_0[i, j] += T.cast(bias[T.min(n_start + j, n - 1)], accum_dtype)
                    for i, j in T.Parallel(half_m, block_n):
                        if i < arows and j < acols:
                            slices[bz, m_start + i, n_start + j] = acc_0[i, j]

                else:
                    T.inc_max_nreg(232)
                    brows = T.max(T.int32(0), T.min(T.int32(half_m), T.int32(m) - m_start - half_m))
                    bcols = T.min(T.int32(block_n), T.int32(n) - n_start)
                    if block_scaled:
                        T.clear(acc_1)
                        for ki in T.Pipelined(k_iters, num_stages=0):
                            _fp8_ws_scaled_step(
                                a_bot,
                                b_smem,
                                sa_stage,
                                sb_stage,
                                ab_full,
                                ab_empty,
                                acc_1,
                                part_1,
                                sa_1,
                                sb_1,
                                ki % num_stages,
                                (ki // num_stages) & 1,
                                row_base=half_m,
                                block_n=block_n,
                            )
                    else:
                        for ki in T.Pipelined(k_iters, num_stages=0):
                            _fp8_ws_plain_step(
                                a_bot,
                                b_smem,
                                ab_full,
                                ab_empty,
                                acc_1,
                                ps1,
                                ki % num_stages,
                                (ki // num_stages) & 1,
                                ki,
                            )
                        _fp8_ws_plain_drain(
                            ab_empty,
                            acc_1,
                            ps1,
                            scale_a,
                            scale_b,
                            num_regs=nr,
                            block_n=block_n,
                        )
                    if has_bias and bz == 0:
                        for i, j in T.Parallel(half_m, block_n):
                            acc_1[i, j] += T.cast(bias[T.min(n_start + j, n - 1)], accum_dtype)
                    for i, j in T.Parallel(half_m, block_n):
                        if i < brows and j < bcols:
                            slices[bz, m_start + half_m + i, n_start + j] = acc_1[i, j]

        return _gemm_fp8_ws_splitk_main

    return _gemm_fp8_ws_splitk_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    traced: bool = False,
    *,
    sm_count: int,
) -> Callable:
    """Hand-written warp-specialized GEMM ``C = op(A) @ op(B)`` for SM90.

    One producer warpgroup (128 threads) issues TMA loads into a double-buffered
    SMEM ring; one consumer warpgroup (128 threads) runs the WGMMA and accumulates
    over K. All four layouts are covered by ``trans_a`` / ``trans_b`` (forwarded to
    the WGMMA transpose flags): ``A`` is $[M \\times K]$ (or $[K \\times M]$ transposed), ``B``
    is $[K \\times N]$ (or $[N \\times K]$ transposed), ``C`` is $[M \\times N]$. fp16 / bf16 inputs,
    fp32 accumulation. The auto warp-specialization pass is disabled so it does not
    fire on top of this manual layout.

    Operands must satisfy TMA's innermost-dimension alignment, which
    ``_tma_misalignment`` states and ``GemmTmaKernel`` refuses on.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed ($[K \\times M]$).
        trans_b: Whether ``B`` is stored transposed ($[N \\times K]$).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).
        traced: Build with in-kernel timeline markers materialized (``True``) or
            stripped to zero cost (``False``). **Part of the cache key**: traced
            and untraced builds are distinct cached kernels, so flipping the
            process trace switch never returns a stale variant. Callers pass
            ``trace.enabled`` explicitly rather than letting the build read the
            global switch.
        sm_count: Device SM count, deciding the multi-wave TMA-store epilogue
            gate. Part of the cache key so a kernel built for one GPU is never
            reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, panel_size)`` returns the compiled ``prim_func``.
        When ``traced``
        it materializes the markers and appends a trailing ``slots`` output (so
        ``out_idx`` returns ``(C, slots)``); otherwise ``C`` is the lone output.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=trace.out_idx(1, traced),
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        panel_size: int = 10,
    ) -> Callable:
        # Manual 2-warpgroup WS: 1 producer WG (128 threads) issues TMA, 1
        # consumer WG (128 threads) runs WGMMA. Barrier arrive_counts (128) are
        # bound to this layout, so threads is fixed at 256.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        b_evict = _b_eviction(m, block_m)
        grid_size = -(-n // block_n) * -(-m // block_m)
        tma_epilogue = (n * 2) % 16 == 0 and grid_size > sm_count

        @T.prim_func
        def _gemm_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                if tma_epilogue:
                    c_smem = T.alloc_shared((block_m, block_n), dtype)
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                            c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        }
                    )

                # Producer→consumer (buffer full) and consumer→producer (buffer
                # empty) barriers, one per ring slot. Each is arrived by exactly
                # one warpgroup (128 threads). Allocated as length-num_stages
                # barrier arrays and indexed by the static slot id.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                # Monotonic per-warpgroup iteration counters; stage = gi %
                # num_stages, phase = (gi // num_stages) % 2.

                m_start = by * block_m
                n_start = bx * block_n

                ps = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: issue TMA loads of A and B tiles. ──
                    # Intern the "producer" group first so it gets gid 0.
                    T.dec_max_nreg(24)
                    with trace.group("producer", lead=0):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            k_start = ki * block_k
                            T.barrier_wait(ab_empty[slot], phase ^ 1)
                            with trace.range("tma", lane="tma"):
                                if trans_a:
                                    T.tma_copy(
                                        a[k_start : k_start + block_k, m_start : m_start + block_m],
                                        a_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                else:
                                    T.tma_copy(
                                        a[m_start : m_start + block_m, k_start : k_start + block_k],
                                        a_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                if trans_b:
                                    T.tma_copy(
                                        b[n_start : n_start + block_n, k_start : k_start + block_k],
                                        b_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                        eviction_policy=b_evict,
                                    )
                                else:
                                    T.tma_copy(
                                        b[k_start : k_start + block_k, n_start : n_start + block_n],
                                        b_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                        eviction_policy=b_evict,
                                    )
                            with trace.range("arrive", lane="barrier"):
                                T.barrier_arrive(ab_full[slot])
                else:
                    # ── Consumer warpgroup: run WGMMA, accumulate over K. ──
                    T.inc_max_nreg(240)
                    num_accum_regs = (block_m * block_n) // 128
                    with trace.group("consumer", lead=128):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            with trace.range("wait", lane="barrier"):
                                T.barrier_wait(ab_full[slot], phase)
                            with trace.range("mma", lane="wgmma"):
                                T.wgmma_gemm(
                                    a_smem[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local,
                                    transpose_A=trans_a,
                                    transpose_B=trans_b,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                            if ki > 0:
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                                T.barrier_arrive(ab_empty[ps[0]])
                            ps[0] = slot

                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                        T.barrier_arrive(ab_empty[ps[0]])
                        if tma_epilogue:
                            with trace.range("epilogue"):
                                T.copy(c_local, c_smem)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(c_smem, c[m_start, n_start])
                        else:
                            with trace.range("epilogue"):
                                for i, j in T.Parallel(block_m, block_n):
                                    if m_start + i < m and n_start + j < n:
                                        c[m_start + i, n_start + j] = c_local[i, j]

                # Build-time flow declaration: producer "arrive" → consumer
                # "wait" (fixed per-iter pairing).
                trace.dag("arrive", "wait")

        # Materialize markers + append ``slots`` when traced; no-op them (identical
        # CUDA to an un-instrumented build) otherwise. Pairs with ``out_idx`` above.
        return trace.finalize(_gemm_main, traced=traced, max_events=1024)

    return _gemm_func


@functools.lru_cache(maxsize=32)
def _gemm_splitk_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Split-K variant of the warp-specialized GEMM mainloop.

    The K contraction is sliced across ``split_k`` CTAs (grid z). Each CTA
    runs the same producer/consumer pipeline as ``_gemm_kernel`` over its
    K slice and writes an fp32 partial tile to the workspace
    ``w[split_k, m, n]``; ``_splitk_reduce_kernel`` then sums the slices and
    casts to the storage dtype. Splitting only pays off when the natural
    (M, N) grid underfills the GPU — see ``GemmTmaKernel.forward`` for the
    dispatch. ``split_k`` must divide the K-tile count evenly.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed (``[K, M]``).
        trans_b: Whether ``B`` is stored transposed (``[N, K]``).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, panel_size, split_k)`` returns the compiled
        ``prim_func`` producing the fp32 workspace.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_splitk_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 4,
        panel_size: int = 16,
        split_k: int = 2,
    ) -> Callable:
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        if k_iters % split_k != 0:
            raise ValueError(
                f"split_k={split_k} must divide the K-tile count evenly "
                f"(k={k}, block_k={block_k} -> {k_iters} tiles)"
            )
        k_slice = k_iters // split_k
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        b_evict = _b_eviction(m, block_m)

        @T.prim_func
        def _gemm_splitk_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (bx, by, bz):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                ps = T.alloc_local((1,), "int32")

                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        k_start = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        if trans_a:
                            T.tma_copy(
                                a[k_start : k_start + block_k, m_start : m_start + block_m],
                                a_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                        else:
                            T.tma_copy(
                                a[m_start : m_start + block_m, k_start : k_start + block_k],
                                a_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                        if trans_b:
                            T.tma_copy(
                                b[n_start : n_start + block_n, k_start : k_start + block_k],
                                b_smem[slot, :, :],
                                barrier=ab_full[slot],
                                eviction_policy=b_evict,
                            )
                        else:
                            T.tma_copy(
                                b[k_start : k_start + block_k, n_start : n_start + block_n],
                                b_smem[slot, :, :],
                                barrier=ab_full[slot],
                                eviction_policy=b_evict,
                            )
                        T.barrier_arrive(ab_full[slot])
                else:
                    T.inc_max_nreg(240)
                    num_accum_regs = (block_m * block_n) // 128
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem[slot, :, :],
                            b_smem[slot, :, :],
                            c_local,
                            transpose_A=trans_a,
                            transpose_B=trans_b,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                            T.barrier_arrive(ab_empty[ps[0]])
                        ps[0] = slot

                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                    T.barrier_arrive(ab_empty[ps[0]])
                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            w[bz, m_start + i, n_start + j] = c_local[i, j]

        return _gemm_splitk_main

    return _gemm_splitk_func


@functools.lru_cache(maxsize=32)
def _splitk_reduce_kernel(
    split_k: int,
    m: int,
    n: int,
    dtype: str = "float16",
    activation: str = "none",
) -> Callable:
    """Reduce the split-K fp32 workspace into the final output.

    Sums ``w[split_k, m, n]`` over the slice axis in fp32 and casts to the
    storage dtype at the boundary. Bandwidth-trivial elementwise kernel; the
    workspace of the shapes worth splitting fits in L2.

    ``C`` is an explicit parameter rather than a JIT-allocated output
    (``out_idx``): the caller allocates it *before* launching the mainloop, so
    the allocation no longer sits between the two launches. On short-mainloop
    shapes the mainloop drains in ~14 us while the host is still allocating,
    and the span metric charges that idle to us (see ``_splitk_pair``).
    """
    accum_dtype = "float"
    if activation not in ("none", "silu_and_mul"):
        raise ValueError(f"unsupported split-K epilogue: {activation}")
    gated = activation == "silu_and_mul"
    if gated and n % 2:
        raise ValueError("silu_and_mul requires an even output width")
    out_n = n // 2 if gated else n

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: gated},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _splitk_reduce_func(elems_per_cta: int = 1024) -> Callable:
        def _slice_sum(w, gi, gj):
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            return expr

        @T.prim_func
        def _splitk_reduce_main(
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
            c: T.Tensor((m, out_n), dtype),  # type: ignore
        ) -> None:
            total = m * out_n
            with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx:
                base = bx * elems_per_cta
                for t in T.Parallel(elems_per_cta):
                    idx = base + t
                    if idx < total:
                        gi = idx // out_n
                        gj = idx % out_n
                        if gated:
                            gate = _slice_sum(w, gi, gj)
                            up = _slice_sum(w, gi, out_n + gj)
                            c[gi, gj] = T.cast(gate * T.sigmoid(gate) * up, dtype)
                        else:
                            c[gi, gj] = T.cast(_slice_sum(w, gi, gj), dtype)

        return _splitk_reduce_main

    return _splitk_reduce_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
) -> Callable:
    """Persistent 2-consumer (cooperative) warp-specialized GEMM for SM90.

    Matches the cuBLAS cooperative (``coopA``) layout: one producer
    warpgroup (128 threads) plus **two** consumer warpgroups (256 threads,
    384 total). A ``block_m x block_n`` output tile is split along M — each
    consumer owns ``block_m // 2`` rows and runs its own WGMMA; the ``B`` tile
    is loaded once and shared (split-A / shared-B). Two math warpgroups double
    the WGMMA issue rate over the single-consumer ``_gemm_kernel``, which is the
    edge on compute-bound prefill shapes (large M, GPU-filling grid).

    A static-wave persistent loop (grid = ``sm_count``; each CTA sweeps tile ids
    ``flat_id = sm_count * w + pid``) overlaps a tile's TMA-store epilogue with
    the next tile's mainloop prologue: the ring counters carry across waves so
    the producer keeps prefetching. A Triton-style grouped tile order
    (``group_size_m``) keeps concurrently-resident CTAs on a shared ``B`` column
    stripe for L2 reuse.

    The epilogue carries across tiles the same way: each consumer stores its tile
    as ``block_n / stage_n`` slices through ``stage_buf`` rotating staging tiles,
    waiting only until ``stage_buf - 1`` stores are in flight, and drains once at
    the end of its persistent loop so no store still reads shared memory at exit.

    NT only (``A[m,k] @ B[n,k]ᵀ``): the split-A layout and shared ``B`` ring are
    specific to a non-transposed ``A`` and transposed ``B``. Other layouts fall
    back to ``_gemm_kernel``. M / N tails are handled by a predicated scalar
    epilogue (full tiles use the TMA store); K tails are TMA zero-filled.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).
        sm_count: Persistent grid width — the device SM count. Part of the cache
            key so a kernel built for one GPU is never reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages, group_size_m, stage_n, stage_buf)`` returns the compiled
        ``prim_func``.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_coop2_kernel is NT-only (trans_a=False, trans_b=True)")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2_func(
        block_n: int = 256,
        block_k: int = 64,
        num_stages: int = 3,
        group_size_m: int = 16,
        stage_n: int = 0,
        stage_buf: int = 0,
    ) -> Callable:
        half_m = block_m // 2
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // 128
        sn = stage_n or (_COOP2_STAGE_N if block_n % _COOP2_STAGE_N == 0 else block_n)
        if block_n % sn:
            raise ValueError(f"coop2 stage_n must divide block_n={block_n}, got {stage_n}")
        n_chunks = block_n // sn
        nbuf = stage_buf or n_chunks
        if sn != _COOP2_STAGE_N and nbuf != 1:
            raise ValueError(
                f"coop2 stage_n must be {_COOP2_STAGE_N} to row-stack {nbuf} staging tiles "
                f"into valid TMA boxes, got {sn}"
            )
        if n_chunks % nbuf:
            raise ValueError(f"coop2 stage_buf must divide the {n_chunks} slices, got {stage_buf}")
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)

        @T.macro
        def decode(flat_id, mt, nt):
            gin = T.int32(group_size_m * num_pid_n)
            gid = flat_id // gin
            first_m = gid * T.int32(group_size_m)
            gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
            mt[0] = first_m + (flat_id % gin) % gsize
            nt[0] = (flat_id % gin) // gsize

        @T.prim_func
        def _gemm_coop2_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((nbuf * half_m, sn), dtype)
                c_smem_1 = T.alloc_shared((nbuf * half_m, sn), dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                ks = ki * block_k
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                T.tma_copy(
                                    a[m_start : m_start + half_m, ks : ks + block_k],
                                    a_smem_top[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.tma_copy(
                                    a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                                    a_smem_bot[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.tma_copy(
                                    b[n_start : n_start + block_n, ks : ks + block_k],
                                    b_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                    eviction_policy=b_evict,
                                )
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.min(T.int32(half_m), T.int32(m) - m_start)
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_0 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_0 // num_stages) & 1)
                                T.wgmma_gemm(
                                    a_smem_top[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local_0,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if ki > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps0[0]])
                                ps0[0] = slot
                                gi_cons_0 = gi_cons_0 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                            T.copy(c_local_0, c_cast_0)
                            if arows == T.int32(half_m) and acols == T.int32(block_n):
                                for ch in range(n_chunks):
                                    c0 = ch * sn
                                    r0 = (ch % nbuf) * half_m
                                    T.tma_store_wait(nbuf - 1)
                                    T.copy(
                                        c_cast_0[:, c0 : c0 + sn],
                                        c_smem_0[r0 : r0 + half_m, :],
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                                    T.tma_copy(
                                        c_smem_0[r0 : r0 + half_m, :],
                                        c[m_start, n_start + c0],
                                    )
                            else:
                                for i, j in T.Parallel(half_m, block_n):
                                    if i < arows and j < acols:
                                        c[m_start + i, n_start + j] = c_cast_0[i, j]
                    T.tma_store_wait(0)

                else:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.max(
                                T.int32(0),
                                T.min(T.int32(half_m), T.int32(m) - m_start - T.int32(half_m)),
                            )
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_1 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_1 // num_stages) & 1)
                                T.wgmma_gemm(
                                    a_smem_bot[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local_1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if ki > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps1[0]])
                                ps1[0] = slot
                                gi_cons_1 = gi_cons_1 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                            T.copy(c_local_1, c_cast_1)
                            if arows == T.int32(half_m) and acols == T.int32(block_n):
                                for ch in range(n_chunks):
                                    c0 = ch * sn
                                    r0 = (ch % nbuf) * half_m
                                    T.tma_store_wait(nbuf - 1)
                                    T.copy(
                                        c_cast_1[:, c0 : c0 + sn],
                                        c_smem_1[r0 : r0 + half_m, :],
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                                    T.tma_copy(
                                        c_smem_1[r0 : r0 + half_m, :],
                                        c[m_start + half_m, n_start + c0],
                                    )
                            elif arows > T.int32(0):
                                for i, j in T.Parallel(half_m, block_n):
                                    if i < arows and j < acols:
                                        c[m_start + half_m + i, n_start + j] = c_cast_1[i, j]
                    T.tma_store_wait(0)

        return _gemm_coop2_main

    return _gemm_coop2_func


@functools.lru_cache(maxsize=32)
def _gemm_pingpong_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    sm_count: int = 132,
) -> Callable:
    """Ping-pong GEMM (NT): two consumer warpgroups on alternate tiles, one at a time.

    Same producer and ring as ``_gemm_coop2_kernel``, but each consumer holds a whole
    ``128 x block_n`` accumulator and takes every other persistent-loop tile. A
    consumer's mainloop starts only once the other has issued its last WGMMA (the
    ``go`` barriers), so one consumer's epilogue runs under the other's mainloop and
    the tensor core never waits for a store; coop2's consumers finish a tile together
    and idle it for 3-5k cycles per tile. The handshake is also what keeps the ring's
    parity waits sound: a consumer two phases ahead reads the older phase as complete.

    Each tile is one consumer's, so ``B`` is not shared and the tile is narrower than
    coop2's; 176 is the tiling cuBLASLt's best Hopper kernel uses. The epilogue
    rotates ``stage_buf`` staging tiles of ``stage_n`` columns, waiting only for the
    store that used the same tile ``stage_buf`` slices ago, and M / N tail tiles store
    through TMA's bounds clipping.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``. Must be a multiple of 8: the ``C`` TMA
            descriptor addresses it in 16-byte units.
        k: Contraction dim.
        trans_a: Must be ``False``.
        trans_b: Must be ``True``.
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).
        sm_count: Persistent grid size.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages, group_size_m, stage_n, stage_buf)`` returns the compiled
        ``prim_func``.

    Raises:
        ValueError: Not NT, ``n % 8``, a ``stage_n`` that is not a multiple of 8
            dividing ``block_n``, or a grid of at most ``sm_count`` tiles, which leaves
            the second consumer provably idle and TileLang then rejects its TMA store.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_pingpong_kernel is NT-only (trans_a=False, trans_b=True)")
    if n % 8:
        raise ValueError(f"_gemm_pingpong_kernel stores C through TMA, needs n % 8 == 0, got {n}")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_pingpong_func(
        block_n: int = 176,
        block_k: int = 64,
        num_stages: int = 5,
        group_size_m: int = 16,
        stage_n: int = 16,
        stage_buf: int = 2,
    ) -> Callable:
        if stage_n % 8 or block_n % stage_n:
            raise ValueError(
                f"pingpong stage_n must be a multiple of 8 dividing block_n={block_n}, got {stage_n}"
            )
        if stage_buf < 1:
            raise ValueError(f"pingpong stage_buf must be at least 1, got {stage_buf}")
        n_chunks = block_n // stage_n
        nbuf = min(stage_buf, n_chunks)
        b_evict = _b_eviction(m, block_m)
        nr = (block_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        if total_tiles <= sm_count:
            raise ValueError(
                f"pingpong needs more than {sm_count} tiles, got {total_tiles} at block_n={block_n}"
            )
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)

        @T.macro
        def decode(flat_id, mt, nt):
            gin = T.int32(group_size_m * num_pid_n)
            gid = flat_id // gin
            first_m = gid * T.int32(group_size_m)
            gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
            mt[0] = first_m + (flat_id % gin) % gsize
            nt[0] = (flat_id % gin) // gsize

        @T.macro
        def consumer(parity, c_local, c_smem, pid, a_smem, b_smem, ab_full, ab_empty, go, c):
            bar_id = _CONSUMER_BAR_WG0 + parity
            ps = T.alloc_local((1,), "int32")
            mt = T.alloc_local((1,), "int32")
            nt = T.alloc_local((1,), "int32")
            for w in T.serial(max_waves):
                flat_id = T.int32(sm_count) * w + pid
                if (w % 2 == parity) and (flat_id < total_tiles):
                    decode(flat_id, mt, nt)
                    m_start = mt[0] * block_m
                    n_start = nt[0] * block_n
                    if w > 0:
                        T.barrier_wait(go[parity], ((w // 2) + parity + 1) & 1)
                    for ki in T.Pipelined(k_iters, num_stages=0):
                        gi = w * k_iters + ki
                        slot = gi % num_stages
                        T.barrier_wait(ab_full[slot], (gi // num_stages) & 1)
                        T.wgmma_gemm(
                            a_smem[slot, :, :],
                            b_smem[slot, :, :],
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps[0]])
                        ps[0] = slot
                    T.barrier_arrive(go[1 - parity])
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps[0]])
                    T.warpgroup_fence_operand(c_local, num_regs=nr)
                    T.tma_store_wait(0)
                    for ch in range(n_chunks):
                        c0 = ch * stage_n
                        r0 = (ch % nbuf) * block_m
                        T.tma_store_wait(nbuf - 1)
                        T.sync_threads(barrier_id=bar_id, arrive_count=128)
                        T.copy(c_local[:, c0 : c0 + stage_n], c_smem[r0 : r0 + block_m, :])
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=bar_id, arrive_count=128)
                        T.tma_copy(c_smem[r0 : r0 + block_m, :], c[m_start, n_start + c0])
            T.tma_store_wait(0)

        @T.prim_func
        def _gemm_pingpong_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_smem = T.alloc_shared((num_stages, block_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_smem_0 = T.alloc_shared((nbuf * block_m, stage_n), dtype)
                c_smem_1 = T.alloc_shared((nbuf * block_m, stage_n), dtype)

                T.annotate_layout(
                    {
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                go = T.alloc_barrier([128, 128])

                gi_prod = T.alloc_var("int32", init=0)
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                ks = ki * block_k
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                T.tma_copy(
                                    a[m_start : m_start + block_m, ks : ks + block_k],
                                    a_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.tma_copy(
                                    b[n_start : n_start + block_n, ks : ks + block_k],
                                    b_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                    eviction_policy=b_evict,
                                )
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1
                elif tx < 256:
                    T.inc_max_nreg(240)
                    consumer(0, c_local_0, c_smem_0, pid, a_smem, b_smem, ab_full, ab_empty, go, c)
                else:
                    T.inc_max_nreg(240)
                    consumer(1, c_local_1, c_smem_1, pid, a_smem, b_smem, ab_full, ab_empty, go, c)

        return _gemm_pingpong_main

    return _gemm_pingpong_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2_splitk_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Split-K variant of the 2-consumer (cooperative) GEMM mainloop (NT).

    For small-M shapes whose natural (M, N) grid underfills the GPU with a
    single K-slice, this slices K across grid-z CTAs. Each CTA runs the coop2
    mainloop (1 producer + 2 math WGs, split-A / shared-B, ``block_m`` fixed at
    128 = two 64-row consumers) over its K-slice and writes an fp32 partial tile
    into ``w[split_k, m, n]``; ``_splitk_reduce_kernel`` then sums the slices and
    casts to the storage dtype.

    The 2-consumer mainloop is the more WGMMA-efficient of the two, so it takes
    the large-K decode shapes, whose slices amortize the reduce round-trip;
    dispatch keeps short-K shapes on ``_gemm_splitk_kernel``.

    NT only. ``split_k`` must divide the K-tile count evenly.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages, split_k)`` returns the compiled ``prim_func`` producing the
        fp32 workspace ``w[split_k, m, n]``.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_coop2_splitk_kernel is NT-only (trans_a=False, trans_b=True)")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2_splitk_func(
        block_n: int = 64, block_k: int = 128, num_stages: int = 4, split_k: int = 4
    ) -> Callable:
        half_m = block_m // 2
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // 128
        k_iters_total = T.ceildiv(k, block_k)
        if k_iters_total % split_k != 0:
            raise ValueError(
                f"split_k={split_k} must divide the K-tile count evenly "
                f"(k={k}, block_k={block_k} -> {k_iters_total} tiles)"
            )
        k_slice = k_iters_total // split_k

        @T.prim_func
        def _gemm_coop2_splitk_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=384) as (
                bx,
                by,
                bz,
            ):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    }
                )
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice
                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        ks = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], ((ki // num_stages) % 2) ^ 1)
                        T.tma_copy(
                            a[m_start : m_start + half_m, ks : ks + block_k],
                            a_smem_top[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                            a_smem_bot[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            b[n_start : n_start + block_n, ks : ks + block_k],
                            b_smem[slot, :, :],
                            barrier=ab_full[slot],
                            eviction_policy=b_evict,
                        )
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_full[slot], (ki // num_stages) % 2)
                        T.wgmma_gemm(
                            a_smem_top[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_0,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    for i, j in T.Parallel(half_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            w[bz, m_start + i, n_start + j] = c_local_0[i, j]
                else:
                    T.inc_max_nreg(240)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_full[slot], (ki // num_stages) % 2)
                        T.wgmma_gemm(
                            a_smem_bot[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    for i, j in T.Parallel(half_m, block_n):
                        if m_start + half_m + i < m and n_start + j < n:
                            w[bz, m_start + half_m + i, n_start + j] = c_local_1[i, j]

        return _gemm_coop2_splitk_main

    return _gemm_coop2_splitk_func


@functools.lru_cache(maxsize=32)
def _splitk_pair(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    coop2: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    panel_size: int,
    split_k: int,
    activation: str = "none",
) -> tuple[Callable, Callable]:
    """Resolve the (mainloop, reduce) compiled pair for a split-K config.

    Both split-K paths run two kernels back to back, so every microsecond the
    host spends between the two launches is GPU idle the span metric charges
    to us: on the short-mainloop shapes the mainloop drains before the reduce
    is even enqueued. Folding the builder lookup and the ``@tilelang.jit``
    factory call of *both* kernels into one cached resolution keeps that
    window to the two launches themselves.

    The other host step that used to land in that window is allocating ``C``.
    ``_splitk_reduce_kernel`` therefore takes it as an explicit parameter, and
    both callers allocate it *before* launching the mainloop, which closes the
    remaining gap to the floor torch reaches on the same rows.
    """
    if coop2:
        mainloop = _gemm_coop2_splitk_kernel(m, n, k, trans_a, trans_b, dtype)(
            block_n, block_k, num_stages, split_k
        )
    else:
        mainloop = _gemm_splitk_kernel(m, n, k, trans_a, trans_b, dtype)(
            block_m, block_n, block_k, num_stages, panel_size, split_k
        )
    elems_per_cta = 256 if activation != "none" else 1024
    return mainloop, _splitk_reduce_kernel(split_k, m, n, dtype, activation)(elems_per_cta)


@functools.lru_cache(maxsize=32)
def _gemm_simple_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Non-warp-specialized pipelined GEMM for short-mainloop shapes (SM90).

    A stock ``T.Pipelined`` + ``T.gemm`` kernel: every thread cooperates in
    both copy and math, and the compiler schedules the cp.async/TMA overlap.
    On short-K skinny-M NT shapes (the decode-down family: ~16 K iterations,
    about one CTA wave) this beats the warp-specialized kernel by ~4% — with
    so short a mainloop the WS producer warpgroup's fixed costs (barrier
    protocol per iteration, idle tail, 128 threads not doing math) outweigh
    the benefit of its hand-managed deeper ring.

    Selected by the dense GEMM heuristic for exactly tiled NT calls. Requires
    tiles that divide the problem exactly; the builder raises ``ValueError``
    otherwise.
    """
    if trans_a:
        raise ValueError("_gemm_simple_kernel supports trans_a=False only")
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_simple_func(
        block_m: int = 64,
        block_n: int = 128,
        block_k: int = 128,
        num_stages: int = 4,
        threads: int = 128,
        panel_size: int = 8,
        cluster_m: int = 1,
    ) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"_gemm_simple_kernel requires exact tiling: got "
                f"m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}"
            )
        if cluster_m > 1:
            if (m // block_m) % cluster_m:
                raise ValueError(
                    f"cluster_m={cluster_m} must divide the M grid ({m // block_m} tiles)"
                )
            if panel_size > 0:
                raise ValueError("cluster_m > 1 requires panel_size == 0")
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        # A cluster multicasts one B tile to all cluster_m of its M-tiles.
        b_evict = _b_eviction(m, block_m * cluster_m)

        def _launch():
            if cluster_m > 1:
                return T.ClusterKernel(
                    n // block_n, m // block_m, cluster_dims=(1, cluster_m, 1), threads=threads
                )
            return T.Kernel(n // block_n, m // block_m, threads=threads)

        @T.prim_func
        def _gemm_simple_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with _launch() as (bx, by):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.clear(c_local)
                for ki in T.Pipelined(k // block_k, num_stages=num_stages):
                    T.copy(a[by * block_m, ki * block_k], a_smem)
                    if trans_b:
                        T.copy(b[bx * block_n, ki * block_k], b_smem, eviction_policy=b_evict)
                    else:
                        T.copy(b[ki * block_k, bx * block_n], b_smem, eviction_policy=b_evict)
                    T.gemm(a_smem, b_smem, c_local, transpose_B=trans_b)
                T.copy(c_local, c[by * block_m, bx * block_n])

        return _gemm_simple_main

    return _gemm_simple_func


@functools.lru_cache(maxsize=32)
def _gemm_swap_ab_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Operand-swapped tiny-m NT GEMM: ``C[m,n] = A[m,k] @ B[n,k]ᵀ``, m <= 8.

    Tiling the output the usual way wastes the M dimension at ``m <= 8``: WGMMA
    needs 64 rows, so ``A`` is padded 8-32x and the grid is only
    ``ceil(n / block_n)`` CTAs, too few to fill the device.

    Computing the transpose instead, ``Cᵀ[n,m] = B[n,k] @ A[m,k]ᵀ``, keeps the
    same NT operand form but puts ``n`` on the 64-row WGMMA axis and ``m`` on
    the 8-wide one: no padding waste, and the grid becomes
    ``ceil(n / block_nn)``. The epilogue stages the ``(block_nn, 8)`` tile
    through SMEM and writes ``c[mi, n0 + j]``, contiguous along ``n``.

    Only worth it when that grid fills enough of the device — see
    ``heuristics._swap_ab_stages``, which also sets ``num_stages``: with
    fewer CTAs resident the ring has to be deeper to hide the same latency.

    Args:
        m: Batch rows (2..8).
        n: Output columns (weight rows).
        k: Contraction dim; innermost for both ``a`` and ``b``.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_nn, block_k,
        num_stages)`` returns the compiled ``prim_func``.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_swap_ab_kernel is NT-only (trans_a=False, trans_b=True)")
    if m > SWAP_AB_MPAD:
        raise ValueError(f"_gemm_swap_ab_kernel serves m <= {SWAP_AB_MPAD}, got m={m}")
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_swap_ab_func(block_nn: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        mpad = SWAP_AB_MPAD
        b_evict = "evict_first"  # one N range per CTA, so B is read once

        @T.prim_func
        def _gemm_swap_ab_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_nn), threads=128) as bx:
                b_smem = T.alloc_shared((block_nn, block_k), dtype)
                a_smem = T.alloc_shared((mpad, block_k), dtype)
                ct_local = T.alloc_fragment((block_nn, mpad), accum_dtype)
                ct_cast = T.alloc_fragment((block_nn, mpad), dtype)
                ct_smem = T.alloc_shared((block_nn, mpad), dtype)
                T.clear(ct_local)
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bx * block_nn, ki * block_k], b_smem, eviction_policy=b_evict)
                    T.copy(a[0, ki * block_k], a_smem)
                    T.gemm(b_smem, a_smem, ct_local, transpose_B=True)
                T.copy(ct_local, ct_cast)
                T.copy(ct_cast, ct_smem)
                for mi, j in T.Parallel(mpad, block_nn):
                    if mi < m and bx * block_nn + j < n:
                        c[mi, bx * block_nn + j] = ct_smem[j, mi]

        return _gemm_swap_ab_main

    return _gemm_swap_ab_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2s_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Single-tile 2-consumer (cooperative) GEMM for small NN shapes (SM90).

    ``_gemm_coop2_kernel`` stripped of its persistent loop: the grid *is* the
    tile grid (``n / block_n`` by ``m / block_m``), so a CTA computes exactly
    one output tile and needs no cross-wave ring carry, no tile decode and no
    grouped swizzle. Small square shapes cannot amortize that machinery: the
    mainloop is only ``k / block_k`` iterations. One producer warpgroup issues
    TMA into a ``num_stages`` ring; two consumer warpgroups each own
    ``block_m // 2 = 64`` rows and share the ``B`` tile (split-A / shared-B).

    NN only (``A[m,k] @ B[k,n]``): ``B`` tiles load as ``(block_k, block_n)``
    and feed WGMMA with ``transpose_B=False``. Requires tiles that divide the
    problem exactly; the builder raises ``ValueError`` otherwise. The dense
    selector considers it for exactly tiled small NN calls.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``B`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NN layout).
        trans_b: Must be ``False`` (NN layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages)`` returns the compiled ``prim_func``.
    """
    if trans_a or trans_b:
        raise ValueError("_gemm_coop2s_kernel is NN-only (trans_a=False, trans_b=False)")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2s_func(block_n: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"_gemm_coop2s_kernel requires exact tiling: got "
                f"m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}"
            )
        half_m = block_m // 2
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // 128
        k_iters = k // block_k

        @T.prim_func
        def _gemm_coop2s_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(n // block_n, m // block_m, threads=384) as (bx, by):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_k, block_n), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, block_n), dtype)
                c_smem_1 = T.alloc_shared((half_m, block_n), dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        ks = ki * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        T.tma_copy(
                            a[m_start : m_start + half_m, ks : ks + block_k],
                            a_smem_top[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                            a_smem_bot[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            b[ks : ks + block_k, n_start : n_start + block_n],
                            b_smem[slot, :, :],
                            barrier=ab_full[slot],
                            eviction_policy=b_evict,
                        )
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem_top[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_0,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.copy(c_local_0, c_cast_0)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_cast_0, c_smem_0)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_smem_0, c[m_start, n_start])
                else:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem_bot[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_1,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.copy(c_local_1, c_cast_1)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_cast_1, c_smem_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_smem_1, c[m_start + half_m, n_start])

        return _gemm_coop2s_main

    return _gemm_coop2s_func


def _(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    panel_size: int,
    split_k: int,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((m, n), dtype=inputs[0].dtype, device=inputs[0].device)


class GemmTmaKernel(Kernel):
    """Dense GEMM kernel family: hand-written SM90 implementations.

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout. The
    default structure is warp-specialized: one producer warpgroup issues TMA
    loads into a multi-stage SMEM ring, one consumer warpgroup runs the WGMMA
    over K. Structure flags in ``config`` select the coop2 / coop2s /
    coop2_splitk / pingpong / simple / split-K variants instead (see ``forward``).
    ``activation="silu_and_mul"`` fuses the gated activation into a split-K
    reduction and returns ``[M, N / 2]``.
    fp16 / bf16 inputs, fp32 accumulation. SM90 only: every structure loads
    through TMA and runs its math on WGMMA.
    """

    supported_archs: list[int] = [90]
    general = True

    _STRUCTURE_FLAGS = ("coop2", "coop2s", "coop2_splitk", "pingpong", "simple", "swap_ab")

    def init_config(self, config: Optional[dict] = None, tune: bool = False) -> None:
        """Take a structure-flagged explicit config verbatim.

        The base merge walks ``default_config``'s keys — right for one schema per
        class, wrong here: it drops the caller's flag and grafts their tile values
        onto whichever structure the selector picked. Asking for ``coop2s`` on a
        shape served by ``coop2`` produced a ``coop2`` config at ``coop2s``' tile
        width, a combination ``_enumerate`` deliberately excludes.

        Selector results already use a complete schema, so this merge behavior
        applies only to partial configs supplied by a caller.
        """
        if config is not None and any(config.get(f) for f in self._STRUCTURE_FLAGS):
            self.config = dict(config)
            print(f"{type(self).__name__} initialized with config: {self.config}")
            return
        super().init_config(config, tune)

    @classmethod
    def applies(cls, call) -> bool:
        return (
            _tma_misalignment(call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b)
            is None
        )

    @classmethod
    def refusal(cls, call) -> Optional[str]:
        archs = cls.supported_archs
        if archs is not None and call.arch not in archs:
            return super().refusal(call)
        return _tma_misalignment(call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b)

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        return _dense_entry(cls, call)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        trans_a: bool = False,
        trans_b: bool = False,
        device_index: Optional[int] = None,
        activation: str = "none",
    ) -> None:
        super().__init__(device_index=device_index)
        if activation not in ("none", "silu_and_mul"):
            raise ValueError("activation must be 'none' or 'silu_and_mul'")
        if activation != "none" and n % 2:
            raise ValueError("silu_and_mul requires an even output width")
        misaligned = _tma_misalignment(m, n, k, dtype, trans_a, trans_b)
        if misaligned is not None:
            raise ValueError(f"{type(self).__name__} cannot serve {m}x{n}x{k}: {misaligned}")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.activation = activation
        self.sm_count = get_sm_count(self.device_index)
        self.device_name = torch.cuda.get_device_name(self.device_index)

        self.kernel = _gemm_kernel(
            m, n, k, trans_a, trans_b, self.dtype_str, sm_count=self.sm_count
        )

        self.init_config(config, tune)
        if activation != "none":
            split_k = self.config.get("split_k", 1)
            unsupported = any(
                self.config.get(flag)
                for flag in ("simple", "swap_ab", "coop2", "coop2s", "pingpong")
            )
            if split_k <= 1 or unsupported:
                raise ValueError("a fused activation requires a split-K GEMM config")

    @property
    def default_config(self) -> dict:
        modal = {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "num_stages": 4,
            "panel_size": 16,
            "split_k": 1,
        }
        scored = best_config(
            self.m, self.n, self.k, self.trans_a, self.trans_b, self.sm_count, self.device_name
        )
        return scored if scored is not None else modal

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.config.get("simple"):
            cfg = self.config
            compiled = _gemm_simple_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(
                cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg.get("threads", 128),
                cfg.get("panel_size", 8),
                cfg.get("cluster_m", 1),
            )
            return compiled(a, b)

        if self.config.get("swap_ab"):
            cfg = self.config
            compiled = _gemm_swap_ab_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(cfg["block_nn"], cfg["block_k"], cfg["num_stages"])
            return compiled(a, b)

        if self.config.get("coop2s"):
            cfg = self.config
            compiled = _gemm_coop2s_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(cfg["block_n"], cfg["block_k"], cfg["num_stages"])
            return compiled(a, b)

        if self.config.get("pingpong"):
            cfg = self.config
            compiled = _gemm_pingpong_kernel(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                sm_count=self.sm_count,
            )(
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg["group_size_m"],
                cfg["stage_n"],
                cfg.get("stage_buf", 2),
            )
            return compiled(a, b)

        if self.config.get("coop2"):
            cfg = self.config
            compiled = _gemm_coop2_kernel(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                sm_count=self.sm_count,
            )(
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg["group_size_m"],
                cfg.get("stage_n", 0),
                cfg.get("stage_buf", 0),
            )
            return compiled(a, b)

        if self.config.get("coop2_splitk"):
            cfg = self.config
            mainloop, reduce_ = _splitk_pair(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                True,
                0,
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                0,
                cfg["split_k"],
                self.activation,
            )
            out_n = self.n // 2 if self.activation != "none" else self.n
            c = torch.empty((self.m, out_n), dtype=a.dtype, device=a.device)
            reduce_(mainloop(a, b), c)
            return c

        split_k = self.config.get("split_k", 1)
        if split_k > 1:
            cfg = self.config
            mainloop, reduce_ = _splitk_pair(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                False,
                cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg["panel_size"],
                split_k,
                self.activation,
            )
            out_n = self.n // 2 if self.activation != "none" else self.n
            c = torch.empty((self.m, out_n), dtype=a.dtype, device=a.device)
            reduce_(mainloop(a, b), c)
            return c

        main_cfg = {k2: v for k2, v in self.config.items() if k2 != "split_k"}
        compiled = _gemm_kernel(
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            self.dtype_str,
            traced=trace.enabled,
            sm_count=self.sm_count,
        )(**main_cfg)
        layout = f"{'T' if self.trans_a else 'N'}{'T' if self.trans_b else 'N'}"
        return trace.run(
            compiled, (a, b), stem=f"gemm_{self.m}x{self.n}x{self.k}_{layout}_{self.dtype_str}"
        )


@functools.lru_cache(maxsize=32)
def _gemm_small_batch_kernel(m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    """Bandwidth-bound NT GEMM ``C[m,n] = A[m,k] @ B[n,k]ᵀ`` for small ``m`` (SM90).

    The weight matrix ``B`` is streamed once through a cp.async SMEM ring and
    each B-tile is reused across all ``m`` rows — the regime where arithmetic
    intensity is ``~m`` and saturating HBM, not tensor-core rate, is the goal.
    One ``tvm_thread_allreduce`` over the ``tk`` reduce lanes runs per output
    row.

    Serves every band of ``GemvKernel``: the two one-row bands at ``m = 1`` and the
    ``lhs_rows`` band at its dispatched ``m`` (the matrix-vector case is this kernel
    with a one-row ``A``, not a separate implementation — ``for mi in T.serial(1)``
    folds away).

    The epilogue write is N-tail guarded, which is load-bearing rather than
    defensive: ``gemv_config`` selects ``block_n = 2`` for ``k >= 12288``, and
    without the guard an odd ``n`` writes one element past ``c[.., n-1]``. The K
    tail needs no mask written here — both tail loads are predicated by the
    buffer extents in the emitted code (see the mainloop comment).

    Args:
        m: Batch rows (``1`` for the GEMV case, else the dispatched small ``m``).
        n: Output columns (weight rows).
        k: Contraction dim; the innermost dim of both ``a`` and ``b``.
        dtype: TileLang dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A JIT factory ``(block_n, reduce_threads, num_stages) -> compiled`` whose
        compiled kernel maps ``(a[m,k], b[n,k]) -> c[m,n]``.
    """
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_small_batch_func(
        block_n: int = 1,
        reduce_threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        tile_k = 128 // (str2dtype[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k
        b_evict = "evict_first"  # one N range per CTA, so B is read once

        @T.prim_func
        def _gemm_small_batch_main(
            a: T.Tensor((m, k), dtype),
            b: T.Tensor((n, k), dtype),
            c: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                c_accum = T.alloc_local((m,), accum_dtype)
                T.clear(c_accum)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((m, tile_k), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(
                        b[bn * block_n, bk * block_k],
                        b_shared,
                        disable_tma=True,
                        eviction_policy=b_evict,
                    )
                    for mi in T.serial(m):
                        for _k in T.vectorized(tile_k):
                            a_local[mi, _k] = a[mi, bk * block_k + tk * tile_k + _k]
                    for mi in T.serial(m):
                        for _k in T.serial(tile_k):
                            c_accum[mi] += a_local[mi, _k].astype(accum_dtype) * b_shared[
                                tn, tk * tile_k + _k
                            ].astype(accum_dtype)

                c_reduced = T.alloc_local((1,), accum_dtype)
                for mi in T.serial(m):
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1), c_accum[mi], True, c_reduced[0], tk, dtype="handle"
                            )
                        )
                    if bn * block_n + tn < n:
                        c[mi, bn * block_n + tn] = c_reduced[0]

        return _gemm_small_batch_main

    return _gemm_small_batch_func


def _(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((n,), dtype=inputs[0].dtype, device=inputs[0].device)


_TILE_K = 8
_SMEM_CAP = 224 * 1024


def _bandwidth_autotune_grid(rts: tuple, bns: tuple, nss: tuple) -> list[dict]:
    """Config grid for the bandwidth-mode kernels, guarded by thread and SMEM caps."""
    return [
        {"block_n": bn, "reduce_threads": rt, "num_stages": ns}
        for rt in rts
        for bn in bns
        if rt * bn <= 1024
        for ns in nss
        if bn * (rt * _TILE_K) * 2 * ns <= _SMEM_CAP
    ]


class GemvKernel(Kernel):
    """The bandwidth-bound band of ``GemmFwdOp``: at most two rows contracted over K.

    Three bands build one body, :func:`_gemm_small_batch_kernel`, which reduces over
    K on CUDA cores: the two layouts a vector operand can take, and the ``m == 2`` NT
    calls whose 64-wide n-tiling still underfills a wave. They differ in the region
    they serve and in the config band they pick, which is what :meth:`band_for` and
    :attr:`default_config` state; the body is the same, so they are one class.

    Args:
        band: Which band this instance serves, one of :attr:`BANDS`.
        m: Rows of the product.
        n: Columns of the product.
        k: Contraction dim.
        dtype: Input/output torch dtype (fp16 / bf16); fp32 accumulation.
        config: Optional explicit config; defaults to :attr:`default_config`.
        tune: Whether to autotune over :attr:`autotune_configs`.
        device_index: Device whose SM count and name pick the config.
    """

    supported_archs: list[int] = [90]

    #: The bands this class serves. ``lhs_row`` and ``rhs_col`` name the vector
    #: operand; ``lhs_rows`` is the two-row NT band.
    BANDS: tuple[str, ...] = ("lhs_row", "rhs_col", "lhs_rows")

    @classmethod
    def band_for(cls, call: GemmCall) -> Optional[str]:
        """The band serving *call*, or ``None`` when this class does not serve it.

        Read by :meth:`applies` and by :meth:`entry_for`, so the region is stated once.
        """
        if call.gemv_mode is not None:
            return call.gemv_mode
        if call.trans_a or not call.trans_b or call.m != 2:
            return None
        if not swap_ab_grid_underfills(call.n, call.sm_count):
            return None
        return "lhs_rows"

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return cls.band_for(call) is not None

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        """The cache identity and the thunk that builds this class for *call*.

        The device is in the identity: the ``lhs_rows`` band's config reads its SM count.
        """
        index = call.device.index if call.device is not None else None
        band = cls.band_for(call)
        identity = (band, call.m, call.n, call.k, call.dtype, index)
        return identity, lambda: cls(
            band,
            call.m,
            call.n,
            call.k,
            call.dtype,
            tune=call.tune,
            device_index=index,
        )

    def __init__(
        self,
        band: str,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if band not in self.BANDS:
            raise ValueError(f"{type(self).__name__} serves bands {self.BANDS}, got {band!r}")
        self.band = band
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        # What the body produces: the other operand's free dim, one row per row of ``a``.
        rows, self.out_len = (m, n) if band == "lhs_rows" else (1, n if band == "lhs_row" else m)
        self.kernel = _gemm_small_batch_kernel(rows, self.out_len, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.band == "lhs_rows":
            return small_batch_config(self.n, self.k, get_sm_count(self.device_index))
        return gemv_config(self.k)

    @property
    def autotune_configs(self) -> list[dict]:
        if self.band == "lhs_rows":
            return _bandwidth_autotune_grid((32, 64, 128), (1, 2, 4), (2, 3, 4, 5))
        return _bandwidth_autotune_grid((32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )
        if self.band == "lhs_rows":
            return kernel(a, b)
        vector, matrix = (a, b) if self.band == "lhs_row" else (b, a)
        out = kernel(vector.reshape(1, -1).contiguous(), matrix)
        return out.reshape(1, self.n) if self.band == "lhs_row" else out.reshape(self.m, 1)


@functools.lru_cache(maxsize=32)
def _gemm_basic_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Pipelined dense GEMM ``C = op(A) @ op(B)`` for any tensor-core target.

    The non-warp-specialized counterpart of ``_gemm_kernel``: the same four
    ``(trans_a, trans_b)`` layouts, but with ``T.Pipelined`` software
    pipelining and plain ``T.gemm`` so it compiles on pre-SM90 targets
    (sm80 / sm86 / sm89). SMEM tile shapes follow the storage layout and the
    ``T.gemm`` transpose flags reconcile them with the logical (M, K) x
    (K, N) contraction (cf. the WGMMA version, which forwards them to WGMMA).
    """
    accum_dtype = "float"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_basic_func(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 2,
        threads: int = 128,
        split_k: int = 1,
    ) -> Callable:
        # SMEM tile shapes follow the storage layout; the T.gemm transpose
        # flags reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        n_exact = n % block_n == 0
        k_tiles = T.ceildiv(k, block_k)
        if split_k < 1 or k_tiles % split_k:
            raise ValueError("split_k must divide the K tile count")
        if split_k > 1 and (m % block_m or not n_exact or k % block_k):
            raise ValueError("split-K basic GEMM requires exact M/N/K tiles")
        k_slice = k_tiles // split_k
        output_shape = (m, n) if split_k == 1 else (split_k, m, n)
        output_dtype = dtype if split_k == 1 else accum_dtype

        @T.prim_func
        def _gemm_basic_main(
            a: T.Tensor(a_shape, dtype),  # type: ignore
            b: T.Tensor(b_shape, dtype),  # type: ignore
            c: T.Tensor(output_shape, output_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (
                bx,
                by,
                bz,
            ):
                a_smem = T.alloc_shared(a_tile, dtype)
                b_smem = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                if split_k == 1:
                    c_smem = T.alloc_shared((block_m, block_n), dtype)
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                            c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        }
                    )

                # L2 rasterization: same panel traversal as the BMM kernel.
                T.use_swizzle(10, enable=True)

                T.clear(c_local)
                m_start = by * block_m
                n_start = bx * block_n

                for ki in T.Pipelined(k_slice, num_stages=num_stages):
                    k_start = (bz * k_slice + ki) * block_k
                    # M/N tail reads land in c_local rows/cols that the
                    # epilogue guard skips; a K tail would corrupt live
                    # outputs, so block_k must divide k (enforced by the
                    # config chain / autotune filter).
                    if trans_a:
                        T.copy(
                            a[k_start : k_start + block_k, m_start : m_start + block_m],
                            a_smem,
                        )
                    else:
                        T.copy(
                            a[m_start : m_start + block_m, k_start : k_start + block_k],
                            a_smem,
                        )
                    if trans_b:
                        T.copy(
                            b[n_start : n_start + block_n, k_start : k_start + block_k],
                            b_smem,
                            eviction_policy="evict_first" if split_k > 1 else None,
                        )
                    elif n_exact:
                        T.copy(
                            b[k_start : k_start + block_k, n_start : n_start + block_n],
                            b_smem,
                        )
                    else:
                        # NN with an N tail: the b tile's innermost (N) dim can
                        # be narrower than one vectorised cp_async transfer
                        # (cp_async only accepts 4/8/16-byte accesses — e.g.
                        # an n=1 fp16 GEMV-replacement shape has 2-byte rows),
                        # so mask the copy instead
                        # (cf. _bmm_fp8_kernel's tail path).
                        for i, j in T.Parallel(block_k, block_n):
                            b_smem[i, j] = T.if_then_else(
                                n_start + j < n,
                                b[k_start + i, n_start + j],
                                T.cast(0, dtype),
                            )
                    T.gemm(
                        a_smem,
                        b_smem,
                        c_local,
                        transpose_A=trans_a,
                        transpose_B=trans_b,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                if split_k == 1:
                    T.copy(c_local, c_smem)
                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            c[m_start + i, n_start + j] = c_smem[i, j]
                else:
                    T.copy(
                        c_local,
                        c[
                            bz,
                            m_start : m_start + block_m,
                            n_start : n_start + block_n,
                        ],
                    )

        return _gemm_basic_main

    return _gemm_basic_func


def _(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((m, n), dtype=a.dtype, device=a.device)


class GemmCpAsyncKernel(Kernel):
    """Dense GEMM kernel: pipelined, architecture-agnostic (sm80+).

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout —
    the same contract as ``GemmTmaKernel`` — via ``T.Pipelined`` + plain
    ``T.gemm`` so it runs on pre-SM90 tensor-core targets (sm80 / sm86 /
    sm89). fp16 / bf16 inputs, fp32 accumulation. ``block_k`` must divide
    ``k`` (the smallest fallback is 16); M / N need not be multiples of the
    block sizes (epilogue guard). A config with ``split_k > 1`` returns the
    reduced ``[M, N]`` result and requires exact M / N / K tiles.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    @staticmethod
    def _narrow_k_row(k: int, dtype: torch.dtype) -> Optional[str]:
        """Why a K row is too narrow for one vectorized load, or ``None``.

        ``_gemm_basic_kernel`` loads the innermost (contiguous) dimension in 4-byte
        units, so a row shorter than that is rejected by the backend. K need not be
        16-aligned beyond this — the backend zero-pads K tails.

        Read three times: by ``applies`` and ``refusal``, so an unservable K is
        refused during selection rather than reaching a builder, and by the
        constructor, which is also entered directly.
        """
        if k * dtype.itemsize >= 4:
            return None
        return (
            f"the pipelined mainloop loads its innermost dimension in 4-byte units, so k "
            f"must span at least one; k={k} of a {dtype.itemsize}-byte dtype does not"
        )

    @classmethod
    def applies(cls, call: Any) -> bool:
        """Every architecture, less the SM90 shapes :class:`GemmTmaKernel` supersedes.

        ``GemmTmaKernel`` serves an SM90 call whose operands TMA can address, so this
        class states that one exclusion and keeps the rest of SM90 — the pipelined
        mainloop loads through ``cp.async`` and has no such requirement. Excluding
        all of SM90 instead left a TMA-misaligned shape with no implementation at
        all, though this one runs it.

        Why the exclusion is here rather than in ``supported_archs``: that list also
        gates direct construction, and this class runs on SM90.
        """
        if cls._narrow_k_row(call.k, call.dtype) is not None:
            return False
        if call.arch != 90:
            return True
        return (
            _tma_misalignment(call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b)
            is not None
        )

    @classmethod
    def refusal(cls, call: Any) -> Optional[str]:
        """The narrow-K reason where that is what refuses, else the base answer."""
        archs = cls.supported_archs
        if archs is not None and call.arch in archs:
            narrow = cls._narrow_k_row(call.k, call.dtype)
            if narrow is not None:
                return narrow
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        return _dense_entry(cls, call)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        trans_a: bool = False,
        trans_b: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        narrow = self._narrow_k_row(k, dtype)
        if narrow is not None:
            raise ValueError(f"{type(self).__name__} cannot serve k={k}: {narrow}")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

        self.kernel = _gemm_basic_kernel(m, n, k, trans_a, trans_b, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Modal winner shape of the pipelined BMM kernel across the manifest
        # workloads, measured on one SM90 board.
        # Prefer block_k dividing k; k with no 32/64 factor (or not
        # 16-aligned at all) falls back to 16 — the mma.sync floor — and
        # the backend zero-pads the K tail.
        block_k = 64 if self.k % 64 == 0 else (32 if self.k % 32 == 0 else 16)
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": block_k,
            "num_stages": 2,
            "threads": 128,
            "split_k": 1,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Prefer block_k that divides k; when k has no 32/64 factor (or is
        # not 16-aligned at all), fall back to 16 — the mma.sync floor — and
        # let the backend zero-pad the K tail (any k with k * itemsize >= 4).
        block_k_options = [bk for bk in (64, 32) if self.k % bk == 0]
        if not block_k_options:
            block_k_options = [16]
        return [
            {
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "num_stages": ns,
                "threads": 128,
                "split_k": 1,
            }
            for bm in [64, 128]
            for bn in [64, 128]
            for bk in block_k_options
            for ns in [2, 3, 4]
        ]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Call the compiled JIT directly (cf. BmmKernel); the torch custom-op
        # is retained only for torch.compile compatibility.
        if not hasattr(self, "_compiled_kernel"):
            jit_config = {k: v for k, v in self.config.items() if k != "pass_configs"}
            self._compiled_kernel = self.kernel(**jit_config)
        result = self._compiled_kernel(a, b)
        split_k = self.config.get("split_k", 1)
        if split_k == 1:
            return result
        if not hasattr(self, "_compiled_reduce"):
            self._compiled_reduce = _splitk_reduce_kernel(split_k, self.m, self.n, self.dtype_str)()
        output = a.new_empty((self.m, self.n))
        self._compiled_reduce(result, output)
        return output
