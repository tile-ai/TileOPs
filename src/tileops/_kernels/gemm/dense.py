import dataclasses
import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    TensorCoreIntrinEmitter as WgmmaEmitter,
)

from tileops._csrc import csrc_include
from tileops._kernels.gemm.call_spec import (
    GEMMCall,
    GEMMFP8Call,
    GEMMFP8FwdInterface,
    GEMMFwdInterface,
)
from tileops._kernels.gemm.heuristics import (
    SWAP_AB_MPAD,
    best_config,
    fp8_ws_config,
    gemv_config,
    small_batch_config,
    two_row_gemv_wins,
)
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.tiling import grouped_tile_id
from tileops._utils import STR_TO_DTYPE, get_sm_count, get_sm_version
from tileops.trace import _passes as passes
from tileops.trace import trace

__all__ = [
    "GEMMCpAsyncKernel",
    "GEMMFP8BlockScaleKernel",
    "GEMMFP8TensorScaleKernel",
    "GEMMTMAKernel",
    "GemvKernel",
]


def _tma_misalignment(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool
) -> Optional[str]:
    """Why TMA cannot address these operands, or ``None`` when it can.

    Every structure ``GEMMTMAKernel`` builds loads its tiles through TMA, whose
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


def _dense_entry(cls: type, call: GEMMCall) -> Entry:
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
        trans_a=call.trans_a,
        trans_b=call.trans_b,
        device_index=index,
    )


class _GEMMFP8Kernel(Kernel, GEMMFP8FwdInterface):
    """Shared body of the two FP8 GEMM kernels; ``BLOCK_SCALED`` picks the scale grid.

    Takes :func:`_gemm_fp8_ws_kernel`, or :func:`_gemm_fp8_kernel` on a call
    :meth:`_ws_refusal` rejects.
    """

    supported_archs: list[int] = [89, 90]

    # Whether this kernel reads block128 scale grids rather than per-tensor scalars.
    BLOCK_SCALED = False

    # The 64 rows one consumer warpgroup covers, a tile holding one or two of them; the
    # K tile is fixed by the block128 scale grid. Not tunable.
    _FP8_WS_HALF_M = 64
    _FP8_WS_BLOCK_K = 128

    # One named-barrier id per consumer warpgroup: each group arrives on its own
    # barrier (``arrive_count=128``) instead of a block-wide sync.
    _CONSUMER_BAR_WG0 = 8
    _CONSUMER_BAR_WG1 = 9

    @staticmethod
    def _ws_refusal(m: int, n: int, k: int, dtype: torch.dtype, arch: int) -> Optional[str]:
        """Why the warp-specialized variant cannot serve this call, or ``None``.

        It loads through TMA, and its epilogue releases a ring slot the mainloop
        named, so it needs at least one K-tile. The fallback carries neither.
        """
        if arch != 90:
            return f"TMA and WGMMA are SM90 instructions, the device is sm{arch}"
        if dtype != torch.float8_e4m3fn:
            return f"the warp-specialized FP8 kernel is e4m3-only, got {dtype}"
        if k == 0:
            return "the warp-specialized mainloop has no K-tile to run at k=0"
        return _tma_misalignment(m, n, k, dtype, trans_a=False, trans_b=True)

    @classmethod
    def entry_for(cls, call: GEMMFP8Call) -> Entry:
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
            device_index=index,
            b_scale_rows=128 if cls.block_scale_grid(call) == "1d2d" else 1,
            arch=call.arch,
            sm_count=call.sm_count,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
        b_scale_rows: int = 1,
        arch: Optional[int] = None,
        sm_count: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.b_scale_rows = b_scale_rows
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = get_sm_count(self.device_index) if sm_count is None else sm_count
        arch = get_sm_version(self.device_index) if arch is None else arch
        self.ws_refusal = self._ws_refusal(m, n, k, dtype, arch)
        self.kernel = self._builder()
        self.init_config(config)
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
                b_scale_rows=self.b_scale_rows,
                half_m=self._FP8_WS_HALF_M,
                block_k=self._FP8_WS_BLOCK_K,
                consumer_bar_wg0=self._CONSUMER_BAR_WG0,
                consumer_bar_wg1=self._CONSUMER_BAR_WG1,
            )
        return _gemm_fp8_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            self.BLOCK_SCALED,
            b_scale_rows=self.b_scale_rows,
        )

    def _split_k_launcher(self, has_bias: bool, split_k: int, tile_config: dict) -> Callable:
        """The launch that runs the sliced mainloop and reduces its fp32 workspace into
        the output."""
        mainloop, reduce = _fp8_ws_splitk_pair(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            self.BLOCK_SCALED,
            has_bias,
            split_k,
            tile_config["block_n"],
            tile_config["num_stages"],
            tile_config["group_size_m"],
            self.b_scale_rows,
            half_m=self._FP8_WS_HALF_M,
            block_k=self._FP8_WS_BLOCK_K,
        )
        mainloop, reduce = CompiledProgram(mainloop), CompiledProgram(reduce)
        m, n, out_dtype = self.m, self.n, self.out_dtype

        def split(a, b, scale_a, scale_b, bias):
            slices = torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
            c = torch.empty((m, n), dtype=out_dtype, device=a.device)
            mainloop(a, b, scale_a, scale_b, self._bias_operand(bias, a), slices)
            reduce(slices, c)
            return c

        return split

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
        return self.program(self._launcher, bias is not None)(a, b, scale_a, scale_b, bias)

    def _launcher(self, cfg: dict, has_bias: bool) -> Callable:
        """What one launch under *cfg* runs on ``(a, b, scale_a, scale_b, bias)``, with or
        without a bias epilogue as *has_bias* says."""
        if self.ws_refusal is None:
            tile_config = {key: value for key, value in cfg.items() if key != "split_k"}
            split_k = cfg.get("split_k", 1)
            if split_k > 1:
                return self._split_k_launcher(has_bias, split_k, tile_config)
            builder = self.kernel
            if has_bias:
                builder = _gemm_fp8_ws_kernel(
                    self.m,
                    self.n,
                    self.k,
                    self.dtype_str,
                    self.out_dtype_str,
                    self.BLOCK_SCALED,
                    has_bias=True,
                    sm_count=self.sm_count,
                    b_scale_rows=self.b_scale_rows,
                    half_m=self._FP8_WS_HALF_M,
                    block_k=self._FP8_WS_BLOCK_K,
                    consumer_bar_wg0=self._CONSUMER_BAR_WG0,
                    consumer_bar_wg1=self._CONSUMER_BAR_WG1,
                )
            ws = CompiledProgram(builder(**tile_config))
            return lambda a, b, scale_a, scale_b, bias: ws(
                a, b, scale_a, scale_b, self._bias_operand(bias, a)
            )
        compiled = CompiledProgram(
            _gemm_fp8_kernel(
                self.m,
                self.n,
                self.k,
                self.dtype_str,
                self.out_dtype_str,
                self.BLOCK_SCALED,
                has_bias=has_bias,
                b_scale_rows=self.b_scale_rows,
            )(**cfg)
        )
        if has_bias:
            return compiled
        return lambda a, b, scale_a, scale_b, bias: compiled(a, b, scale_a, scale_b)


class GEMMFP8TensorScaleKernel(_GEMMFP8Kernel):
    """FP8 NT GEMM for per-tensor scales; the two scalars land in the epilogue."""

    BLOCK_SCALED = False

    @classmethod
    def refusal(cls, call: GEMMFP8Call) -> "str | None":
        if call.scale_a_shape != (1, 1) or call.scale_b_shape != (1, 1):
            return "requires one scale per tensor"
        return super().refusal(call)


class GEMMFP8BlockScaleKernel(_GEMMFP8Kernel):
    """FP8 NT GEMM for block128 scale grids; each K-step's partial is scaled and folded in.

    Serves ``scale_b`` per row and per 128x128 block alike, on every shape, bias and
    output dtype the op admits, so it runs wherever no specialised block kernel does.
    """

    BLOCK_SCALED = True
    general = True

    @classmethod
    def refusal(cls, call: GEMMFP8Call) -> "str | None":
        if cls.block_scale_grid(call) is None:
            return (
                f"scales of shape {call.scale_a_shape} and {call.scale_b_shape} form no "
                f"block128 grid"
            )
        return super().refusal(call)


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
    b_scale_rows: int = 1,
    *,
    half_m: int,
    block_k: int,
) -> tuple[Callable, Callable]:
    """The compiled (mainloop, reduce) pair for one split-K configuration.

    Resolved together so the host is not building the second launch while the
    first is already draining.
    """
    mainloop = _gemm_fp8_ws_splitk_kernel(
        m,
        n,
        k,
        dtype,
        out_dtype,
        block_scaled,
        has_bias,
        split_k=split_k,
        b_scale_rows=b_scale_rows,
        half_m=half_m,
        block_k=block_k,
    )(block_n, num_stages, group_size_m)
    return mainloop, splitk_reduce_kernel(split_k, m, n, out_dtype)()


@functools.lru_cache(maxsize=32)
def _gemm_fp8_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool = False,
    b_scale_rows: int = 1,
) -> Callable:
    """Each K tile's product is added into ``c_local`` in fp32, as the block-scaled path does."""
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
        scale_b_shape = (-(-n // b_scale_rows), scale_k) if block_scaled else (1, 1)

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
                partial = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
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
                                scale_b[(n_start + j) // b_scale_rows, scale_idx],
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
                    if block_scaled:
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += partial[i, j] * scale_a_local[i] * scale_b_local[j]
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            c_local[i, j] += partial[i, j]

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
                partial = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
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
                                scale_b[(n_start + j) // b_scale_rows, scale_idx],
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
                    if block_scaled:
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += partial[i, j] * scale_a_local[i] * scale_b_local[j]
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            c_local[i, j] += partial[i, j]

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
    b_scale_rows: int,
    block_m: int,
    half_m: int,
    block_k: int,
):
    """One K-step of the producer: the step's A boxes, its B box and its two scale vectors.

    ``block_m`` is one or two consumers' 64 rows; ``a_bot`` is loaded only for the second.
    ``b_scale_rows`` is how many rows of ``b`` one ``scale_b`` row covers: 1 for a
    per-row grid, 128 for a per-128x128-block grid.
    """
    ks = kb * block_k
    T.tma_copy(
        a[m_start : m_start + half_m, ks : ks + block_k],
        a_top[slot, :, :],
        barrier=ab_full[slot],
    )
    if block_m > half_m:
        T.tma_copy(
            a[m_start + half_m : m_start + block_m, ks : ks + block_k],
            a_bot[slot, :, :],
            barrier=ab_full[slot],
        )
    T.tma_copy(
        b[n_start : n_start + block_n, ks : ks + block_k],
        b_smem[slot, :, :],
        barrier=ab_full[slot],
    )
    if block_scaled:
        for i in T.Parallel(block_m):
            sa_stage[slot, i] = scale_a[T.min(m_start + i, m - 1), kb]
        for j in T.Parallel(block_n):
            sb_stage[slot, j] = scale_b[T.min(n_start + j, n - 1) // b_scale_rows, kb]
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
    half_m: int,
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
    T.copy(sa_stage[slot, row_base : row_base + half_m], sa_f)
    T.copy(sb_stage[slot, :], sb_f)
    T.wait_wgmma(0)
    for i, j in T.Parallel(half_m, block_n):
        acc[i, j] += part[i, j] * sa_f[i] * sb_f[j]
    T.barrier_arrive(ab_empty[slot])


@T.macro
def _fp8_ws_plain_step(
    a_smem, b_smem, ab_full, ab_empty, acc, prev, slot, phase, ki, *, native_n: int = 0
):
    """One K-step of a per-tensor consumer: WGMMA accumulates, the previous slot is released.

    The release trails by one step because step ``ki``'s WGMMA still reads slot
    ``ki``; ``prev`` carries the slot the next release belongs to. A nonzero
    ``native_n`` issues the step as ``native_n``-wide WGMMAs through
    ``dense_wgmma_helper.h`` instead of ``T.wgmma_gemm``'s ``gcd(block_n, 256)``-wide ones.
    """
    T.barrier_wait(ab_full[slot], phase)
    if native_n:
        T.call_extern(
            "handle",
            f"tileops::dense_wgmma_fp8_64x{native_n}_k128",
            acc.data,
            T.address_of(a_smem[slot, 0, 0]),
            T.address_of(b_smem[slot, 0, 0]),
            ki > 0,
        )
    else:
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
def _fp8_ws_plain_drain(
    ab_empty, acc, prev, scale_a, scale_b, *, num_regs: int, block_n: int, half_m: int
):
    """Close a per-tensor consumer's mainloop and apply the two scalars."""
    T.wait_wgmma(0)
    T.barrier_arrive(ab_empty[prev[0]])
    T.warpgroup_fence_operand(acc, num_regs=num_regs)
    for i, j in T.Parallel(half_m, block_n):
        acc[i, j] *= scale_a[0, 0] * scale_b[0, 0]


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
    b_scale_rows: int = 1,
    half_m: int,
    block_k: int,
    consumer_bar_wg0: int,
    consumer_bar_wg1: int,
) -> Callable:
    """Persistent FP8 NT GEMM with split-A/shared-B consumer warpgroups.

    Block128 scaling stages scales with each K tile and rescales partial accumulators;
    per-tensor scaling applies in the epilogue. TMA zero-fills input tails and stores
    mask output tails. ``sm_count`` is part of the build cache key.

    Returns:
        JIT factory taking ``(block_n, num_stages, group_size_m)``."""
    accum_dtype = "float"
    scale_k = (k + 127) // 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            *csrc_include("dense_wgmma_helper.h"),
        ],
    )
    def _gemm_fp8_ws_func(
        block_n: int = 128,
        num_stages: int = 4,
        group_size_m: int = 8,
        block_m: int = 2 * half_m,
    ) -> Callable:
        if block_m not in (half_m, 2 * half_m):
            raise ValueError(f"block_m must be {half_m} or {2 * half_m}, got {block_m}")

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

        # One consumer warpgroup per 64 rows of the tile, beside the producer warpgroup.
        consumers = block_m // half_m
        # T.wgmma_gemm issues ``gcd(block_n, 256)``-wide WGMMAs: a 176-wide tile takes
        # eleven 16-wide ones a K step, measured 2x slower than one 176-wide one.
        # ``dense_wgmma_helper.h`` issues the tile's own width for the widths it
        # instantiates, under per-tensor scales, whose accumulator nothing else reads
        # inside the mainloop.
        native_widths = (176,)
        native_n = block_n if not block_scaled and block_n in native_widths else 0

        def native_acc_layout(buf, first_thread: int):
            """The accumulator layout of one consumer's 64 x ``block_n`` WGMMA, whose
            warpgroup starts at thread ``first_thread``."""
            base = WgmmaEmitter(
                a_dtype=T.float8_e4m3fn,
                b_dtype=T.float8_e4m3fn,
                accum_dtype=T.float32,
                a_transposed=False,
                b_transposed=True,
                block_row_warps=4,
                block_col_warps=1,
                warp_row_tiles=half_m // 4,
                warp_col_tiles=block_n,
                chunk=block_k,
            ).make_mma_store_layout(buf)
            return T.Fragment(
                [half_m, block_n],
                forward_thread_fn=lambda i, j: base.map_forward_thread([i, j])[0] + first_thread,
                forward_index_fn=lambda i, j: base.map_forward_index([i, j]),
            )

        nr = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        grid = min(sm_count, total_tiles)
        max_waves = -(-total_tiles // grid) + 1
        k_iters = -(-k // block_k)
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (-(-n // b_scale_rows), scale_k) if block_scaled else (1, 1)
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
            with T.Kernel(grid, threads=128 * (1 + consumers)) as (pid,):
                a_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                if consumers == 2:
                    a_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                else:
                    # Nothing loads or reads a second consumer's rows when there is none.
                    a_bot = a_top
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_smem_0 = T.alloc_shared((store_rows, block_n), out_dtype)
                c_smem_1 = (
                    T.alloc_shared((store_rows, block_n), out_dtype) if consumers == 2 else c_smem_0
                )
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
                if native_n:
                    # No T.wgmma_gemm infers these; the producer warpgroup holds threads
                    # 0..127, then one consumer warpgroup each.
                    layouts[acc_0] = native_acc_layout(acc_0, 128)
                    layouts[out_0] = native_acc_layout(out_0, 128)
                    layouts[acc_1] = native_acc_layout(acc_1, 256)
                    layouts[out_1] = native_acc_layout(out_1, 256)
                T.annotate_layout(layouts)

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128 * consumers] * num_stages)

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
                            grouped_tile_id(
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
                                    b_scale_rows=b_scale_rows,
                                    block_m=block_m,
                                    half_m=half_m,
                                    block_k=block_k,
                                )
                                gi_prod = gi_prod + 1

                elif tx < 256:
                    T.inc_max_nreg(232)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            grouped_tile_id(
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
                                        half_m=half_m,
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
                                        native_n=native_n,
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
                                    half_m=half_m,
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
                                bar=consumer_bar_wg0,
                                block_n=block_n,
                                n=n,
                                out_dtype=out_dtype,
                                has_bias=has_bias,
                                stage_store=stage_store,
                            )

                elif consumers == 2:
                    T.inc_max_nreg(232)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            grouped_tile_id(
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
                                        half_m=half_m,
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
                                        native_n=native_n,
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
                                    half_m=half_m,
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
                                bar=consumer_bar_wg1,
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
    b_scale_rows: int = 1,
    half_m: int,
    block_k: int,
) -> Callable:
    """Split-K FP8 NT GEMM writing fp32 ``slices[split_k, m, n]``.

    ``split_k`` must divide the block128 K-tile count. Only slice zero adds bias;
    ``splitk_reduce_kernel`` sums and casts the partials.

    Returns:
        JIT factory taking ``(block_n, num_stages, group_size_m)``."""
    accum_dtype = "float"
    block_m = 2 * half_m
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
        nr = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        k_iters = k_iters_total // split_k
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (-(-n // b_scale_rows), scale_k) if block_scaled else (1, 1)
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

                grouped_tile_id(
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
                            b_scale_rows=b_scale_rows,
                            block_m=block_m,
                            half_m=half_m,
                            block_k=block_k,
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
                                half_m=half_m,
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
                            half_m=half_m,
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
                                half_m=half_m,
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
                            half_m=half_m,
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
    """SM90 GEMM with one TMA producer and one WGMMA consumer warpgroup.

    Supports all transpose layouts with fp32 accumulation. Operands must satisfy
    ``_tma_misalignment``. Automatic warp specialization is disabled for this schedule.
    ``traced`` and ``sm_count`` are explicit cache keys; traced builds append a
    ``slots`` output after C.

    Returns:
        JIT factory taking ``(block_m, block_n, block_k, num_stages, panel_size)``."""
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
    """Split-K GEMM writing fp32 ``w[split_k, m, n]`` for ``splitk_reduce_kernel``.

    ``split_k`` must divide the K-tile count.

    Returns:
        JIT factory taking ``(block_m, block_n, block_k, num_stages, panel_size, split_k)``."""
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
def splitk_reduce_kernel(
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


@dataclasses.dataclass(frozen=True)
class _NarrowGrid:
    """Tile counts of a persistent 128-row grid whose last ``narrow_rows`` row tiles start
    with ``narrow_cols`` tiles ``narrow_n`` wide. Tile ids run the full rows, then those
    rows' wide tiles, then the narrow tiles; ``narrow_n`` 0 is the uniform grid."""

    rows: int
    cols: int
    block_n: int
    narrow_n: int
    narrow_rows: int
    narrow_cols: int
    full_rows: int
    tail_cols: int
    full_tiles: int
    wide_tiles: int
    total_tiles: int


def _narrow_grid(
    m: int, n: int, block_n: int, narrow_n: int, narrow_rows: int, narrow_cols: int
) -> _NarrowGrid:
    """The grid of 128 x ``block_n`` tiles over ``m x n`` with the narrow tiles given."""
    rows, cols = -(-m // 128), -(-n // block_n)
    full_rows, tail_cols = rows, 0
    if narrow_n:
        span = narrow_cols * narrow_n
        if not (
            0 < narrow_n < block_n
            and narrow_n % 8 == 0
            and 0 < narrow_rows <= rows
            and 0 < span <= n
        ):
            raise ValueError(
                f"narrow tiles ({narrow_rows} rows, {narrow_cols} x {narrow_n}) do not fit "
                f"a {m} x {n} grid of 128 x {block_n} tiles"
            )
        full_rows, tail_cols = rows - narrow_rows, -(-(n - span) // block_n)
    full_tiles = full_rows * cols
    wide_tiles = full_tiles + narrow_rows * tail_cols
    return _NarrowGrid(
        rows,
        cols,
        block_n,
        narrow_n,
        narrow_rows,
        narrow_cols,
        full_rows,
        tail_cols,
        full_tiles,
        wide_tiles,
        wide_tiles + narrow_rows * narrow_cols,
    )


@T.macro
def _grouped_rows(f, mt, nt, *, rows: int, cols: int, row0: int, group_size_m: int):
    """Grouped-raster tile ``f`` of a ``rows x cols`` block starting at row tile ``row0``."""
    gin = T.int32(group_size_m * cols)
    gid = f // gin
    first_m = gid * T.int32(group_size_m)
    gsize = T.min(T.int32(group_size_m), T.int32(rows) - first_m)
    mt[0] = row0 + first_m + (f % gin) % gsize
    nt[0] = (f % gin) // gsize


@T.macro
def _narrow_grid_tile(flat_id, mt, nt, *, grid: _NarrowGrid, group_size_m: int):
    """Write tile ``flat_id``'s row tile to ``mt[0]``. On a uniform grid ``nt[0]`` gets its
    column tile; with narrow tiles, its first column, and ``nt[1]`` 1 for a narrow tile."""
    g = grid
    if g.narrow_n == 0:
        _grouped_rows(flat_id, mt, nt, rows=g.rows, cols=g.cols, row0=0, group_size_m=group_size_m)
    else:
        # An empty segment gets no branch (its raster would divide by zero).
        if g.wide_tiles == 0:
            _narrow_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m)
        elif flat_id < T.int32(g.wide_tiles):
            if g.full_tiles == 0:
                _tail_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m)
            elif g.wide_tiles == g.full_tiles or flat_id < T.int32(g.full_tiles):
                _grouped_rows(
                    flat_id,
                    mt,
                    nt,
                    rows=g.full_rows,
                    cols=g.cols,
                    row0=0,
                    group_size_m=group_size_m,
                )
                nt[0] = nt[0] * g.block_n
                nt[1] = 0
            else:
                _tail_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m)
        else:
            _narrow_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m)


@T.macro
def _tail_tile(flat_id, mt, nt, *, grid: _NarrowGrid, group_size_m: int):
    g = grid
    _grouped_rows(
        flat_id - T.int32(g.full_tiles),
        mt,
        nt,
        rows=g.narrow_rows,
        cols=g.tail_cols,
        row0=g.full_rows,
        group_size_m=group_size_m,
    )
    nt[0] = g.narrow_cols * g.narrow_n + nt[0] * g.block_n
    nt[1] = 0


@T.macro
def _narrow_tile(flat_id, mt, nt, *, grid: _NarrowGrid, group_size_m: int):
    g = grid
    _grouped_rows(
        flat_id - T.int32(g.wide_tiles),
        mt,
        nt,
        rows=g.narrow_rows,
        cols=g.narrow_cols,
        row0=g.full_rows,
        group_size_m=group_size_m,
    )
    nt[0] = nt[0] * g.narrow_n
    nt[1] = 1


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
    consumer_bar_wg0: int,
    consumer_bar_wg1: int,
) -> Callable:
    """Persistent GEMM with two consumers sharing B and splitting A's rows.

    Ring counters carry across waves to overlap TMA stores with the next tile.
    K tails are TMA zero-filled; output tails use TMA stores where aligned and
    predicated row stores otherwise. Widths 176/192 use ``dense_wgmma_helper.h``
    to keep WGMMA at the full N width. ``sm_count`` is part of the cache key.

    With ``narrow_n`` set, the last ``narrow_rows`` row tiles start with
    ``narrow_cols`` tiles ``narrow_n`` wide that run after every wide tile.

    Every layout loads its SMEM tiles as stored and lets the WGMMA transpose flags
    reconcile them; the helper widths and narrow tiles take K-major operands, so NT only.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages, group_size_m, stage_n,
        narrow_n, narrow_rows, narrow_cols)``."""
    accum_dtype = "float"
    block_m = 128
    # ``dense_wgmma_helper.h``'s name for each dtype.
    wgmma_dtype_tag = {"bfloat16": "bf16", "float16": "fp16"}[dtype]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
    )
    def _gemm_coop2_func(
        block_n: int = 256,
        block_k: int = 64,
        num_stages: int = 3,
        group_size_m: int = 16,
        stage_n: int = 0,
        narrow_n: int = 0,
        narrow_rows: int = 0,
        narrow_cols: int = 0,
    ) -> Callable:
        half_m = block_m // 2
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // 128
        sn = block_n if stage_n <= 0 else stage_n
        # The epilogue stores the tile ``sn`` columns at a time, so a chunk that does not
        # divide the tile would drop its last columns.
        if block_n % sn or sn % 8:
            raise ValueError(f"stage_n must divide block_n and be a multiple of 8, got {sn}")
        n_chunks = block_n // sn
        grid = _narrow_grid(m, n, block_n, narrow_n, narrow_rows, narrow_cols)
        mixed = narrow_n > 0
        # Narrow tiles leave through whole TMA-stored chunks over 64-wide K steps.
        if mixed and not (narrow_n >= sn and n % 8 == 0 and block_k == 64):
            raise ValueError(
                f"narrow_n {narrow_n} needs n % 8 == 0, {sn}-wide chunks at most and "
                f"64-wide K steps"
            )
        if (trans_a or not trans_b) and (block_n in (176, 192) or mixed):
            raise ValueError("176/192-wide and narrow coop2 tiles are NT-only")
        a_tile = (block_k, half_m) if trans_a else (half_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        total_tiles = grid.total_tiles
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)
        # TileLang 0.1.12 issues a ``T.wgmma_gemm`` of a width that does not divide 256
        # as gcd(width, 256)-wide WGMMAs, each re-reading A: 176 as eleven n16, 192 as
        # three n64. ``dense_wgmma_helper.h`` issues these two widths as one N-wide
        # WGMMA per 16 of K, over 64-wide K steps; other K steps keep T.wgmma_gemm.
        wide_wgmma = (block_n in (176, 192) or mixed) and block_k == 64
        wgmma_helper = f"tileops::dense_wgmma_{wgmma_dtype_tag}_64x{block_n}_k64"
        narrow_helper = f"tileops::dense_wgmma_{wgmma_dtype_tag}_64x{narrow_n}_k64"
        # The accumulator layout T.wgmma_gemm gives one consumer's 64 x block_n tile;
        # the helper writes the same registers.
        acc_emitter = WgmmaEmitter(
            a_dtype=dtype,
            b_dtype=dtype,
            accum_dtype="float32",
            a_transposed=False,
            b_transposed=True,
            block_row_warps=4,
            block_col_warps=1,
            warp_row_tiles=half_m // 4,
            warp_col_tiles=block_n,
            chunk=block_k,
        )

        @T.macro
        def consumer_wgmma(a_ring, b_ring, slot, acc, ki, narrow):
            if narrow:
                T.call_extern(
                    "handle",
                    narrow_helper,
                    acc.data,
                    T.address_of(a_ring[slot, 0, 0]),
                    T.address_of(b_ring[slot, 0, 0]),
                    T.if_then_else(ki > 0, 1, 0),
                )
            elif wide_wgmma:
                T.call_extern(
                    "handle",
                    wgmma_helper,
                    acc.data,
                    T.address_of(a_ring[slot, 0, 0]),
                    T.address_of(b_ring[slot, 0, 0]),
                    T.if_then_else(ki > 0, 1, 0),
                )
            else:
                T.wgmma_gemm(
                    a_ring[slot, :, :],
                    b_ring[slot, :, :],
                    acc,
                    transpose_A=trans_a,
                    transpose_B=trans_b,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=(ki == 0),
                )

        # TMA addresses the output only where its 16-bit rows are a whole number of
        # 16-byte units; there it also drops what an M / N tail puts past the output.
        tma_rows = n % 8 == 0

        # A last chunk overlapping the one before waits for its global writes.
        narrow_chunks = -(-narrow_n // sn)
        narrow_overlap = mixed and narrow_n % sn != 0

        @T.macro
        def store_half_narrow(c, c_cast, c_smem, row_start, n_start, bar):
            for i in T.unroll(narrow_chunks):
                c0 = T.min(i * sn, narrow_n - sn)
                if narrow_overlap and i == narrow_chunks - 1:
                    T.tma_store_wait(0, read=False)
                else:
                    T.tma_store_wait(0)
                T.sync_threads(barrier_id=bar, arrive_count=128)
                T.copy(c_cast[:, c0 : c0 + sn], c_smem)
                T.fence_proxy_async()
                T.sync_threads(barrier_id=bar, arrive_count=128)
                T.tma_copy(
                    c_smem,
                    c[row_start : row_start + half_m, n_start + c0 : n_start + c0 + sn],
                )

        @T.macro
        def store_half(c, c_cast, c_smem, row_start, rows, n_start, bar):
            """Store one consumer's ``rows`` (at most ``half_m``) of the tile."""
            if tma_rows:
                for ch in range(n_chunks):
                    c0 = ch * sn
                    # The previous store may still read c_smem; waiting here rather
                    # than after issuing it lets that store overlap the mainloop
                    # that ran since.
                    T.tma_store_wait(0)
                    T.sync_threads(barrier_id=bar, arrive_count=128)
                    T.copy(c_cast[:, c0 : c0 + sn], c_smem)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=bar, arrive_count=128)
                    T.tma_copy(
                        c_smem,
                        c[row_start : row_start + half_m, n_start + c0 : n_start + c0 + sn],
                    )
            else:
                # A warp stores one row at a time, its lanes on consecutive columns. The
                # row loop stays rolled: unrolled, each thread holds every store address
                # live and the consumer spills its accumulator.
                lane = T.get_thread_binding() % 32
                warp = T.get_thread_binding() % 128 // 32
                for ch in range(n_chunks):
                    c0 = ch * sn
                    cols = T.min(T.int32(sn), T.int32(n) - n_start - c0)
                    T.sync_threads(barrier_id=bar, arrive_count=128)
                    T.copy(c_cast[:, c0 : c0 + sn], c_smem)
                    T.sync_threads(barrier_id=bar, arrive_count=128)
                    for r in T.serial(half_m // 4):
                        i = r * 4 + warp
                        if i < rows:
                            for t in T.unroll(-(-sn // 32)):
                                j = t * 32 + lane
                                if j < cols:
                                    c[row_start + i, n_start + c0 + j] = c_smem[i, j]

        @T.macro
        def load_a(a, a_smem, ab_full, slot, row_start, ks):
            if trans_a:
                T.tma_copy(
                    a[ks : ks + block_k, row_start : row_start + half_m],
                    a_smem[slot, :, :],
                    barrier=ab_full[slot],
                )
            else:
                T.tma_copy(
                    a[row_start : row_start + half_m, ks : ks + block_k],
                    a_smem[slot, :, :],
                    barrier=ab_full[slot],
                )

        @T.macro
        def load_b(b, b_smem, ab_full, slot, n_start, ks, width):
            if trans_b:
                T.tma_copy(
                    b[n_start : n_start + width, ks : ks + block_k],
                    b_smem[slot, :, :] if width == block_n else b_smem[slot, 0:width, :],
                    barrier=ab_full[slot],
                    eviction_policy=b_evict,
                )
            else:
                T.tma_copy(
                    b[ks : ks + block_k, n_start : n_start + block_n],
                    b_smem[slot, :, :],
                    barrier=ab_full[slot],
                    eviction_policy=b_evict,
                )

        @T.prim_func
        def _gemm_coop2_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_smem_top = T.alloc_shared((num_stages,) + a_tile, dtype)
                a_smem_bot = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, sn), dtype)
                c_smem_1 = T.alloc_shared((half_m, sn), dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )
                if wide_wgmma:
                    T.annotate_layout(
                        {
                            c_local_0: acc_emitter.make_mma_store_layout(c_local_0),
                            c_local_1: acc_emitter.make_mma_store_layout(c_local_1),
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
                # nt[0]: column tile, or first column under narrow tiles; nt[1]: narrow.
                nt = T.alloc_local((2 if mixed else 1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            _narrow_grid_tile(flat_id, mt, nt, grid=grid, group_size_m=group_size_m)
                            m_start = mt[0] * block_m
                            n_start = nt[0] if mixed else nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                ks = ki * block_k
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                load_a(a, a_smem_top, ab_full, slot, m_start, ks)
                                load_a(a, a_smem_bot, ab_full, slot, m_start + half_m, ks)
                                if mixed and nt[1] == 1:
                                    load_b(b, b_smem, ab_full, slot, n_start, ks, narrow_n)
                                else:
                                    load_b(b, b_smem, ab_full, slot, n_start, ks, block_n)
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            _narrow_grid_tile(flat_id, mt, nt, grid=grid, group_size_m=group_size_m)
                            m_start = mt[0] * block_m
                            n_start = nt[0] if mixed else nt[0] * block_n
                            # One mainloop per width, so a K step tests nothing.
                            if mixed and nt[1] == 1:
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_0 % num_stages
                                    T.barrier_wait(ab_full[slot], (gi_cons_0 // num_stages) & 1)
                                    consumer_wgmma(a_smem_top, b_smem, slot, c_local_0, ki, 1)
                                    if ki > 0:
                                        T.wait_wgmma(1)
                                        T.barrier_arrive(ab_empty[ps0[0]])
                                    ps0[0] = slot
                                    gi_cons_0 = gi_cons_0 + 1
                            else:
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_0 % num_stages
                                    T.barrier_wait(ab_full[slot], (gi_cons_0 // num_stages) & 1)
                                    consumer_wgmma(a_smem_top, b_smem, slot, c_local_0, ki, 0)
                                    if ki > 0:
                                        T.wait_wgmma(1)
                                        T.barrier_arrive(ab_empty[ps0[0]])
                                    ps0[0] = slot
                                    gi_cons_0 = gi_cons_0 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                            T.copy(c_local_0, c_cast_0)
                            if mixed and nt[1] == 1:
                                store_half_narrow(
                                    c, c_cast_0, c_smem_0, m_start, n_start, consumer_bar_wg0
                                )
                            else:
                                store_half(
                                    c,
                                    c_cast_0,
                                    c_smem_0,
                                    m_start,
                                    T.min(T.int32(half_m), T.int32(m) - m_start),
                                    n_start,
                                    consumer_bar_wg0,
                                )
                    # Drain the last store before the CTA's shared memory goes.
                    T.tma_store_wait(0)

                else:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            _narrow_grid_tile(flat_id, mt, nt, grid=grid, group_size_m=group_size_m)
                            m_start = mt[0] * block_m
                            n_start = nt[0] if mixed else nt[0] * block_n
                            # Rows the second consumer holds; none in a tile ending
                            # inside the first consumer's half.
                            arows = T.int32(m) - m_start - T.int32(half_m)
                            # One mainloop per width, so a K step tests nothing.
                            if mixed and nt[1] == 1:
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_1 % num_stages
                                    T.barrier_wait(ab_full[slot], (gi_cons_1 // num_stages) & 1)
                                    consumer_wgmma(a_smem_bot, b_smem, slot, c_local_1, ki, 1)
                                    if ki > 0:
                                        T.wait_wgmma(1)
                                        T.barrier_arrive(ab_empty[ps1[0]])
                                    ps1[0] = slot
                                    gi_cons_1 = gi_cons_1 + 1
                            else:
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_cons_1 % num_stages
                                    T.barrier_wait(ab_full[slot], (gi_cons_1 // num_stages) & 1)
                                    consumer_wgmma(a_smem_bot, b_smem, slot, c_local_1, ki, 0)
                                    if ki > 0:
                                        T.wait_wgmma(1)
                                        T.barrier_arrive(ab_empty[ps1[0]])
                                    ps1[0] = slot
                                    gi_cons_1 = gi_cons_1 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                            T.copy(c_local_1, c_cast_1)
                            if mixed and nt[1] == 1:
                                store_half_narrow(
                                    c,
                                    c_cast_1,
                                    c_smem_1,
                                    m_start + half_m,
                                    n_start,
                                    consumer_bar_wg1,
                                )
                            elif arows > T.int32(0):
                                store_half(
                                    c,
                                    c_cast_1,
                                    c_smem_1,
                                    m_start + half_m,
                                    arows,
                                    n_start,
                                    consumer_bar_wg1,
                                )
                    T.tma_store_wait(0)

        return _gemm_coop2_main

    return _gemm_coop2_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2_swap_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
    consumer_bar_wg0: int,
    consumer_bar_wg1: int,
) -> Callable:
    """Persistent GEMM over 256 x ``block_n`` tiles of ``C.T = op(B).T @ op(A).T``.

    Two consumers each hold 128 rows of the transposed tile, that is 128 columns
    of C, so ``block_n`` sets the tile's extent in M and need not divide 256: each
    16 of K issues one ``block_n``-wide WGMMA per 64 rows through
    ``dense_wgmma_helper.h``. The epilogue stores the tile transposed, 32 rows of
    C at a time, through ``stmatrix.trans`` and TMA, so C's rows must be whole
    16-byte units. ``sm_count`` is part of the cache key.

    With ``narrow_n`` set, ``row_mix = (split, wide_a, narrow_a, wide_b, narrow_b)``
    covers each 256-row block of the first ``split`` with ``wide_a`` tiles
    ``block_n`` wide then ``narrow_a`` tiles ``narrow_n`` wide, and the other blocks
    with ``wide_b`` and ``narrow_b``; every wide tile runs before every narrow one, each
    run of blocks in raster groups of ``group_size_m``. With ``trans_a``, a ``block_n`` that is a multiple of 64 stages
    ``op(A).T`` in 64-column swizzle atoms, so ``narrow_n`` must be a multiple of 64.

    Returns:
        JIT factory taking ``(block_n, num_stages, group_size_m, narrow_n, row_mix)``."""
    if n % 8:
        raise ValueError(f"coop2_swap stores C through TMA and needs n % 8 == 0, got n={n}")
    accum_dtype = "float"
    block_k, store_m = 64, 32
    a_mn, b_mn = ("true" if mn else "false" for mn in (not trans_b, trans_a))
    wgmma_dtype = "kBFloat16" if dtype == "bfloat16" else "kFloat16"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
    )
    def _gemm_coop2_swap_func(
        block_n: int = 160,
        num_stages: int = 4,
        group_size_m: int = 1,
        narrow_n: int = 0,
        row_mix: tuple = (),
    ) -> Callable:
        if block_n % store_m or narrow_n % store_m or narrow_n >= block_n:
            raise ValueError(
                f"block_n and narrow_n must be multiples of {store_m} with narrow_n < block_n,"
                f" got {block_n} and {narrow_n}"
            )
        if trans_a and block_n % 64 == 0 and narrow_n % 64:
            raise ValueError(
                f"with trans_a, a block_n of {block_n} needs narrow_n a multiple of 64,"
                f" got {narrow_n}"
            )
        if bool(narrow_n) != bool(row_mix) or len(row_mix) not in (0, 5):
            raise ValueError(
                f"narrow_n needs a five-entry row_mix and row_mix needs narrow_n,"
                f" got narrow_n={narrow_n}, row_mix={row_mix}"
            )
        rows, cols = -(-n // 256), -(-m // block_n)
        if narrow_n:
            split, wide_a, narrow_a, wide_b, narrow_b = row_mix
            if not (
                0 <= split <= rows
                and (split == 0 or wide_a * block_n + narrow_a * narrow_n >= m)
                and (split == rows or wide_b * block_n + narrow_b * narrow_n >= m)
            ):
                raise ValueError(f"row_mix {row_mix} does not cover {m} columns of C.T")
            # (first id, first block, blocks, tiles per block, first column, width, narrow)
            # per run.
            runs = [
                (split, 0, wide_a, 0, block_n, 0),
                (rows - split, split, wide_b, 0, block_n, 0),
                (split, 0, narrow_a, wide_a * block_n, narrow_n, 1),
                (rows - split, split, narrow_b, wide_b * block_n, narrow_n, 1),
            ]
            segs, total_tiles = [], 0
            for blocks, row0, per_row, col0, width, narrow in runs:
                if blocks and per_row:
                    segs.append((total_tiles, row0, blocks, per_row, col0, width, narrow))
                    total_tiles += blocks * per_row
            segs = tuple(segs)
        else:
            total_tiles = rows * cols
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)
        wgmma = f"tileops::dense_wgmma_128xN_k64<tl::DataType::{wgmma_dtype}, {block_n}, {a_mn}, {b_mn}>"
        narrow_wgmma = (
            f"tileops::dense_wgmma_128xN_k64<tl::DataType::{wgmma_dtype}, {narrow_n}, {a_mn},"
            f" {b_mn}, {block_n}>"
        )
        store = f"tileops::dense_store_trans_64<{store_m}>"
        a_half = (128, block_k) if trans_b else (block_k, 128)
        b_tile = (block_k, block_n) if trans_a else (block_n, block_k)
        acc_emitter = WgmmaEmitter(
            a_dtype=dtype,
            b_dtype=dtype,
            accum_dtype="float32",
            a_transposed=False,
            b_transposed=True,
            block_row_warps=4,
            block_col_warps=1,
            warp_row_tiles=16,
            warp_col_tiles=block_n,
            chunk=block_k,
        )

        @T.macro
        def run_tile(flat_id, mt, nt, run):
            first, row0, blocks, per_row, col0, width, narrow = run
            _grouped_rows(
                flat_id - first,
                mt,
                nt,
                rows=blocks,
                cols=per_row,
                row0=row0,
                group_size_m=group_size_m,
            )
            nt[0] = col0 + nt[0] * width
            nt[1] = narrow

        @T.macro
        def mixed_tile(flat_id, mt, nt, runs):
            if len(runs) == 1 or flat_id < runs[1][0]:
                run_tile(flat_id, mt, nt, runs[0])
            else:
                mixed_tile(flat_id, mt, nt, runs[1:])

        @T.macro
        def tile_at(flat_id, mt, nt):
            """Tile ``flat_id``'s 256-row block to ``mt[0]``, its first column of C.T to
            ``nt[0]`` and, with narrow tiles, 1 for a narrow tile to ``nt[1]``."""
            if narrow_n:
                mixed_tile(flat_id, mt, nt, segs)
            else:
                _grouped_rows(
                    flat_id, mt, nt, rows=rows, cols=cols, row0=0, group_size_m=group_size_m
                )
                nt[0] = nt[0] * block_n

        @T.macro
        def load_a(b, a_smem, full, slot, row_start, ks):
            if not trans_b:
                T.tma_copy(
                    b[ks : ks + block_k, row_start : row_start + 128],
                    a_smem[slot, :, :],
                    barrier=full[slot],
                )
            else:
                T.tma_copy(
                    b[row_start : row_start + 128, ks : ks + block_k],
                    a_smem[slot, :, :],
                    barrier=full[slot],
                )

        @T.macro
        def load_b(a, b_smem, full, slot, col_start, ks, width):
            if trans_a:
                T.tma_copy(
                    a[ks : ks + block_k, col_start : col_start + width],
                    b_smem[slot, :, 0:width],
                    barrier=full[slot],
                )
            else:
                T.tma_copy(
                    a[col_start : col_start + width, ks : ks + block_k],
                    b_smem[slot, 0:width, :],
                    barrier=full[slot],
                )

        @T.macro
        def mainloop(a_smem, b_smem, full, empty, lo, hi, gi, ps, helper, width):
            for ki in T.Pipelined(k_iters, num_stages=0):
                slot = gi[0] % num_stages
                T.barrier_wait(full[slot], (gi[0] // num_stages) & 1)
                T.call_extern(  # lint: extern-target allow: wgmma or narrow_wgmma, by tile width
                    "handle",
                    helper,
                    lo.data,
                    hi.data,
                    T.address_of(a_smem[slot, 0, 0]),
                    T.address_of(b_smem[slot, 0, 0]),
                    T.if_then_else(ki > 0, 1, 0),
                )
                if ki > 0:
                    T.wait_wgmma(1)
                    T.barrier_arrive(empty[ps[0]])
                ps[0] = slot
                gi[0] = gi[0] + 1
            T.wait_wgmma(0)
            T.barrier_arrive(empty[ps[0]])
            T.warpgroup_fence_operand(lo, num_regs=width // 2)
            T.warpgroup_fence_operand(hi, num_regs=width // 2)

        @T.macro
        def store_chunk(c, lo, hi, c_smem, c0, row_start, ch, bar):
            """Columns ``[ch * store_m, (ch + 1) * store_m)`` of both halves, stored as rows
            ``c0 ...`` of C."""
            T.tma_store_wait(0)
            T.sync_threads(barrier_id=bar, arrive_count=128)
            T.call_extern(
                "handle", store, lo.data, T.address_of(c_smem[0, 0, 0]), ch * store_m // 8
            )
            T.call_extern(
                "handle", store, hi.data, T.address_of(c_smem[1, 0, 0]), ch * store_m // 8
            )
            T.fence_proxy_async()
            T.sync_threads(barrier_id=bar, arrive_count=128)
            for q in T.unroll(2):
                T.tma_copy(
                    c_smem[q, :, :],
                    c[c0 : c0 + store_m, row_start + q * 64 : row_start + q * 64 + 64],
                )

        @T.macro
        def consumer(
            c, a_smem, b_smem, full, empty, lo, hi, c_smem, gi, ps, pid, mt, nt, half, bar
        ):
            """One consumer's 128 rows of every tile it owns, mainloop then transposed store."""
            for w in T.serial(max_waves):
                flat_id = T.int32(sm_count) * w + pid
                if flat_id < total_tiles:
                    tile_at(flat_id, mt, nt)
                    row_start = mt[0] * 256 + half * 128
                    col_start = nt[0]
                    if narrow_n and nt[1] == 1:
                        mainloop(
                            a_smem, b_smem, full, empty, lo, hi, gi, ps, narrow_wgmma, narrow_n
                        )
                    else:
                        mainloop(a_smem, b_smem, full, empty, lo, hi, gi, ps, wgmma, block_n)
                    for ch in T.unroll(block_n // store_m):
                        if ch < narrow_n // store_m or not narrow_n or nt[1] == 0:
                            c0 = col_start + ch * store_m
                            store_chunk(c, lo, hi, c_smem, c0, row_start, ch, bar)
            T.tma_store_wait(0)

        @T.prim_func
        def _gemm_coop2_swap_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_smem_top = T.alloc_shared((num_stages,) + a_half, dtype)
                a_smem_bot = T.alloc_shared((num_stages,) + a_half, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                lo_0 = T.alloc_fragment((64, block_n), accum_dtype)
                hi_0 = T.alloc_fragment((64, block_n), accum_dtype)
                lo_1 = T.alloc_fragment((64, block_n), accum_dtype)
                hi_1 = T.alloc_fragment((64, block_n), accum_dtype)
                c_smem_0 = T.alloc_shared((2, store_m, 64), dtype)
                c_smem_1 = T.alloc_shared((2, store_m, 64), dtype)
                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                        **{
                            f: acc_emitter.make_mma_store_layout(f)
                            for f in (lo_0, hi_0, lo_1, hi_1)
                        },
                    }
                )
                full = T.alloc_barrier([128] * num_stages)
                empty = T.alloc_barrier([256] * num_stages)
                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_local((1,), "int32")
                gi_cons_1 = T.alloc_local((1,), "int32")
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((2 if narrow_n else 1,), "int32")

                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            tile_at(flat_id, mt, nt)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                ks = ki * block_k
                                T.barrier_wait(empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                load_a(b, a_smem_top, full, slot, mt[0] * 256, ks)
                                load_a(b, a_smem_bot, full, slot, mt[0] * 256 + 128, ks)
                                if narrow_n and nt[1] == 1:
                                    load_b(a, b_smem, full, slot, nt[0], ks, narrow_n)
                                else:
                                    load_b(a, b_smem, full, slot, nt[0], ks, block_n)
                                T.barrier_arrive(full[slot])
                                gi_prod = gi_prod + 1
                elif tx < 256:
                    T.inc_max_nreg(240)
                    gi_cons_0[0] = 0
                    consumer(
                        c,
                        a_smem_top,
                        b_smem,
                        full,
                        empty,
                        lo_0,
                        hi_0,
                        c_smem_0,
                        gi_cons_0,
                        ps0,
                        pid,
                        mt,
                        nt,
                        0,
                        consumer_bar_wg0,
                    )
                else:
                    T.inc_max_nreg(240)
                    gi_cons_1[0] = 0
                    consumer(
                        c,
                        a_smem_bot,
                        b_smem,
                        full,
                        empty,
                        lo_1,
                        hi_1,
                        c_smem_1,
                        gi_cons_1,
                        ps1,
                        pid,
                        mt,
                        nt,
                        1,
                        consumer_bar_wg1,
                    )

        return _gemm_coop2_swap_main

    return _gemm_coop2_swap_func


@functools.lru_cache(maxsize=32)
def _gemm_pingpong_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
    consumer_bar_wg0: int,
    consumer_bar_wg1: int,
) -> Callable:
    """Persistent NT GEMM with two consumers alternating whole 128-row tiles.

    Turn barriers overlap one consumer's epilogue with the other's mainloop.
    TMA stores require N to be a multiple of eight; K tails are zero-filled.
    Widths 176/192 use ``dense_wgmma_helper.h`` for full-width WGMMA.

    With ``narrow_n`` set, the last ``narrow_rows`` row tiles start with
    ``narrow_cols`` tiles ``narrow_n`` wide that run after every wide tile.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages, group_size_m, stage_n,
        epi_bufs, narrow_n, narrow_rows, narrow_cols)``."""
    if trans_a or not trans_b:
        raise ValueError("_gemm_pingpong_kernel is NT-only (trans_a=False, trans_b=True)")
    # Every tile leaves through the TMA store, which addresses 16-bit output rows only
    # in whole 16-byte units.
    if n % 8:
        raise ValueError(f"_gemm_pingpong_kernel needs n to be a multiple of 8, got {n}")
    accum_dtype = "float"
    block_m = 128
    half_m = 64
    # ``dense_wgmma_helper.h``'s name for each dtype.
    wgmma_dtype_tag = {"bfloat16": "bf16", "float16": "fp16"}[dtype]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            *csrc_include("dense_wgmma_helper.h"),
        ],
    )
    def _gemm_pingpong_func(
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 4,
        group_size_m: int = 16,
        stage_n: int = 0,
        epi_bufs: int = 1,
        narrow_n: int = 0,
        narrow_rows: int = 0,
        narrow_cols: int = 0,
    ) -> Callable:
        # A consumer holds 128 x block_n f32 in 240 registers a thread; past 192
        # columns the accumulators spill.
        if block_n > 192:
            raise ValueError(f"block_n must be at most 192, got {block_n}")
        # The epilogue casts and stages the tile ``sn`` columns at a time, straight from
        # the accumulators; a TMA store box row is a multiple of 16 bytes.
        sn = block_n if stage_n <= 0 else stage_n
        if block_n % sn or sn % 8:
            raise ValueError(f"stage_n must divide block_n and be a multiple of 8, got {sn}")
        n_chunks = block_n // sn
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // 128
        grid = _narrow_grid(m, n, block_n, narrow_n, narrow_rows, narrow_cols)
        mixed = narrow_n > 0
        if mixed and narrow_n % sn:
            raise ValueError(f"narrow_n {narrow_n} must be whole {sn}-wide epilogue chunks")
        total_tiles = grid.total_tiles
        narrow_chunks = narrow_n // sn
        max_waves = -(-total_tiles // sm_count)
        # Each consumer takes every other wave.
        consumer_waves = -(-max_waves // 2)
        k_iters = -(-k // block_k)
        wide_wgmma = block_n in (176, 192) and block_k == 64
        wgmma_helper = f"tileops::dense_wgmma_{wgmma_dtype_tag}_64x{block_n}_k64"
        if mixed and not wide_wgmma:
            raise ValueError("narrow tiles run on dense_wgmma_helper.h's 64-wide K steps")
        narrow_helper = f"tileops::dense_wgmma_{wgmma_dtype_tag}_64x{narrow_n}_k64"
        # The accumulator layout T.wgmma_gemm gives a 64 x block_n tile over one
        # warpgroup; the helper writes the same registers.
        acc_emitter = WgmmaEmitter(
            a_dtype=dtype,
            b_dtype=dtype,
            accum_dtype="float32",
            a_transposed=False,
            b_transposed=True,
            block_row_warps=4,
            block_col_warps=1,
            warp_row_tiles=half_m // 4,
            warp_col_tiles=block_n,
            chunk=block_k,
        )

        @T.macro
        def half_wgmma(a_ring, b_ring, slot, acc, ki, narrow):
            if narrow:
                # A narrow WGMMA writes a prefix of the wide one's accumulator registers.
                T.call_extern(
                    "handle",
                    narrow_helper,
                    acc.data,
                    T.address_of(a_ring[slot, 0, 0]),
                    T.address_of(b_ring[slot, 0, 0]),
                    T.if_then_else(ki > 0, 1, 0),
                )
            elif wide_wgmma:
                T.call_extern(
                    "handle",
                    wgmma_helper,
                    acc.data,
                    T.address_of(a_ring[slot, 0, 0]),
                    T.address_of(b_ring[slot, 0, 0]),
                    T.if_then_else(ki > 0, 1, 0),
                )
            else:
                T.wgmma_gemm(
                    a_ring[slot, :, :],
                    b_ring[slot, :, :],
                    acc,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=(ki == 0),
                )

        @T.macro
        def load_b(b, b_smem, ab_full, slot, n_start, ks, width):
            T.tma_copy(
                b[n_start : n_start + width, ks : ks + block_k],
                b_smem[slot, :, :] if width == block_n else b_smem[slot, 0:width, :],
                barrier=ab_full[slot],
                eviction_policy=b_evict,
            )

        @T.macro
        def advance(pos):
            # The next ring position (slot, phase), without dividing by num_stages.
            pos[0] = pos[0] + 1
            if pos[0] == num_stages:
                pos[0] = 0
                pos[1] = pos[1] ^ 1

        @T.macro
        def mainloop(a_top, a_bot, b_smem, ab_full, ab_empty, acc_top, acc_bot, prev, pos, narrow):
            for ki in T.Pipelined(k_iters, num_stages=0):
                slot = pos[0]
                T.barrier_wait(ab_full[slot], pos[1])
                half_wgmma(a_top, b_smem, slot, acc_top, ki, narrow)
                half_wgmma(a_bot, b_smem, slot, acc_bot, ki, narrow)
                if ki > 0:
                    T.wait_wgmma(2)
                    T.barrier_arrive(ab_empty[prev[0]])
                prev[0] = slot
                advance(pos)

        @T.macro
        def store_chunk(acc_top, acc_bot, c_smem, c, m_start, n_start, ch, epilogue_bar):
            c0 = ch * sn
            # A buffer is free with epi_bufs - 1 stores in flight; the first chunk waits
            # for all, since the previous tile's stores may hold any buffer.
            if ch == 0:
                T.tma_store_wait(0)
            else:
                T.tma_store_wait(epi_bufs - 1)
            T.sync_threads(barrier_id=epilogue_bar, arrive_count=128)
            T.copy(acc_top[:, c0 : c0 + sn], c_smem[ch % epi_bufs, 0:half_m, :])
            T.copy(acc_bot[:, c0 : c0 + sn], c_smem[ch % epi_bufs, half_m:block_m, :])
            T.fence_proxy_async()
            T.sync_threads(barrier_id=epilogue_bar, arrive_count=128)
            T.tma_copy(
                c_smem[ch % epi_bufs, :, :],
                c[m_start : m_start + block_m, n_start + c0 : n_start + c0 + sn],
            )

        @T.macro
        def consumer(
            wg,
            pid,
            c,
            a_top,
            a_bot,
            b_smem,
            ab_full,
            ab_empty,
            turn,
            acc_top,
            acc_bot,
            c_smem,
            mt,
            nt,
            prev,
            pos,
            epilogue_bar,
        ):
            for j in T.serial(consumer_waves):
                w = 2 * j + wg
                flat_id = T.int32(sm_count) * w + pid
                if flat_id < total_tiles:
                    _narrow_grid_tile(flat_id, mt, nt, grid=grid, group_size_m=group_size_m)
                    m_start = mt[0] * block_m
                    n_start = nt[0] if mixed else nt[0] * block_n
                    # This consumer's j-th turn on the tensor cores.
                    T.barrier_wait(turn[wg], j & 1)
                    pos[0] = (w * k_iters) % num_stages
                    pos[1] = ((w * k_iters) // num_stages) & 1
                    # One mainloop per width, so a K step tests nothing.
                    if mixed and nt[1] == 1:
                        mainloop(
                            a_top, a_bot, b_smem, ab_full, ab_empty, acc_top, acc_bot, prev, pos, 1
                        )
                    else:
                        mainloop(
                            a_top, a_bot, b_smem, ab_full, ab_empty, acc_top, acc_bot, prev, pos, 0
                        )
                    # The other consumer's next tile may start its WGMMAs; they queue
                    # behind this tile's last.
                    if flat_id + T.int32(sm_count) < total_tiles:
                        T.barrier_arrive(turn[1 - wg])
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[prev[0]])
                    T.warpgroup_fence_operand(acc_top, num_regs=nr)
                    T.warpgroup_fence_operand(acc_bot, num_regs=nr)
                    for ch in T.unroll(n_chunks):
                        if mixed and ch >= narrow_chunks:
                            # A narrow tile ends after its narrow_chunks chunks.
                            if nt[1] == 0:
                                store_chunk(
                                    acc_top, acc_bot, c_smem, c, m_start, n_start, ch, epilogue_bar
                                )
                        else:
                            store_chunk(
                                acc_top, acc_bot, c_smem, c, m_start, n_start, ch, epilogue_bar
                            )
            # Drain the last store before the CTA's shared memory goes.
            T.tma_store_wait(0)

        @T.prim_func
        def _gemm_pingpong_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                acc_top_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_bot_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_top_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_bot_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_smem_0 = T.alloc_shared((epi_bufs, block_m, sn), dtype)
                c_smem_1 = T.alloc_shared((epi_bufs, block_m, sn), dtype)
                T.annotate_layout(
                    {
                        a_top: tilelang.layout.make_swizzled_layout(a_top),
                        a_bot: tilelang.layout.make_swizzled_layout(a_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )
                if wide_wgmma:
                    T.annotate_layout(
                        {
                            acc_top_0: acc_emitter.make_mma_store_layout(acc_top_0),
                            acc_bot_0: acc_emitter.make_mma_store_layout(acc_bot_0),
                            acc_top_1: acc_emitter.make_mma_store_layout(acc_top_1),
                            acc_bot_1: acc_emitter.make_mma_store_layout(acc_bot_1),
                        }
                    )
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                # One per consumer: completes a phase when the other consumer hands it
                # the tensor cores.
                turn = T.alloc_barrier([128, 128])
                # Ring positions (slot, phase): the producer's, and each consumer's.
                rp = T.alloc_local((2,), "int32")
                pos_0 = T.alloc_local((2,), "int32")
                pos_1 = T.alloc_local((2,), "int32")
                rp[0] = 0
                rp[1] = 0
                prev_0 = T.alloc_local((1,), "int32")
                prev_1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                # nt[0]: column tile, or first column under narrow tiles; nt[1]: narrow.
                nt = T.alloc_local((2 if mixed else 1,), "int32")
                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            _narrow_grid_tile(flat_id, mt, nt, grid=grid, group_size_m=group_size_m)
                            m_start = mt[0] * block_m
                            n_start = nt[0] if mixed else nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = rp[0]
                                ks = ki * block_k
                                T.barrier_wait(ab_empty[slot], rp[1] ^ 1)
                                T.tma_copy(
                                    a[m_start : m_start + half_m, ks : ks + block_k],
                                    a_top[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.tma_copy(
                                    a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                                    a_bot[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                if mixed and nt[1] == 1:
                                    load_b(b, b_smem, ab_full, slot, n_start, ks, narrow_n)
                                else:
                                    load_b(b, b_smem, ab_full, slot, n_start, ks, block_n)
                                T.barrier_arrive(ab_full[slot])
                                advance(rp)
                elif tx < 256:
                    T.inc_max_nreg(240)
                    consumer(
                        0,
                        pid,
                        c,
                        a_top,
                        a_bot,
                        b_smem,
                        ab_full,
                        ab_empty,
                        turn,
                        acc_top_0,
                        acc_bot_0,
                        c_smem_0,
                        mt,
                        nt,
                        prev_0,
                        pos_0,
                        consumer_bar_wg0,
                    )
                else:
                    T.inc_max_nreg(240)
                    # The first consumer's first tile goes first.
                    if pid < total_tiles:
                        T.barrier_arrive(turn[0])
                    consumer(
                        1,
                        pid,
                        c,
                        a_top,
                        a_bot,
                        b_smem,
                        ab_full,
                        ab_empty,
                        turn,
                        acc_top_1,
                        acc_bot_1,
                        c_smem_1,
                        mt,
                        nt,
                        prev_1,
                        pos_1,
                        consumer_bar_wg1,
                    )

        return _gemm_pingpong_main

    return _gemm_pingpong_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2_splitk_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Split-K NT GEMM with two 64-row consumers sharing B.

    Writes fp32 ``w[split_k, m, n]`` for ``splitk_reduce_kernel``.
    ``split_k`` must divide the K-tile count.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages, split_k)``."""
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
    host spends between the two launches is GPU idle that the span metric
    counts: on short-mainloop shapes the mainloop drains before the reduce is
    enqueued. Folding the builder lookup and the ``@tilelang.jit``
    factory call of *both* kernels into one cached resolution keeps that
    window to the two launches themselves.

    Allocating ``C`` would also fall in that window, so
    ``splitk_reduce_kernel`` takes it as an explicit parameter and both
    callers allocate it *before* launching the mainloop.
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
    return mainloop, splitk_reduce_kernel(split_k, m, n, dtype, activation)(elems_per_cta)


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
                c_smem = T.alloc_shared((block_m, block_n), dtype)
                T.clear(c_local)
                for ki in T.Pipelined(k // block_k, num_stages=num_stages):
                    T.copy(a[by * block_m, ki * block_k], a_smem)
                    if trans_b:
                        T.copy(b[bx * block_n, ki * block_k], b_smem, eviction_policy=b_evict)
                    else:
                        T.copy(b[ki * block_k, bx * block_n], b_smem, eviction_policy=b_evict)
                    T.gemm(a_smem, b_smem, c_local, transpose_B=trans_b)
                # One TMA store from shared memory, not per-thread global stores.
                T.copy(c_local, c_smem)
                T.copy(c_smem, c[by * block_m, bx * block_n])

        return _gemm_simple_main

    return _gemm_simple_func


@T.macro
def swap_ab_mainloop(a, b, ct, a_smem, b_smem, bx, k, block_nn, block_k, num_stages):
    """Accumulate one CTA's transposed tile ``ct = b[bx * block_nn:, :] @ a^T`` over K.

    Each CTA reads its rows of ``b`` once, so they are evicted first; ``a`` is what L2 keeps.
    """
    for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
        T.copy(b[bx * block_nn, ki * block_k], b_smem, eviction_policy="evict_first")
        T.copy(a[0, ki * block_k], a_smem)
        T.gemm(b_smem, a_smem, ct, transpose_B=True)


@T.macro
def swap_ab_store(ct_cast, ct_smem, c, bx, m, n, block_nn, mpad):
    """Write a cast transposed tile back as ``c[mi, bx * block_nn + j]``, through shared memory."""
    T.copy(ct_cast, ct_smem)
    for mi, j in T.Parallel(mpad, block_nn):
        if mi < m and bx * block_nn + j < n:
            c[mi, bx * block_nn + j] = ct_smem[j, mi]


@functools.lru_cache(maxsize=32)
def _gemm_swap_ab_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Tiny-M NT GEMM computing ``C.T = B @ A.T`` to avoid padding M to 64.

    The epilogue transposes the shared tile into contiguous output columns.
    ``_tiny_m_score_us`` ranks its stage counts against the 64-row split-K tile.

    Returns:
        JIT factory taking ``(block_nn, block_k, num_stages)``."""
    if trans_a or not trans_b:
        raise ValueError("_gemm_swap_ab_kernel is NT-only (trans_a=False, trans_b=True)")
    if m > SWAP_AB_MPAD:
        raise ValueError(f"_gemm_swap_ab_kernel serves m <= {SWAP_AB_MPAD}, got m={m}")
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_swap_ab_func(block_nn: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        mpad = SWAP_AB_MPAD

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
                swap_ab_mainloop(
                    a, b, ct_local, a_smem, b_smem, bx, k, block_nn, block_k, num_stages
                )
                T.copy(ct_local, ct_cast)
                swap_ab_store(ct_cast, ct_smem, c, bx, m, n, block_nn, mpad)

        return _gemm_swap_ab_main

    return _gemm_swap_ab_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2s_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    consumer_bar_wg0: int,
    consumer_bar_wg1: int,
) -> Callable:
    """Single-tile NN GEMM with two consumers splitting A and sharing B.

    Tiles must divide the problem exactly. Each CTA computes one tile without
    persistent scheduling or ring carry between tiles.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages)``."""
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
                    T.sync_threads(barrier_id=consumer_bar_wg0, arrive_count=128)
                    T.copy(c_cast_0, c_smem_0)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=consumer_bar_wg0, arrive_count=128)
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
                    T.sync_threads(barrier_id=consumer_bar_wg1, arrive_count=128)
                    T.copy(c_cast_1, c_smem_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=consumer_bar_wg1, arrive_count=128)
                    T.copy(c_smem_1, c[m_start + half_m, n_start])

        return _gemm_coop2s_main

    return _gemm_coop2s_func


class GEMMTMAKernel(Kernel, GEMMFwdInterface):
    """Dense GEMM kernel family: hand-written SM90 implementations.

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout. The
    default structure is warp-specialized: one producer warpgroup issues TMA
    loads into a multi-stage SMEM ring, one consumer warpgroup runs the WGMMA
    over K. Structure flags in ``config`` select the coop2 / coop2s /
    coop2_splitk / coop2_swap / pingpong / simple / split-K variants instead (see
    ``forward``).
    ``activation="silu_and_mul"`` fuses the gated activation into a split-K
    reduction and returns ``[M, N / 2]``.
    fp16 / bf16 inputs, fp32 accumulation. SM90 only: every structure loads
    through TMA and runs its math on WGMMA, so it claims the shapes whose operands
    TMA can address and ``GEMMCpAsyncKernel`` takes the rest.
    """

    supported_archs: list[int] = [90]

    _STRUCTURE_FLAGS = (
        "coop2",
        "coop2s",
        "coop2_splitk",
        "coop2_swap",
        "pingpong",
        "simple",
        "swap_ab",
    )

    # One named-barrier id per consumer warpgroup: each group arrives on its own
    # barrier (``arrive_count=128``) instead of a block-wide sync.
    _CONSUMER_BAR_WG0 = 8
    _CONSUMER_BAR_WG1 = 9

    def init_config(self, config: Optional[dict] = None) -> None:
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
            return
        super().init_config(config)

    @classmethod
    def refusal(cls, call: GEMMCall) -> Optional[str]:
        return _tma_misalignment(call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b)

    @classmethod
    def entry_for(cls, call: GEMMCall) -> Entry:
        return _dense_entry(cls, call)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
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

        self.init_config(config)
        if activation != "none":
            split_k = self.config.get("split_k", 1)
            unsupported = any(
                self.config.get(flag)
                for flag in ("simple", "swap_ab", "coop2", "coop2s", "coop2_swap", "pingpong")
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
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            self.sm_count,
            self.device_name,
            # The raster group of a persistent tile depends on what L2 holds.
            l2_bytes=torch.cuda.get_device_properties(self.device_index).L2_cache_size,
        )
        return scored if scored is not None else modal

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # The traced build is another program, chosen by the switch at launch.
        return self.program(self._launcher, trace.enabled)(a, b)

    def _launcher(self, cfg: dict, traced: bool) -> Callable:
        """What one launch under *cfg* runs on ``(a, b)``: the compiled program of the
        decomposition *cfg* selects, or a split-K mainloop and reduction with their output.
        Only the default decomposition has a traced build, which *traced* selects."""
        m, n, k, trans_a, trans_b, dtype = (
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            self.dtype_str,
        )
        if cfg.get("simple"):
            return _gemm_simple_kernel(m, n, k, trans_a, trans_b, dtype)(
                cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg.get("threads", 128),
                cfg.get("panel_size", 8),
                cfg.get("cluster_m", 1),
            )
        if cfg.get("swap_ab"):
            return _gemm_swap_ab_kernel(m, n, k, trans_a, trans_b, dtype)(
                cfg["block_nn"], cfg["block_k"], cfg["num_stages"]
            )
        if cfg.get("coop2s"):
            return _gemm_coop2s_kernel(
                m,
                n,
                k,
                trans_a,
                trans_b,
                dtype,
                consumer_bar_wg0=self._CONSUMER_BAR_WG0,
                consumer_bar_wg1=self._CONSUMER_BAR_WG1,
            )(cfg["block_n"], cfg["block_k"], cfg["num_stages"])
        narrow = {
            "narrow_n": cfg.get("narrow_n", 0),
            "narrow_rows": cfg.get("narrow_rows", 0),
            "narrow_cols": cfg.get("narrow_cols", 0),
        }
        if cfg.get("coop2"):
            return _gemm_coop2_kernel(
                m,
                n,
                k,
                trans_a,
                trans_b,
                dtype,
                sm_count=self.sm_count,
                consumer_bar_wg0=self._CONSUMER_BAR_WG0,
                consumer_bar_wg1=self._CONSUMER_BAR_WG1,
            )(
                block_n=cfg["block_n"],
                block_k=cfg["block_k"],
                num_stages=cfg["num_stages"],
                group_size_m=cfg["group_size_m"],
                stage_n=cfg.get("stage_n", 0),
                **narrow,
            )
        if cfg.get("coop2_swap"):
            return _gemm_coop2_swap_kernel(
                m,
                n,
                k,
                trans_a,
                trans_b,
                dtype,
                sm_count=self.sm_count,
                consumer_bar_wg0=self._CONSUMER_BAR_WG0,
                consumer_bar_wg1=self._CONSUMER_BAR_WG1,
            )(
                cfg["block_n"],
                cfg["num_stages"],
                cfg["group_size_m"],
                cfg.get("narrow_n", 0),
                tuple(cfg.get("row_mix", ())),
            )
        if cfg.get("pingpong"):
            return _gemm_pingpong_kernel(
                m,
                n,
                k,
                trans_a,
                trans_b,
                dtype,
                sm_count=self.sm_count,
                consumer_bar_wg0=self._CONSUMER_BAR_WG0,
                consumer_bar_wg1=self._CONSUMER_BAR_WG1,
            )(
                block_n=cfg["block_n"],
                block_k=cfg["block_k"],
                num_stages=cfg["num_stages"],
                group_size_m=cfg["group_size_m"],
                stage_n=cfg.get("stage_n", 0),
                epi_bufs=cfg.get("epi_bufs", 1),
                **narrow,
            )
        coop2 = bool(cfg.get("coop2_splitk"))
        split_k = cfg["split_k"] if coop2 else cfg.get("split_k", 1)
        if coop2 or split_k > 1:
            mainloop, reduce_ = _splitk_pair(
                m,
                n,
                k,
                trans_a,
                trans_b,
                dtype,
                coop2,
                0 if coop2 else cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                0 if coop2 else cfg["panel_size"],
                split_k,
                self.activation,
            )
            out_n = n // 2 if self.activation != "none" else n
            mainloop, reduce_ = CompiledProgram(mainloop), CompiledProgram(reduce_)

            def splitk(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                c = torch.empty((m, out_n), dtype=a.dtype, device=a.device)
                reduce_(mainloop(a, b), c)
                return c

            return splitk
        main_cfg = {key: value for key, value in cfg.items() if key != "split_k"}
        compiled = _gemm_kernel(
            m, n, k, trans_a, trans_b, dtype, traced=traced, sm_count=self.sm_count
        )(**main_cfg)
        if not passes.is_traced(compiled):
            return compiled
        stem = f"gemm_{m}x{n}x{k}_{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}_{dtype}"
        launch = CompiledProgram(compiled)
        return lambda a, b: trace.run(launch, (a, b), stem=stem, traced=True)


@functools.lru_cache(maxsize=32)
def _gemm_small_batch_kernel(m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    """Small-M NT GEMM reusing each streamed B tile across all M rows.

    One-row products support SM89/SM90; the two-row band is calibrated on SM90.
    N-tail stores are guarded because ``gemv_config`` can select two-column tiles
    for odd N. Buffer extents predicate K-tail loads.

    Returns:
        JIT factory taking ``(block_n, reduce_threads, num_stages)`` and producing C."""
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_small_batch_func(
        block_n: int = 1,
        reduce_threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        tile_k = 128 // (STR_TO_DTYPE[dtype].itemsize * 8)
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


def _bandwidth_autotune_grid(rts: tuple, bns: tuple, nss: tuple, *, smem_cap: int) -> list[dict]:
    """Config grid for the bandwidth-mode kernels, guarded by thread and SMEM caps."""
    # K elements one reduce thread loads per step: 128 bits of a 16-bit dtype.
    tile_k = 8
    # Shared-memory ceiling the staged ``B`` ring (2 bytes per element) must fit under.
    return [
        {"block_n": bn, "reduce_threads": rt, "num_stages": ns}
        for rt in rts
        for bn in bns
        if rt * bn <= 1024
        for ns in nss
        if bn * (rt * tile_k) * 2 * ns <= smem_cap
    ]


class GemvKernel(Kernel, GEMMFwdInterface):
    """The bandwidth-bound band of ``GEMMFwdOp``: at most two rows contracted over K.

    Three bands build one body, :func:`_gemm_small_batch_kernel`, which reduces over
    K on CUDA cores: the two layouts a vector operand can take, and the ``m == 2`` NT
    calls where it outscores every tiny-m tile (``two_row_gemv_wins``). They differ in
    the region they serve and in the config band they pick, which is what
    :meth:`band_for` and :attr:`default_config` state; the body is the same, so they
    are one class.

    Args:
        band: Which band this instance serves, one of :attr:`BANDS`.
        m: Rows of the product.
        n: Columns of the product.
        k: Contraction dim.
        dtype: Input/output torch dtype (fp16 / bf16); fp32 accumulation.
        config: Optional explicit config; defaults to :attr:`default_config`.
        device_index: Device whose SM count and name pick the config.
    """

    supported_archs: list[int] = [89, 90]
    # A vector operand is TMA-addressable as often as any other shape, so this band
    # overlaps the warp-specialized kernel; reducing over K on CUDA cores wins there.
    preferred_over = frozenset({"gemm_tma"})

    #: The bands this class serves. ``lhs_row`` and ``rhs_col`` name the vector
    #: operand; ``lhs_rows`` is the two-row NT band.
    BANDS: tuple[str, ...] = ("lhs_row", "rhs_col", "lhs_rows")

    @classmethod
    def _vector_operand(cls, call: GEMMCall) -> Optional[str]:
        """Which operand is the vector: ``"lhs_row"``, ``"rhs_col"``, or neither.

        ``a`` is a single row with ``b`` transposed, or ``b`` is a single column with
        neither transposed; the other two layouts have no GEMV form here.
        """
        if call.trans_a:
            return None
        if call.m == 1 and call.trans_b:
            return "lhs_row"
        if call.n == 1 and not call.trans_b:
            return "rhs_col"
        return None

    @classmethod
    def band_for(cls, call: GEMMCall) -> Optional[str]:
        """The band serving *call*, or ``None`` when this class does not serve it.

        Read by :meth:`refusal` and by :meth:`entry_for`, so the region is stated once.
        """
        vector = cls._vector_operand(call)
        if vector is not None:
            return vector
        # The two-row crossover is calibrated on SM90 only.
        if call.arch != 90 or call.trans_a or not call.trans_b or call.m != 2:
            return None
        if not two_row_gemv_wins(call.n, call.k, call.sm_count, call.calibration):
            return None
        return "lhs_rows"

    @classmethod
    def refusal(cls, call: GEMMCall) -> "str | None":
        if cls.band_for(call) is None:
            return "no GEMV band serves this shape"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GEMMCall) -> Entry:
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
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        if self.band == "lhs_rows":
            return small_batch_config(self.n, self.k, get_sm_count(self.device_index))
        return gemv_config(self.k)

    @property
    def autotune_configs(self) -> list[dict]:
        smem_cap = torch.cuda.get_device_properties(self.device_index).shared_memory_per_block_optin
        if self.band == "lhs_rows":
            return _bandwidth_autotune_grid(
                (32, 64, 128), (1, 2, 4), (2, 3, 4, 5), smem_cap=smem_cap
            )
        return _bandwidth_autotune_grid(
            (32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6), smem_cap=smem_cap
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        kernel = self.program(self._compile)
        if self.band == "lhs_rows":
            return kernel(a, b)
        vector, matrix = (a, b) if self.band == "lhs_row" else (b, a)
        out = kernel(vector.reshape(1, -1).contiguous(), matrix)
        return out.reshape(1, self.n) if self.band == "lhs_row" else out.reshape(self.m, 1)

    def _compile(self, cfg: dict) -> Callable:
        """The small-batch program, compiled from *cfg*."""
        return self.kernel(cfg["block_n"], cfg["reduce_threads"], cfg["num_stages"])


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
        # Built here, not closed over: the autotuner refuses a closure that holds a tuple.
        a_shape = (k, m) if trans_a else (m, k)
        b_shape = (n, k) if trans_b else (k, n)
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


class GEMMCpAsyncKernel(Kernel, GEMMFwdInterface):
    """Dense GEMM kernel: pipelined, architecture-agnostic (sm80+).

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout —
    the same contract as ``GEMMTMAKernel`` — via ``T.Pipelined`` + plain
    ``T.gemm`` so it runs on pre-SM90 tensor-core targets (sm80 / sm86 /
    sm89). It loads through ``cp.async``, which addresses any extent, so it is the
    general implementation: it serves every shape no other one claims, SM90 shapes
    TMA cannot address included. fp16 / bf16 inputs, fp32 accumulation. ``block_k``
    must divide ``k`` (the smallest fallback is 16); M / N need not be multiples of
    the block sizes (epilogue guard). A config with ``split_k > 1`` returns the
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

        Read by ``refusal``, so an unservable K is refused during selection rather
        than reaching a builder, and by the constructor, which is also entered directly.
        """
        if k * dtype.itemsize >= 4:
            return None
        return (
            f"the pipelined mainloop loads its innermost dimension in 4-byte units, so k "
            f"must span at least one; k={k} of a {dtype.itemsize}-byte dtype does not"
        )

    @classmethod
    def refusal(cls, call: GEMMCall) -> Optional[str]:
        return cls._narrow_k_row(call.k, call.dtype)

    @classmethod
    def entry_for(cls, call: GEMMCall) -> Entry:
        return _dense_entry(cls, call)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        trans_a: bool = False,
        trans_b: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

        self.kernel = _gemm_basic_kernel(m, n, k, trans_a, trans_b, self.dtype_str)

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Modal winner shape of the pipelined BMM kernel across the manifest
        # workloads, fitted on the calibration device.
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
        return self.program(self._launcher)(a, b)

    def _launcher(self, cfg: dict) -> Callable:
        """What one launch under *cfg* runs on ``(a, b)``: the pipelined program, followed
        for ``split_k > 1`` by the reduction of its fp32 slices into the output."""
        jit_config = {k: v for k, v in cfg.items() if k != "pass_configs"}
        mainloop = self.kernel(**jit_config)
        split_k = cfg.get("split_k", 1)
        if split_k == 1:
            return mainloop
        mainloop = CompiledProgram(mainloop)
        reduce_ = CompiledProgram(splitk_reduce_kernel(split_k, self.m, self.n, self.dtype_str)())
        m, n = self.m, self.n

        def split(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            result = mainloop(a, b)
            output = a.new_empty((m, n))
            reduce_(result, output)
            return output

        return split
