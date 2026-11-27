import dataclasses
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    TensorCoreIntrinEmitter as WgmmaEmitter,
)

from tileops._csrc import csrc_include
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import (
    MAX_BLOCK_THREADS,
    QUANT_SCALE_BLOCK,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.gemm.call_spec import (
    GEMMCall,
    GEMMFP8Call,
    GEMMFP8FwdInterface,
    GEMMFwdInterface,
)
from tileops._kernels.gemm.heuristics import (
    COOP2_SWAP_STAGING,
    SWAP_AB_MPAD,
    best_config,
    fp8_ws_config,
    gemv_config,
    gemv_registers,
    small_batch_config,
    stream_width,
    two_row_gemv_wins,
)
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.tiling import grouped_tile_id
from tileops._utils import STR_TO_DTYPE, WARP_LANES, WARP_SHUFFLE_STAGES
from tileops.trace import _passes as passes
from tileops.trace import trace

__all__ = [
    "GEMMCpAsyncKernel",
    "GEMMFP8BlockScaleKernel",
    "GEMMFP8TensorScaleKernel",
    "GEMMTMAKernel",
    "GemvKernel",
]


# One named-barrier id per consumer warpgroup of the two-consumer kernels: each group arrives
# on its own barrier (``arrive_count=128``) instead of a block-wide sync.
_CONSUMER_BAR_WG0 = 8
_CONSUMER_BAR_WG1 = 9


def _tma_misalignment(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool
) -> str | None:
    """Why TMA cannot address these operands, or ``None`` when it can.

    TMA addresses each operand's innermost dimension in 16-byte units, so that extent
    must be a multiple of ``16 / itemsize`` elements: ``K`` for a non-transposed ``A``
    and a transposed ``B``, ``M`` for a transposed ``A``, ``N`` for a non-transposed
    ``B``. Checked here because TileLang's own descriptor check names nothing to change.
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


def _b_eviction(m: int, block_m: int) -> str | None:
    """``"evict_first"`` for a ``B`` tile the grid reads at most twice, else ``None``.

    Each of the ``ceil(m / block_m)`` M-tiles reads the whole of ``B``. At one or two of
    them ``B`` is streamed and the L2 it would hold belongs to ``A``, which every N-tile
    re-reads; above two ``B`` is itself the reused operand.
    """
    return "evict_first" if -(-m // block_m) <= 2 else None


@T.macro
def _add_bias(acc, bias, start, n, *, by_row=False, helper_layout=False):
    """Add FP32 column bias, or row bias for a transposed output tile.

    Out-of-range lanes read the last bias and are excluded from output stores.
    ``helper_layout`` uses the 64-row WGMMA fragment layout through a C++ helper:
    ``T.Parallel`` over a fragment whose layout starts at thread 0 but that a later
    warpgroup holds drops the loop."""
    rows, cols = acc.shape
    if helper_layout:
        T.call_extern(
            "handle",
            f"tileops::dense_add_{'row' if by_row else 'col'}_bias_64xN<{int(cols)}>",
            acc.data,
            T.address_of(bias[0]),
            start,
            n,
        )
    else:
        for i, j in T.Parallel(rows, cols):
            acc[i, j] += T.cast(bias[T.min(start + (i if by_row else j), n - 1)], acc.dtype)


def _a_row_load(m: int, block_m: int, block_k: int, dtype: str) -> tuple[int, int]:
    """Rows of a K-major A that one K step loads into a ``block_m``-row tile, and the K
    elements each TMA copy of them covers.

    A grid of one M tile loads only A's rows, rounded up to an 8-row swizzle atom, because TMA's
    zero fill of rows past ``m`` costs more than loading real rows. The tile's other rows keep
    stale data, and their outputs are never stored. Fewer rows than the tile are copied one
    128-byte swizzle atom of K at a time, so each copy fills a contiguous shared region; a
    ``block_k`` the atom does not divide loads the full tile.
    """
    atom = min(block_k, 128 // STR_TO_DTYPE[dtype].itemsize)
    if -(-m // block_m) > 1 or block_k % atom:
        return block_m, atom
    return min(block_m, -(-m // 8) * 8), atom


@T.macro
def _load_a(a, a_smem, ab_full, slot, m_start, k_start, block_m, block_k, rows, atom, trans_a):
    """TMA-load one K step of A into ring slot *slot*: ``rows`` rows of a K-major A, ``atom``
    K elements a copy (see :func:`_a_row_load`)."""
    if trans_a:
        T.tma_copy(
            a[k_start : k_start + block_k, m_start : m_start + block_m],
            a_smem[slot, :, :],
            barrier=ab_full[slot],
        )
    elif rows < block_m:
        for kc in T.unroll(block_k // atom):
            T.tma_copy(
                a[m_start : m_start + rows, k_start + kc * atom : k_start + kc * atom + atom],
                a_smem[slot, 0:rows, kc * atom : kc * atom + atom],
                barrier=ab_full[slot],
            )
    else:
        T.tma_copy(
            a[m_start : m_start + block_m, k_start : k_start + block_k],
            a_smem[slot, :, :],
            barrier=ab_full[slot],
        )


@T.macro
def _wgmma_step(a_ring, b_ring, slot, acc, ki, helper, trans_a, trans_b):
    """One K step of ring slot *slot* into *acc*: one ``dense_wgmma_helper.h`` *helper* call,
    or ``T.wgmma_gemm`` where *helper* is ``None``."""
    if helper:
        T.call_extern(  # lint: extern-target allow: a dense_wgmma_helper.h WGMMA, by tile width
            "handle",
            helper,
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


def wgmma_acc_layout(buf, rows: int, cols: int, chunk: int, operand_dtype, accum_dtype):
    """The layout ``T.wgmma_gemm`` gives one warpgroup's ``rows x cols`` NT accumulator over K
    steps of *chunk*, *operand_dtype* operands accumulating in *accum_dtype*; the
    ``dense_wgmma_helper.h`` WGMMAs write the same registers."""
    return WgmmaEmitter(
        a_dtype=operand_dtype,
        b_dtype=operand_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=4,
        block_col_warps=1,
        warp_row_tiles=rows // 4,
        warp_col_tiles=cols,
        chunk=chunk,
    ).make_mma_store_layout(buf)


def native_acc_layout(
    buf, rows: int, cols: int, chunk: int, operand_dtype, accum_dtype, first_thread: int
):
    """:func:`wgmma_acc_layout` for the warpgroup starting at thread *first_thread*."""
    base = wgmma_acc_layout(buf, rows, cols, chunk, operand_dtype, accum_dtype)
    return T.Fragment(
        [rows, cols],
        forward_thread_fn=lambda i, j: base.map_forward_thread([i, j])[0] + first_thread,
        forward_index_fn=lambda i, j: base.map_forward_index([i, j]),
    )


def _main(
    name: str,
    body: Callable,
    operands: tuple,
    c: tuple,
    bias: tuple | None = None,
) -> Callable:
    """The program running ``body(*operands, bias, c)``, with ``bias`` a parameter
    between the operands and C, or ``None`` when *bias* is ``None``.

    *operands* holds one ``(shape, dtype)`` pair, the parameter ``w``, or two, ``a`` and
    ``b``; *c* and *bias* are one pair each. The program's symbol is ``{name}_bias_main``
    with a bias, else ``{name}_main``.
    """
    c_shape, c_dtype = c
    if len(operands) == 1:
        ((w_shape, w_dtype),) = operands
        if bias is None:

            @T.prim_func
            def main(
                w: T.Tensor(w_shape, w_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(w, None, c)

        else:
            bias_shape, bias_dtype = bias

            @T.prim_func
            def main(
                w: T.Tensor(w_shape, w_dtype),  # type: ignore
                bias: T.Tensor(bias_shape, bias_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(w, bias, c)

    else:
        (a_shape, a_dtype), (b_shape, b_dtype) = operands
        if bias is None:

            @T.prim_func
            def main(
                a: T.Tensor(a_shape, a_dtype),  # type: ignore
                b: T.Tensor(b_shape, b_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(a, b, None, c)

        else:
            bias_shape, bias_dtype = bias

            @T.prim_func
            def main(
                a: T.Tensor(a_shape, a_dtype),  # type: ignore
                b: T.Tensor(b_shape, b_dtype),  # type: ignore
                bias: T.Tensor(bias_shape, bias_dtype),  # type: ignore
                c: T.Tensor(c_shape, c_dtype),  # type: ignore
            ) -> None:
                body(a, b, bias, c)

    symbol = f"{name}_main" if bias is None else f"{name}_bias_main"
    return main.with_attr("global_symbol", symbol)


def _dense_entry(cls: type, call: GEMMCall) -> Entry:
    """The entry for a kernel taking ``(m, n, k, dtype)``, both flags and the output dtype.

    The device is in the identity: its SM count and name pick the config.
    """
    index = call.device_index
    identity = (
        call.m,
        call.n,
        call.k,
        call.dtype,
        call.trans_a,
        call.trans_b,
        call.out_dtype,
        index,
    )
    return identity, lambda: cls(
        call.m,
        call.n,
        call.k,
        call.dtype,
        trans_a=call.trans_a,
        trans_b=call.trans_b,
        device_index=index,
        out_dtype=call.out_dtype,
    )


class _GEMMFP8Kernel(Kernel, GEMMFP8FwdInterface):
    """Shared body of the two FP8 GEMM kernels; ``BLOCK_SCALED`` picks the scale grid.

    Takes :func:`_gemm_fp8_ws_kernel`, or :func:`_gemm_fp8_kernel` on a call
    :meth:`_ws_refusal` rejects.
    """

    supported_archs: list[int] = [89, 90]

    # Whether this kernel reads block128 scale grids rather than per-tensor scalars.
    BLOCK_SCALED = False

    @staticmethod
    def _ws_refusal(m: int, n: int, k: int, dtype: torch.dtype, arch: int) -> str | None:
        """Why the warp-specialized variant cannot serve this call, or ``None``.

        It loads through TMA, and its epilogue releases a ring slot the mainloop
        named, so it needs at least one K-tile. The fallback carries neither.
        """
        if arch != 90:
            return f"TMA and WGMMA are SM90 instructions, the device is sm{arch}"
        if k == 0:
            return "the warp-specialized mainloop has no K-tile to run at k=0"
        return _tma_misalignment(m, n, k, dtype, trans_a=False, trans_b=True)

    @classmethod
    def entry_for(cls, call: GEMMFP8Call) -> Entry:
        index = call.device_index
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
            b_scale_rows=QUANT_SCALE_BLOCK if cls.block_scale_grid(call) == "1d2d" else 1,
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
        config: dict | None = None,
        device_index: int | None = None,
        b_scale_rows: int = 1,
        arch: int | None = None,
        sm_count: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.b_scale_rows = b_scale_rows
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = self.build_target.sm_count if sm_count is None else sm_count
        arch = self.build_target.arch if arch is None else arch
        self.ws_refusal = self._ws_refusal(m, n, k, dtype, arch)
        self.init_config(config)
        self._unused_bias: torch.Tensor | None = None

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

    def _bias_operand(self, bias: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.program(self._launcher, bias is not None)(a, b, scale_a, scale_b, bias)

    def _launcher(self, cfg: dict, has_bias: bool) -> Callable:
        """What one launch under *cfg* runs on ``(a, b, scale_a, scale_b, bias)``, with or
        without a bias epilogue as *has_bias* says."""
        if self.ws_refusal is None:
            tile_config = {key: value for key, value in cfg.items() if key != "split_k"}
            split_k = cfg.get("split_k", 1)
            if split_k > 1:
                return self._split_k_launcher(has_bias, split_k, tile_config)
            builder = _gemm_fp8_ws_kernel(
                self.m,
                self.n,
                self.k,
                self.dtype_str,
                self.out_dtype_str,
                self.BLOCK_SCALED,
                has_bias=has_bias,
                sm_count=self.sm_count,
                b_scale_rows=self.b_scale_rows,
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


def _fp8_scale_shapes(
    m: int, n: int, k: int, block_scaled: bool, b_scale_rows: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The shapes of ``scale_a`` and ``scale_b``: block128 grids, or one scalar each."""
    if not block_scaled:
        return (1, 1), (1, 1)
    scale_k = -(-k // QUANT_SCALE_BLOCK)
    return (m, scale_k), (-(-n // b_scale_rows), scale_k)


@target_cache(maxsize=32)
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
    )(block_n, num_stages, group_size_m)
    return mainloop, splitk_reduce_kernel(split_k, m, n, out_dtype)()


@target_cache(maxsize=32)
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
    """FP8 NT GEMM without TMA or warp specialization, for calls the warp-specialized
    kernel refuses; fp32 accumulation per K tile.

    Returns:
        JIT factory taking ``(block_m, block_n, block_k, num_stages, threads)``."""
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
        if block_scaled and QUANT_SCALE_BLOCK % block_k:
            raise ValueError(f"block_k must divide {QUANT_SCALE_BLOCK} for block128 scaling")
        scale_a_shape, scale_b_shape = _fp8_scale_shapes(m, n, k, block_scaled, b_scale_rows)

        def _epilogue(acc, scale_a, scale_b, bias, col):
            """One output: the per-tensor scales applied, then the bias added."""
            value = acc if block_scaled else acc * scale_a[0, 0] * scale_b[0, 0]
            return value + bias[col] if has_bias else value

        @T.macro
        def _gemm_fp8_body(a, b, scale_a, scale_b, bias, c):
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
                        scale_idx = kk * block_k // QUANT_SCALE_BLOCK
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
                        c[m_start + i, n_start + j] = _epilogue(
                            c_local[i, j], scale_a, scale_b, bias, n_start + j
                        )

        # Only the program returned is parsed: the other would read the bias it lacks.
        if has_bias:

            @T.prim_func
            def _gemm_fp8_bias_main(
                a: T.Tensor((m, k), dtype),  # type: ignore
                b: T.Tensor((n, k), dtype),  # type: ignore
                scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
                scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
                bias: T.Tensor((n,), out_dtype),  # type: ignore
                c: T.Tensor((m, n), out_dtype),  # type: ignore
            ) -> None:
                _gemm_fp8_body(a, b, scale_a, scale_b, bias, c)

            return _gemm_fp8_bias_main

        @T.prim_func
        def _gemm_fp8_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            _gemm_fp8_body(a, b, scale_a, scale_b, None, c)

        return _gemm_fp8_main

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
):
    """One K-step of the producer: the step's A boxes, its B box and its two scale vectors.

    ``block_m`` is one or two consumers' 64 rows; ``a_bot`` is loaded only for the second.
    ``b_scale_rows`` is how many rows of ``b`` one ``scale_b`` row covers: 1 for a
    per-row grid, 128 for a per-128x128-block grid.
    """
    ks = kb * QUANT_SCALE_BLOCK
    T.tma_copy(
        a[m_start : m_start + WGMMA_ROWS, ks : ks + QUANT_SCALE_BLOCK],
        a_top[slot, :, :],
        barrier=ab_full[slot],
    )
    if block_m > WGMMA_ROWS:
        T.tma_copy(
            a[m_start + WGMMA_ROWS : m_start + block_m, ks : ks + QUANT_SCALE_BLOCK],
            a_bot[slot, :, :],
            barrier=ab_full[slot],
        )
    T.tma_copy(
        b[n_start : n_start + block_n, ks : ks + QUANT_SCALE_BLOCK],
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
):
    """One K-step of a block128 consumer: WGMMA into a fresh accumulator, then add it
    scaled into *acc*. ``row_base`` is this consumer's offset into the staged ``A`` row
    scales."""
    T.barrier_wait(ab_full[slot], phase)
    T.wgmma_gemm(
        a_smem[slot, :, :],
        b_smem[slot, :, :],
        part,
        transpose_B=True,
        policy=T.GemmWarpPolicy.FullRow,
        clear_accum=True,
    )
    T.copy(sa_stage[slot, row_base : row_base + WGMMA_ROWS], sa_f)
    T.copy(sb_stage[slot, :], sb_f)
    T.wait_wgmma(0)
    for i, j in T.Parallel(WGMMA_ROWS, block_n):
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
    ``dense_wgmma_helper.h``.
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
def _fp8_ws_plain_drain(ab_empty, acc, prev, scale_a, scale_b, *, num_regs: int, block_n: int):
    """Close a per-tensor consumer's mainloop and apply the two scalars."""
    T.wait_wgmma(0)
    T.barrier_arrive(ab_empty[prev[0]])
    T.warpgroup_fence_operand(acc, num_regs=num_regs)
    for i, j in T.Parallel(WGMMA_ROWS, block_n):
        acc[i, j] *= scale_a[0, 0] * scale_b[0, 0]


@target_cache(maxsize=32)
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
) -> Callable:
    """Persistent FP8 NT GEMM with split-A/shared-B consumer warpgroups.

    Block128 scales apply per K tile; per-tensor scales apply in the epilogue.
    ``sm_count`` is part of the build cache key.

    Returns:
        JIT factory taking ``(block_n, num_stages, group_size_m)``."""
    accum_dtype = "float"
    # One consumer warpgroup's rows; the K tile is the block128 scale grid's.
    half_m, block_k = WGMMA_ROWS, QUANT_SCALE_BLOCK
    scale_a_shape, scale_b_shape = _fp8_scale_shapes(m, n, k, block_scaled, b_scale_rows)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            *CUDA_COMPILE_FLAGS,
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

        # One consumer warpgroup per 64 rows of the tile, beside the producer warpgroup.
        consumers = block_m // half_m
        # ``dense_wgmma_helper.h`` issues the tile's own width for the widths it
        # instantiates, under per-tensor scales, whose accumulator nothing else reads
        # inside the mainloop.
        native_widths = (176,)
        native_n = block_n if not block_scaled and block_n in native_widths else 0

        def consumer_acc_layout(buf, first_thread: int):
            """The accumulator layout of one consumer's 64 x ``block_n`` WGMMA."""
            return native_acc_layout(
                buf, half_m, block_n, block_k, T.float8_e4m3fn, T.float32, first_thread
            )

        nr = (half_m * block_n) // WARPGROUP_THREADS
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        grid = min(sm_count, total_tiles)
        max_waves = -(-total_tiles // grid) + 1
        k_iters = -(-k // block_k)
        bias_shape = (n,) if has_bias else (1,)
        stage_rows = num_stages if block_scaled else 1
        stage_store = m > half_m
        store_rows = half_m if stage_store else 1

        @T.macro
        def consumer(
            wg,
            a_smem,
            acc,
            part,
            sa,
            sb,
            out,
            c_smem,
            prev,
            gi: T.Ref,
            c,
            bias,
            scale_a,
            scale_b,
            b_smem,
            sa_stage,
            sb_stage,
            ab_full,
            ab_empty,
            mt,
            nt,
            pid,
        ):
            """Consumer warpgroup *wg*: the ``half_m`` rows from ``wg * half_m`` of each tile."""
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
                    m_start = mt[0] * block_m + half_m if wg else mt[0] * block_m
                    n_start = nt[0] * block_n
                    # The second consumer of a tile ending inside the first's rows holds none.
                    rows = (
                        T.max(T.int32(0), T.min(T.int32(half_m), T.int32(m) - m_start))
                        if wg
                        else T.min(T.int32(half_m), T.int32(m) - m_start)
                    )
                    cols = T.min(T.int32(block_n), T.int32(n) - n_start)
                    if block_scaled:
                        T.clear(acc)
                        for _ki in T.Pipelined(k_iters, num_stages=0):
                            slot = gi % num_stages
                            _fp8_ws_scaled_step(
                                a_smem,
                                b_smem,
                                sa_stage,
                                sb_stage,
                                ab_full,
                                ab_empty,
                                acc,
                                part,
                                sa,
                                sb,
                                slot,
                                (gi // num_stages) & 1,
                                row_base=wg * half_m,
                                block_n=block_n,
                            )
                            gi = gi + 1
                    else:
                        for ki in T.Pipelined(k_iters, num_stages=0):
                            slot = gi % num_stages
                            _fp8_ws_plain_step(
                                a_smem,
                                b_smem,
                                ab_full,
                                ab_empty,
                                acc,
                                prev,
                                slot,
                                (gi // num_stages) & 1,
                                ki,
                                native_n=native_n,
                            )
                            gi = gi + 1
                        _fp8_ws_plain_drain(
                            ab_empty,
                            acc,
                            prev,
                            scale_a,
                            scale_b,
                            num_regs=nr,
                            block_n=block_n,
                        )
                    # Cast this consumer's rows and store them: one TMA box when full,
                    # elements otherwise. Without stage_store no row tile is full.
                    bar = _CONSUMER_BAR_WG1 if wg else _CONSUMER_BAR_WG0
                    if has_bias:
                        for i, j in T.Parallel(half_m, block_n):
                            out[i, j] = T.cast(
                                acc[i, j] + T.cast(bias[T.min(n_start + j, n - 1)], accum_dtype),
                                out_dtype,
                            )
                    else:
                        T.copy(acc, out)
                    if not stage_store:
                        if rows > T.int32(0):
                            for i, j in T.Parallel(half_m, block_n):
                                if i < rows and j < cols:
                                    c[m_start + i, n_start + j] = out[i, j]
                    elif rows == T.int32(half_m) and cols == T.int32(block_n):
                        T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
                        if native_n:
                            # stmatrix offsets the branch's first thread; this layout is
                            # absolute, so loop.
                            for i, j in T.Parallel(half_m, block_n):
                                c_smem[i, j] = out[i, j]
                        else:
                            T.copy(out, c_smem)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
                        T.copy(c_smem, c[m_start, n_start])
                    elif rows > T.int32(0):
                        for i, j in T.Parallel(half_m, block_n):
                            if i < rows and j < cols:
                                c[m_start + i, n_start + j] = out[i, j]

        @T.prim_func
        def _gemm_fp8_ws_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(grid, threads=WARPGROUP_THREADS * (1 + consumers)) as (pid,):
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
                    layouts[acc_0] = consumer_acc_layout(acc_0, WARPGROUP_THREADS)
                    layouts[out_0] = consumer_acc_layout(out_0, WARPGROUP_THREADS)
                    layouts[acc_1] = consumer_acc_layout(acc_1, 2 * WARPGROUP_THREADS)
                    layouts[out_1] = consumer_acc_layout(out_1, 2 * WARPGROUP_THREADS)
                T.annotate_layout(layouts)

                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([WARPGROUP_THREADS * consumers] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
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
                                )
                                gi_prod = gi_prod + 1

                elif tx < 2 * WARPGROUP_THREADS:
                    consumer(
                        0,
                        a_top,
                        acc_0,
                        part_0,
                        sa_0,
                        sb_0,
                        out_0,
                        c_smem_0,
                        ps0,
                        gi_cons_0,
                        c,
                        bias,
                        scale_a,
                        scale_b,
                        b_smem,
                        sa_stage,
                        sb_stage,
                        ab_full,
                        ab_empty,
                        mt,
                        nt,
                        pid,
                    )
                elif consumers == 2:
                    consumer(
                        1,
                        a_bot,
                        acc_1,
                        part_1,
                        sa_1,
                        sb_1,
                        out_1,
                        c_smem_1,
                        ps1,
                        gi_cons_1,
                        c,
                        bias,
                        scale_a,
                        scale_b,
                        b_smem,
                        sa_stage,
                        sb_stage,
                        ab_full,
                        ab_empty,
                        mt,
                        nt,
                        pid,
                    )

        return _gemm_fp8_ws_main

    return _gemm_fp8_ws_func


@target_cache(maxsize=32)
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
) -> Callable:
    """Split-K FP8 NT GEMM writing fp32 ``slices[split_k, m, n]``.

    ``split_k`` must divide the block128 K-tile count. Only slice zero adds bias;
    ``splitk_reduce_kernel`` sums and casts the partials.

    Returns:
        JIT factory taking ``(block_n, num_stages, group_size_m)``."""
    accum_dtype = "float"
    half_m, block_k = WGMMA_ROWS, QUANT_SCALE_BLOCK
    block_m = 2 * half_m
    scale_a_shape, scale_b_shape = _fp8_scale_shapes(m, n, k, block_scaled, b_scale_rows)
    k_iters_total = -(-k // block_k)
    if k_iters_total % split_k:
        raise ValueError(
            f"split_k={split_k} must divide the K-tile count evenly "
            f"(k={k}, block_k={block_k} -> {k_iters_total} tiles)"
        )

    @tilelang.jit(
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", "-DENABLE_FP8"],
    )
    def _gemm_fp8_ws_splitk_func(
        block_n: int = 64,
        num_stages: int = 5,
        group_size_m: int = 8,
    ) -> Callable:
        nr = (half_m * block_n) // WARPGROUP_THREADS
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        k_iters = k_iters_total // split_k
        stage_rows = num_stages if block_scaled else 1
        bias_shape = (n,) if has_bias else (1,)

        @T.macro
        def consumer(
            wg,
            a_smem,
            acc,
            part,
            sa,
            sb,
            prev,
            scale_a,
            scale_b,
            bias,
            slices,
            b_smem,
            sa_stage,
            sb_stage,
            ab_full,
            ab_empty,
            m_start,
            n_start,
            bz,
        ):
            """Consumer warpgroup *wg*: the ``half_m`` rows from ``wg * half_m`` of the tile."""
            T.inc_max_nreg(232)
            rows = (
                T.max(T.int32(0), T.min(T.int32(half_m), T.int32(m) - m_start - half_m))
                if wg
                else T.min(T.int32(half_m), T.int32(m) - m_start)
            )
            cols = T.min(T.int32(block_n), T.int32(n) - n_start)
            if block_scaled:
                T.clear(acc)
                for ki in T.Pipelined(k_iters, num_stages=0):
                    _fp8_ws_scaled_step(
                        a_smem,
                        b_smem,
                        sa_stage,
                        sb_stage,
                        ab_full,
                        ab_empty,
                        acc,
                        part,
                        sa,
                        sb,
                        ki % num_stages,
                        (ki // num_stages) & 1,
                        row_base=wg * half_m,
                        block_n=block_n,
                    )
            else:
                for ki in T.Pipelined(k_iters, num_stages=0):
                    _fp8_ws_plain_step(
                        a_smem,
                        b_smem,
                        ab_full,
                        ab_empty,
                        acc,
                        prev,
                        ki % num_stages,
                        (ki // num_stages) & 1,
                        ki,
                    )
                _fp8_ws_plain_drain(
                    ab_empty,
                    acc,
                    prev,
                    scale_a,
                    scale_b,
                    num_regs=nr,
                    block_n=block_n,
                )
            if has_bias and bz == 0:
                for i, j in T.Parallel(half_m, block_n):
                    acc[i, j] += T.cast(bias[T.min(n_start + j, n - 1)], accum_dtype)
            for i, j in T.Parallel(half_m, block_n):
                if i < rows and j < cols:
                    slices[bz, m_start + wg * half_m + i, n_start + j] = acc[i, j]

        @T.prim_func
        def _gemm_fp8_ws_splitk_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            slices: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(total_tiles, split_k, threads=3 * WARPGROUP_THREADS) as (pid, bz):
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

                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([2 * WARPGROUP_THREADS] * num_stages)

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

                if tx < WARPGROUP_THREADS:
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
                        )

                elif tx < 2 * WARPGROUP_THREADS:
                    consumer(
                        0,
                        a_top,
                        acc_0,
                        part_0,
                        sa_0,
                        sb_0,
                        ps0,
                        scale_a,
                        scale_b,
                        bias,
                        slices,
                        b_smem,
                        sa_stage,
                        sb_stage,
                        ab_full,
                        ab_empty,
                        m_start,
                        n_start,
                        bz,
                    )
                else:
                    consumer(
                        1,
                        a_bot,
                        acc_1,
                        part_1,
                        sa_1,
                        sb_1,
                        ps1,
                        scale_a,
                        scale_b,
                        bias,
                        slices,
                        b_smem,
                        sa_stage,
                        sb_stage,
                        ab_full,
                        ab_empty,
                        m_start,
                        n_start,
                        bz,
                    )

        return _gemm_fp8_ws_splitk_main

    return _gemm_fp8_ws_splitk_func


@target_cache(maxsize=32)
def _gemm_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    traced: bool = False,
    *,
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """SM90 GEMM with one TMA producer and one WGMMA consumer warpgroup.

    Supports all transpose layouts with fp32 accumulation. Operands must satisfy
    ``_tma_misalignment``. Traced builds append a ``slots`` output after C.
    *has_bias* builds the program taking ``bias[n]`` before C. C is *out_dtype*, or
    *dtype* when it is ``None``. ``cluster_m`` launches that many M tiles of one N
    column as a cluster, so the SMs reading the same B tile run together.

    Returns:
        JIT factory taking ``(block_m, block_n, block_k, num_stages, panel_size,
        cluster_m)``."""
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    wgmma_dtype = "kBFloat16" if dtype == "bfloat16" else "kFloat16"

    @tilelang.jit(
        out_idx=trace.out_idx(1, traced),
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
    )
    def _gemm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        panel_size: int = 10,
        cluster_m: int = 1,
    ) -> Callable:
        # One TMA producer and one WGMMA consumer warpgroup; the barrier counts assume it.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        b_evict = _b_eviction(m, block_m)
        a_rows, a_atom = _a_row_load(m, block_m, block_k, dtype)
        # TMA stores the output where its rows are a whole number of 16-byte units.
        tma_epilogue = (n * STR_TO_DTYPE[c_dtype].itemsize) % 16 == 0
        if cluster_m > 1 and (-(-m // block_m) % cluster_m or panel_size > 0):
            raise ValueError(
                f"cluster_m={cluster_m} must divide the M grid and runs without a panel swizzle"
            )
        # The helper's accumulator reaches global memory only through the shared-memory
        # TMA store: TileLang drops its layout in the predicated direct-store loop.
        native_wgmma = (
            tma_epilogue
            and not trans_a
            and trans_b
            and block_m == 64
            and block_k == 64
            and 256 % block_n != 0
        )
        native_helper = f"tileops::dense_wgmma_64xN_k64<tl::DataType::{wgmma_dtype}, {block_n}>"
        # The first thread of the helper accumulator's layout; see its annotation.
        acc_first = WARPGROUP_THREADS if c_dtype == "float32" else 0

        def _launch():
            if cluster_m > 1:
                return T.ClusterKernel(
                    T.ceildiv(n, block_n),
                    T.ceildiv(m, block_m),
                    cluster_dims=(1, cluster_m, 1),
                    threads=threads,
                )
            return T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads)

        @T.macro
        def _gemm_body(a, b, bias, c):
            with _launch() as (bx, by):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                # A ring of num_stages A/B slots: K step ki takes slot ki % num_stages, whose
                # barrier phase flips every num_stages steps.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                smem = [a_smem, b_smem]
                if tma_epilogue:
                    c_smem = T.alloc_shared((block_m, block_n), c_dtype)
                    smem.append(c_smem)
                T.annotate_layout({buf: tilelang.layout.make_swizzled_layout(buf) for buf in smem})
                if native_wgmma:
                    # The accumulator layout T.wgmma_gemm gives a 64 x block_n tile; the
                    # helper writes the same registers. TileLang copies a 16-bit tile out
                    # with stmatrix, which places the warpgroup by the branch's threads, and
                    # a float32 one with a loop over the layout's own threads, which must be
                    # the consumer's: from threads 0-127 it stores nothing.
                    T.annotate_layout(
                        {
                            c_local: native_acc_layout(
                                c_local, block_m, block_n, block_k, dtype, "float32", acc_first
                            )
                        }
                    )
                if has_bias:
                    bias_smem = T.alloc_shared((block_n,), dtype)
                    bias_full = T.alloc_barrier([WARPGROUP_THREADS])

                # One full (producer to consumer) and one empty (consumer to producer)
                # barrier per ring slot, each arrived on by one warpgroup.
                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)

                m_start = by * block_m
                n_start = bx * block_n

                ps = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
                    # Producer warpgroup. Intern the "producer" group first so it gets gid 0.
                    T.dec_max_nreg(24)
                    with trace.group("producer", lead=0):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            k_start = ki * block_k
                            T.barrier_wait(ab_empty[slot], phase ^ 1)
                            with trace.range("tma", lane="tma"):
                                _load_a(
                                    a,
                                    a_smem,
                                    ab_full,
                                    slot,
                                    m_start,
                                    k_start,
                                    block_m,
                                    block_k,
                                    a_rows,
                                    a_atom,
                                    trans_a,
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
                    if has_bias:
                        # Stage the tile's bias in SMEM after the last loads; the epilogue
                        # reads it from there.
                        for j in T.serial(T.ceildiv(block_n, WARPGROUP_THREADS)):
                            col = j * WARPGROUP_THREADS + tx
                            if col < block_n:
                                bias_smem[col] = bias[T.min(n_start + col, n - 1)]
                        T.barrier_arrive(bias_full[0])
                else:
                    # Consumer warpgroup.
                    T.inc_max_nreg(240)
                    num_accum_regs = (block_m * block_n) // WARPGROUP_THREADS
                    with trace.group("consumer", lead=WARPGROUP_THREADS):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            with trace.range("wait", lane="barrier"):
                                T.barrier_wait(ab_full[slot], phase)
                            with trace.range("mma", lane="wgmma"):
                                if native_wgmma:
                                    T.call_extern(
                                        "handle",
                                        native_helper,
                                        c_local.data,
                                        T.address_of(a_smem[slot, 0, 0]),
                                        T.address_of(b_smem[slot, 0, 0]),
                                        T.if_then_else(ki > 0, 1, 0),
                                    )
                                else:
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
                        if has_bias:
                            T.barrier_wait(bias_full[0], 0)
                            _add_bias(c_local, bias_smem, 0, block_n, helper_layout=native_wgmma)
                        if tma_epilogue:
                            with trace.range("epilogue"):
                                T.copy(c_local, c_smem)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=WARPGROUP_THREADS)
                                T.copy(c_smem, c[m_start, n_start])
                        else:
                            with trace.range("epilogue"):
                                for i, j in T.Parallel(block_m, block_n):
                                    if m_start + i < m and n_start + j < n:
                                        c[m_start + i, n_start + j] = c_local[i, j]

                # Each producer "arrive" pairs with the consumer "wait" of the same K step.
                trace.dag("arrive", "wait")

        main = _main(
            "_gemm",
            _gemm_body,
            (((k, m) if trans_a else (m, k), dtype), ((n, k) if trans_b else (k, n), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

        # Traced: materialize markers and append ``slots`` (matches ``out_idx``); untraced:
        # drop them, giving the un-instrumented CUDA.
        return trace.finalize(main, traced=traced, max_events=1024)

    return _gemm_func


@target_cache(maxsize=32)
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
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
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
        a_rows, a_atom = _a_row_load(m, block_m, block_k, dtype)
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

                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ps = T.alloc_local((1,), "int32")

                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice

                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        k_start = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        _load_a(
                            a,
                            a_smem,
                            ab_full,
                            slot,
                            m_start,
                            k_start,
                            block_m,
                            block_k,
                            a_rows,
                            a_atom,
                            trans_a,
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
                    num_accum_regs = (block_m * block_n) // WARPGROUP_THREADS
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


@target_cache(maxsize=32)
def splitk_reduce_kernel(
    split_k: int,
    m: int,
    n: int,
    dtype: str = "float16",
    activation: str = "none",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Reduce the split-K fp32 workspace into the final output.

    Sums ``w[split_k, m, n]`` over the slice axis in fp32, adds ``bias[n]`` (in *dtype*)
    under *has_bias*, applies *activation* (``"silu_and_mul"`` takes columns ``[0, n/2)``
    as gate and ``[n/2, n)`` as up, giving ``[m, n/2]``), and casts to *out_dtype*, or
    *dtype* when it is ``None``.

    ``C`` is a parameter, not a JIT-allocated output, so no allocation sits between the
    mainloop and reduce launches.

    Returns:
        JIT factory taking ``(elems_per_cta)``."""
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    if activation not in ("none", "silu_and_mul"):
        raise ValueError(f"unsupported split-K epilogue: {activation}")
    gated = activation == "silu_and_mul"
    if gated and n % 2:
        raise ValueError("silu_and_mul requires an even output width")
    out_n = n // 2 if gated else n

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: gated},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _splitk_reduce_func(elems_per_cta: int = 1024) -> Callable:
        def _slice_sum(w, bias, gi, gj):
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            if has_bias:
                expr = expr + T.cast(bias[gj], accum_dtype)
            return expr

        @T.macro
        def _splitk_reduce_body(w, bias, c):
            total = m * out_n
            with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx:
                base = bx * elems_per_cta
                for t in T.Parallel(elems_per_cta):
                    idx = base + t
                    if idx < total:
                        gi = idx // out_n
                        gj = idx % out_n
                        if gated:
                            gate = _slice_sum(w, bias, gi, gj)
                            up = _slice_sum(w, bias, gi, out_n + gj)
                            c[gi, gj] = T.cast(gate * T.sigmoid(gate) * up, c_dtype)
                        else:
                            c[gi, gj] = T.cast(_slice_sum(w, bias, gi, gj), c_dtype)

        return _main(
            "_splitk_reduce",
            _splitk_reduce_body,
            (((split_k, m, n), accum_dtype),),
            ((m, out_n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _splitk_reduce_func


def _persistent_ctas(total_tiles: int, sm_count: int) -> int:
    """The fewest CTAs that run ``total_tiles`` uniform tiles in as many waves as
    ``sm_count`` CTAs do.

    The busiest CTA runs as many tiles either way, while each wave reads fewer distinct
    panels.
    """
    return -(-total_tiles // -(-total_tiles // sm_count))


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
            _segment_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m, narrow=True)
        elif flat_id < T.int32(g.wide_tiles):
            if g.full_tiles == 0:
                _segment_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m, narrow=False)
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
                _segment_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m, narrow=False)
        else:
            _segment_tile(flat_id, mt, nt, grid=g, group_size_m=group_size_m, narrow=True)


@T.macro
def _segment_tile(flat_id, mt, nt, *, grid: _NarrowGrid, group_size_m: int, narrow: bool):
    """Tile ``flat_id`` of the last ``narrow_rows`` row tiles: a narrow tile, or a wide one
    right of the narrow tiles."""
    g = grid
    _grouped_rows(
        flat_id - T.int32(g.wide_tiles if narrow else g.full_tiles),
        mt,
        nt,
        rows=g.narrow_rows,
        cols=g.narrow_cols if narrow else g.tail_cols,
        row0=g.full_rows,
        group_size_m=group_size_m,
    )
    if narrow:
        nt[0] = nt[0] * g.narrow_n
        nt[1] = 1
    else:
        nt[0] = g.narrow_cols * g.narrow_n + nt[0] * g.block_n
        nt[1] = 0


@target_cache(maxsize=32)
def _gemm_coop2_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Persistent GEMM with two consumers sharing B and splitting A's rows.

    All layouts; a ``block_n`` that does not divide 256, and narrow tiles, are NT only.
    ``sm_count`` is part of the cache key. With ``narrow_n`` set, the last
    ``narrow_rows`` row tiles start with ``narrow_cols`` tiles ``narrow_n`` wide that run
    after every wide tile. *has_bias* builds the program taking ``bias[n]`` before C.
    C is *out_dtype*, or *dtype* when it is ``None``.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages, group_size_m, stage_n,
        narrow_n, narrow_rows, narrow_cols)``."""
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    cast = c_dtype != "float32"
    block_m = 128
    # ``dense_wgmma_helper.h``'s name for each dtype.
    wgmma_dtype_tag = {"bfloat16": "bf16", "float16": "fp16"}[dtype]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
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
        nr = (half_m * block_n) // WARPGROUP_THREADS
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
        if (trans_a or not trans_b) and (256 % block_n or mixed):
            raise ValueError(
                "coop2 tiles whose width does not divide 256, and narrow ones, are NT-only"
            )
        a_tile = (block_k, half_m) if trans_a else (half_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        total_tiles = grid.total_tiles
        ctas = sm_count if mixed else _persistent_ctas(total_tiles, sm_count)
        max_waves = -(-total_tiles // ctas) + 1
        k_iters = T.ceildiv(k, block_k)
        # ``dense_wgmma_helper.h`` issues a width that does not divide 256, and a narrow
        # tile's, as one N-wide WGMMA per 16 of K over 64-wide K steps; other K steps keep
        # T.wgmma_gemm.
        wide_wgmma = (256 % block_n or mixed) and block_k == 64
        wgmma_helper = f"tileops::dense_wgmma_{wgmma_dtype_tag}_64x{block_n}_k64"
        narrow_helper = f"tileops::dense_wgmma_{wgmma_dtype_tag}_64x{narrow_n}_k64"

        def helper(narrow):
            return narrow_helper if narrow else wgmma_helper if wide_wgmma else None

        # The accumulator layout T.wgmma_gemm gives one consumer's 64 x block_n tile; the
        # helper writes the same registers. TileLang places the warpgroup by the branch's
        # threads when it copies a 16-bit tile out with stmatrix, or a float32 one in
        # chunks, but copies a whole float32 tile with a loop over the layout's own
        # threads: then the layout starts at the consumer's first thread.
        acc_first = WARPGROUP_THREADS if not cast and n_chunks == 1 else 0

        # TMA addresses the output only where its rows are a whole number of 16-byte
        # units; there it also drops what an M / N tail puts past the output.
        tma_rows = (n * STR_TO_DTYPE[c_dtype].itemsize) % 16 == 0

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
                T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
                T.copy(c_cast[:, c0 : c0 + sn], c_smem)
                T.fence_proxy_async()
                T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
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
                    T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
                    T.copy(c_cast[:, c0 : c0 + sn], c_smem)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
                    T.tma_copy(
                        c_smem,
                        c[row_start : row_start + half_m, n_start + c0 : n_start + c0 + sn],
                    )
            else:
                # A warp stores one row at a time, its lanes on consecutive columns. The
                # row loop stays rolled: unrolled, each thread holds every store address
                # live and the consumer spills its accumulator.
                lane = T.get_thread_binding() % 32
                warp = T.get_thread_binding() % WARPGROUP_THREADS // 32
                for ch in range(n_chunks):
                    c0 = ch * sn
                    cols = T.min(T.int32(sn), T.int32(n) - n_start - c0)
                    T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
                    T.copy(c_cast[:, c0 : c0 + sn], c_smem)
                    T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
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

        @T.macro
        def mainloop(a_smem, b_smem, ab_full, ab_empty, acc, prev, gi: T.Ref, narrow):
            for ki in T.Pipelined(k_iters, num_stages=0):
                slot = gi % num_stages
                T.barrier_wait(ab_full[slot], (gi // num_stages) & 1)
                _wgmma_step(a_smem, b_smem, slot, acc, ki, helper(narrow), trans_a, trans_b)
                if ki > 0:
                    T.wait_wgmma(1)
                    T.barrier_arrive(ab_empty[prev[0]])
                prev[0] = slot
                gi = gi + 1

        @T.macro
        def consumer(
            wg,
            a_smem,
            acc,
            acc_cast,
            c_smem,
            prev,
            gi: T.Ref,
            c,
            bias,
            b_smem,
            ab_full,
            ab_empty,
            mt,
            nt,
            pid,
        ):
            """Consumer warpgroup *wg*: the ``half_m`` rows from ``wg * half_m`` of each tile."""
            T.inc_max_nreg(240)
            bar = _CONSUMER_BAR_WG1 if wg else _CONSUMER_BAR_WG0
            for w in T.serial(max_waves):
                flat_id = T.int32(ctas) * w + pid
                if flat_id < total_tiles:
                    _narrow_grid_tile(flat_id, mt, nt, grid=grid, group_size_m=group_size_m)
                    m_start = mt[0] * block_m
                    n_start = nt[0] if mixed else nt[0] * block_n
                    if wg:
                        # Rows the second consumer holds; none in a tile ending inside the
                        # first consumer's half.
                        arows = T.int32(m) - m_start - T.int32(half_m)
                    # One mainloop per width, so a K step tests nothing.
                    if mixed and nt[1] == 1:
                        mainloop(a_smem, b_smem, ab_full, ab_empty, acc, prev, gi, 1)
                    else:
                        mainloop(a_smem, b_smem, ab_full, ab_empty, acc, prev, gi, 0)
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[prev[0]])
                    T.warpgroup_fence_operand(acc, num_regs=nr)
                    if has_bias:
                        _add_bias(acc, bias, n_start, n, helper_layout=wide_wgmma)
                    if cast:
                        T.copy(acc, acc_cast)
                    if mixed and nt[1] == 1:
                        store_half_narrow(
                            c, acc_cast, c_smem, m_start + half_m if wg else m_start, n_start, bar
                        )
                    elif not wg:
                        store_half(
                            c,
                            acc_cast,
                            c_smem,
                            m_start,
                            T.min(T.int32(half_m), T.int32(m) - m_start),
                            n_start,
                            bar,
                        )
                    elif arows > T.int32(0):
                        store_half(c, acc_cast, c_smem, m_start + half_m, arows, n_start, bar)
            # Drain the last store before the CTA's shared memory goes.
            T.tma_store_wait(0)

        @T.macro
        def _gemm_coop2_body(a, b, bias, c):
            with T.Kernel(ctas, threads=3 * WARPGROUP_THREADS) as (pid,):
                a_smem_top = T.alloc_shared((num_stages,) + a_tile, dtype)
                a_smem_bot = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), c_dtype) if cast else c_local_0
                c_cast_1 = T.alloc_fragment((half_m, block_n), c_dtype) if cast else c_local_1
                c_smem_0 = T.alloc_shared((half_m, sn), c_dtype)
                c_smem_1 = T.alloc_shared((half_m, sn), c_dtype)

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
                            c_local_0: native_acc_layout(
                                c_local_0, half_m, block_n, block_k, dtype, "float32", acc_first
                            ),
                            c_local_1: native_acc_layout(
                                c_local_1,
                                half_m,
                                block_n,
                                block_k,
                                dtype,
                                "float32",
                                2 * acc_first,
                            ),
                        }
                    )

                ab_full = T.alloc_barrier([32] * num_stages)
                ab_empty = T.alloc_barrier([2 * WARPGROUP_THREADS] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                # nt[0]: column tile, or first column under narrow tiles; nt[1]: narrow.
                nt = T.alloc_local((2 if mixed else 1,), "int32")

                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
                    T.dec_max_nreg(24)
                    # One producer warp avoids redundant barrier work under the power cap.
                    if tx < 32:
                        for w in T.serial(max_waves):
                            flat_id = T.int32(ctas) * w + pid
                            if flat_id < total_tiles:
                                _narrow_grid_tile(
                                    flat_id, mt, nt, grid=grid, group_size_m=group_size_m
                                )
                                m_start = mt[0] * block_m
                                n_start = nt[0] if mixed else nt[0] * block_n
                                for ki in T.Pipelined(k_iters, num_stages=0):
                                    slot = gi_prod % num_stages
                                    ks = ki * block_k
                                    T.barrier_wait(
                                        ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1
                                    )
                                    load_a(a, a_smem_top, ab_full, slot, m_start, ks)
                                    load_a(a, a_smem_bot, ab_full, slot, m_start + half_m, ks)
                                    if mixed and nt[1] == 1:
                                        load_b(b, b_smem, ab_full, slot, n_start, ks, narrow_n)
                                    else:
                                        load_b(b, b_smem, ab_full, slot, n_start, ks, block_n)
                                    T.barrier_arrive(ab_full[slot])
                                    gi_prod = gi_prod + 1

                elif tx < 2 * WARPGROUP_THREADS:
                    consumer(
                        0,
                        a_smem_top,
                        c_local_0,
                        c_cast_0,
                        c_smem_0,
                        ps0,
                        gi_cons_0,
                        c,
                        bias,
                        b_smem,
                        ab_full,
                        ab_empty,
                        mt,
                        nt,
                        pid,
                    )
                else:
                    consumer(
                        1,
                        a_smem_bot,
                        c_local_1,
                        c_cast_1,
                        c_smem_1,
                        ps1,
                        gi_cons_1,
                        c,
                        bias,
                        b_smem,
                        ab_full,
                        ab_empty,
                        mt,
                        nt,
                        pid,
                    )

        return _main(
            "_gemm_coop2",
            _gemm_coop2_body,
            (((k, m) if trans_a else (m, k), dtype), ((n, k) if trans_b else (k, n), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_coop2_func


@target_cache(maxsize=32)
def _gemm_coop2_swap_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Persistent GEMM over 256 x ``block_n`` tiles of ``C.T = op(B).T @ op(A).T``.

    Two consumers each hold 128 rows of the transposed tile, that is 128 columns
    of C; ``block_n`` is the tile's extent in M and need not divide 256. C is stored
    through TMA, so its rows must be whole 16-byte units. ``sm_count`` is part of the
    cache key.

    With ``narrow_n`` set, ``row_mix = (split, wide_a, narrow_a, wide_b, narrow_b)``
    covers each 256-row block of the first ``split`` with ``wide_a`` tiles
    ``block_n`` wide then ``narrow_a`` tiles ``narrow_n`` wide, and the other blocks
    with ``wide_b`` and ``narrow_b``; every wide tile runs before every narrow one, each
    run of blocks in raster groups of ``group_size_m``. With ``trans_a``, a ``block_n`` that
    is a multiple of 64 stages ``op(A).T`` in 64-column swizzle atoms, so ``narrow_n`` must
    be a multiple of 64.
    *has_bias* builds the program taking ``bias[n]`` before C. C is in *dtype*:
    ``stmatrix.trans`` stores 16-bit elements only, so an *out_dtype* other than *dtype*
    is refused.

    Returns:
        JIT factory taking ``(block_n, num_stages, group_size_m, narrow_n, row_mix)``."""
    if out_dtype not in (None, dtype):
        raise ValueError(
            f"coop2_swap stores C through stmatrix.trans, {dtype} only, not {out_dtype}"
        )
    if n % 8:
        raise ValueError(f"coop2_swap stores C through TMA and needs n % 8 == 0, got n={n}")
    accum_dtype = "float"
    block_k, store_m = 64, COOP2_SWAP_STAGING[1]
    a_mn, b_mn = ("true" if mn else "false" for mn in (not trans_b, trans_a))
    wgmma_dtype = "kBFloat16" if dtype == "bfloat16" else "kFloat16"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
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
        ctas = sm_count if narrow_n else _persistent_ctas(total_tiles, sm_count)
        max_waves = -(-total_tiles // ctas) + 1
        k_iters = T.ceildiv(k, block_k)
        wgmma = f"tileops::dense_wgmma_128xN_k64<tl::DataType::{wgmma_dtype}, {block_n}, {a_mn}, {b_mn}>"
        narrow_wgmma = (
            f"tileops::dense_wgmma_128xN_k64<tl::DataType::{wgmma_dtype}, {narrow_n}, {a_mn},"
            f" {b_mn}, {block_n}>"
        )
        store = f"tileops::dense_store_trans_64<{store_m}>"
        a_half = (128, block_k) if trans_b else (block_k, 128)
        b_tile = (block_k, block_n) if trans_a else (block_n, block_k)

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
            T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
            T.call_extern(
                "handle", store, lo.data, T.address_of(c_smem[0, 0, 0]), ch * store_m // 8
            )
            T.call_extern(
                "handle", store, hi.data, T.address_of(c_smem[1, 0, 0]), ch * store_m // 8
            )
            T.fence_proxy_async()
            T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
            for q in T.unroll(2):
                T.tma_copy(
                    c_smem[q, :, :],
                    c[c0 : c0 + store_m, row_start + q * 64 : row_start + q * 64 + 64],
                )

        @T.macro
        def consumer(
            c, bias, a_smem, b_smem, full, empty, lo, hi, c_smem, gi, ps, pid, mt, nt, half, bar
        ):
            """One consumer's 128 rows of every tile it owns, mainloop then transposed store."""
            for w in T.serial(max_waves):
                flat_id = T.int32(ctas) * w + pid
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
                    if has_bias:
                        _add_bias(lo, bias, row_start, n, by_row=True, helper_layout=True)
                        _add_bias(hi, bias, row_start + 64, n, by_row=True, helper_layout=True)
                    for ch in T.unroll(block_n // store_m):
                        if ch < narrow_n // store_m or not narrow_n or nt[1] == 0:
                            c0 = col_start + ch * store_m
                            store_chunk(c, lo, hi, c_smem, c0, row_start, ch, bar)
            T.tma_store_wait(0)

        @T.macro
        def _gemm_coop2_swap_body(a, b, bias, c):
            with T.Kernel(ctas, threads=3 * WARPGROUP_THREADS) as (pid,):
                a_smem_top = T.alloc_shared((num_stages,) + a_half, dtype)
                a_smem_bot = T.alloc_shared((num_stages,) + a_half, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                lo_0 = T.alloc_fragment((64, block_n), accum_dtype)
                hi_0 = T.alloc_fragment((64, block_n), accum_dtype)
                lo_1 = T.alloc_fragment((64, block_n), accum_dtype)
                hi_1 = T.alloc_fragment((64, block_n), accum_dtype)
                c_smem_0 = T.alloc_shared(COOP2_SWAP_STAGING, dtype)
                c_smem_1 = T.alloc_shared(COOP2_SWAP_STAGING, dtype)
                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                        **{
                            f: wgmma_acc_layout(f, 64, block_n, block_k, dtype, "float32")
                            for f in (lo_0, hi_0, lo_1, hi_1)
                        },
                    }
                )
                full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                empty = T.alloc_barrier([2 * WARPGROUP_THREADS] * num_stages)
                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_local((1,), "int32")
                gi_cons_1 = T.alloc_local((1,), "int32")
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((2 if narrow_n else 1,), "int32")

                tx = T.get_thread_binding()
                if tx < WARPGROUP_THREADS:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(ctas) * w + pid
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
                elif tx < 2 * WARPGROUP_THREADS:
                    T.inc_max_nreg(240)
                    gi_cons_0[0] = 0
                    consumer(
                        c,
                        bias,
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
                        _CONSUMER_BAR_WG0,
                    )
                else:
                    T.inc_max_nreg(240)
                    gi_cons_1[0] = 0
                    consumer(
                        c,
                        bias,
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
                        _CONSUMER_BAR_WG1,
                    )

        return _main(
            "_gemm_coop2_swap",
            _gemm_coop2_swap_body,
            (((k, m) if trans_a else (m, k), dtype), ((n, k) if trans_b else (k, n), dtype)),
            ((m, n), dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_coop2_swap_func


@target_cache(maxsize=32)
def _gemm_pingpong_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Persistent NT GEMM with two consumers alternating whole 128-row tiles, one's
    epilogue overlapping the other's mainloop.

    C is stored through TMA, so its rows must be whole 16-byte units. *has_bias* builds
    the program taking ``bias[n]`` before C. C is *out_dtype*, or *dtype* when it is
    ``None``. With ``narrow_n`` set, the last ``narrow_rows`` row tiles start with
    ``narrow_cols`` tiles ``narrow_n`` wide that run after every wide tile.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages, group_size_m, stage_n,
        epi_bufs, narrow_n, narrow_rows, narrow_cols)``."""
    if trans_a or not trans_b:
        raise ValueError("_gemm_pingpong_kernel is NT-only (trans_a=False, trans_b=True)")
    c_dtype = out_dtype or dtype
    # Every tile leaves through the TMA store, which addresses output rows only in whole
    # 16-byte units.
    if (n * STR_TO_DTYPE[c_dtype].itemsize) % 16:
        raise ValueError(f"_gemm_pingpong_kernel needs {c_dtype} rows of 16-byte units, got n={n}")
    accum_dtype = "float"
    block_m = 128
    half_m = WGMMA_ROWS
    # ``dense_wgmma_helper.h``'s name for each dtype.
    wgmma_dtype_tag = {"bfloat16": "bf16", "float16": "fp16"}[dtype]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
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
        nr = (half_m * block_n) // WARPGROUP_THREADS
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

        def helper(narrow):
            return narrow_helper if narrow else wgmma_helper if wide_wgmma else None

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
                # A narrow WGMMA writes a prefix of the wide one's accumulator registers.
                _wgmma_step(a_top, b_smem, slot, acc_top, ki, helper(narrow), False, True)
                _wgmma_step(a_bot, b_smem, slot, acc_bot, ki, helper(narrow), False, True)
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
            T.sync_threads(barrier_id=epilogue_bar, arrive_count=WARPGROUP_THREADS)
            T.copy(acc_top[:, c0 : c0 + sn], c_smem[ch % epi_bufs, 0:half_m, :])
            T.copy(acc_bot[:, c0 : c0 + sn], c_smem[ch % epi_bufs, half_m:block_m, :])
            T.fence_proxy_async()
            T.sync_threads(barrier_id=epilogue_bar, arrive_count=WARPGROUP_THREADS)
            T.tma_copy(
                c_smem[ch % epi_bufs, :, :],
                c[m_start : m_start + block_m, n_start + c0 : n_start + c0 + sn],
            )

        @T.macro
        def consumer(
            wg,
            pid,
            c,
            bias,
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
                    if has_bias:
                        _add_bias(acc_top, bias, n_start, n, helper_layout=wide_wgmma)
                        _add_bias(acc_bot, bias, n_start, n, helper_layout=wide_wgmma)
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

        @T.macro
        def _gemm_pingpong_body(a, b, bias, c):
            with T.Kernel(sm_count, threads=3 * WARPGROUP_THREADS) as (pid,):
                a_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                acc_top_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_bot_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_top_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                acc_bot_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_smem_0 = T.alloc_shared((epi_bufs, block_m, sn), c_dtype)
                c_smem_1 = T.alloc_shared((epi_bufs, block_m, sn), c_dtype)
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
                            f: wgmma_acc_layout(f, half_m, block_n, block_k, dtype, "float32")
                            for f in (acc_top_0, acc_bot_0, acc_top_1, acc_bot_1)
                        }
                    )
                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                # One per consumer: completes a phase when the other consumer hands it
                # the tensor cores.
                turn = T.alloc_barrier([WARPGROUP_THREADS] * 2)
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

                if tx < WARPGROUP_THREADS:
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
                elif tx < 2 * WARPGROUP_THREADS:
                    T.inc_max_nreg(240)
                    consumer(
                        0,
                        pid,
                        c,
                        bias,
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
                        _CONSUMER_BAR_WG0,
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
                        bias,
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
                        _CONSUMER_BAR_WG1,
                    )

        return _main(
            "_gemm_pingpong",
            _gemm_pingpong_body,
            (((m, k), dtype), ((n, k), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_pingpong_func


@target_cache(maxsize=32)
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
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _gemm_coop2_splitk_func(
        block_n: int = 64, block_k: int = 128, num_stages: int = 4, split_k: int = 4
    ) -> Callable:
        half_m = block_m // 2
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // WARPGROUP_THREADS
        k_iters_total = T.ceildiv(k, block_k)
        if k_iters_total % split_k != 0:
            raise ValueError(
                f"split_k={split_k} must divide the K-tile count evenly "
                f"(k={k}, block_k={block_k} -> {k_iters_total} tiles)"
            )
        k_slice = k_iters_total // split_k

        @T.macro
        def consumer(wg, a_smem, acc, prev, b_smem, ab_full, ab_empty, w, m_start, n_start, bz):
            """Consumer warpgroup *wg*: the ``half_m`` rows from ``wg * half_m`` of the tile."""
            T.inc_max_nreg(240)
            for ki in T.Pipelined(k_slice, num_stages=0):
                slot = ki % num_stages
                T.barrier_wait(ab_full[slot], (ki // num_stages) % 2)
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
            T.wait_wgmma(0)
            T.barrier_arrive(ab_empty[prev[0]])
            T.warpgroup_fence_operand(acc, num_regs=nr)
            for i, j in T.Parallel(half_m, block_n):
                if m_start + wg * half_m + i < m and n_start + j < n:
                    w[bz, m_start + wg * half_m + i, n_start + j] = acc[i, j]

        @T.prim_func
        def _gemm_coop2_splitk_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=3 * WARPGROUP_THREADS
            ) as (
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
                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([2 * WARPGROUP_THREADS] * num_stages)

                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice
                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
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
                elif tx < 2 * WARPGROUP_THREADS:
                    consumer(
                        0,
                        a_smem_top,
                        c_local_0,
                        ps0,
                        b_smem,
                        ab_full,
                        ab_empty,
                        w,
                        m_start,
                        n_start,
                        bz,
                    )
                else:
                    consumer(
                        1,
                        a_smem_bot,
                        c_local_1,
                        ps1,
                        b_smem,
                        ab_full,
                        ab_empty,
                        w,
                        m_start,
                        n_start,
                        bz,
                    )

        return _gemm_coop2_splitk_main

    return _gemm_coop2_splitk_func


@target_cache(maxsize=32)
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
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> tuple[Callable, Callable]:
    """Resolve the (mainloop, reduce) compiled pair for a split-K config; *has_bias*
    builds the reduction that adds ``bias[n]``, and the reduction writes *out_dtype*.

    Both resolve in one cached lookup because host time between the two launches is
    GPU idle.
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
    reduce_ = splitk_reduce_kernel(split_k, m, n, dtype, activation, has_bias, out_dtype)
    return mainloop, reduce_(elems_per_cta)


@target_cache(maxsize=32)
def _gemm_simple_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """``T.Pipelined`` + ``T.gemm`` GEMM without warp specialization, for short mainloops.

    Tiles must divide the problem exactly; the builder raises ``ValueError`` otherwise.
    *has_bias* builds the program taking ``bias[n]`` before C. C is *out_dtype*, or *dtype*
    when it is ``None``.
    """
    if trans_a:
        raise ValueError("_gemm_simple_kernel supports trans_a=False only")
    accum_dtype = "float"
    c_dtype = out_dtype or dtype

    @tilelang.jit(out_idx=[-1], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
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

        @T.macro
        def _gemm_simple_body(a, b, bias, c):
            with _launch() as (bx, by):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_smem = T.alloc_shared((block_m, block_n), c_dtype)
                T.clear(c_local)
                for ki in T.Pipelined(k // block_k, num_stages=num_stages):
                    T.copy(a[by * block_m, ki * block_k], a_smem)
                    if trans_b:
                        T.copy(b[bx * block_n, ki * block_k], b_smem, eviction_policy=b_evict)
                    else:
                        T.copy(b[ki * block_k, bx * block_n], b_smem, eviction_policy=b_evict)
                    T.gemm(a_smem, b_smem, c_local, transpose_B=trans_b)
                if has_bias:
                    _add_bias(c_local, bias, bx * block_n, n)
                # One TMA store from shared memory, not per-thread global stores.
                T.copy(c_local, c_smem)
                T.copy(c_smem, c[by * block_m, bx * block_n])

        return _main(
            "_gemm_simple",
            _gemm_simple_body,
            (((m, k), dtype), ((n, k) if trans_b else (k, n), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_simple_func


@target_cache(maxsize=32)
def _gemm_swap_ab_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Tiny-M NT GEMM computing ``C.T = B @ A.T`` to avoid padding M to 64.

    Serves ``m <= SWAP_AB_MPAD``. *has_bias* builds the program taking ``bias[n]`` before C. C
    is *out_dtype*, or *dtype* when it is ``None``.

    Returns:
        JIT factory taking ``(block_nn, block_k, num_stages)``."""
    if trans_a or not trans_b:
        raise ValueError("_gemm_swap_ab_kernel is NT-only (trans_a=False, trans_b=True)")
    if m > SWAP_AB_MPAD:
        raise ValueError(f"_gemm_swap_ab_kernel serves m <= {SWAP_AB_MPAD}, got m={m}")
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    # A float32 C stores the accumulator as it is.
    cast = c_dtype != "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
    def _gemm_swap_ab_func(block_nn: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        mpad = SWAP_AB_MPAD

        @T.macro
        def _gemm_swap_ab_body(a, b, bias, c):
            with T.Kernel(T.ceildiv(n, block_nn), threads=128) as bx:
                b_smem = T.alloc_shared((block_nn, block_k), dtype)
                a_smem = T.alloc_shared((mpad, block_k), dtype)
                ct_local = T.alloc_fragment((block_nn, mpad), accum_dtype)
                ct_cast = T.alloc_fragment((block_nn, mpad), c_dtype) if cast else ct_local
                ct_smem = T.alloc_shared((block_nn, mpad), c_dtype)
                T.clear(ct_local)
                # Each CTA reads its rows of ``b`` once, so they are evicted first; ``a`` is
                # what L2 keeps.
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bx * block_nn, ki * block_k], b_smem, eviction_policy="evict_first")
                    T.copy(a[0, ki * block_k], a_smem)
                    T.gemm(b_smem, a_smem, ct_local, transpose_B=True)
                if has_bias:
                    _add_bias(ct_local, bias, bx * block_nn, n, by_row=True)
                if cast:
                    T.copy(ct_local, ct_cast)
                # The cast tile is written back transposed, through shared memory.
                T.copy(ct_cast, ct_smem)
                for mi, j in T.Parallel(mpad, block_nn):
                    if mi < m and bx * block_nn + j < n:
                        c[mi, bx * block_nn + j] = ct_smem[j, mi]

        return _main(
            "_gemm_swap_ab",
            _gemm_swap_ab_body,
            (((m, k), dtype), ((n, k), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_swap_ab_func


@target_cache(maxsize=32)
def _gemm_stream_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Small-M NT GEMM streaming ``B`` one 64-row block at a time: ``C.T = B @ A.T``.

    Serves ``m <= 256``. Each CTA runs ``blocks`` consecutive 64-row blocks of ``B``, each
    over all of K. *has_bias* builds the program taking ``bias[n]`` before C. C is
    *out_dtype*, or *dtype* when it is ``None``.

    Returns:
        JIT factory taking ``(block_k, num_stages, blocks)``."""
    if trans_a or not trans_b:
        raise ValueError("_gemm_stream_kernel is NT-only (trans_a=False, trans_b=True)")
    # A's rows are the WGMMA's N, at most 256.
    mpad, helper_width = stream_width(m)
    if mpad > 256:
        raise ValueError(f"_gemm_stream_kernel serves m <= 256, got m={m}")
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    # The output tile stores through TMA, which needs 16-byte output rows.
    tma_epilogue = (n * STR_TO_DTYPE[c_dtype].itemsize) % 16 == 0
    wgmma_dtype = "kBFloat16" if dtype == "bfloat16" else "kFloat16"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", *csrc_include("dense_wgmma_helper.h")],
    )
    def _gemm_stream_func(block_k: int, num_stages: int, blocks: int) -> Callable:
        # One WGMMA's M: the rows of B one block streams.
        rows = 64
        ctas = T.ceildiv(T.ceildiv(n, rows), blocks)
        k_iters = T.ceildiv(k, block_k)
        # One TMA copy per 128-byte swizzle atom of K: a window of a two-atom swizzled tile
        # is not contiguous in shared memory.
        atom = min(block_k, 128 // STR_TO_DTYPE[dtype].itemsize)
        num_accum_regs = rows * mpad // WARPGROUP_THREADS
        # The helper issues one 64-wide K chunk a call.
        native_wgmma = helper_width and block_k % 64 == 0
        native_helper = f"tileops::dense_wgmma_64xN_k64<tl::DataType::{wgmma_dtype}, {mpad}>"
        # A float32 C is staged from the accumulator itself.
        cast = c_dtype != "float32"
        acc_first = 0 if cast else WARPGROUP_THREADS

        @T.macro
        def _gemm_stream_body(a, b, bias, c):
            with T.Kernel(ctas, threads=256) as bx:
                b_smem = T.alloc_shared((num_stages, rows, block_k), dtype)
                a_smem = T.alloc_shared((num_stages, mpad, block_k), dtype)
                ct = T.alloc_fragment((rows, mpad), accum_dtype)
                c_smem = T.alloc_shared((mpad, rows), c_dtype)
                if native_wgmma:
                    ct_smem = T.alloc_shared((rows, mpad), c_dtype)
                    # The accumulator layout T.wgmma_gemm gives a 64 x mpad tile; the helper
                    # writes the same registers. Its first thread follows the copy below, as
                    # in _gemm_kernel: 0 for stmatrix, the consumer's 128 for a float32 loop.
                    T.annotate_layout(
                        {ct: native_acc_layout(ct, rows, mpad, 64, dtype, "float32", acc_first)}
                    )
                elif cast:
                    ct_cast = T.alloc_fragment((rows, mpad), c_dtype)
                else:
                    ct_cast = ct
                T.annotate_layout(
                    {
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                    }
                )
                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ps = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
                    T.dec_max_nreg(24)
                    for s in T.serial(blocks):
                        n_start = (bx * blocks + s) * rows
                        for ki in T.serial(k_iters):
                            step = s * k_iters + ki
                            slot = step % num_stages
                            phase = (step // num_stages) % 2
                            k_start = ki * block_k
                            T.barrier_wait(ab_empty[slot], phase ^ 1)
                            if n_start < n:
                                for kc in T.unroll(block_k // atom):
                                    k0 = k_start + kc * atom
                                    # Each block reads its rows of B once.
                                    T.tma_copy(
                                        b[n_start : n_start + rows, k0 : k0 + atom],
                                        b_smem[slot, :, kc * atom : kc * atom + atom],
                                        barrier=ab_full[slot],
                                        eviction_policy="evict_first",
                                    )
                                    T.tma_copy(
                                        a[0:mpad, k0 : k0 + atom],
                                        a_smem[slot, :, kc * atom : kc * atom + atom],
                                        barrier=ab_full[slot],
                                    )
                            T.barrier_arrive(ab_full[slot])
                else:
                    T.inc_max_nreg(240)
                    for s in T.serial(blocks):
                        n_start = (bx * blocks + s) * rows
                        for ki in T.serial(k_iters):
                            step = s * k_iters + ki
                            slot = step % num_stages
                            phase = (step // num_stages) % 2
                            T.barrier_wait(ab_full[slot], phase)
                            if native_wgmma:
                                # One commit group per 64 of K.
                                for kc in T.unroll(block_k // 64):
                                    T.call_extern(
                                        "handle",
                                        native_helper,
                                        ct.data,
                                        T.address_of(b_smem[slot, 0, kc * 64]),
                                        T.address_of(a_smem[slot, 0, kc * 64]),
                                        T.if_then_else(ki + kc > 0, 1, 0),
                                    )
                            else:
                                T.wgmma_gemm(
                                    b_smem[slot, :, :],
                                    a_smem[slot, :, :],
                                    ct,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                            if ki > 0:
                                T.wait_wgmma(block_k // 64 if native_wgmma else 1)
                                T.warpgroup_fence_operand(ct, num_regs=num_accum_regs)
                                T.barrier_arrive(ab_empty[ps[0]])
                            ps[0] = slot
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(ct, num_regs=num_accum_regs)
                        T.barrier_arrive(ab_empty[ps[0]])
                        if has_bias:
                            _add_bias(ct, bias, n_start, n, by_row=True, helper_layout=native_wgmma)
                        if native_wgmma:
                            # A transposed read of the helper's accumulator makes ptxas
                            # serialize its WGMMAs (C7514), so it goes through a plain shared
                            # copy first.
                            T.copy(ct, ct_smem)
                            T.sync_threads(barrier_id=4, arrive_count=WARPGROUP_THREADS)
                            for mi, j in T.Parallel(mpad, rows):
                                c_smem[mi, j] = ct_smem[j, mi]
                        else:
                            if cast:
                                T.copy(ct, ct_cast)
                            for j, mi in T.Parallel(rows, mpad):
                                c_smem[mi, j] = ct_cast[j, mi]
                        if tma_epilogue:
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=WARPGROUP_THREADS)
                            if n_start < n:
                                T.copy(c_smem, c[0, n_start])
                        else:
                            T.sync_threads(barrier_id=4, arrive_count=WARPGROUP_THREADS)
                            for mi, j in T.Parallel(mpad, rows):
                                if mi < m and n_start + j < n:
                                    c[mi, n_start + j] = c_smem[mi, j]
                        # The next block's tile reuses c_smem.
                        T.sync_threads(barrier_id=4, arrive_count=WARPGROUP_THREADS)

        return _main(
            "_gemm_stream",
            _gemm_stream_body,
            (((m, k), dtype), ((n, k), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_stream_func


@target_cache(maxsize=32)
def _gemm_coop2s_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Single-tile NN GEMM with two consumers splitting A and sharing B.

    Tiles must divide the problem exactly; one tile per CTA. *has_bias* builds the program
    taking ``bias[n]`` before C. C is *out_dtype*, or *dtype* when it is ``None``; a
    float32 C is staged from the accumulator itself.

    Returns:
        JIT factory taking ``(block_n, block_k, num_stages)``."""
    if trans_a or trans_b:
        raise ValueError("_gemm_coop2s_kernel is NN-only (trans_a=False, trans_b=False)")
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    cast = c_dtype != "float32"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _gemm_coop2s_func(block_n: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"_gemm_coop2s_kernel requires exact tiling: got "
                f"m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}"
            )
        half_m = block_m // 2
        b_evict = _b_eviction(m, block_m)
        nr = (half_m * block_n) // WARPGROUP_THREADS
        k_iters = k // block_k

        @T.macro
        def consumer(
            wg,
            a_smem,
            acc,
            acc_cast,
            c_smem,
            prev,
            b_smem,
            ab_full,
            ab_empty,
            bias,
            c,
            m_start,
            n_start,
        ):
            """Consumer warpgroup *wg*: the ``half_m`` rows from ``wg * half_m`` of the tile."""
            T.inc_max_nreg(240)
            bar = _CONSUMER_BAR_WG1 if wg else _CONSUMER_BAR_WG0
            for ki in T.serial(k_iters):
                slot = ki % num_stages
                phase = (ki // num_stages) % 2
                T.barrier_wait(ab_full[slot], phase)
                T.wgmma_gemm(
                    a_smem[slot, :, :],
                    b_smem[slot, :, :],
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=(ki == 0),
                )
                if ki > 0:
                    T.wait_wgmma(1)
                    T.barrier_arrive(ab_empty[prev[0]])
                    T.warpgroup_fence_operand(acc, num_regs=nr)
                prev[0] = slot
            T.wait_wgmma(0)
            T.warpgroup_fence_operand(acc, num_regs=nr)
            T.barrier_arrive(ab_empty[prev[0]])
            if has_bias:
                _add_bias(acc, bias, n_start, n)
            if cast:
                T.copy(acc, acc_cast)
            T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
            T.copy(acc_cast, c_smem)
            T.fence_proxy_async()
            T.sync_threads(barrier_id=bar, arrive_count=WARPGROUP_THREADS)
            T.copy(c_smem, c[m_start + wg * half_m, n_start])

        @T.macro
        def _gemm_coop2s_body(a, b, bias, c):
            with T.Kernel(n // block_n, m // block_m, threads=3 * WARPGROUP_THREADS) as (bx, by):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_k, block_n), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), c_dtype) if cast else c_local_0
                c_cast_1 = T.alloc_fragment((half_m, block_n), c_dtype) if cast else c_local_1
                c_smem_0 = T.alloc_shared((half_m, block_n), c_dtype)
                c_smem_1 = T.alloc_shared((half_m, block_n), c_dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([WARPGROUP_THREADS] * num_stages)
                ab_empty = T.alloc_barrier([2 * WARPGROUP_THREADS] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < WARPGROUP_THREADS:
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
                elif tx < 2 * WARPGROUP_THREADS:
                    consumer(
                        0,
                        a_smem_top,
                        c_local_0,
                        c_cast_0,
                        c_smem_0,
                        ps0,
                        b_smem,
                        ab_full,
                        ab_empty,
                        bias,
                        c,
                        m_start,
                        n_start,
                    )
                else:
                    consumer(
                        1,
                        a_smem_bot,
                        c_local_1,
                        c_cast_1,
                        c_smem_1,
                        ps1,
                        b_smem,
                        ab_full,
                        ab_empty,
                        bias,
                        c,
                        m_start,
                        n_start,
                    )

        return _main(
            "_gemm_coop2s",
            _gemm_coop2s_body,
            (((m, k), dtype), ((k, n), dtype)),
            ((m, n), c_dtype),
            ((n,), dtype) if has_bias else None,
        )

    return _gemm_coop2s_func


class GEMMTMAKernel(Kernel, GEMMFwdInterface):
    """SM90 dense GEMM ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout.

    The default structure is one TMA producer and one WGMMA consumer warpgroup; a
    structure flag in ``config`` (one of ``_STRUCTURE_FLAGS``) or ``split_k > 1`` selects
    another. ``activation="silu_and_mul"`` requires a split-K config and returns
    ``[M, N / 2]``. fp16 / bf16 inputs, fp32 accumulation. The output is ``out_dtype``,
    or the input dtype when it is ``None``; coop2_swap writes the input dtype only.
    Claims only shapes whose operands TMA can address; ``GEMMCpAsyncKernel`` takes the
    rest.
    """

    supported_archs: list[int] = [90]

    _STRUCTURE_FLAGS = (
        "coop2",
        "coop2s",
        "coop2_splitk",
        "coop2_swap",
        "pingpong",
        "simple",
        "stream",
        "swap_ab",
    )

    # The basic tile a board without a calibration runs, and the schema an unflagged
    # caller config fills where the selector picked a config outside that schema.
    _MODAL_CONFIG = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "num_stages": 4,
        "panel_size": 16,
        "split_k": 1,
    }

    def init_config(self, config: dict | None = None) -> None:
        """Take a flagged config verbatim, and fill an unflagged one into the basic schema.

        An unflagged config states a basic or split-K tile, so it fills the modal basic tile
        where the selector's config has other keys: a flagged structure's, or the cluster of
        a fill tile.
        """
        if config is not None and any(config.get(f) for f in self._STRUCTURE_FLAGS):
            self.config = dict(config)
            return
        selected = self.default_config
        if config is not None and selected.keys() != self._MODAL_CONFIG.keys():
            selected = self._MODAL_CONFIG
        self.config = (
            selected
            if config is None
            else {k: config[k] if config.get(k) is not None else v for k, v in selected.items()}
        )

    @classmethod
    def refusal(cls, call: GEMMCall) -> str | None:
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
        config: dict | None = None,
        trans_a: bool = False,
        trans_b: bool = False,
        device_index: int | None = None,
        activation: str = "none",
        out_dtype: torch.dtype | None = None,
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
        self.out_dtype = out_dtype or dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.activation = activation
        self.sm_count = self.build_target.sm_count
        self.init_config(config)
        if activation != "none":
            split_k = self.config.get("split_k", 1)
            # Only the split-K structures reduce through the kernel that fuses the activation.
            unsupported = any(
                self.config.get(flag) for flag in self._STRUCTURE_FLAGS if flag != "coop2_splitk"
            )
            if split_k <= 1 or unsupported:
                raise ValueError("a fused activation requires a split-K GEMM config")

    @property
    def default_config(self) -> dict:
        scored = best_config(
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            self.sm_count,
            self.build_target.calibration,
            # The raster group of a persistent tile depends on what L2 holds.
            l2_bytes=self.build_target.l2_bytes,
            out_bytes=self.out_dtype.itemsize,
        )
        return scored if scored is not None else dict(self._MODAL_CONFIG)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # The traced build is another program, chosen by the switch at launch.
        biases = () if bias is None else (bias,)
        return self.program(self._launcher, trace.enabled, bias is not None)(a, b, *biases)

    def _launcher(self, cfg: dict, traced: bool, has_bias: bool) -> Callable:
        """Build the configured GEMM launcher with optional bias.

        Split-K launchers include the FP32 reduction. Tracing is available only
        for the default decomposition."""
        m, n, k, trans_a, trans_b, dtype, out = (
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            self.dtype_str,
            self.out_dtype_str,
        )
        if cfg.get("simple"):
            return _gemm_simple_kernel(m, n, k, trans_a, trans_b, dtype, has_bias, out)(
                cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg.get("threads", 128),
                cfg.get("panel_size", 8),
                cfg.get("cluster_m", 1),
            )
        if cfg.get("swap_ab"):
            return _gemm_swap_ab_kernel(m, n, k, trans_a, trans_b, dtype, has_bias, out)(
                cfg["block_nn"], cfg["block_k"], cfg["num_stages"]
            )
        if cfg.get("stream"):
            return _gemm_stream_kernel(m, n, k, trans_a, trans_b, dtype, has_bias, out)(
                cfg["block_k"], cfg["num_stages"], cfg["blocks"]
            )
        if cfg.get("coop2s"):
            return _gemm_coop2s_kernel(
                m,
                n,
                k,
                trans_a,
                trans_b,
                dtype,
                has_bias=has_bias,
                out_dtype=out,
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
                has_bias=has_bias,
                out_dtype=out,
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
                has_bias=has_bias,
                out_dtype=out,
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
                has_bias=has_bias,
                out_dtype=out,
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
                has_bias,
                out,
            )
            out_n = n // 2 if self.activation != "none" else n
            mainloop, reduce_ = CompiledProgram(mainloop), CompiledProgram(reduce_)
            out_dtype = self.out_dtype

            def splitk(a: torch.Tensor, b: torch.Tensor, *bias: torch.Tensor) -> torch.Tensor:
                c = torch.empty((m, out_n), dtype=out_dtype, device=a.device)
                reduce_(mainloop(a, b), *bias, c)
                return c

            return splitk
        main_cfg = {key: value for key, value in cfg.items() if key != "split_k"}
        compiled = _gemm_kernel(
            m,
            n,
            k,
            trans_a,
            trans_b,
            dtype,
            traced=traced,
            has_bias=has_bias,
            out_dtype=out,
        )(**main_cfg)
        if not passes.is_traced(compiled):
            return compiled
        stem = f"gemm_{m}x{n}x{k}_{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}_{dtype}"
        launch = CompiledProgram(compiled)
        return lambda *args: trace.run(launch, args, stem=stem, traced=True)


@target_cache(maxsize=32)
def _gemm_small_batch_kernel(
    m: int, n: int, k: int, dtype: str = "float16", bias_len: int = 0, out_dtype: str | None = None
) -> Callable:
    """Small-M NT GEMM reusing each streamed B tile across all M rows, reducing over K
    on CUDA cores.

    A nonzero *bias_len* builds the program taking ``bias[bias_len]`` before C, added to
    each fp32 sum after the K reduction: ``n`` adds one per column, 1 adds ``bias[0]``
    to every output. C is *out_dtype*, or *dtype* when it is ``None``.

    Returns:
        JIT factory taking ``(block_n, reduce_threads, num_stages)`` and producing C."""
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    flags = [
        *CUDA_COMPILE_FLAGS,
        "-DENABLE_BF16",
        *(csrc_include("streaming_load.h") if bias_len else ()),
    ]

    @tilelang.jit(out_idx=[-1], compile_flags=flags)
    def _gemm_small_batch_func(
        block_n: int = 1,
        reduce_threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        tile_k = 128 // (STR_TO_DTYPE[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k
        b_evict = "evict_first"  # one N range per CTA, so B is read once

        @T.macro
        def _gemm_small_batch_body(a, b, bias, c):
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                c_accum = T.alloc_local((m,), accum_dtype)
                T.clear(c_accum)
                if bias_len:
                    col = 0 if bias_len == 1 else T.min(bn * block_n + tn, n - 1)
                    bias_f = T.alloc_local((1,), accum_dtype)
                    bias_f[0] = T.cast(0, accum_dtype)
                    if tk == 0:
                        bias_f[0] = T.call_extern(
                            accum_dtype, "tileops::load_before_barrier_f32", T.address_of(bias[col])
                        )
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
                    # block_n need not divide n.
                    if bn * block_n + tn < n:
                        if bias_len:
                            if tk == 0:
                                c[mi, bn * block_n + tn] = c_reduced[0] + bias_f[0]
                        else:
                            c[mi, bn * block_n + tn] = c_reduced[0]

        return _main(
            "_gemm_small_batch",
            _gemm_small_batch_body,
            (((m, k), dtype), ((n, k), dtype)),
            ((m, n), c_dtype),
            ((bias_len,), dtype) if bias_len else None,
        )

    return _gemm_small_batch_func


@target_cache(maxsize=32)
def _gemv_row_kernel(
    n: int, k: int, dtype: str = "float16", bias_len: int = 0, out_dtype: str | None = None
) -> Callable:
    """One-row GEMV, ``c[0, :] = a[0, :] @ b.T``, one warp per ``rows`` rows of ``b``.

    Each lane loads its 16-byte slices of a row straight into registers, ``depth`` slices
    per burst, and adds their products into ``reduce_threads / 32`` float32 sums, one per
    reduction lane it stands in for: reduction lane ``t`` sums K positions
    ``[i * reduce_threads * 8 + t * 8, +8)`` in order of ``i``, then the lanes are added
    pairwise from the farthest. That is the order :func:`_gemm_small_batch_kernel` adds in
    with ``reduce_threads`` threads, so both give the same bits. Buffer extents predicate
    the K-tail loads, which read zero. A nonzero *bias_len* adds ``bias[col]``, or
    ``bias[0]`` at 1, after the reduction; C is *out_dtype*, or *dtype* when it is ``None``.

    Returns:
        JIT factory taking ``(reduce_threads, warps, rows, depth)``, ``reduce_threads`` a
        power-of-two number of warps and ``depth`` a multiple of that number, and producing C.
    """
    accum_dtype = "float"
    c_dtype = out_dtype or dtype
    vec = 128 // (STR_TO_DTYPE[dtype].itemsize * 8)
    flags = [*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"]

    @tilelang.jit(out_idx=[-1], compile_flags=flags)
    def _gemv_row_func(
        reduce_threads: int = 128, warps: int = 1, rows: int = 1, depth: int = 32
    ) -> Callable:
        groups = reduce_threads // WARP_LANES
        if reduce_threads % WARP_LANES or groups & (groups - 1) or depth % groups:
            raise ValueError(
                f"reduce_threads {reduce_threads} must be a power-of-two number of warps and "
                f"depth {depth} a multiple of it"
            )
        block_k = reduce_threads * vec
        slices = -(-k // block_k) * groups
        burst = min(depth, slices)
        bursts, tail = divmod(slices, burst)
        tile_rows = warps * rows
        folds = tuple(groups >> (i + 1) for i in range(groups.bit_length() - 1))

        def slice_col(s, lane):
            return (s // groups) * block_k + (s % groups * WARP_LANES + lane) * vec

        @T.macro
        def load_slice(b, b_local, row0, col, j):
            for r in T.unroll(rows, explicit=True):
                row = T.min(row0 + r, n - 1)
                for e in T.vectorized(vec):
                    b_local[r, j, e] = b[row, col + e]

        @T.macro
        def add_slice(a, b_local, acc, col, j):
            a_local = T.alloc_local((vec,), dtype)
            for e in T.vectorized(vec):
                a_local[e] = a[0, col + e]
            for r in T.unroll(rows, explicit=True):
                for e in T.unroll(vec, explicit=True):
                    acc[r, j % groups] = acc[r, j % groups] + T.cast(
                        a_local[e], accum_dtype
                    ) * T.cast(b_local[r, j, e], accum_dtype)

        @T.macro
        def _gemv_row_body(a, b, bias, c):
            with T.Kernel(T.ceildiv(n, tile_rows), threads=warps * WARP_LANES) as bx:
                tx = T.get_thread_binding(0)
                lane = tx % WARP_LANES
                row0 = bx * tile_rows + tx // WARP_LANES * rows
                b_local = T.alloc_local((rows, burst, vec), dtype)
                acc = T.alloc_local((rows, groups), accum_dtype)
                T.clear(acc)
                for j in T.unroll(burst, explicit=True):
                    load_slice(b, b_local, row0, slice_col(j, lane), j)
                if bursts > 1:
                    for ib in T.serial(bursts - 1):
                        for j in T.unroll(burst, explicit=True):
                            add_slice(a, b_local, acc, slice_col(ib * burst + j, lane), j)
                            load_slice(b, b_local, row0, slice_col((ib + 1) * burst + j, lane), j)
                for j in T.unroll(burst, explicit=True):
                    add_slice(a, b_local, acc, slice_col((bursts - 1) * burst + j, lane), j)
                    if j < tail:
                        load_slice(b, b_local, row0, slice_col(bursts * burst + j, lane), j)
                if tail:
                    for j in T.unroll(tail, explicit=True):
                        add_slice(a, b_local, acc, slice_col(bursts * burst + j, lane), j)
                for r in T.unroll(rows, explicit=True):
                    for g in T.unroll(groups, explicit=True):
                        for st in T.unroll(WARP_SHUFFLE_STAGES, explicit=True):
                            acc[r, g] = acc[r, g] + T.shfl_xor(acc[r, g], (WARP_LANES // 2) >> st)
                    for fold in folds:
                        for g in T.unroll(fold, explicit=True):
                            acc[r, g] = acc[r, g] + acc[r, g + fold]
                    if lane == 0 and row0 + r < n:
                        if bias_len:
                            col_bias = 0 if bias_len == 1 else row0 + r
                            c[0, row0 + r] = T.cast(
                                acc[r, 0] + T.cast(bias[col_bias], accum_dtype), c_dtype
                            )
                        else:
                            c[0, row0 + r] = T.cast(acc[r, 0], c_dtype)

        return _main(
            "_gemv_row",
            _gemv_row_body,
            (((1, k), dtype), ((n, k), dtype)),
            ((1, n), c_dtype),
            ((bias_len,), dtype) if bias_len else None,
        )

    return _gemv_row_func


class GemvKernel(Kernel, GEMMFwdInterface):
    """GEMV and two-row GEMM on :func:`_gemm_small_batch_kernel`.

    Serves three bands: the two layouts a vector operand can take, and SM90 ``m == 2``
    NT calls where ``two_row_gemv_wins``. A one-row band over rows ``gemv_registers``
    admits reads them into registers with :func:`_gemv_row_kernel`, which adds a row's
    products in the ring's order.

    Args:
        band: Which band this instance serves, one of :attr:`BANDS`.
        m: Rows of the product.
        n: Columns of the product.
        k: Contraction dim.
        dtype: Input torch dtype (fp16 / bf16); fp32 accumulation.
        config: Optional explicit config; defaults to :attr:`default_config`.
        device_index: Device whose SM count and name pick the config.
        out_dtype: Output torch dtype; ``None`` for *dtype*.
    """

    supported_archs: list[int] = [89, 90]
    # A vector operand is TMA-addressable as often as any other shape, so this band
    # overlaps the warp-specialized kernel; reducing over K on CUDA cores wins there.
    preferred_over = frozenset({"gemm_tma"})

    #: The bands this class serves. ``lhs_row`` and ``rhs_col`` name the vector
    #: operand; ``lhs_rows`` is the two-row NT band.
    BANDS: tuple[str, ...] = ("lhs_row", "rhs_col", "lhs_rows")

    @classmethod
    def _vector_operand(cls, call: GEMMCall) -> str | None:
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
    def band_for(cls, call: GEMMCall) -> str | None:
        """The band serving *call*, or ``None`` when this class does not serve it."""
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
        index = call.device_index
        band = cls.band_for(call)
        identity = (band, call.m, call.n, call.k, call.dtype, call.out_dtype, index)
        return identity, lambda: cls(
            band,
            call.m,
            call.n,
            call.k,
            call.dtype,
            device_index=index,
            out_dtype=call.out_dtype,
        )

    def __init__(
        self,
        band: str,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: dict | None = None,
        device_index: int | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if band not in self.BANDS:
            raise ValueError(f"{type(self).__name__} serves bands {self.BANDS}, got {band!r}")
        self.band = band
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype or dtype
        # What the body produces: the other operand's free dim, one row per row of ``a``.
        self.rows, self.out_len = (
            (m, n) if band == "lhs_rows" else (1, n if band == "lhs_row" else m)
        )
        self.staged = band == "lhs_rows" or not gemv_registers(k)
        self.kernel = self._builder(bias_len=0)
        self.init_config(config)

    def _builder(self, bias_len: int) -> Callable:
        """The JIT factory of this band's body, taking ``bias[bias_len]`` when nonzero."""
        if self.staged:
            return _gemm_small_batch_kernel(
                self.rows,
                self.out_len,
                self.k,
                self.dtype_str,
                bias_len=bias_len,
                out_dtype=self.out_dtype_str,
            )
        return _gemv_row_kernel(
            self.out_len, self.k, self.dtype_str, bias_len=bias_len, out_dtype=self.out_dtype_str
        )

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        if self.band == "lhs_rows":
            return small_batch_config(self.n, self.k, self.build_target.sm_count)
        return gemv_config(self.k)

    @property
    def autotune_configs(self) -> list[dict]:
        """Configs within the block's thread cap whose staged ``B`` ring, 8 K elements of 2
        bytes per reduce thread and stage, fits the shared-memory budget."""
        if self.band == "lhs_rows":
            rts, bns, nss = (32, 64, 128), (1, 2, 4), (2, 3, 4, 5)
        elif self.staged:
            rts, bns, nss = (32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6)
        else:
            return [
                {"reduce_threads": rt, "warps": warps, "rows": rows, "depth": depth}
                for rt in (32, 64, 128, 256)
                for warps in (1, 2, 4)
                for rows in (1, 2, 4)
                for depth in (16, 32)
            ]
        smem_cap = self.build_target.smem_budget
        return [
            {"block_n": bn, "reduce_threads": rt, "num_stages": ns}
            for rt in rts
            for bn in bns
            if rt * bn <= MAX_BLOCK_THREADS
            for ns in nss
            if bn * rt * 8 * 2 * ns <= smem_cap
        ]

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        kernel = self.program(self._compile, bias is not None)
        biases = () if bias is None else (bias,)
        if self.band == "lhs_rows":
            return kernel(a, b, *biases)
        vector, matrix = (a, b) if self.band == "lhs_row" else (b, a)
        out = kernel(vector.reshape(1, -1).contiguous(), matrix, *biases)
        return out.reshape(1, self.n) if self.band == "lhs_row" else out.reshape(self.m, 1)

    def _compile(self, cfg: dict, has_bias: bool) -> Callable:
        """Compile this band's body with optional column bias.

        The ``rhs_col`` band broadcasts ``bias[0]`` over its output column."""
        builder = self._builder(self.n) if has_bias else self.kernel
        if self.staged:
            return builder(cfg["block_n"], cfg["reduce_threads"], cfg["num_stages"])
        return builder(cfg["reduce_threads"], cfg["warps"], cfg["rows"], cfg["depth"])


@target_cache(maxsize=32)
def _gemm_basic_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    has_bias: bool = False,
    out_dtype: str | None = None,
) -> Callable:
    """Pipelined dense GEMM ``C = op(A) @ op(B)`` for any tensor-core target.

    All four ``(trans_a, trans_b)`` layouts; compiles on sm80 / sm86 / sm89 / sm90.
    *has_bias* builds a ``split_k == 1`` program taking ``bias[n]`` before C; split
    slices leave the bias to the reduction. A ``split_k == 1`` C is *out_dtype*, or
    *dtype* when it is ``None``; ``split_k > 1`` writes fp32 ``[split_k, m, n]``.
    """
    accum_dtype = "float"
    c_dtype = out_dtype or dtype

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
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
        output_dtype = c_dtype if split_k == 1 else accum_dtype

        @T.macro
        def _gemm_basic_body(a, b, bias, c):
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

                smem = [a_smem, b_smem]
                if split_k == 1:
                    c_smem = T.alloc_shared((block_m, block_n), c_dtype)
                    smem.append(c_smem)
                T.annotate_layout({buf: tilelang.layout.make_swizzled_layout(buf) for buf in smem})

                # Panel-swizzled raster for L2 reuse.
                T.use_swizzle(10, enable=True)

                T.clear(c_local)
                m_start = by * block_m
                n_start = bx * block_n

                for ki in T.Pipelined(k_slice, num_stages=num_stages):
                    k_start = (bz * k_slice + ki) * block_k
                    # M/N tail reads land in c_local rows/cols that the epilogue guard
                    # skips; T.copy zero-fills a K tail.
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
                        # NN with an N tail: the b tile's N rows can be narrower than
                        # the 4/8/16-byte cp_async accesses, so mask the copy instead.
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
                    if has_bias:
                        _add_bias(c_local, bias, n_start, n)
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

        return _main(
            "_gemm_basic",
            _gemm_basic_body,
            ((a_shape, dtype), (b_shape, dtype)),
            (output_shape, output_dtype),
            ((n,), dtype) if has_bias and split_k == 1 else None,
        )

    return _gemm_basic_func


class GEMMCpAsyncKernel(Kernel, GEMMFwdInterface):
    """Pipelined dense GEMM ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout,
    sm80+.

    The general implementation: it loads through ``cp.async``, which addresses any
    extent, so it serves every shape no other kernel claims. fp16 / bf16 inputs, fp32
    accumulation, an output in ``out_dtype`` or, when it is ``None``, the input dtype.
    M / N / K need not be multiples of the block sizes. A config with ``split_k > 1``
    returns the reduced ``[M, N]`` result and requires exact M / N / K tiles.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    @staticmethod
    def _narrow_row(axis: str, extent: int, dtype: torch.dtype) -> str | None:
        """Why an operand row of *extent* elements along *axis* is too narrow for one
        vectorized load, or ``None``.

        ``_gemm_basic_kernel`` loads each operand's innermost (contiguous) dimension in
        4-byte units, so a row shorter than that is rejected by the backend. K need not be
        16-aligned beyond this — the backend zero-pads K tails.
        """
        if extent * dtype.itemsize >= 4:
            return None
        return (
            f"the pipelined mainloop loads its innermost dimension in 4-byte units, so "
            f"{axis} must span at least one; {axis}={extent} of a {dtype.itemsize}-byte "
            f"dtype does not"
        )

    @classmethod
    def refusal(cls, call: GEMMCall) -> str | None:
        # At k=0 the operands hold no element and the mainloop runs no K step, so no row is
        # loaded and the program writes the bias, or zeros.
        if call.k == 0:
            return None
        reason = cls._narrow_row("k", call.k, call.dtype)
        if reason is None:
            return None
        # A transposed A and non-transposed B have contiguous M and N axes;
        # a short contraction does not make their physical rows short.
        a_row = ("m", call.m) if call.trans_a else ("k", call.k)
        b_row = ("k", call.k) if call.trans_b else ("n", call.n)
        return cls._narrow_row(*min(a_row, b_row, key=lambda row: row[1]), call.dtype)

    @classmethod
    def entry_for(cls, call: GEMMCall) -> Entry:
        return _dense_entry(cls, call)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: dict | None = None,
        trans_a: bool = False,
        trans_b: bool = False,
        device_index: int | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype or dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

        self.kernel = _gemm_basic_kernel(
            m, n, k, trans_a, trans_b, self.dtype_str, out_dtype=self.out_dtype_str
        )

        self.init_config(config)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": self._block_ks()[0],
            "num_stages": 2,
            "threads": 128,
            "split_k": 1,
        }

    @property
    def autotune_configs(self) -> list[dict]:
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
            for bk in self._block_ks()
            for ns in [2, 3, 4]
        ]

    def _block_ks(self) -> list[int]:
        """The K steps that divide k, widest first; otherwise 16, the mma.sync floor, whose
        K tail the loads zero-fill."""
        return [bk for bk in (64, 32) if self.k % bk == 0] or [16]

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        biases = () if bias is None else (bias,)
        return self.program(self._launcher, bias is not None)(a, b, *biases)

    def _launcher(self, cfg: dict, has_bias: bool) -> Callable:
        """Build the pipelined GEMM launcher with optional bias.

        Split-K bias is applied once in the FP32 reduction."""
        split_k = cfg.get("split_k", 1)
        builder = self.kernel
        out = self.out_dtype_str
        if has_bias and split_k == 1:
            builder = _gemm_basic_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str, has_bias, out
            )
        mainloop = builder(**cfg)
        if split_k == 1:
            return mainloop
        mainloop = CompiledProgram(mainloop)
        reduce_ = CompiledProgram(
            splitk_reduce_kernel(
                split_k, self.m, self.n, self.dtype_str, has_bias=has_bias, out_dtype=out
            )()
        )
        m, n, out_dtype = self.m, self.n, self.out_dtype

        def split(a: torch.Tensor, b: torch.Tensor, *bias: torch.Tensor) -> torch.Tensor:
            result = mainloop(a, b)
            output = a.new_empty((m, n), dtype=out_dtype)
            reduce_(result, *bias, output)
            return output

        return split
