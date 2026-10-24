"""MoE adapter for the shared persistent grouped GEMM template, and its MMA counterpart."""

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops.kernels.buffer_utils import tensors_overlap
from tileops.kernels.elementwise._erf import erf
from tileops.kernels.gemm.persistent.heuristics import (
    ACTIVATIONS,
    INTERLEAVED_ACTIVATIONS,
    SWIGLUOAI_ALPHA,
    SWIGLUOAI_LIMIT,
    GemmType,
)
from tileops.kernels.gemm.persistent.template import GemmTemplate
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, MGroupedGemmFwdInterface
from tileops.manifest.primitives import moe_layout_metadata

__all__ = ["MoEGroupedGemmKernel", "MoEGroupedGemmMMAKernel"]


@functools.lru_cache(maxsize=32)
def _moe_grouped_gemm_mma_kernel(
    gemm_type: str,
    num_groups: int,
    n: int,
    k: int,
    alignment: int,
    max_m: int,
    activation: str,
    ab_dtype: str,
    cd_dtype: str,
    has_bias: bool,
) -> Callable:
    """One ``T.gemm`` tile per block over the rows of one group.

    Masked and aligned per-row layouts map a block to a tile directly. The prefix-sum
    layouts number each group's tiles after the previous groups', so a block walks the
    group ends to find its own; the tight per-row layout walks them too, finding each end
    by binary search over the sorted row ids. ``ceil(m / block_m)`` plus one per group
    bounds the count, and the blocks past it do nothing.
    """
    accum_dtype = "float"
    gtype = GemmType(gemm_type)
    masked = gtype is GemmType.M_GROUPED_MASKED
    per_row = gtype is GemmType.M_GROUPED_ALIGNED_PER_ROW
    tight_per_row = gtype is GemmType.M_GROUPED_TIGHT_PER_ROW
    tight = gtype in (GemmType.M_GROUPED_TIGHT_PSUM, GemmType.M_GROUPED_TIGHT_PER_ROW)
    fused = activation != "none"
    c_cols = n // 2 if fused else n
    # An interleaved B (row 2j gate j, row 2j + 1 up j) is taken as [G, n / 2, 2, k], and
    # its bias as [G, n / 2, 2]; a stacked one holds the up rows c_cols past the gate rows.
    interleaved = activation in INTERLEAVED_ACTIVATIONS
    b_shape = (num_groups, c_cols, 2, k) if interleaved else (num_groups, n, k)
    bias_shape = (num_groups, c_cols, 2) if interleaved else (num_groups, n)

    @tilelang.jit(
        compile_flags=["-O3", "-DENABLE_BF16"],
        # Under WS the producer reads its own copy of the group index, which find_tile
        # never writes, so B would be loaded from an arbitrary group.
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def _moe_grouped_gemm_mma_func(
        block_m: int, block_n: int, block_k: int, num_stages: int, threads: int
    ) -> Callable:
        m = T.dynamic("m")
        a_shape = (num_groups, max_m, k) if masked else (m, k)
        c_shape = (num_groups, max_m, c_cols) if masked else (m, c_cols)
        layout_shape = (m,) if per_row or tight_per_row else (num_groups,)
        tile_n = block_n // 2 if fused else block_n
        tiles_per_slab = tilelang.cdiv(max_m, block_m) if masked else 0
        if masked:
            m_tiles = num_groups * tiles_per_slab
        elif per_row:
            m_tiles = T.ceildiv(m, block_m)
        else:
            m_tiles = T.ceildiv(m, block_m) + num_groups

        @T.macro
        def find_tile(layout, bx, group, row0, row_end, seen, prev_end):
            """The group, first row and row bound of block *bx*'s tile; group -1 for none."""
            group[0] = -1
            if masked:
                row0[0] = bx % tiles_per_slab * block_m
                row_end[0] = T.min(layout[bx // tiles_per_slab], max_m)
                if row0[0] < row_end[0]:
                    group[0] = bx // tiles_per_slab
            elif per_row:
                # Segments start on a multiple of the alignment, which block_m divides,
                # so a tile holds the rows of one group and its padding.
                row0[0] = bx * block_m
                row_end[0] = m
                if (layout[row0[0]] >= 0) & (layout[row0[0]] < num_groups):
                    group[0] = layout[row0[0]]
            else:
                seen[0] = 0
                prev_end[0] = 0
                for g in T.serial(num_groups):
                    if not tight_per_row or group[0] < 0:
                        if tight_per_row:
                            end = T.alloc_var("int32", init=0)
                            lo = T.alloc_var("int32", init=0)
                            hi = T.alloc_var("int32", init=m)
                            while lo < hi:
                                mid = (lo + hi) >> T.int32(1)
                                if layout[mid] <= g:
                                    lo = mid + T.int32(1)
                                else:
                                    hi = mid
                            end = lo
                        start = (
                            prev_end[0] if tight else T.ceildiv(prev_end[0], alignment) * alignment
                        )
                        tiles = T.ceildiv(
                            T.max((end if tight_per_row else layout[g]) - start, 0), block_m
                        )
                        if (group[0] < 0) & (bx < seen[0] + tiles):
                            group[0] = g
                            row0[0] = start + (bx - seen[0]) * block_m
                            row_end[0] = end if tight_per_row else layout[g]
                        seen[0] += tiles
                        prev_end[0] = end if tight_per_row else layout[g]

        def activate(gate, up):
            """act(gate, up) in float32."""
            half = T.cast(0.5, accum_dtype)
            if activation == "silu_and_mul":
                return gate * (half + half * T.tanh(half * gate)) * up
            if activation == "swigluoai":
                # swigluoai as MoEGroupedGemmFwdOp defines it, the sigmoid in silu's tanh form.
                limit = T.cast(SWIGLUOAI_LIMIT, accum_dtype)
                clamped_gate = T.min(gate, limit)
                clamped_up = T.max(T.min(up, limit), -limit)
                sig = half + half * T.tanh(T.cast(SWIGLUOAI_ALPHA / 2, accum_dtype) * clamped_gate)
                return clamped_gate * sig * (clamped_up + T.cast(1.0, accum_dtype))
            inv_sqrt2 = T.cast(0.7071067811865476, accum_dtype)
            return half * gate * (T.cast(1.0, accum_dtype) + erf(gate * inv_sqrt2, cd_dtype)) * up

        def b_rows(B, g, n0, part, k0):
            """*tile_n* gate (*part* 0) or up (*part* 1) rows of group *g* at column *n0*."""
            if interleaved:
                return B[g, n0 : n0 + tile_n, part, k0 : k0 + block_k]
            row0 = n0 if part == 0 else c_cols + n0
            return B[g, row0 : row0 + tile_n, k0 : k0 + block_k]

        def bias_of(Bias, g, col, part):
            """The bias of output column *col*'s gate (*part* 0) or up (*part* 1) row."""
            col = T.min(col, c_cols - 1)  # columns past the output are not stored
            if not fused:
                return T.cast(Bias[g, col], accum_dtype)
            if interleaved:
                return T.cast(Bias[g, col, part], accum_dtype)
            return T.cast(Bias[g, col + part * c_cols], accum_dtype)

        @T.macro
        def tile_body(A, B, layout, C, Bias, bx, by):
            """One block's tile: find its group, run the K loop, apply the epilogue, store."""
            A_s = T.alloc_shared((block_m, block_k), ab_dtype)
            B_s = T.alloc_shared((tile_n, block_k), ab_dtype)
            C_s = T.alloc_shared((block_m, tile_n), cd_dtype)
            acc = T.alloc_fragment((block_m, tile_n), accum_dtype)
            if fused:
                U_s = T.alloc_shared((tile_n, block_k), ab_dtype)
                acc_up = T.alloc_fragment((block_m, tile_n), accum_dtype)
            group = T.alloc_local([1], "int32")
            row0 = T.alloc_local([1], "int32")
            row_end = T.alloc_local([1], "int32")
            seen = T.alloc_local([1], "int32")
            prev_end = T.alloc_local([1], "int32")

            find_tile(layout, bx, group, row0, row_end, seen, prev_end)
            if group[0] >= 0:
                g = group[0]
                r0 = row0[0]
                n0 = by * tile_n
                T.clear(acc)
                if fused:
                    T.clear(acc_up)
                for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k0 = ko * block_k
                    if masked:
                        T.copy(A[g, r0 : r0 + block_m, k0 : k0 + block_k], A_s)
                    else:
                        T.copy(A[r0 : r0 + block_m, k0 : k0 + block_k], A_s)
                    T.copy(b_rows(B, g, n0, 0, k0), B_s)
                    T.gemm(A_s, B_s, acc, transpose_B=True)
                    if fused:
                        T.copy(b_rows(B, g, n0, 1, k0), U_s)
                        T.gemm(A_s, U_s, acc_up, transpose_B=True)
                if fused and has_bias:
                    for i, j in T.Parallel(block_m, tile_n):
                        C_s[i, j] = T.cast(
                            activate(
                                acc[i, j] + bias_of(Bias, g, n0 + j, 0),
                                acc_up[i, j] + bias_of(Bias, g, n0 + j, 1),
                            ),
                            cd_dtype,
                        )
                elif fused:
                    for i, j in T.Parallel(block_m, tile_n):
                        C_s[i, j] = T.cast(activate(acc[i, j], acc_up[i, j]), cd_dtype)
                elif has_bias:
                    for i, j in T.Parallel(block_m, tile_n):
                        C_s[i, j] = T.cast(acc[i, j] + bias_of(Bias, g, n0 + j, 0), cd_dtype)
                else:
                    T.copy(acc, C_s)
                # C_s is written and read under different thread mappings.
                T.sync_threads()
                for i, j in T.Parallel(block_m, tile_n):
                    if (r0 + i < row_end[0]) & (n0 + j < c_cols):
                        if masked:
                            C[g, r0 + i, n0 + j] = C_s[i, j]
                        else:
                            C[r0 + i, n0 + j] = C_s[i, j]

        # The bias is a kernel argument only where the call passes one.
        if has_bias:

            @T.prim_func
            def _moe_grouped_gemm_mma_main(
                A: T.Tensor(a_shape, ab_dtype),  # type: ignore
                B: T.Tensor(b_shape, ab_dtype),  # type: ignore
                layout: T.Tensor(layout_shape, "int32"),  # type: ignore
                C: T.Tensor(c_shape, cd_dtype),  # type: ignore
                Bias: T.Tensor(bias_shape, ab_dtype),  # type: ignore
            ) -> None:
                # M tiles vary fastest, so the blocks reading one group's weight tile run
                # together.
                with T.Kernel(m_tiles, T.ceildiv(c_cols, tile_n), threads=threads) as (bx, by):
                    tile_body(A, B, layout, C, Bias, bx, by)

        else:

            @T.prim_func
            def _moe_grouped_gemm_mma_main(
                A: T.Tensor(a_shape, ab_dtype),  # type: ignore
                B: T.Tensor(b_shape, ab_dtype),  # type: ignore
                layout: T.Tensor(layout_shape, "int32"),  # type: ignore
                C: T.Tensor(c_shape, cd_dtype),  # type: ignore
            ) -> None:
                # M tiles vary fastest, so the blocks reading one group's weight tile run
                # together.
                with T.Kernel(m_tiles, T.ceildiv(c_cols, tile_n), threads=threads) as (bx, by):
                    tile_body(A, B, layout, C, None, bx, by)

        return _moe_grouped_gemm_mma_main

    return _moe_grouped_gemm_mma_func


class MoEGroupedGemmKernel(Kernel, MGroupedGemmFwdInterface):
    """Adapt staged MoE grouped-GEMM calls to the shared GEMM template."""

    supported_archs: list[int] = [90]
    # Where both run, a caller's replacement of this key wins over the MMA kernel.
    preferred_over = frozenset({"grouped_gemm_mma"})

    _TYPES: dict[tuple[str, Optional[str], Optional[str]], GemmType] = {
        ("contiguous", "tight", "physical_psum"): GemmType.M_GROUPED_TIGHT_PSUM,
        ("contiguous", "tight", "per_row"): GemmType.M_GROUPED_TIGHT_PER_ROW,
        ("contiguous", "aligned", "physical_psum"): GemmType.M_GROUPED_ALIGNED_PSUM,
        ("contiguous", "aligned", "per_row"): GemmType.M_GROUPED_ALIGNED_PER_ROW,
        ("masked", None, None): GemmType.M_GROUPED_MASKED,
    }
    _ALIGNED_TILE_HEIGHTS = (64, 128, 256)

    @classmethod
    def refusal(cls, call: MGroupedGemmCall) -> Optional[str]:
        """The layouts, dtypes and extents the template instantiates."""
        if (call.kind, call.packing, call.metadata_kind) not in cls._TYPES:
            return f"does not serve the {call.kind} {call.packing} {call.metadata_kind} layout"
        if call.ab_dtype not in (torch.bfloat16, torch.float16):
            return f"serves bfloat16 and float16 operands, got {call.ab_dtype}"
        if call.cd_dtype not in (call.ab_dtype, torch.float32):
            return f"writes the operand dtype or float32, got {call.cd_dtype}"
        if call.packing == "aligned" and call.alignment not in cls._ALIGNED_TILE_HEIGHTS:
            return f"aligns groups to a tile height in {cls._ALIGNED_TILE_HEIGHTS}, got {call.alignment}"
        if call.activation is not None and call.activation not in ACTIVATIONS:
            return f"fuses only {sorted(ACTIVATIONS)}, got {call.activation}"
        if call.k % 8:
            return f"steps K by 8, got K = {call.k}"
        n_step = 8 if call.activation is None else 16
        if call.n % n_step:
            return f"needs N a multiple of {n_step} here, got N = {call.n}"
        return None

    def __init__(self, call: MGroupedGemmCall) -> None:
        device_index = call.device.index if call.device is not None else None
        super().__init__(device_index=device_index)
        self.call = call
        self.inner = GemmTemplate(
            self._TYPES[(call.kind, call.packing, call.metadata_kind)],
            num_groups=call.num_groups,
            m_alignment=call.alignment if call.packing == "aligned" else 128,
            cd_dtype=None if call.cd_dtype is call.ab_dtype else call.cd_dtype,
            activation="none" if call.activation is None else call.activation,
            device_index=device_index,
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run each expert's grouped product, with its bias and fused activation when requested."""
        return self.inner(a, b, grouped_layout=layout_metadata, out=out, bias=bias)


class MoEGroupedGemmMMAKernel(MoEGroupedGemmKernel):
    """The same grouped GEMM on MMA tiles, for GPUs without the SM90 template."""

    supported_archs: list[int] = [80, 86, 89]
    preferred_over = frozenset()

    def __init__(self, call: MGroupedGemmCall) -> None:
        device_index = call.device.index if call.device is not None else None
        Kernel.__init__(self, device_index=device_index)
        self.call = call
        self.kernel = _moe_grouped_gemm_mma_kernel(
            self._TYPES[(call.kind, call.packing, call.metadata_kind)].value,
            call.num_groups,
            call.n,
            call.k,
            call.alignment,
            call.max_m or 0,
            "none" if call.activation is None else call.activation,
            self.dtype_to_str(call.ab_dtype),
            self.dtype_to_str(call.cd_dtype),
            call.has_bias,
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        # block_m divides every aligned layout's segment height, so a tile never spans two
        # groups' segments.
        return {"block_m": 64, "block_n": 128, "block_k": 64, "num_stages": 2, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": 64, "block_n": n, "block_k": k, "num_stages": s, "threads": t}
            for n, k, s, t in itertools.product([64, 128], [32, 64], [2, 3], [128, 256])
        ]

    @property
    def autotune_supply_prog(self) -> Callable:
        return self._supply_prog_for(self.call)

    @classmethod
    def _supply_prog_for(cls, call: MGroupedGemmCall) -> Callable:
        """Supply autotuning the call's own rows, split over the groups as the manifest's
        workloads split them, so each candidate walks the tiles a real call does."""
        from tilelang.utils.device import get_current_device

        masked = call.kind == "masked"
        rows = call.num_groups * call.max_m if masked else call.m
        lead = (call.num_groups, call.max_m) if masked else (rows,)
        cols = call.n // 2 if call.activation not in (None, "none") else call.n
        # The kernel's own view of b and the bias: [G, n / 2, 2, ..] when interleaved.
        rows_b = (cols, 2) if call.activation in INTERLEAVED_ACTIVATIONS else (call.n,)
        names = ("a", "b", "layout", "c", "bias") if call.has_bias else ("a", "b", "layout", "c")

        def supply_prog(params: list) -> list:
            if len(params) != len(names):
                raise RuntimeError(
                    f"autotuning {cls.__name__} expects {len(names)} parameters "
                    f"({', '.join(names)}), got {len(params)}"
                )
            device = get_current_device()
            metadata = moe_layout_metadata(call, rows, call.num_groups)
            tensors = [
                torch.randn(*lead, call.k, dtype=call.ab_dtype, device=device),
                torch.randn(call.num_groups, *rows_b, call.k, dtype=call.ab_dtype, device=device),
                torch.tensor(metadata, dtype=torch.int32, device=device),
                torch.empty(*lead, cols, dtype=call.cd_dtype, device=device),
            ]
            if call.has_bias:
                tensors.append(
                    torch.randn(call.num_groups, *rows_b, dtype=call.ab_dtype, device=device)
                )
            return tensors

        return supply_prog

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run each expert's grouped product, with its bias and fused activation when requested."""
        call = self.call
        if (bias is not None) != call.has_bias:
            raise ValueError(f"this kernel was built with has_bias={call.has_bias}")
        cols = call.n // 2 if call.activation not in (None, "none") else call.n
        if out is None:
            rows = (call.num_groups, call.max_m) if call.kind == "masked" else (a.shape[0],)
            out = torch.empty((*rows, cols), dtype=call.cd_dtype, device=a.device)
        else:
            # Tiles store while other tiles still load, so out shares no byte with a tensor the
            # kernel reads; disjoint slices of one workspace buffer are accepted.
            reads = (("a", a), ("b", b), ("layout_metadata", layout_metadata), ("bias", bias))
            for name, t in reads:
                if t is not None and tensors_overlap(out, t):
                    raise ValueError(f"out must not overlap {name} in memory")
        if out.numel() == 0:
            return out
        if call.activation in INTERLEAVED_ACTIVATIONS:
            # The kernel takes an interleaved b and its bias as [.., n / 2, 2, ..].
            b = b.unflatten(1, (cols, 2))
            bias = None if bias is None else bias.unflatten(1, (cols, 2))
        kernel = self.kernel(**self.config)
        if bias is None:
            kernel(a, b, layout_metadata, out)
        else:
            kernel(a, b, layout_metadata, out, bias)
        return out
