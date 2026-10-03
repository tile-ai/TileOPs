"""MoE adapter for the shared persistent grouped GEMM template, and its MMA counterpart."""

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops.kernels.elementwise._erf import erf
from tileops.kernels.gemm.persistent.heuristics import ACTIVATIONS, GemmType
from tileops.kernels.gemm.persistent.template import GemmTemplate
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, MGroupedGemmFwdInterface
from tileops.manifest.primitives import moe_layout_metadata

__all__ = ["MoeGroupedGemmKernel", "MoeGroupedGemmMmaKernel"]


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
) -> Callable:
    """One ``T.gemm`` tile per block over the rows of one group.

    Masked and per-row layouts map a block to a tile directly. The prefix-sum layouts
    number each group's tiles after the previous groups', so a block walks the group ends
    to find its own; ``ceil(m / block_m)`` plus one per group bounds the count, and the
    blocks past it do nothing.
    """
    accum_dtype = "float"
    gtype = GemmType(gemm_type)
    masked = gtype is GemmType.M_GROUPED_MASKED
    per_row = gtype is GemmType.M_GROUPED_ALIGNED_PER_ROW
    tight = gtype is GemmType.M_GROUPED_TIGHT_PSUM
    fused = activation != "none"
    c_cols = n // 2 if fused else n

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
        layout_shape = (m,) if per_row else (num_groups,)
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
                    start = prev_end[0] if tight else T.ceildiv(prev_end[0], alignment) * alignment
                    tiles = T.ceildiv(T.max(layout[g] - start, 0), block_m)
                    if (group[0] < 0) & (bx < seen[0] + tiles):
                        group[0] = g
                        row0[0] = start + (bx - seen[0]) * block_m
                        row_end[0] = layout[g]
                    seen[0] += tiles
                    prev_end[0] = layout[g]

        def activate(gate, up):
            """act(gate) * up in float32, as the SM90 template's epilogue computes it."""
            half = T.cast(0.5, accum_dtype)
            if activation == "silu_and_mul":
                return gate * (half + half * T.tanh(half * gate)) * up
            inv_sqrt2 = T.cast(0.7071067811865476, accum_dtype)
            return half * gate * (T.cast(1.0, accum_dtype) + erf(gate * inv_sqrt2, cd_dtype)) * up

        @T.prim_func
        def _moe_grouped_gemm_mma_main(
            A: T.Tensor(a_shape, ab_dtype),  # type: ignore
            B: T.Tensor((num_groups, n, k), ab_dtype),  # type: ignore
            layout: T.Tensor(layout_shape, "int32"),  # type: ignore
            C: T.Tensor(c_shape, cd_dtype),  # type: ignore
        ) -> None:
            # M tiles vary fastest, so the blocks reading one group's weight tile run together.
            with T.Kernel(m_tiles, T.ceildiv(c_cols, tile_n), threads=threads) as (bx, by):
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
                        T.copy(B[g, n0 : n0 + tile_n, k0 : k0 + block_k], B_s)
                        T.gemm(A_s, B_s, acc, transpose_B=True)
                        if fused:
                            # The up projection sits c_cols rows past the gate in B.
                            T.copy(B[g, c_cols + n0 : c_cols + n0 + tile_n, k0 : k0 + block_k], U_s)
                            T.gemm(A_s, U_s, acc_up, transpose_B=True)
                    if fused:
                        for i, j in T.Parallel(block_m, tile_n):
                            C_s[i, j] = T.cast(activate(acc[i, j], acc_up[i, j]), cd_dtype)
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

        return _moe_grouped_gemm_mma_main

    return _moe_grouped_gemm_mma_func


class MoeGroupedGemmKernel(Kernel, MGroupedGemmFwdInterface):
    """Adapt staged MoE grouped-GEMM calls to the shared GEMM template."""

    supported_archs: list[int] = [90]
    # Where both run, a caller's replacement of this key wins over the MMA kernel.
    preferred_over = frozenset({"grouped_gemm_mma"})

    _TYPES: dict[tuple[str, Optional[str], Optional[str]], GemmType] = {
        ("contiguous", "tight", "physical_psum"): GemmType.M_GROUPED_TIGHT_PSUM,
        ("contiguous", "aligned", "physical_psum"): GemmType.M_GROUPED_ALIGNED_PSUM,
        ("contiguous", "aligned", "per_row"): GemmType.M_GROUPED_ALIGNED_PER_ROW,
        ("masked", None, None): GemmType.M_GROUPED_MASKED,
    }
    _ALIGNED_TILE_HEIGHTS = (64, 128, 256)

    @classmethod
    def applies(cls, call: MGroupedGemmCall) -> bool:
        n_step = 8 if call.activation is None else 16
        return (
            (call.kind, call.packing, call.metadata_kind) in cls._TYPES
            and call.ab_dtype in (torch.bfloat16, torch.float16)
            and call.cd_dtype in (call.ab_dtype, torch.float32)
            and (call.packing != "aligned" or call.alignment in cls._ALIGNED_TILE_HEIGHTS)
            and (call.activation is None or call.activation in ACTIVATIONS)
            and call.k % 8 == 0
            and call.n % n_step == 0
        )

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
    ) -> torch.Tensor:
        """Run each expert's grouped product, including a fused activation when requested."""
        return self.inner(a, b, grouped_layout=layout_metadata, out=out)


class MoeGroupedGemmMmaKernel(MoeGroupedGemmKernel):
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
        """Supply autotuning the call's own rows, split over the groups as the manifest's
        workloads split them, so each candidate walks the tiles a real call does."""
        from tilelang.utils.device import get_current_device

        call = self.call
        masked = call.kind == "masked"
        rows = call.num_groups * call.max_m if masked else call.m
        lead = (call.num_groups, call.max_m) if masked else (rows,)
        cols = call.n // 2 if call.activation not in (None, "none") else call.n

        def supply_prog(params: list) -> list:
            if len(params) != 4:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects 4 parameters (a, b, layout, c), "
                    f"got {len(params)}"
                )
            device = get_current_device()
            metadata = moe_layout_metadata(call, rows, call.num_groups)
            return [
                torch.randn(*lead, call.k, dtype=call.ab_dtype, device=device),
                torch.randn(call.num_groups, call.n, call.k, dtype=call.ab_dtype, device=device),
                torch.tensor(metadata, dtype=torch.int32, device=device),
                torch.empty(*lead, cols, dtype=call.cd_dtype, device=device),
            ]

        return supply_prog

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run each expert's grouped product, including a fused activation when requested."""
        call = self.call
        cols = call.n // 2 if call.activation not in (None, "none") else call.n
        if out is None:
            rows = (call.num_groups, call.max_m) if call.kind == "masked" else (a.shape[0],)
            out = torch.empty((*rows, cols), dtype=call.cd_dtype, device=a.device)
        if out.numel() == 0:
            return out
        self.kernel(**self.config)(a, b, layout_metadata, out)
        return out
