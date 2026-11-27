"""MXFP4 grouped GEMM for decode-sized calls on SM90: few rows of ``a`` per expert.

One CTA computes one 64-row weight tile of one expert against a tile of up to eight of the
expert's rows, so every SM streams weights at once. The weights are decoded to e4m3 in
registers and contracted on an fp8 WGMMA, as ``moe_grouped_gemm_mxfp4``'s register-fragment
path does, step for step.
"""

from collections.abc import Callable
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops._csrc import csrc_include
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import (
    QUANT_SCALE_BLOCK,
    TMA_DTYPE_UINT8,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_L2_PROMOTION_256B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_64B,
    TMA_SWIZZLE_128B,
    TMA_SWIZZLE_NONE,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import MGroupedGEMMMXFP4Call, MGroupedGEMMMXFP4FwdInterface
from tileops.manifest.primitives import moe_layout_metadata

__all__ = ["MoEGroupedGEMMMXFP4DecodeKernel"]

_ROW_TILE = 8
_B_SCALE_BLOCK = 32
_CODE_STEPS = 4
_LOOKUP_ENDS = 16
_THREADS = WARPGROUP_THREADS + 32


def _decode_smem_bytes(num_stages: int, scale_k: int) -> int:
    """Bytes of a config: the ring of row and packed-weight stages, its code boxes (four K
    steps' e8m0 codes of the tile's rows each), the rows' token scales, the run lookup, the vote
    and the barriers."""
    ring = num_stages * (_ROW_TILE * QUANT_SCALE_BLOCK + WGMMA_ROWS * QUANT_SCALE_BLOCK // 2)
    codes = max(2, num_stages // _CODE_STEPS) * WGMMA_ROWS * _CODE_STEPS * 4
    tokens = _ROW_TILE * scale_k * 4
    return ring + codes + tokens + 32 + (2 * num_stages + 1) * 8


@target_cache(maxsize=32)
def _moe_grouped_gemm_mxfp4_decode_kernel(num_groups: int, n: int, k: int) -> Callable:
    """Build the decode MXFP4 grouped GEMM over tight contiguous rows, ``physical_psum``
    metadata and e8m0 scales.

    The grid is one CTA per row of ``a`` and 64-column weight tile; the CTA of row ``r`` runs
    when ``r`` starts a tile of eight of its expert's rows. A producer thread streams each K
    step's rows and packed weights, and every fourth step's weight codes, through a ring. The
    math warp-group folds each step as it decodes it (``mxfp4_decode_fold_step``) and checks
    that it may; a tile with a step that may not is streamed again and contracted sub-block by
    sub-block.
    """
    block_k = QUANT_SCALE_BLOCK
    packed_k = block_k // 2
    scale_k = k // block_k
    block_n = WGMMA_ROWS
    math_threads = WARPGROUP_THREADS
    vote_barrier = 13
    row_bytes = _ROW_TILE * block_k
    weight_bytes = block_n * packed_k
    code_bytes = block_n * _CODE_STEPS * 4

    @tilelang.jit(
        pass_configs={
            PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        },
        compile_flags=[
            *CUDA_COMPILE_FLAGS,
            "--use_fast_math",
            "-DENABLE_BF16",
            *csrc_include("moe_grouped_gemm_mxfp4_decode_helper.h"),
        ],
    )
    def _moe_grouped_gemm_mxfp4_decode_func(num_stages: int) -> Callable:
        m = T.dynamic("m")
        code_slots = max(2, num_stages // _CODE_STEPS)
        rounds = scale_k // num_stages

        @T.macro
        def issue(
            a_desc, b_desc, s_desc, a_shared, b_packed, codes, full, empty, row, n0, g, again
        ):
            """Stream the tile's K steps through the ring in round ``again``: each step's eight
            rows and packed weights, and every fourth step the codes of four steps, once the
            last step that read their box before has released it."""
            for kk in T.serial(scale_k):
                slot = kk % num_stages
                phase = (kk // num_stages + again * rounds) & 1
                reader = kk + again * scale_k - _CODE_STEPS * code_slots + _CODE_STEPS - 1
                if T.And(kk % _CODE_STEPS == 0, reader >= 0):
                    T.barrier_wait(empty[reader % num_stages], (reader // num_stages) & 1)
                T.barrier_wait(empty[slot], phase ^ 1)
                if kk % _CODE_STEPS == 0:
                    T.mbarrier_arrive_expect_tx(full[slot], row_bytes + weight_bytes + code_bytes)
                    T.tma_load(
                        s_desc,
                        full[slot],
                        T.address_of(codes[(kk + again * scale_k) // _CODE_STEPS % code_slots, 0]),
                        kk * _CODE_STEPS,
                        n0,
                        g,
                        0,
                    )
                else:
                    T.mbarrier_arrive_expect_tx(full[slot], row_bytes + weight_bytes)
                T.tma_load(
                    a_desc, full[slot], T.address_of(a_shared[slot, 0, 0]), kk * block_k, row, 0
                )
                T.tma_load(
                    b_desc, full[slot], T.address_of(b_packed[slot, 0, 0]), kk * packed_k, n0, g, 0
                )

        @T.macro
        def decode(fold, again, b_packed, codes, full, frags, scales, fits, kk):
            """K step ``kk``'s weight fragments and row scales once its stage has landed: folded,
            clearing ``fits`` unless the step may fold, or split by sub-block."""
            T.barrier_wait(full[kk % num_stages], (kk // num_stages + again * rounds) & 1)
            box = T.address_of(codes[(kk + again * scale_k) // _CODE_STEPS % code_slots, 0])
            if fold:
                T.call_extern(
                    "handle",
                    "tileops::mxfp4_decode_fold_step",
                    T.address_of(b_packed[kk % num_stages, 0, 0]),
                    box,
                    kk % _CODE_STEPS,
                    frags.data,
                    scales.data,
                    fits.data,
                )
            else:
                T.call_extern(
                    "handle",
                    f"tileops::mxfp4_weight_step<tileops::kMxfp4SplitE8M0, {_CODE_STEPS}, 0, "
                    f"{block_n}>",
                    T.address_of(b_packed[kk % num_stages, 0, 0]),
                    box,
                    0,
                    frags.data,
                    scales.data,
                    kk % _CODE_STEPS,
                )

        @T.macro
        def run_step(
            fold,
            again,
            a_shared,
            b_packed,
            codes,
            tokens,
            full,
            empty,
            final,
            step,
            frags,
            scales,
            frags_next,
            scales_next,
            fits,
            kk,
            tx,
        ):
            """K step ``kk``: its WGMMAs while the next step's weights decode, the release of its
            stage, then its promotion."""
            slot = kk % num_stages
            T.call_extern(
                "handle",
                f"tileops::mxfp4_decode_wgmma<{1 if fold else 4}>",
                step.data,
                frags.data,
                T.address_of(a_shared[slot, 0, 0]),
            )
            if kk + 1 < scale_k:
                decode(
                    fold,
                    again,
                    b_packed,
                    codes,
                    full,
                    frags_next,
                    scales_next,
                    fits,
                    kk + 1,
                )
            T.wait_wgmma(0)
            if tx % 32 == 0:
                T.barrier_arrive(empty[slot])
            T.call_extern(
                "handle",
                f"tileops::mxfp4_decode_promote<{str(fold).lower()}>",
                step.data,
                final.data,
                scales.data,
                T.address_of(tokens[kk * _ROW_TILE]),
            )

        @T.macro
        def steps(
            fold,
            again,
            a_shared,
            b_packed,
            codes,
            tokens,
            full,
            empty,
            final,
            step,
            frags0,
            scales0,
            frags1,
            scales1,
            fits,
            tx,
        ):
            """The tile's K steps, folded or split by sub-block, in round ``again``."""
            decode(fold, again, b_packed, codes, full, frags0, scales0, fits, 0)
            for kk in T.unroll(scale_k, unroll_factor=num_stages):
                if kk % 2 == 0:
                    run_step(
                        fold,
                        again,
                        a_shared,
                        b_packed,
                        codes,
                        tokens,
                        full,
                        empty,
                        final,
                        step,
                        frags0,
                        scales0,
                        frags1,
                        scales1,
                        fits,
                        kk,
                        tx,
                    )
                else:
                    run_step(
                        fold,
                        again,
                        a_shared,
                        b_packed,
                        codes,
                        tokens,
                        full,
                        empty,
                        final,
                        step,
                        frags1,
                        scales1,
                        frags0,
                        scales0,
                        fits,
                        kk,
                        tx,
                    )

        @T.prim_func
        def _moe_grouped_gemm_mxfp4_decode_main(
            A: T.Tensor((m, k), "float8_e4m3fn"),  # type: ignore
            A_scale: T.Tensor((m, scale_k), "uint8"),  # type: ignore
            B: T.Tensor((num_groups, n, k // 2), "uint8"),  # type: ignore
            B_scale: T.Tensor((num_groups, n, k // _B_SCALE_BLOCK), "uint8"),  # type: ignore
            layout: T.Tensor((num_groups,), "int32"),  # type: ignore
            C: T.Tensor((m, n), "bfloat16"),  # type: ignore
        ) -> None:
            with T.Kernel(m, n // block_n, threads=_THREADS) as (row, tile):
                a_shared = T.alloc_shared((num_stages, _ROW_TILE, block_k), "float8_e4m3fn")
                b_packed = T.alloc_shared((num_stages, block_n, packed_k), "uint8")
                codes = T.alloc_shared((code_slots, block_n * _CODE_STEPS), "uint32")
                tokens = T.alloc_shared((_ROW_TILE * scale_k,), "float32")
                run_shared = T.alloc_shared((4,), "int32", scope="shared")
                vote = T.alloc_shared((1,), "int32", scope="shared")
                T.annotate_layout({a_shared: tilelang.layout.make_swizzled_layout(a_shared)})
                T.annotate_min_blocks_per_sm(3)
                full = T.alloc_barrier([1] * num_stages)
                empty = T.alloc_barrier([math_threads // 32] * num_stages)
                voted = T.alloc_barrier([1])
                run = T.alloc_local((3,), "int32")
                fits = T.alloc_local((1,), "int32")
                final = T.alloc_local((4,), "float32")
                step = T.alloc_local((16,), "float32")
                frags0 = T.alloc_local((16,), "uint32")
                frags1 = T.alloc_local((16,), "uint32")
                scales0 = T.alloc_local((8,), "float32")
                scales1 = T.alloc_local((8,), "float32")
                folds = T.alloc_var("int32")
                tx = T.get_thread_binding()
                a_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8,
                    2,
                    A.data,
                    k,
                    m,
                    1,
                    k,
                    block_k,
                    _ROW_TILE,
                    1,
                    1,
                    TMA_INTERLEAVE_NONE,
                    TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_256B,
                    TMA_OOB_FILL_NONE,
                )
                b_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8,
                    3,
                    B.data,
                    k // 2,
                    n,
                    num_groups,
                    1,
                    k // 2,
                    n * k // 2,
                    packed_k,
                    block_n,
                    1,
                    1,
                    1,
                    1,
                    TMA_INTERLEAVE_NONE,
                    TMA_SWIZZLE_64B,
                    TMA_L2_PROMOTION_256B,
                    TMA_OOB_FILL_NONE,
                )
                s_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8,
                    3,
                    B_scale.data,
                    k // _B_SCALE_BLOCK,
                    n,
                    num_groups,
                    1,
                    k // _B_SCALE_BLOCK,
                    n * k // _B_SCALE_BLOCK,
                    _CODE_STEPS * 4,
                    block_n,
                    1,
                    1,
                    1,
                    1,
                    TMA_INTERLEAVE_NONE,
                    TMA_SWIZZLE_NONE,
                    TMA_L2_PROMOTION_128B,
                    TMA_OOB_FILL_NONE,
                )
                if tx < math_threads:
                    T.call_extern(
                        "handle",
                        f"tileops::mxfp4_decode_tokens<{scale_k}, {math_threads}>",
                        A_scale.data,
                        row,
                        m,
                        T.address_of(tokens[0]),
                        tx,
                    )
                found = T.call_extern(
                    "bool",
                    f"tileops::mxfp4_decode_find_run<{_THREADS}, {num_groups}>",
                    layout.data,
                    row,
                    T.address_of(run_shared[0]),
                    run.data,
                )
                g = run[0]
                valid = T.min(T.min(run[2], m) - row, _ROW_TILE)
                n0 = tile * block_n
                if T.And(found, (row - run[1]) % _ROW_TILE == 0):
                    if tx == math_threads:
                        issue(
                            a_desc,
                            b_desc,
                            s_desc,
                            a_shared,
                            b_packed,
                            codes,
                            full,
                            empty,
                            row,
                            n0,
                            g,
                            0,
                        )
                        T.barrier_wait(voted[0], 0)
                        if T.call_extern("int32", "tileops::mxfp4_decode_read", vote.data) == 0:
                            issue(
                                a_desc,
                                b_desc,
                                s_desc,
                                a_shared,
                                b_packed,
                                codes,
                                full,
                                empty,
                                row,
                                n0,
                                g,
                                1,
                            )
                    elif tx < math_threads:
                        T.clear(final)
                        fits[0] = 1
                        steps(
                            True,
                            0,
                            a_shared,
                            b_packed,
                            codes,
                            tokens,
                            full,
                            empty,
                            final,
                            step,
                            frags0,
                            scales0,
                            frags1,
                            scales1,
                            fits,
                            tx,
                        )
                        folds = T.call_extern(
                            "int32",
                            "tileops::mxfp4_all_threads",
                            fits[0],
                            vote_barrier,
                            math_threads,
                        )
                        if tx == 0:
                            T.call_extern(
                                "handle",
                                "tileops::mxfp4_decode_publish",
                                vote.data,
                                folds,
                                T.address_of(voted[0]),
                            )
                        if folds == 0:
                            T.clear(final)
                            steps(
                                False,
                                1,
                                a_shared,
                                b_packed,
                                codes,
                                tokens,
                                full,
                                empty,
                                final,
                                step,
                                frags0,
                                scales0,
                                frags1,
                                scales1,
                                fits,
                                tx,
                            )
                        T.call_extern(
                            "handle",
                            "tileops::mxfp4_decode_store",
                            final.data,
                            T.access_ptr(C[row, n0], "w"),
                            n,
                            valid,
                        )

        return _moe_grouped_gemm_mxfp4_decode_main

    return _moe_grouped_gemm_mxfp4_decode_func


class MoEGroupedGEMMMXFP4DecodeKernel(Kernel, MGroupedGEMMMXFP4FwdInterface):
    """MXFP4 grouped GEMM over few rows per expert: decode-sized calls (``call.decode``) over
    tight contiguous rows, ``physical_psum`` metadata and e8m0 scales.

    Each CTA takes one 64-row weight tile of one expert against up to eight of its rows, the
    weights decoded to e4m3 in registers and each 128-wide K step contracted on an fp8 WGMMA.
    A tile whose rows' scales each lie in ``[2 ** -63, 2 ** 63]`` and span at most five binades
    within every step folds each step's four sub-blocks into one contraction under the step's
    largest scale; another contracts sub-block by sub-block, its scale products split into
    mantissa and exponent. A config is ``{num_stages}``, the depth of the ring of row and
    weight stages, a multiple of four that divides the K steps.
    """

    supported_archs: list[int] = [90]
    preferred_over: ClassVar[frozenset[str]] = frozenset({"grouped_gemm_mxfp4"})
    _NUM_STAGES: ClassVar[tuple[int, ...]] = (4, 8, 16)
    _SMEM_HEADROOM: ClassVar[int] = 2048

    @classmethod
    def refusal(cls, call: MGroupedGEMMMXFP4Call) -> str | None:
        """Decode-sized tight ``physical_psum`` runs under e8m0 scales, experts the run lookup
        scans in ``_LOOKUP_ENDS`` ends a thread, ``K`` a positive multiple of 512, ``N`` of 64,
        and the shallowest ring fitting in shared memory."""
        if call.kind != "contiguous" or call.packing != "tight":
            return f"reads tight contiguous rows, got {call.kind} {call.packing}"
        if call.metadata_kind != "physical_psum":
            return f"finds runs from physical_psum ends, got {call.metadata_kind}"
        if call.scale_dtype != torch.float8_e8m0fnu:
            return f"folds e8m0 scales only, got {call.scale_dtype}"
        if not call.decode:
            return f"serves decode-sized calls, got {call.m} rows over {call.num_groups} experts"
        most = _LOOKUP_ENDS * _THREADS
        if call.num_groups > most:
            return f"looks up a row's run among at most {most} experts, got {call.num_groups}"
        step_k = QUANT_SCALE_BLOCK * _CODE_STEPS
        if call.k <= 0 or call.k % step_k:
            return f"loads four K steps' codes a box, so needs K a multiple of {step_k}"
        if call.n % WGMMA_ROWS:
            return f"tiles N by {WGMMA_ROWS} weight rows, got N = {call.n}"
        least = _decode_smem_bytes(min(cls._NUM_STAGES), call.k // QUANT_SCALE_BLOCK)
        room = call.smem_budget - cls._SMEM_HEADROOM
        if least > room:
            return f"takes {least} bytes of shared memory on the shallowest ring, past {room}"
        return super().refusal(call)

    def __init__(self, call: MGroupedGEMMMXFP4Call) -> None:
        super().__init__(device_index=call.device_index)
        self.call = call
        self.scale_k = call.k // QUANT_SCALE_BLOCK
        self.smem_limit = self.build_target.smem_budget - self._SMEM_HEADROOM
        self.kernel = _moe_grouped_gemm_mxfp4_decode_kernel(call.num_groups, call.n, call.k)
        self.init_config()

    @property
    def autotune_supply_prog(self) -> Callable:
        """Supply the call's rows spread evenly over its experts, under e8m0 codes, for tuning."""
        from tilelang.utils.device import get_current_device

        call = self.call

        def supply_prog(params: list) -> list:
            device = get_current_device()
            metadata = moe_layout_metadata(call, call.m, call.num_groups)

            def codes(*shape):
                return torch.randint(123, 129, shape, dtype=torch.uint8, device=device)

            return [
                torch.randn(call.m, call.k, device=device).to(torch.float8_e4m3fn),
                codes(call.m, self.scale_k),
                torch.randint(
                    0, 256, (call.num_groups, call.n, call.k // 2), dtype=torch.uint8, device=device
                ),
                codes(call.num_groups, call.n, call.k // _B_SCALE_BLOCK),
                torch.tensor(metadata, dtype=torch.int32, device=device),
                torch.empty(call.m, call.n, dtype=torch.bfloat16, device=device),
            ]

        return supply_prog

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"num_stages": stages}
            for stages in self._NUM_STAGES
            if self.scale_k % stages == 0
            and _decode_smem_bytes(stages, self.scale_k) <= self.smem_limit
        ]

    @property
    def default_config(self) -> dict:
        return {"num_stages": min(8, max(c["num_stages"] for c in self.autotune_configs))}

    def forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run each expert's block-scaled product over its rows, into *out* if given."""
        if out is None:
            out = torch.empty((*a.shape[:-1], self.call.n), dtype=torch.bfloat16, device=a.device)
        if out.numel() == 0:
            return out
        self.program(self._compile)(
            a, a_scale.view(torch.uint8), b, b_scale.view(torch.uint8), layout_metadata, out
        )
        return out

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(**cfg)
