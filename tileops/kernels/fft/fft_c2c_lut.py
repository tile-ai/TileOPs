import math
from typing import Any, Callable, Dict

import tilelang
import tilelang.language as T
import torch

from ..kernel import Kernel

# Sufficient precision for twiddle angles
_PI = 3.14159265358979323846


def _fft_c2c_lut_kernel(n: int, dtype: str = 'complex64') -> Callable:
    """
    1D Complex-to-Complex FFT kernel with pre-computed twiddle LUT and
    shared-memory (SMEM) stage fusion.

    Design (two-phase, single-launch for SMEM stages):

    Phase 1 — Fused bit-reversal + SMEM butterfly stages (one kernel launch):
      Each block reads input elements at their bit-reversed positions directly
      into shared memory, eliminating the separate bit-reversal kernel.  It
      then performs all log2(min(n, 2*threads)) butterfly stages entirely in
      SMEM using on-the-fly trig.  Result: one global-memory read + one write
      for the entire SMEM phase, regardless of how many stages it covers.

    Phase 2 — LUT stages (remaining stages, one kernel each):
      For stages where butterfly strides exceed the SMEM chunk size, each
      stage is launched as a separate kernel that reads twiddle factors from
      a pre-computed GPU LUT instead of recomputing expensive sin/cos.

    LUT layout (flat array, size n-1):
      Stage s (half_m = 2^s) occupies LUT[half_m-1 : 2*half_m-1].
      LUT[half_m-1 + k] = exp(-2πi * k / m),  m = 2*half_m,  k = 0..half_m-1.

    Reference: cuFFT optimization guide — precomputed twiddle tables + shared
    memory butterflies eliminate redundant trig evaluation and global memory
    traffic for early stages.

    Args:
        n:     FFT size (must be a power of 2).
        dtype: 'complex64' or 'complex128'.

    Returns:
        A JIT-decorated function _fft_lut_func(block_size, threads) that
        itself returns a compiled prim_func taking
        (x_real, x_imag, lut_real, lut_imag) and producing (y_real, y_imag).
    """
    if dtype == 'complex64':
        real_dtype = 'float32'
    elif dtype == 'complex128':
        real_dtype = 'float64'
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if n & (n - 1) != 0 or n == 0:
        raise ValueError(f"FFT size must be a power of 2, got {n}")

    log2n = int(math.log2(n))
    lut_size = n - 1  # total number of twiddle factors across all stages

    @tilelang.jit(
        out_idx=[4, 5],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3"]
    )
    def _fft_lut_func(block_size: int, threads: int) -> Callable:
        # --------------- compile-time constants --------------------------------
        # Each SMEM block covers smem_per_block consecutive elements.
        # Butterfly pairs for stages s satisfy stride = 2^s < smem_per_block,
        # so all pairs fit within one block for s in 0..smem_stages-1.
        smem_per_block = min(n, 2 * threads)
        smem_blocks = n // smem_per_block       # >= 1 (both are powers of 2)
        smem_stages = int(math.log2(smem_per_block))   # stages handled in SMEM
        # remaining stages (stride >= smem_per_block) use the LUT
        lut_stage_start = smem_stages
        lut_stage_count = log2n - smem_stages
        # float32 for complex64 (FP32 throughput 30× > FP64 on H200);
        # float64 for complex128 — identical behaviour, no regression.
        accum_dtype = real_dtype
        # -----------------------------------------------------------------------

        @T.macro
        def fused_bitrev_smem_stages(
            x_real: T.Tensor((n,), real_dtype),
            x_imag: T.Tensor((n,), real_dtype),
            y_real: T.Tensor((n,), real_dtype),
            y_imag: T.Tensor((n,), real_dtype),
        ):
            """
            Fused bit-reversal load + SMEM butterfly stages in a single kernel.

            Each of the smem_blocks blocks:
              1. Loads smem_per_block elements from x (at bit-reversed indices)
                 into SMEM — this replaces the separate bit-reversal kernel.
              2. Performs smem_stages butterfly stages in-place on SMEM data.
              3. Writes smem_per_block results back to y.

            The fused load eliminates one full kernel launch and one round-trip
            to global memory compared to the two-kernel (bit-reversal + SMEM)
            approach.  Random reads from the bit-reversed positions are the same
            cost regardless of when they happen.
            """
            with T.Kernel(smem_blocks, threads=threads) as bx:
                smem_r = T.alloc_shared([smem_per_block], real_dtype)
                smem_i = T.alloc_shared([smem_per_block], real_dtype)

                # ---- load first half: smem[0..threads-1] from bit-reversed x ----
                for i in T.Parallel(threads):
                    if i < smem_per_block:
                        idx = bx * smem_per_block + i
                        rev_idx = T.alloc_var("int32")
                        temp = T.alloc_var("int32")
                        rev_idx = 0
                        temp = idx
                        for _bit in T.serial(log2n):
                            rev_idx = (rev_idx << 1) | (temp & 1)
                            temp = temp >> 1
                        smem_r[i] = x_real[rev_idx]
                        smem_i[i] = x_imag[rev_idx]

                # ---- load second half: smem[threads..smem_per_block-1] ----
                for i in T.Parallel(threads):
                    j = i + threads
                    if j < smem_per_block:
                        idx = bx * smem_per_block + j
                        rev_idx = T.alloc_var("int32")
                        temp = T.alloc_var("int32")
                        rev_idx = 0
                        temp = idx
                        for _bit in T.serial(log2n):
                            rev_idx = (rev_idx << 1) | (temp & 1)
                            temp = temp >> 1
                        smem_r[j] = x_real[rev_idx]
                        smem_i[j] = x_imag[rev_idx]

                T.sync_threads()

                # ---- butterfly stages (Python-level loop → compile-time unroll) ----
                for s in range(smem_stages):
                    m_s = 1 << (s + 1)    # compile-time constant
                    half_m_s = 1 << s     # compile-time constant

                    for i in T.Parallel(threads):
                        if i < smem_per_block // 2:
                            group = i // half_m_s
                            k = i % half_m_s
                            j_idx = group * m_s + k
                            l_idx = j_idx + half_m_s

                            angle = (
                                -2.0 * _PI
                                * T.cast(k, "float64")
                                / T.cast(m_s, "float64")
                            )
                            tw_r = T.cos(T.cast(angle, accum_dtype))
                            tw_i = T.sin(T.cast(angle, accum_dtype))

                            u_r = T.cast(smem_r[j_idx], accum_dtype)
                            u_i = T.cast(smem_i[j_idx], accum_dtype)
                            v_r = T.cast(smem_r[l_idx], accum_dtype)
                            v_i = T.cast(smem_i[l_idx], accum_dtype)

                            t_r = v_r * tw_r - v_i * tw_i
                            t_i = v_r * tw_i + v_i * tw_r

                            smem_r[j_idx] = T.cast(u_r + t_r, real_dtype)
                            smem_i[j_idx] = T.cast(u_i + t_i, real_dtype)
                            smem_r[l_idx] = T.cast(u_r - t_r, real_dtype)
                            smem_i[l_idx] = T.cast(u_i - t_i, real_dtype)

                    T.sync_threads()

                # ---- store ----
                for i in T.Parallel(threads):
                    if i < smem_per_block:
                        y_real[bx * smem_per_block + i] = smem_r[i]
                        y_imag[bx * smem_per_block + i] = smem_i[i]

                for i in T.Parallel(threads):
                    j = i + threads
                    if j < smem_per_block:
                        y_real[bx * smem_per_block + j] = smem_r[j]
                        y_imag[bx * smem_per_block + j] = smem_i[j]

        @T.macro
        def lut_butterfly_stage(
            y_real: T.Tensor((n,), real_dtype),
            y_imag: T.Tensor((n,), real_dtype),
            lut_real: T.Tensor((lut_size,), real_dtype),
            lut_imag: T.Tensor((lut_size,), real_dtype),
            stage,
        ):
            """
            One butterfly stage using pre-computed LUT twiddle factors.

            LUT offset for (stage, k): half_m - 1 + k,  half_m = 2^stage.
            Avoids sin/cos entirely for large-stride stages where trig is expensive.

            When stage is a compile-time Python int (called from range()),
            half_m and m are compile-time constants, enabling the compiler to
            replace division/modulo with shift/mask operations.
            """
            half_m = 1 << stage
            m = half_m * 2
            lut_base = half_m - 1

            with T.Kernel(T.ceildiv(n // 2, block_size), threads=threads) as bx:
                for i in T.Parallel(block_size):
                    butterfly_idx = bx * block_size + i
                    if butterfly_idx < n // 2:
                        group = butterfly_idx // half_m
                        k = butterfly_idx % half_m
                        j_idx = group * m + k
                        l_idx = j_idx + half_m

                        lut_offset = lut_base + k
                        tw_r = T.cast(lut_real[lut_offset], accum_dtype)
                        tw_i = T.cast(lut_imag[lut_offset], accum_dtype)

                        u_r = T.cast(y_real[j_idx], accum_dtype)
                        u_i = T.cast(y_imag[j_idx], accum_dtype)
                        v_r = T.cast(y_real[l_idx], accum_dtype)
                        v_i = T.cast(y_imag[l_idx], accum_dtype)

                        t_r = v_r * tw_r - v_i * tw_i
                        t_i = v_r * tw_i + v_i * tw_r

                        y_real[j_idx] = T.cast(u_r + t_r, real_dtype)
                        y_imag[j_idx] = T.cast(u_i + t_i, real_dtype)
                        y_real[l_idx] = T.cast(u_r - t_r, real_dtype)
                        y_imag[l_idx] = T.cast(u_i - t_i, real_dtype)

        def _butterfly(u_r, u_i, v_r, v_i, tw_r, tw_i):
            """Radix-2 butterfly: returns (u+t, u-t) where t = v * tw."""
            t_r = v_r * tw_r - v_i * tw_i
            t_i = v_r * tw_i + v_i * tw_r
            return u_r + t_r, u_i + t_i, u_r - t_r, u_i - t_i

        @T.macro
        def lut_butterfly_radix4(
            y_real: T.Tensor((n,), real_dtype),
            y_imag: T.Tensor((n,), real_dtype),
            lut_real: T.Tensor((lut_size,), real_dtype),
            lut_imag: T.Tensor((lut_size,), real_dtype),
            stage_lo,
        ):
            """
            Fused radix-4 butterfly combining two consecutive LUT stages
            (stage_lo and stage_lo+1) into a single kernel launch.

            Each thread processes 4 elements instead of 2, halving the number
            of kernel launches and global memory round-trips for LUT stages.

            Math: apply stage_lo butterflies (stride q) on pairs (a,b) and
            (c,d), then stage_lo+1 butterflies (stride 2q) on pairs (a',c')
            and (b',d').  The second pair's twiddle is w2 * (-i).
            """
            q = 1 << stage_lo           # stride for stage_lo
            m4 = q * 4                  # combined group size
            lut_base_lo = q - 1         # LUT offset for stage_lo
            lut_base_hi = 2 * q - 1     # LUT offset for stage_lo+1
            n_groups = n // 4           # total radix-4 groups

            with T.Kernel(T.ceildiv(n_groups, block_size), threads=threads) as bx:
                for i in T.Parallel(block_size):
                    gidx = bx * block_size + i
                    if gidx < n_groups:
                        grp = gidx // q
                        k = gidx % q
                        base = grp * m4 + k

                        # Load 4 elements
                        a_r = T.cast(y_real[base], accum_dtype)
                        a_i = T.cast(y_imag[base], accum_dtype)
                        b_r = T.cast(y_real[base + q], accum_dtype)
                        b_i = T.cast(y_imag[base + q], accum_dtype)
                        c_r = T.cast(y_real[base + 2 * q], accum_dtype)
                        c_i = T.cast(y_imag[base + 2 * q], accum_dtype)
                        d_r = T.cast(y_real[base + 3 * q], accum_dtype)
                        d_i = T.cast(y_imag[base + 3 * q], accum_dtype)

                        # Twiddle factors from LUT
                        w1_r = T.cast(lut_real[lut_base_lo + k], accum_dtype)
                        w1_i = T.cast(lut_imag[lut_base_lo + k], accum_dtype)
                        w2_r = T.cast(lut_real[lut_base_hi + k], accum_dtype)
                        w2_i = T.cast(lut_imag[lut_base_hi + k], accum_dtype)

                        # Stage_lo: butterflies (a,b) and (c,d) with w1
                        ap_r, ap_i, bp_r, bp_i = _butterfly(a_r, a_i, b_r, b_i, w1_r, w1_i)
                        cp_r, cp_i, dp_r, dp_i = _butterfly(c_r, c_i, d_r, d_i, w1_r, w1_i)

                        # Stage_lo+1: (a',c') with w2, (b',d') with w2*(-i)
                        tc_r = cp_r * w2_r - cp_i * w2_i
                        tc_i = cp_r * w2_i + cp_i * w2_r
                        # w2*(-i) = (w2_i, -w2_r)
                        td2_r = dp_r * w2_i - dp_i * (-w2_r)
                        td2_i = dp_r * (-w2_r) + dp_i * w2_i

                        # Store 4 results
                        y_real[base] = T.cast(ap_r + tc_r, real_dtype)
                        y_imag[base] = T.cast(ap_i + tc_i, real_dtype)
                        y_real[base + 2 * q] = T.cast(ap_r - tc_r, real_dtype)
                        y_imag[base + 2 * q] = T.cast(ap_i - tc_i, real_dtype)
                        y_real[base + q] = T.cast(bp_r + td2_r, real_dtype)
                        y_imag[base + q] = T.cast(bp_i + td2_i, real_dtype)
                        y_real[base + 3 * q] = T.cast(bp_r - td2_r, real_dtype)
                        y_imag[base + 3 * q] = T.cast(bp_i - td2_i, real_dtype)

        @T.macro
        def lut_butterfly_radix8(
            y_real: T.Tensor((n,), real_dtype),
            y_imag: T.Tensor((n,), real_dtype),
            lut_real: T.Tensor((lut_size,), real_dtype),
            lut_imag: T.Tensor((lut_size,), real_dtype),
            stage_lo,
        ):
            """
            Fused radix-8 butterfly combining three consecutive LUT stages
            (stage_lo, stage_lo+1, stage_lo+2) into a single kernel launch.

            Each thread processes 8 elements, reducing kernel launches by 3×
            compared to radix-2 and by 1.5× compared to radix-4.

            Algorithm:
              Stage s  (stride q):   4 butterfly pairs with w1
              Stage s+1 (stride 2q): 4 butterfly pairs with w2a, w2b
              Stage s+2 (stride 4q): 4 butterfly pairs with w3a..w3d
            """
            q = 1 << stage_lo             # stride for stage_lo
            m8 = q * 8                    # combined group size
            lut1 = q - 1                  # LUT base for stage_lo
            lut2 = 2 * q - 1             # LUT base for stage_lo+1
            lut3 = 4 * q - 1             # LUT base for stage_lo+2
            n_groups = n // 8             # total radix-8 groups

            with T.Kernel(T.ceildiv(n_groups, block_size), threads=threads) as bx:
                for i in T.Parallel(block_size):
                    gidx = bx * block_size + i
                    if gidx < n_groups:
                        grp = gidx // q
                        k = gidx % q
                        base = grp * m8 + k

                        # Load 8 elements: x0..x7 at offsets 0, q, 2q, ..., 7q
                        x0_r = T.cast(y_real[base], accum_dtype)
                        x0_i = T.cast(y_imag[base], accum_dtype)
                        x1_r = T.cast(y_real[base + q], accum_dtype)
                        x1_i = T.cast(y_imag[base + q], accum_dtype)
                        x2_r = T.cast(y_real[base + 2 * q], accum_dtype)
                        x2_i = T.cast(y_imag[base + 2 * q], accum_dtype)
                        x3_r = T.cast(y_real[base + 3 * q], accum_dtype)
                        x3_i = T.cast(y_imag[base + 3 * q], accum_dtype)
                        x4_r = T.cast(y_real[base + 4 * q], accum_dtype)
                        x4_i = T.cast(y_imag[base + 4 * q], accum_dtype)
                        x5_r = T.cast(y_real[base + 5 * q], accum_dtype)
                        x5_i = T.cast(y_imag[base + 5 * q], accum_dtype)
                        x6_r = T.cast(y_real[base + 6 * q], accum_dtype)
                        x6_i = T.cast(y_imag[base + 6 * q], accum_dtype)
                        x7_r = T.cast(y_real[base + 7 * q], accum_dtype)
                        x7_i = T.cast(y_imag[base + 7 * q], accum_dtype)

                        # ---- Stage s (stride q): 4 butterfly pairs with w1 ----
                        # w1 = LUT[lut1 + k]
                        w1_r = T.cast(lut_real[lut1 + k], accum_dtype)
                        w1_i = T.cast(lut_imag[lut1 + k], accum_dtype)

                        # ---- Stage s (stride q): 4 butterfly pairs with w1 ----
                        a0_r, a0_i, a1_r, a1_i = _butterfly(x0_r, x0_i, x1_r, x1_i, w1_r, w1_i)
                        a2_r, a2_i, a3_r, a3_i = _butterfly(x2_r, x2_i, x3_r, x3_i, w1_r, w1_i)
                        a4_r, a4_i, a5_r, a5_i = _butterfly(x4_r, x4_i, x5_r, x5_i, w1_r, w1_i)
                        a6_r, a6_i, a7_r, a7_i = _butterfly(x6_r, x6_i, x7_r, x7_i, w1_r, w1_i)

                        # ---- Stage s+1 (stride 2q): pairs at distance 2q ----
                        w2a_r = T.cast(lut_real[lut2 + k], accum_dtype)
                        w2a_i = T.cast(lut_imag[lut2 + k], accum_dtype)
                        w2b_r = T.cast(lut_real[lut2 + k + q], accum_dtype)
                        w2b_i = T.cast(lut_imag[lut2 + k + q], accum_dtype)

                        b0_r, b0_i, b2_r, b2_i = _butterfly(a0_r, a0_i, a2_r, a2_i, w2a_r, w2a_i)
                        b1_r, b1_i, b3_r, b3_i = _butterfly(a1_r, a1_i, a3_r, a3_i, w2b_r, w2b_i)
                        b4_r, b4_i, b6_r, b6_i = _butterfly(a4_r, a4_i, a6_r, a6_i, w2a_r, w2a_i)
                        b5_r, b5_i, b7_r, b7_i = _butterfly(a5_r, a5_i, a7_r, a7_i, w2b_r, w2b_i)

                        # ---- Stage s+2 (stride 4q): pairs at distance 4q ----
                        w3a_r = T.cast(lut_real[lut3 + k], accum_dtype)
                        w3a_i = T.cast(lut_imag[lut3 + k], accum_dtype)
                        w3b_r = T.cast(lut_real[lut3 + k + q], accum_dtype)
                        w3b_i = T.cast(lut_imag[lut3 + k + q], accum_dtype)
                        w3c_r = T.cast(lut_real[lut3 + k + 2 * q], accum_dtype)
                        w3c_i = T.cast(lut_imag[lut3 + k + 2 * q], accum_dtype)
                        w3d_r = T.cast(lut_real[lut3 + k + 3 * q], accum_dtype)
                        w3d_i = T.cast(lut_imag[lut3 + k + 3 * q], accum_dtype)

                        c0_r, c0_i, c4_r, c4_i = _butterfly(b0_r, b0_i, b4_r, b4_i, w3a_r, w3a_i)
                        c1_r, c1_i, c5_r, c5_i = _butterfly(b1_r, b1_i, b5_r, b5_i, w3b_r, w3b_i)
                        c2_r, c2_i, c6_r, c6_i = _butterfly(b2_r, b2_i, b6_r, b6_i, w3c_r, w3c_i)
                        c3_r, c3_i, c7_r, c7_i = _butterfly(b3_r, b3_i, b7_r, b7_i, w3d_r, w3d_i)

                        # Store 8 results
                        y_real[base] = T.cast(c0_r, real_dtype)
                        y_imag[base] = T.cast(c0_i, real_dtype)
                        y_real[base + q] = T.cast(c1_r, real_dtype)
                        y_imag[base + q] = T.cast(c1_i, real_dtype)
                        y_real[base + 2 * q] = T.cast(c2_r, real_dtype)
                        y_imag[base + 2 * q] = T.cast(c2_i, real_dtype)
                        y_real[base + 3 * q] = T.cast(c3_r, real_dtype)
                        y_imag[base + 3 * q] = T.cast(c3_i, real_dtype)
                        y_real[base + 4 * q] = T.cast(c4_r, real_dtype)
                        y_imag[base + 4 * q] = T.cast(c4_i, real_dtype)
                        y_real[base + 5 * q] = T.cast(c5_r, real_dtype)
                        y_imag[base + 5 * q] = T.cast(c5_i, real_dtype)
                        y_real[base + 6 * q] = T.cast(c6_r, real_dtype)
                        y_imag[base + 6 * q] = T.cast(c6_i, real_dtype)
                        y_real[base + 7 * q] = T.cast(c7_r, real_dtype)
                        y_imag[base + 7 * q] = T.cast(c7_i, real_dtype)

        @T.prim_func
        def _fft_lut_main(
            x_real: T.Tensor((n,), real_dtype),         # type: ignore
            x_imag: T.Tensor((n,), real_dtype),         # type: ignore
            lut_real: T.Tensor((lut_size,), real_dtype),  # type: ignore
            lut_imag: T.Tensor((lut_size,), real_dtype),  # type: ignore
            y_real: T.Tensor((n,), real_dtype),         # type: ignore
            y_imag: T.Tensor((n,), real_dtype),         # type: ignore
        ) -> None:
            # Fused bit-reversal + SMEM butterfly stages: single kernel launch,
            # one global-memory read of x, one write of y for all SMEM stages.
            fused_bitrev_smem_stages(x_real, x_imag, y_real, y_imag)

            # LUT stages: use radix-8 (fused triples) where possible,
            # then radix-4 or radix-2 for the remainder.
            r8_count = (lut_stage_count // 3) * 3
            for s in range(0, r8_count, 3):
                lut_butterfly_radix8(
                    y_real, y_imag, lut_real, lut_imag,
                    s + lut_stage_start,
                )
            remainder = lut_stage_count - r8_count
            if remainder == 2:
                lut_butterfly_radix4(
                    y_real, y_imag, lut_real, lut_imag,
                    r8_count + lut_stage_start,
                )
            elif remainder == 1:
                lut_butterfly_stage(
                    y_real, y_imag, lut_real, lut_imag,
                    r8_count + lut_stage_start,
                )

        return _fft_lut_main

    return _fft_lut_func


@torch.library.custom_op("top::fft_c2c_lut_wrapped_kernel", mutates_args=())
def _fft_c2c_lut_wrapped_kernel(
    n: int,
    dtype: str,
    block_size: int,
    threads: int,
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    lut_real: torch.Tensor,
    lut_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fft_c2c_lut_kernel(n, dtype)(block_size, threads)(
        x_real, x_imag, lut_real, lut_imag
    )


@_fft_c2c_lut_wrapped_kernel.register_fake
def _(
    n: int,
    dtype: str,
    block_size: int,
    threads: int,
    *inputs: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    real_dtype = inputs[0].dtype
    device = inputs[0].device
    return (
        torch.empty((n,), dtype=real_dtype, device=device),
        torch.empty((n,), dtype=real_dtype, device=device),
    )


class FFTC2CLUTKernel(Kernel):
    """
    1D Complex-to-Complex FFT kernel with pre-computed twiddle LUT and
    shared-memory stage fusion.

    Combines two complementary optimizations over the baseline FFTC2CKernel:

    1. SMEM stage fusion: the first log2(min(n, 2*threads))+1 butterfly stages
       are processed entirely in shared memory within a single kernel launch,
       reducing global memory traffic from O(log N) to O(1) for small-stride
       stages.

    2. Pre-computed twiddle LUT: the remaining large-stride stages look up
       twiddle factors from a GPU-resident LUT (built by FFTC2CLUTOp at
       construction time), eliminating repeated sin/cos evaluation.

    Together these match cuFFT-level performance on modern NVIDIA GPUs.

    Args:
        x_real:    Real part of input, shape (n,), float32 or float64.
        x_imag:    Imaginary part of input, shape (n,), float32 or float64.
        lut_real:  Pre-computed twiddle real parts, shape (n-1,).
        lut_imag:  Pre-computed twiddle imaginary parts, shape (n-1,).

    Note:
        n must be a power of 2.
        The LUT tensors are managed and passed by FFTC2CLUTOp, not this class.

    Optimization notes and benchmark results: https://github.com/tile-ai/TileOPs/issues/310
    """

    supported_archs = [75, 80, 86, 89, 90]

    def __init__(self,
                 n: int,
                 dtype: torch.dtype = torch.complex64,
                 tune: bool = False) -> None:
        super().__init__()
        self.n = n
        self.dtype = dtype

        self.kernel = _fft_c2c_lut_kernel(n, self.dtype_str)
        self.init_config(tune=tune)

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "block_size": 1024,
            "threads": 512,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # block_size controls LUT-stage work per block (higher → better ILP).
        # threads controls SMEM coverage (2*threads elements per block).
        # Decoupled search: threads for SMEM depth, block_size for LUT throughput.
        configs = [
            {"block_size": bs, "threads": bs}
            for bs in [128, 256, 512, 1024]
        ]
        configs += [
            {"block_size": 1024, "threads": 256},
            {"block_size": 1024, "threads": 512},
            {"block_size": 2048, "threads": 512},
            {"block_size": 2048, "threads": 1024},
        ]
        return configs

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        lut_real: torch.Tensor,
        lut_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 1D DFT on pre-split real/imaginary parts using a pre-built LUT.

        Args:
            x_real:   Real part, shape (n,).
            x_imag:   Imaginary part, shape (n,).
            lut_real: Twiddle real parts, shape (n-1,).
            lut_imag: Twiddle imaginary parts, shape (n-1,).

        Returns:
            (y_real, y_imag): FFT output real and imaginary parts, shape (n,).
        """
        return _fft_c2c_lut_wrapped_kernel(
            self.n,
            self.dtype_str,
            self.config["block_size"],
            self.config["threads"],
            x_real,
            x_imag,
            lut_real,
            lut_imag,
        )
