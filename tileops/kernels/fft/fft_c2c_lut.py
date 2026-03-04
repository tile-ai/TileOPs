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
            stage: T.int32,
        ):
            """
            One butterfly stage using pre-computed LUT twiddle factors.

            LUT offset for (stage, k): half_m - 1 + k,  half_m = 2^stage.
            Avoids sin/cos entirely for large-stride stages where trig is expensive.
            """
            m = T.int32(1) << (stage + T.int32(1))
            half_m = T.int32(1) << stage

            with T.Kernel(T.ceildiv(n // 2, block_size), threads=threads) as bx:
                for i in T.Parallel(block_size):
                    butterfly_idx = bx * block_size + i
                    if butterfly_idx < n // 2:
                        group = butterfly_idx // half_m
                        k = butterfly_idx % half_m
                        j_idx = group * m + k
                        l_idx = j_idx + half_m

                        lut_offset = half_m - T.int32(1) + k
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

            # LUT butterfly stages for strides that exceed the SMEM chunk
            for stage in T.serial(lut_stage_count):
                lut_butterfly_stage(
                    y_real, y_imag, lut_real, lut_imag,
                    stage + lut_stage_start,
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
            "block_size": 256,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_size": bs, "threads": bs}
            for bs in [32, 64, 128, 256, 512, 1024]
        ]

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
