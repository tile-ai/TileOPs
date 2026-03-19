import functools
import math
from typing import Any, Callable, Dict

import tilelang
import tilelang.language as T
import torch

from ..kernel import Kernel


@functools.lru_cache(maxsize=32)
def _fft_c2c_kernel(n: int, dtype: str = 'complex64') -> Callable:
    """
    1D Complex-to-Complex FFT kernel using Cooley-Tukey radix-2 algorithm.

    Reference: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

    This implements an iterative O(N log N) FFT using decimation-in-time.
    Requires n to be a power of 2.
    """

    # Determine real dtype for complex components
    if dtype == 'complex64':
        real_dtype = 'float32'
    elif dtype == 'complex128':
        real_dtype = 'float64'
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Verify n is a power of 2
    if n & (n - 1) != 0 or n == 0:
        raise ValueError(f"FFT size must be a power of 2, got {n}")

    log2n = int(math.log2(n))

    @tilelang.jit(
        out_idx=[2, 3],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3"]
    )
    def _fft_func(block_size: int, threads: int) -> Callable:

        @T.macro
        def bit_reversal_permutation(
            x_real: T.Tensor((n,), real_dtype),
            x_imag: T.Tensor((n,), real_dtype),
            y_real: T.Tensor((n,), real_dtype),
            y_imag: T.Tensor((n,), real_dtype),
        ):
            # Stage 1: Bit-reversal permutation with shared memory staging
            with T.Kernel(T.ceildiv(n, block_size), threads=threads) as bx:
                # Allocate shared memory for staging
                y_real_shared = T.alloc_shared([block_size], real_dtype)
                y_imag_shared = T.alloc_shared([block_size], real_dtype)

                # Load and bit-reverse in parallel
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < n:
                        # Compute bit-reversed index
                        rev_idx = T.alloc_var("int32")
                        temp_idx = T.alloc_var("int32")
                        rev_idx = 0
                        temp_idx = idx
                        for _bit in T.serial(log2n):
                            rev_idx = (rev_idx << 1) | (temp_idx & 1)
                            temp_idx = temp_idx >> 1

                        # Load from bit-reversed position into shared memory
                        y_real_shared[i] = x_real[rev_idx]
                        y_imag_shared[i] = x_imag[rev_idx]

                # Coalesced write back to global memory
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < n:
                        y_real[idx] = y_real_shared[i]
                        y_imag[idx] = y_imag_shared[i]

        @T.macro
        def butterfly_stage(
            y_real: T.Tensor((n,), real_dtype),
            y_imag: T.Tensor((n,), real_dtype),
            stage: T.int32,
        ):
            # Single butterfly stage with optimized thread utilization
            # Each thread processes one butterfly (not one element)
            m = 1 << (stage + 1)  # 2^(stage+1)
            half_m = 1 << stage    # 2^stage

            with T.Kernel(T.ceildiv(n // 2, block_size), threads=threads) as bx:
                # Each thread processes one butterfly
                # This improves thread utilization from ~50% to ~100%
                for i in T.Parallel(block_size):
                    butterfly_idx = bx * block_size + i
                    if butterfly_idx < n // 2:
                        # Compute butterfly pair indices
                        # For stage s, butterflies are grouped in blocks of size m
                        group = butterfly_idx // half_m
                        k = butterfly_idx % half_m
                        j = group * m + k
                        l = j + half_m

                        # Twiddle factor: exp(-2πi * k / m)
                        # Use float64 for angle computation for precision
                        angle = -2.0 * 3.14159265358979323846 * T.cast(k, "float64") / T.cast(m, "float64")
                        twiddle_real = T.cos(angle)
                        twiddle_imag = T.sin(angle)

                        # Load values from global memory
                        u_real = T.cast(y_real[j], "float64")
                        u_imag = T.cast(y_imag[j], "float64")
                        v_real = T.cast(y_real[l], "float64")
                        v_imag = T.cast(y_imag[l], "float64")

                        # Complex multiplication: t = v * twiddle
                        t_real = v_real * twiddle_real - v_imag * twiddle_imag
                        t_imag = v_real * twiddle_imag + v_imag * twiddle_real

                        # Butterfly: (u, v) -> (u + t, u - t)
                        # Compute results in float64 for numerical stability
                        result_j_real = u_real + t_real
                        result_j_imag = u_imag + t_imag
                        result_l_real = u_real - t_real
                        result_l_imag = u_imag - t_imag

                        # Write back to global memory
                        y_real[j] = T.cast(result_j_real, real_dtype)
                        y_imag[j] = T.cast(result_j_imag, real_dtype)
                        y_real[l] = T.cast(result_l_real, real_dtype)
                        y_imag[l] = T.cast(result_l_imag, real_dtype)

        @T.prim_func
        def _fft_main(
            x_real: T.Tensor((n,), real_dtype),  # type: ignore
            x_imag: T.Tensor((n,), real_dtype),  # type: ignore
            y_real: T.Tensor((n,), real_dtype),  # type: ignore
            y_imag: T.Tensor((n,), real_dtype),  # type: ignore
        ) -> None:
            # Cooley-Tukey FFT: iterative radix-2 decimation-in-time
            bit_reversal_permutation(x_real, x_imag, y_real, y_imag)

            # Stage 2: Iterative FFT butterfly operations
            # Note: Cannot use T.Pipelined here because each stage depends on
            # the previous stage's output (data dependency prevents overlap)
            for stage in T.serial(log2n):
                butterfly_stage(y_real, y_imag, stage)

        return _fft_main

    return _fft_func


@torch.library.custom_op("top::fft_c2c_wrapped_kernel", mutates_args=())
def _fft_c2c_wrapped_kernel(
    n: int,
    dtype: str,
    block_size: int,
    threads: int,
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fft_c2c_kernel(n, dtype)(block_size, threads)(x_real, x_imag)


@_fft_c2c_wrapped_kernel.register_fake
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
        torch.empty((n,), dtype=real_dtype, device=device)
    )


class FFTC2CKernel(Kernel):
    """
    1D Complex-to-Complex Fast Fourier Transform kernel.

    Implements the Cooley-Tukey radix-2 FFT algorithm with O(N log N) complexity
    using iterative decimation-in-time.

    Args:
        x_real: Real part of input, shape (n,), float32 or float64
        x_imag: Imaginary part of input, shape (n,), float32 or float64

    Computation:
        Performs 1D FFT using Cooley-Tukey radix-2 algorithm:
        1. Bit-reversal permutation to reorder input elements
        2. log₂(N) butterfly stages, each performing N/2 butterfly operations
        Each butterfly combines two complex numbers using twiddle factors exp(-2πi*k/m).

    Reference:
        https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

    Note:
        This implementation requires n to be a power of 2.
        Uses float64 accumulation for numerical stability.
    """

    supported_archs = [75, 80, 86, 89, 90]

    def __init__(self,
                 n: int,
                 dtype: torch.dtype = torch.complex64,
                 tune: bool = False) -> None:
        super().__init__()
        self.n = n
        self.dtype = dtype

        self.kernel = _fft_c2c_kernel(n, self.dtype_str)
        self.init_config(tune=tune)

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "block_size": 256,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # block_size == threads: each thread computes exactly one output element.
        # Vary block size to trade occupancy vs. shared-memory pressure.
        return [
            {"block_size": bs, "threads": bs}
            for bs in [32, 64, 128, 256, 512]
        ]

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 1D DFT on pre-split real and imaginary parts.

        Args:
            x_real: Real part of input, shape (n,), float32 or float64
            x_imag: Imaginary part of input, shape (n,), float32 or float64

        Returns:
            Tuple of (y_real, y_imag) output tensors of shape (n,)
        """
        return _fft_c2c_wrapped_kernel(
            self.n,
            self.dtype_str,
            self.config["block_size"],
            self.config["threads"],
            x_real,
            x_imag,
        )
