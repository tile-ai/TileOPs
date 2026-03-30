import math
from typing import Dict, Optional

import torch

from tileops.kernels.fft import FFTC2CKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ['FFTC2COp']


class FFTC2COp(Op):
    """
    1D Complex-to-Complex Fast Fourier Transform operation.

    Computes the one-dimensional discrete Fourier transform of complex input.
    This is equivalent to torch.fft.fft.

    Supports batched input: any leading dimensions are flattened into a single
    batch dimension, processed in parallel by the kernel, and reshaped back.

    Uses pre-computed twiddle factor LUT and shared-memory butterfly fusion
    for optimal GPU performance.

    Args:
        n: FFT size (number of complex samples, must be a power of 2)
        dtype: Data type for computation (default: torch.complex64)
        tune: Whether to enable autotuning (default: False)
        kernel_map: Optional custom kernel mapping for testing
    """

    def __init__(self,
                 n: int,
                 dtype: torch.dtype = torch.complex64,
                 tune: bool = False,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.n = n
        self.dtype = dtype
        self._tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[int, Kernel] = {}
        # Default kernel for Op base class compatibility
        self.kernel = self._get_kernel(1)

        # Pre-compute twiddle LUT on CPU and move to GPU once (shared across batch sizes)
        self.twiddle_real, self.twiddle_imag = self._build_lut(n, dtype)

    def _get_kernel(self, batch_size: int) -> Kernel:
        if batch_size not in self._kernel_cache:
            self._kernel_cache[batch_size] = self.kernel_map["fft_c2c_kernel"](
                self.n, batch_size, self.dtype, tune=self._tune,
            )
        return self._kernel_cache[batch_size]

    @staticmethod
    def _build_lut(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-compute the full twiddle factor LUT for all butterfly stages.

        For stage s, half_m = 2^s twiddle factors are stored at offset (half_m - 1):
            LUT[half_m - 1 + k] = exp(-2πi * k / m),  k = 0..half_m-1,  m = 2*half_m

        All angles are computed in float64 for precision, then cast to the
        real component dtype (float32 for complex64, float64 for complex128).
        """
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        log2n = int(math.log2(n))
        lut_size = n - 1

        angles = torch.zeros(lut_size, dtype=torch.float64)
        for s in range(log2n):
            half_m = 1 << s
            m = half_m * 2
            k_vals = torch.arange(half_m, dtype=torch.float64)
            angles[half_m - 1:2 * half_m - 1] = -2.0 * math.pi * k_vals / m

        lut_real = torch.cos(angles).to(real_dtype).cuda()
        lut_imag = torch.sin(angles).to(real_dtype).cuda()
        return lut_real, lut_imag

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fft_c2c_kernel": FFTC2CKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D FFT of complex input.

        Args:
            x: Input tensor of shape (..., n) with complex dtype

        Returns:
            Output tensor of same shape as input with FFT applied along last dimension
        """
        x_real = x.real.contiguous()
        x_imag = x.imag.contiguous()
        original_shape = x.shape

        if x.ndim > 1:
            batch_size = x_real[..., 0].numel()
            x_real = x_real.reshape(batch_size, self.n)
            x_imag = x_imag.reshape(batch_size, self.n)
        else:
            batch_size = 1
            x_real = x_real.unsqueeze(0)
            x_imag = x_imag.unsqueeze(0)

        kernel = self._get_kernel(batch_size)
        y_real, y_imag = kernel(x_real, x_imag,
                                self.twiddle_real, self.twiddle_imag)

        if x.ndim > 1:
            y_real = y_real.reshape(original_shape)
            y_imag = y_imag.reshape(original_shape)
        else:
            y_real = y_real.squeeze(0)
            y_imag = y_imag.squeeze(0)

        return torch.complex(y_real, y_imag)
