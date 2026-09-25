from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.fft import FFTC2CCall, FFTC2CDecomposedKernel, FFTC2COneCTAKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["FFTC2CFwdOp"]


class FFTC2CFwdOp(Op):
    """1D Complex-to-Complex Fast Fourier Transform (FFT), equivalent to ``torch.fft.fft``.

    Transforms the last axis; leading dimensions are batched. Every power-of-two
    length from 1 to 2**28 is served in complex64 and complex128 on sm_80 and
    sm_90. sm_86 and sm_89 (99 KB of shared memory per block) refuse 8192
    (complex128), 16384 (complex64), 2**22 through 2**24, and 2**27 (complex128).

    * ``n = 1`` returns a copy of the input.
    * Up to 16384 (8192 at complex128), one launch holds a whole transform.
    * Longer lengths are a four-step decomposition: two launches up to 2**24,
      three above. Each call allocates one intermediate buffer the size of the
      input.

    Relative error ``max|got - ref| / max|ref|`` against a float64 reference is
    at most 9.8e-07 (complex64) and 1.6e-15 (complex128), within 2.2x of cuFFT's.

    Raises:
        ValueError: The input is not a CUDA tensor, is not complex64 or
            complex128, is 0-dimensional, or its last axis is not a power of two
            from 1 through 2**28.
    """

    def __init__(
        self,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            tune: Whether to enable autotuning (default: False)
            kernel_map: Optional custom kernel mapping for testing
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.target = target
        self.n = None
        self.input_shape = None
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "fft_c2c_one_cta_kernel": FFTC2COneCTAKernel,
            "fft_c2c_decomposed_kernel": FFTC2CDecomposedKernel,
        }

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: ``same_as(input)`` — a transform moves no axis."""
        return {"output": tuple(input_shape)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute 1D FFT of complex input.

        Args:
            input: Input tensor of shape (..., n) with complex dtype.

        Returns:
            Output tensor of same shape as input with FFT applied along the
            last dimension.
        """
        x = input
        if not x.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        if x.dtype not in (torch.complex64, torch.complex128):
            raise ValueError(f"input.dtype must be complex64 or complex128, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("input must be at least 1D")
        n = x.shape[-1]
        if n <= 0 or n & (n - 1) != 0:
            raise ValueError(f"FFT size must be a positive power of 2, got {n}")
        if n > 1 << 28:
            raise ValueError(f"FFT size must be at most 2**28, got {n}")
        self.n = n
        self.dtype = x.dtype
        # What the manifest roofline resolves ``input`` through: the batch extent
        # is the call's, not the kernel cache's.
        self.input_shape = tuple(x.shape)
        if n == 1:
            self.kernel = None
            return x.clone()

        # The kernels read the interleaved (real, imag) pair directly.
        x_pair = torch.view_as_real(x.resolve_conj().contiguous()).reshape(x.numel() // n, n, 2)
        call = FFTC2CCall(n=n, dtype=x.dtype, device=x.device, tune=self.tune)
        self.kernel = self.kernel_for("fft_c2c", (input,), call)
        y_pair = self.kernel(x_pair)
        return torch.view_as_complex(y_pair.reshape(*x.shape, 2))
