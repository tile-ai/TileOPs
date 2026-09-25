from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.fft import FFTC2CCall, FFTC2CDecomposedKernel, FFTC2COneCTAKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["FFTC2CFwdOp"]


class FFTC2CFwdOp(Op):
    """1D Complex-to-Complex Fast Fourier Transform (FFT) operation.

    Computes the one-dimensional discrete Fourier transform of complex input
    along the last axis, equivalently to ``torch.fft.fft``. Any leading
    dimensions are flattened into one batch dimension, transformed in parallel,
    and reshaped back.

    Every power-of-two length from 1 to 2**28 is served, in complex64 and
    complex128, on sm_80 and sm_90. sm_86 and sm_89 give a block 99 KB of shared
    memory and serve every length except 8192 (complex128), 16384 (complex64),
    2**22 through 2**24, and 2**27 (complex128). Which of three shapes a call
    takes is decided by what one thread block can hold:

    * ``n = 1`` is the identity and launches nothing; the input is returned as
      a copy, since a transform is out of place.
    * ``2`` to ``16384`` is one launch, the whole transform register-resident
      that keeps the whole length in one block. The exception is 16384 at
      complex128, whose four-pass plan needs more shared memory than a block
      can be given.
    * ``32768`` to ``2**24``, plus 16384 at complex128, is two launches: the
      four-step decomposition into two factors, a column pass and a row pass.
    * ``2**25`` to ``2**28`` is three launches, the same decomposition applied
      twice.

    A decomposed length costs one intermediate buffer whatever its factor count.
    Such a call allocates and frees ``input.numel() * input.element_size()``
    bytes of device memory on top of its result; a one-launch length allocates
    only the result.

    Accuracy, measured across the whole range against a CPU float64 reference:
    the relative error ``max|got - ref| / max|ref|`` stays at or below 9.8e-07
    for complex64 and 1.6e-15 for complex128, and at most 2.2x the error cuFFT
    shows at the same length. Against ``torch.fft.fft`` on unit-variance input,
    results agree to ``atol=rtol=1e-4`` for complex64 and ``1e-8`` for
    complex128 through n = 16384. Past that a transform's outputs carry sqrt(n)
    times the input's scale, so the absolute half of a fixed tolerance has to
    grow with sqrt(n) to mean the same thing; the relative half does not.

    Raises:
        ValueError: The input is not a CUDA tensor, is not complex64 or
            complex128, is 0-dimensional, or its last axis is not a supported
            power of two from 1 through 2**28.
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
