from typing import ClassVar, Dict, Mapping, Optional

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
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fft_c2c_one_cta_kernel": FFTC2COneCTAKernel,
        "fft_c2c_decomposed_kernel": FFTC2CDecomposedKernel,
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel mapping for testing
            tune: Whether to enable autotuning (default: False)
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute 1D FFT of complex input.

        Args:
            input: Input tensor of shape (..., n) with complex dtype.

        Returns:
            Output tensor of same shape as input with FFT applied along the
            last dimension.
        """
        n = input.shape[-1]
        if n == 1:
            self.kernel = None
            return input.clone()
        call = FFTC2CCall(n=n, dtype=input.dtype, device=input.device, tune=self.tune)
        self.kernel = self.kernel_for("fft_c2c", (input,), call)
        return self.kernel(input)
