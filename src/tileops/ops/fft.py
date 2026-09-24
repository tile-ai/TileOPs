import math
from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.fft_c2c import (
    FFTC2CDecomposedKernel,
    FFTC2COneCTAKernel,
    FFTPlan,
)
from tileops.kernels.fft_call_spec import FFTC2CCall
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
    complex128, on sm_80 through sm_90. Which of three shapes a call takes is
    decided by what one thread block can hold:

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
        # The one-launch kernel reads a full-circle table and the pass-2 bases of
        # its own length; a decomposed one reads those per factor plus the
        # four-step primitives below. All of it is data the op owns and caches,
        # not kernels.
        self._circle_cache: Dict[
            tuple[int, torch.dtype, int | None], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # The four-step kernels' tables, in the tuples one call passes.
        self._four_step_cache: Dict[tuple[tuple, torch.dtype, int | None], tuple] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "fft_c2c_one_cta_kernel": FFTC2COneCTAKernel,
            "fft_c2c_decomposed_kernel": FFTC2CDecomposedKernel,
        }

    def _get_circle_lut(
        self, n: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One length's pass tables: the full circle, and its pass-2 bases.

        ``circle[k] = exp(-2 pi i k / n)`` interleaved, and the pass-2 bases used
        by the three-pass plans (a one-row compatibility tensor for smaller plans).
        Built in float64 and cast once, so the table itself is not the error term.
        """
        key = (n, dtype, device.index)
        if key not in self._circle_cache:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            k = torch.arange(n, dtype=torch.float64)
            ang = -2.0 * math.pi * k / n
            circle = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
            # Compatibility input for the packed builders. Tiny and packed kernels
            # ignore it, while every three-pass plan reads n / (16*16) values.
            m = torch.arange(max(1, n // 256), dtype=torch.float64)
            ang2 = -2.0 * math.pi * m / max(1, n // 16)
            base2 = torch.stack([torch.cos(ang2), torch.sin(ang2)], dim=1)
            self._circle_cache[key] = (
                circle.to(real_dtype).to(device),
                base2.to(real_dtype).to(device),
            )
        return self._circle_cache[key]

    @staticmethod
    def _quadrant_twiddle(idx: torch.Tensor, n: int) -> torch.Tensor:
        """``exp(-2 pi i m / n)`` for each m in *idx*, in float64, one quadrant of trig.

        Evaluating the angle straight from m puts up to 2*pi into it, and the
        rounding of that angle is what the cosine carries out: the table's worst
        entry is then 6.7e-16 off. Taking the trigonometry over one quadrant and
        turning the result by i**q -- an exact swap and sign flip -- caps the angle
        at pi/2 and the table at 2.0e-16. Only the four-step factor reads this; the
        pass twiddles read at most a few degrees into their own circle, where the
        two constructions agree.

        Args:
            idx: Integer exponents, already reduced mod n.
            n: The circle they are exponents on.
        """
        quadrant = idx // (n // 4)
        ang = -2.0 * math.pi * (idx % (n // 4)).to(torch.float64) / n
        re, im = torch.cos(ang), torch.sin(ang)
        # (re + i*im) * (-i)**q, written as the swap and sign flip it is.
        out_re = torch.where(
            quadrant == 0, re, torch.where(quadrant == 1, im, torch.where(quadrant == 2, -re, -im))
        )
        out_im = torch.where(
            quadrant == 0, im, torch.where(quadrant == 1, -re, torch.where(quadrant == 2, -im, re))
        )
        return torch.stack([out_re, out_im], dim=1)

    def _get_four_step_twiddle(
        self, plan: FFTPlan, level: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """One column kernel's per-column twiddle primitives, ``[R, m2, 2]``.

        Built on demand, not cached here: ``_get_four_step_tables`` is what the
        call path reads, and it caches the whole bundle.

        Column kernel *level* splits the length ``m`` it is handed a row of into
        ``m1 * m2``, so its four-step factor is ``W_m^(j_a*k_b)``, and its last
        pass emits ``k_b = lane + (m1/16)*u`` with ``lane`` a runtime index and
        ``u`` a compile-time one over ``[0, 16)``. So the factor is a per-lane base
        times one power for each half of ``u``, and row *r* of this table is
        ``W_m^(e_r * j_a)`` for the exponents that kernel was emitted against --
        ``plan.twiddle_exp[level]``, which names them so the two sides cannot
        drift. Each entry is one exactly-rounded lookup rather than a step of a
        rotor, so the table is not the error term.

        Args:
            plan: The record the kernel was built from.
            level: Which column kernel of that plan, counted from the outermost.
            dtype: The call's complex dtype; the table is stored as its real half.
            device: Where the table is wanted.
        """
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        m = math.prod(plan.factors[level:])
        m2 = math.prod(plan.factors[level + 1 :])
        ja = torch.arange(m2, dtype=torch.int64)
        rows = [self._quadrant_twiddle((e * ja) % m, m) for e in plan.twiddle_exp[level]]
        return torch.stack(rows, dim=0).to(real_dtype).to(device).contiguous()

    def _get_four_step_tables(
        self, plan: FFTPlan, dtype: torch.dtype, device: torch.device
    ) -> tuple:
        """The three tuples one four-step call is handed, built once per plan.

        Assembled here rather than looked up one table at a time because this is
        the per-call path: a three-factor plan wants seven tensors in three
        tuples, and rebuilding those tuples on every call measured 4 us of host
        time against a 40 us budget.

        Args:
            plan: The record the kernel was built from.
            dtype: The call's complex dtype; the tables are its real half.
            device: Where the tables are wanted.

        Returns:
            ``(w1, w2, twlut)`` -- per kernel its pass-1 and pass-2 tables, then
            one four-step table per column kernel.
        """
        key = (plan.factors, dtype, device.index)
        if key not in self._four_step_cache:
            luts = [self._get_circle_lut(f, dtype, device) for f in plan.factors]
            self._four_step_cache[key] = (
                tuple(lut[0] for lut in luts),
                tuple(lut[1] for lut in luts),
                tuple(
                    self._get_four_step_twiddle(plan, level, dtype, device)
                    for level in range(len(plan.factors) - 1)
                ),
            )
        return self._four_step_cache[key]

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
        if n == 1:
            self.n = n
            self.dtype = x.dtype
            self.input_shape = tuple(x.shape)
            self.kernel = None
            return x.clone()

        # A view, not a copy, when x is already contiguous: the kernel reads the
        # interleaved (real, imag) pair directly, so no separate real/imag
        # planes are built on the host.
        x_pair = torch.view_as_real(x.contiguous())
        original_shape = x.shape

        # Flatten all batch dimensions into a single batch dimension. Counted
        # from the element count, not from a view built to be counted: this sits
        # on the per-call path.
        batch_size = x.numel() // n
        x_pair = x_pair.reshape(batch_size, n, 2)

        self.n = n
        self.dtype = x.dtype
        # What the manifest roofline resolves ``input`` through: the batch extent
        # is the call's, not the kernel cache's.
        self.input_shape = tuple(original_shape)
        call = FFTC2CCall(
            n=n,
            dtype=x.dtype,
            device_index=x.device.index,
            device=x.device,
            tune=self.tune,
        )
        kernel = self.kernel_for("fft_c2c", (input,), call)
        self.kernel = kernel
        # The two implementations read different tables, so each is handed its own.
        if isinstance(kernel, FFTC2CDecomposedKernel):
            # The plan is the kernel's, not the length's: the two dtypes of one
            # length are not always decomposed the same way.
            w1, w2, twlut = self._get_four_step_tables(kernel.plan, x.dtype, x.device)
            # The intermediate buffer is allocated per call and dropped with the
            # call, not held on the kernel: one cached on the instance would pin
            # n*batch*itemsize of device memory and make a single kernel object
            # unsafe for two overlapping streams.
            t_pair = torch.empty_like(x_pair)
            y_pair = kernel(x_pair, t_pair, w1, w2, twlut)
        else:
            circle, base2 = self._get_circle_lut(n, x.dtype, x.device)
            y_pair = kernel(x_pair, circle, base2)

        # The kernel writes the final butterfly directly in interleaved layout;
        # view_as_complex is metadata-only and launches no packing kernel.
        return torch.view_as_complex(y_pair.reshape(*original_shape, 2))
