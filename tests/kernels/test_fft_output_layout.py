"""Kernel-level coverage for FFT's interleaved output layout.

The public Op turns the internal ``(batch, n, 2)`` buffer into a complex view,
which is metadata-only and would not catch a buffer that is the right numbers in
the wrong layout. So the layout -- shape, real dtype, contiguity -- is asserted
here, against the kernel, one case per real width.
"""

import math

import pytest
import torch

from tileops.kernels.fft_c2c import FFTC2COneCTAKernel


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.complex64, id="complex64"),
        pytest.param(torch.complex128, id="complex128"),
    ),
)
def test_one_cta_kernel_writes_interleaved_output(dtype: torch.dtype) -> None:
    n = 4096
    batch_size = 2
    x = torch.randn(batch_size, n, device="cuda", dtype=dtype)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    k = torch.arange(n, dtype=torch.float64)
    circle = (
        torch.stack([torch.cos(-2.0 * math.pi * k / n), torch.sin(-2.0 * math.pi * k / n)], dim=1)
        .to(real_dtype)
        .to(x.device)
    )
    r3 = max(1, n // 256)
    m = torch.arange(r3, dtype=torch.float64)
    base2 = (
        torch.stack(
            [
                torch.cos(-2.0 * math.pi * m / (n // 16)),
                torch.sin(-2.0 * math.pi * m / (n // 16)),
            ],
            dim=1,
        )
        .to(real_dtype)
        .to(x.device)
    )

    output_pair = FFTC2COneCTAKernel(n, dtype)(torch.view_as_real(x.contiguous()), circle, base2)

    assert output_pair.shape == (*x.shape, 2)
    assert output_pair.dtype == real_dtype
    assert output_pair.is_contiguous()
    tolerance = 1e-4 if dtype == torch.complex64 else 1e-8
    torch.testing.assert_close(
        torch.view_as_complex(output_pair),
        torch.fft.fft(x),
        atol=tolerance,
        rtol=tolerance,
    )
