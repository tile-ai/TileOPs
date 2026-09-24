"""Kernel-level coverage for FFT's interleaved output layout and real dtype."""

import pytest
import torch

from tileops.kernels.fft_c2c import FFTC2COneCTAKernel
from tileops.ops import FFTC2CFwdOp


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
    circle, base2 = FFTC2CFwdOp()._get_circle_lut(n, dtype, x.device)

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
