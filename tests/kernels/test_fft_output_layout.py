"""Kernel-level coverage for FFT's interleaved output layout.

The public Op converts the internal ``(batch, n, 2)`` buffer to a complex
view, so its layout requires direct kernel coverage.
"""

import pytest
import torch

from tileops.kernels.fft import FFTC2CKernel
from tileops.ops import FFTC2CFwdOp


@pytest.mark.full
@pytest.mark.parametrize(
    ("config", "dtype"),
    [
        pytest.param(
            {"block_size": 128, "threads": 128},
            torch.complex64,
            id="radix8-then-radix2-c64",
        ),
        pytest.param(
            {"block_size": 256, "threads": 256},
            torch.complex64,
            id="radix8-c64",
        ),
        pytest.param(
            {"block_size": 1024, "threads": 512},
            torch.complex64,
            id="radix4-c64",
        ),
        pytest.param(
            {"block_size": 256, "threads": 256},
            torch.complex128,
            id="radix8-c128",
        ),
    ],
)
def test_fft_final_stage_writes_interleaved_output(
    config: dict[str, int],
    dtype: torch.dtype,
) -> None:
    n = 4096
    batch_size = 2
    x = torch.randn(batch_size, n, device="cuda", dtype=dtype)
    lut_real, lut_imag = FFTC2CFwdOp._build_lut(n, dtype, x.device)
    kernel = FFTC2CKernel(n, batch_size, dtype, config=config)

    output_pair = kernel(
        x.real.contiguous(),
        x.imag.contiguous(),
        lut_real,
        lut_imag,
    )

    assert output_pair.shape == (batch_size, n, 2)
    assert output_pair.dtype == (torch.float32 if dtype == torch.complex64 else torch.float64)
    assert output_pair.is_contiguous()
    tolerance = 1e-4 if dtype == torch.complex64 else 1e-8
    torch.testing.assert_close(
        torch.view_as_complex(output_pair),
        torch.fft.fft(x),
        atol=tolerance,
        rtol=tolerance,
    )
