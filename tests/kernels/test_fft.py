"""FFT kernels select the plan their device's record names."""

import pytest
import torch

from tileops.kernels import fft as fft_kernels
from tileops.kernels.fft import FFT_NARROW_PLANS, FFT_PLANS, FFTC2CFourStepKernel


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "arch, table",
    [pytest.param(89, FFT_NARROW_PLANS, id="sm89"), pytest.param(90, FFT_PLANS, id="sm90")],
)
def test_decomposed_kernel_selects_its_devices_record(
    monkeypatch: pytest.MonkeyPatch, arch: int, table: dict
) -> None:
    monkeypatch.setattr(fft_kernels, "get_sm_version", lambda index=None: arch)
    kernel = FFTC2CFourStepKernel(1 << 22, torch.complex64)
    assert kernel.plan is table[1 << 22, "complex64"]
    assert kernel.config["tile"] == kernel.plan.tile
